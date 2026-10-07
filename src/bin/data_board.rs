//! BENCH DATA board: receive TUM1 image / IMU1 inertial records over SoftAP
//! TCP and relay them to the VO board over 4-wire SPI.
//!
//! Wired link = ESP-IDF SPI slave-HD driver, segment mode. The slave hardware
//! follows the master's command mask, so the VO master selects the line mode
//! per transaction: QIO `RDDMA|0xA0` (addr+data 4-bit) for bench rate, plain
//! `RDDMA` (1-bit) as bring-up fallback. No ENQPI/QPI state is used.
//!
//! Control is VO-driven (DATA never pushes). One input buffer, three states:
//!   FILLING:   STATUS=BUSY, DATA blocking on TCP. Laptop stalls on its
//!              unacked send = backpressure; every frame delivered once, in order.
//!   READY:     frame parked, LEN/SEQ/CRC valid. DATA polls CMD every ~50 ms,
//!              reads TCP nothing. VO writes CMD="INIT" only on stable READY.
//!   STREAMING: lock held. DATA resets READY_N=0 then runs CHUNK_BYTES TX
//!              queue/get cycles; VO gates every RDDMA on READY_N > consumed
//!              (double-read), closes each chunk with CMD8.
//! On streaming timeout DATA keeps the frame, bumps XFER, returns to READY.
//! VO discovers frames by XFER change and identifies them by SEQ, so a VO
//! reboot or laptop resend can never skip or mix frames (VO must also skip
//! any frame with SEQ == last fully received SEQ).
//!
//! Shared regs, all u32 LE (slave-HD direct buffer, 64 B):
//!   0:READY(0xEE) 4:MAX 8:LEN 12:SEQ 16:CRC 20:READY_N(per transfer)
//!   24:STATUS(0 BUSY/1 READY/2 STREAMING) 28:CMD(VO writes "INIT")
//!   32:XFER (bumps every time a frame (re-)becomes READY)
//! DATA acts only on stable reads (two consecutive equal polls); the frame
//! CRC backstops any torn snapshot. TCP acks `b"ACK1" | seq | crc` on success
//! only — senders must wait for it (long/no timeout) and resend on drop.
//! Both record types share one sender-assigned SEQ space (payload bytes 4..8)
//! so VO dedups uniformly; IMU batches keep it small so stop-and-wait holds
//! 200 Hz worth of samples at ~20 records/s.
//!
//! Record grammars (all ints LE, `u32 n` length prefix = bytes after it):
//!   TUM1 image:  n | "TUM1" | u32 seq | u64 t_us | u16 w | u16 h | w*h gray
//!   IMU1 batch:  n | "IMU1" | u32 seq | u64 t0_us | u16 nsamp | u16 dt_us
//!                | nsamp x 6xf32 (ax,ay,az,wx,wy,wz, m/s^2 + rad/s).
//!   IMU1 limits: 1 <= nsamp <= 64, dt_us >= 100 (u16; EuRoC nominal 5000).
//!
//! ACK timing: IMU1 is ACKed as soon as it is parked (the VO side may be busy
//! with a rare image, and a lost 50 ms IMU batch is harmless), so the sender
//! keeps streaming inertial data while VO computes. A TUM1 image is not ACKed
//! until VO has consumed it — losing one costs 5 s of drift. Both types are
//! still parked one at a time, so the FIFO is the socket buffer.

use std::io::{ErrorKind, Read, Write};
use std::time::Duration;

use std::net::{Ipv4Addr, TcpListener, TcpStream};

use esp_idf_hal::peripherals::Peripherals;
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys::EspError;
use esp_idf_svc::wifi::{
    AccessPointConfiguration, AuthMethod, BlockingWifi, Configuration, EspWifi,
};
use esp_idf_sys::{esp_err_t, spi_bus_config_t, spi_host_device_t_SPI2_HOST};

// --- slave-HD FFI (esp-idf-sys bindgens spi_slave.h only, not spi_slave_hd.h;
// --- signatures/structs copied from esp_driver_spi/include/driver/spi_slave_hd.h).
mod slave_hd {
    use std::ffi::c_void;

    use esp_idf_sys::{esp_err_t, spi_bus_config_t};

    pub const CHAN_TX: u32 = 0;

    #[repr(C)]
    pub struct Data {
        pub data: *mut u8,
        pub len: u32,
        pub trans_len: u32,
        pub flags: u32,
        pub arg: *mut c_void,
    }

    // 7 ISR callbacks + arg, all NULL ( polled queue/get only ).
    #[repr(C)]
    pub struct Callbacks {
        pub cb: [*const c_void; 7],
        pub arg: *mut c_void,
    }

    #[repr(C)]
    pub struct Slot {
        pub mode: u8,
        pub _pad: [u8; 3],
        pub spics_io_num: u32,
        pub flags: u32,
        pub command_bits: u32,
        pub address_bits: u32,
        pub dummy_bits: u32,
        pub queue_size: u32,
        pub dma_chan: i32,
        pub cb_config: Callbacks,
    }

    extern "C" {
        pub fn spi_slave_hd_init(
            host: u32,
            bus: *const spi_bus_config_t,
            cfg: *const Slot,
        ) -> esp_err_t;
        pub fn spi_slave_hd_queue_trans(
            host: u32,
            chan: u32,
            trans: *mut Data,
            timeout: u32,
        ) -> esp_err_t;
        pub fn spi_slave_hd_get_trans_res(
            host: u32,
            chan: u32,
            out_trans: *mut *mut Data,
            timeout: u32,
        ) -> esp_err_t;
        pub fn spi_slave_hd_write_buffer(host: u32, addr: i32, data: *const u8, len: u32);
        pub fn spi_slave_hd_read_buffer(host: u32, addr: i32, data: *mut u8, len: u32);
    }
}

const AP_SSID: &str = "vo-box-data";
const AP_PASS: &str = "vobox1234";
const TCP_PORT: u16 = 5000;
const AP_IP_FALLBACK: Ipv4Addr = Ipv4Addr::new(192, 168, 71, 1);

const TUM1: &[u8; 4] = b"TUM1";
const IMU1: &[u8; 4] = b"IMU1";
const ACK1: &[u8; 4] = b"ACK1";
// IMU1 payload header: magic(4) + seq(4) + t0_us(8) + nsamp(2) + dt_us(2).
const IMU_HDR: usize = 20;
const IMU_MAX_SAMP: usize = 64;
const INIT_MAGIC: [u8; 4] = *b"INIT";
const MAX_FRAME: usize = 640 * 480 + 20;

// DATA MCU -> VO MCU pin assignment from AGENTS.md (all 4 data lines live now).
const QSPI_CS: i32 = 1;
const QSPI_CLK: i32 = 14;
const QSPI_IO0: i32 = 21;
const QSPI_IO1: i32 = 47;
const QSPI_IO2: i32 = 41;
const QSPI_IO3: i32 = 42;

// One TX queue/get per chunk; VO derives the count from LEN + this.
// Must match vo_replay.rs. Multiple of 4 (slave-HD DMA rule).
const CHUNK_BYTES: usize = 4096;

// Shared-reg map (VO reads via RDBUF, writes CMD via WRBUF).
const REG_READY: i32 = 0;
const REG_MAX: i32 = 4;
const REG_LEN: i32 = 8;
const REG_SEQ: i32 = 12;
const REG_CRC: i32 = 16;
const REG_READY_N: i32 = 20;
const REG_STATUS: i32 = 24;
const REG_CMD: i32 = 28;
const REG_XFER: i32 = 32;
const READY_FLAG: u32 = 0xEE;

const STATUS_BUSY: u32 = 0;
const STATUS_READY: u32 = 1;
const STATUS_STREAMING: u32 = 2;

// FreeRTOS ticks (default 100 Hz). GET covers one chunk's RDDMA segments +
// CMD8; VO never localizes mid-transfer, so 5 s is already generous.
const QUEUE_TIMEOUT: u32 = 100;
const GET_TIMEOUT: u32 = 500;
// 2 ms: IMU batches arrive at ~20/s as single chunks, so the INIT handshake
// must not pace the link (50 ms would cap ~20 records/s before transfer).
// DATA-board power is not measured, so spinning here is free.
const CMD_POLL_MS: u32 = 2;

const HOST: u32 = spi_host_device_t_SPI2_HOST as u32;

fn check(result: esp_err_t) -> Result<(), EspError> {
    EspError::convert(result)
}

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    if let Err(e) = run() {
        log::error!("data board stopped: {e}");
        panic!("data board: {e}");
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let peripherals = Peripherals::take()?;
    let sysloop = EspSystemEventLoop::take()?;
    let nvs = EspDefaultNvsPartition::take()?;
    let mut wifi = BlockingWifi::wrap(
        EspWifi::new(peripherals.modem, sysloop.clone(), Some(nvs))?,
        sysloop,
    )?;
    wifi.set_configuration(&Configuration::AccessPoint(AccessPointConfiguration {
        ssid: AP_SSID.try_into().unwrap(),
        password: AP_PASS.try_into().unwrap(),
        auth_method: AuthMethod::WPA2Personal,
        ..AccessPointConfiguration::default()
    }))?;
    wifi.start()?;

    let ap_ip = {
        let mut ip = None;
        for _ in 0..100 {
            let netif = wifi.wifi().ap_netif();
            if netif.is_up()? {
                ip = Some(netif.get_ip_info()?.ip);
                break;
            }
            esp_idf_hal::delay::FreeRtos::delay_ms(100);
        }
        ip.unwrap_or(AP_IP_FALLBACK)
    };
    log::info!("DATA SoftAP {AP_SSID} up at {ap_ip}:{TCP_PORT}");

    init_wired_link()?;
    // Single DMA staging buffer, reused for every chunk of every record.
    let mut staging = DmaBuf::alloc(CHUNK_BYTES)?;
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, TCP_PORT))?;
    log::info!("DATA waiting for TUM1/IMU1 records; SPI2-HD slave CS={QSPI_CS} CLK={QSPI_CLK} IO0..3={QSPI_IO0},{QSPI_IO1},{QSPI_IO2},{QSPI_IO3}");

    loop {
        let (mut stream, peer) = match listener.accept() {
            Ok(x) => x,
            Err(e) => {
                log::warn!("DATA accept failed: {e}");
                esp_idf_hal::delay::FreeRtos::delay_ms(500);
                continue;
            }
        };
        log::info!("laptop connected: {peer}");
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
        // At most one parked record; acked_at_park = IMU already ACKed on park.
        let mut parked: Option<Vec<u8>> = None;
        let mut acked_at_park = false;
        let mut xfer: u32 = 0;
        loop {
            if parked.is_none() {
                // ---- FILLING: laptop backpressures here until VO takes records.
                write_reg(REG_STATUS, STATUS_BUSY);
                match receive_record(&mut stream) {
                    Ok(Some(frame)) => parked = Some(frame),
                    Ok(None) => {
                        log::info!("laptop disconnected");
                        break;
                    }
                    Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                        log::info!("laptop closed between frames");
                        break;
                    }
                    Err(e) => {
                        log::warn!("record receive failed: {e}");
                        break;
                    }
                }
                // Publish LEN+CRC before SEQ; XFER+READY last.
                if let Some(frame) = parked.as_ref() {
                    let kind = kind_str(&frame[4..8]);
                    let seq = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]);
                    let crc = crc32(frame);
                    write_reg(REG_LEN, frame.len() as u32);
                    write_reg(REG_CRC, crc);
                    write_reg(REG_SEQ, seq);
                    xfer += 1;
                    write_reg(REG_XFER, xfer);
                    write_reg(REG_STATUS, STATUS_READY);
                    log::info!("{kind} seq={seq} parked xfer={xfer} ({} B)", frame.len());
                    // IMU is acked now so the sender is never gated by VO; the
                    // record still waits here until VO streams it.
                    acked_at_park = false;
                    if &frame[4..8] == IMU1 {
                        if let Err(e) = stream.write_all(&ack_record(seq, crc)) {
                            log::warn!("imu ack write failed: {e}");
                            break;
                        }
                        acked_at_park = true;
                    }
                }
            } else {
                // ---- READY: wait for VO's INIT, then STREAMING under lock.
                wait_init();
                let frame = parked.as_ref().unwrap();
                let kind = kind_str(&frame[4..8]);
                let seq = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]);
                write_reg(REG_STATUS, STATUS_STREAMING);
                write_reg(REG_READY_N, 0);
                match stream_chunks(frame, &mut staging) {
                    Ok(crc) => {
                        log::info!("{kind} seq={seq} streamed crc={crc:08x}");
                        if !acked_at_park {
                            if let Err(e) = stream.write_all(&ack_record(seq, crc)) {
                                // VO already consumed it; the sender resends on
                                // reconnect and VO dedups by SEQ.
                                log::warn!("ack write failed: {e}");
                                break;
                            }
                        }
                        parked = None;
                    }
                    Err(e) => {
                        // Keep the record, re-present under a new XFER so any
                        // VO (rebooted or timed out) rediscovers it cleanly.
                        log::error!("{kind} stream seq={seq} failed: {e}; re-presenting");
                        xfer += 1;
                        write_reg(REG_XFER, xfer);
                        write_reg(REG_STATUS, STATUS_READY);
                    }
                }
            }
        }
    }
}

fn ack_record(seq: u32, crc: u32) -> [u8; 12] {
    let mut ack = [0u8; 12];
    ack[..4].copy_from_slice(ACK1);
    ack[4..8].copy_from_slice(&seq.to_le_bytes());
    ack[8..12].copy_from_slice(&crc.to_le_bytes());
    ack
}

fn kind_str(magic: &[u8]) -> &'static str {
    if magic == TUM1 { "TUM1" } else if magic == IMU1 { "IMU1" } else { "????" }
}

/// One length-prefixed record of either type (the stored `record` keeps its
/// 4-byte length prefix, so payload SEQ sits at record[8..12] for both).
fn receive_record(stream: &mut TcpStream) -> std::io::Result<Option<Vec<u8>>> {
    let mut len_bytes = [0u8; 4];
    match stream.read_exact(&mut len_bytes) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let payload_len = u32::from_le_bytes(len_bytes) as usize;
    if !(24..=MAX_FRAME).contains(&payload_len) {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!("bad record payload length {payload_len}"),
        ));
    }

    let mut payload = vec![0u8; payload_len];
    stream.read_exact(&mut payload)?;
    match &payload[..4] {
        m if m == TUM1 => {
            let w = u16::from_le_bytes([payload[16], payload[17]]) as usize;
            let h = u16::from_le_bytes([payload[18], payload[19]]) as usize;
            if w != 640 || h != 480 || payload_len != 20 + w * h {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("bad TUM1 dimensions {w}x{h}"),
                ));
            }
        }
        m if m == IMU1 => {
            let nsamp = u16::from_le_bytes([payload[16], payload[17]]) as usize;
            let dt_us = u16::from_le_bytes([payload[18], payload[19]]) as usize;
            // dt_us is u16 (EuRoC nominal 5000); reject degenerate rates.
            if !(1..=IMU_MAX_SAMP).contains(&nsamp)
                || dt_us < 100
                || payload_len != IMU_HDR + 24 * nsamp
            {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("bad IMU1 header nsamp={nsamp} dt_us={dt_us}"),
                ));
            }
        }
        _ => {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "record has unknown magic",
            ));
        }
    }

    let mut record = Vec::with_capacity(payload.len() + 4);
    record.extend_from_slice(&len_bytes);
    record.extend_from_slice(&payload);
    Ok(Some(record))
}

fn init_wired_link() -> Result<(), EspError> {
    let mut bus = spi_bus_config_t::default();
    bus.sclk_io_num = QSPI_CLK;
    bus.__bindgen_anon_1.mosi_io_num = QSPI_IO0;
    bus.__bindgen_anon_2.miso_io_num = QSPI_IO1;
    // Unlike the full-duplex slave driver, slave-HD uses all four lines.
    bus.__bindgen_anon_3.quadwp_io_num = QSPI_IO2;
    bus.__bindgen_anon_4.quadhd_io_num = QSPI_IO3;
    // flags 0: spi_common auto-derives QUAD from the WP/HD pins above.
    bus.max_transfer_sz = CHUNK_BYTES as i32;

    let slot = slave_hd::Slot {
        mode: 0, // must match the VO master
        _pad: [0; 3],
        spics_io_num: QSPI_CS as u32,
        flags: 0,
        command_bits: 8,
        address_bits: 8,
        dummy_bits: 8,
        queue_size: 2,
        dma_chan: esp_idf_sys::spi_common_dma_t_SPI_DMA_CH_AUTO as i32,
        cb_config: slave_hd::Callbacks {
            cb: [std::ptr::null(); 7],
            arg: std::ptr::null_mut(),
        },
    };
    check(unsafe { slave_hd::spi_slave_hd_init(HOST, &bus, &slot) })?;
    write_reg(REG_READY, READY_FLAG);
    write_reg(REG_MAX, (MAX_FRAME + 4) as u32);
    write_reg(REG_STATUS, STATUS_BUSY);
    write_reg(REG_CMD, 0);
    write_reg(REG_XFER, 0);
    Ok(())
}

fn write_reg(addr: i32, val: u32) {
    let bytes = val.to_le_bytes();
    unsafe { slave_hd::spi_slave_hd_write_buffer(HOST, addr, bytes.as_ptr(), 4) };
}

fn read_reg(addr: i32) -> u32 {
    let mut bytes = [0u8; 4];
    unsafe { slave_hd::spi_slave_hd_read_buffer(HOST, addr, bytes.as_mut_ptr(), 4) };
    u32::from_le_bytes(bytes)
}

// Blocks until VO writes CMD="INIT" (two stable polls), then clears it to
// acknowledge. Clears any other stable nonzero value as stale garbage.
fn wait_init() {
    let mut prev = read_reg(REG_CMD);
    loop {
        esp_idf_hal::delay::FreeRtos::delay_ms(CMD_POLL_MS);
        let cur = read_reg(REG_CMD);
        if cur == prev {
            if cur == u32::from_le_bytes(INIT_MAGIC) {
                write_reg(REG_CMD, 0);
                return;
            }
            if cur != 0 {
                write_reg(REG_CMD, 0);
            }
        }
        prev = cur;
    }
}

// Streams one parked frame; READY_N counts chunks from 0 for this transfer.
// Returns the frame CRC. Fails only on VO timeout — caller keeps the frame.
fn stream_chunks(record: &[u8], staging: &mut DmaBuf) -> Result<u32, EspError> {
    let crc = crc32(record);
    let mut ready_n: u32 = 0;
    let stage = staging.as_slice_mut();
    for chunk in record.chunks(CHUNK_BYTES) {
        stage[..chunk.len()].copy_from_slice(chunk);
        let mut trans = slave_hd::Data {
            data: stage.as_mut_ptr(),
            len: chunk.len() as u32,
            trans_len: 0,
            flags: 0, // staging is DMA-capable, no bounce needed
            arg: std::ptr::null_mut(),
        };
        check(unsafe {
            slave_hd::spi_slave_hd_queue_trans(HOST, slave_hd::CHAN_TX, &mut trans, QUEUE_TIMEOUT)
        })?;
        // Bump only after the buffer is accepted: VO gates each RDDMA on this.
        ready_n += 1;
        write_reg(REG_READY_N, ready_n);
        let mut done: *mut slave_hd::Data = std::ptr::null_mut();
        check(unsafe {
            slave_hd::spi_slave_hd_get_trans_res(
                HOST,
                slave_hd::CHAN_TX,
                &mut done,
                GET_TIMEOUT,
            )
        })?;
    }
    Ok(crc)
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// DMA-capable staging buffer (slave-HD has no bounce flag set, so the buffer
// itself must qualify; a Rust Vec may live in PSRAM).
struct DmaBuf {
    ptr: *mut u8,
    len: usize,
}

const MALLOC_CAP_8BIT: u32 = 1 << 2;
const MALLOC_CAP_DMA: u32 = 1 << 3;

extern "C" {
    fn heap_caps_malloc(size: usize, caps: u32) -> *mut std::ffi::c_void;
    fn heap_caps_free(ptr: *mut std::ffi::c_void);
}

impl DmaBuf {
    fn alloc(len: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ptr = unsafe { heap_caps_malloc(len, MALLOC_CAP_8BIT | MALLOC_CAP_DMA) as *mut u8 };
        if ptr.is_null() {
            return Err(format!("DMA alloc {len} B failed").into());
        }
        Ok(Self { ptr, len })
    }

    fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for DmaBuf {
    fn drop(&mut self) {
        unsafe { heap_caps_free(self.ptr as *mut std::ffi::c_void) };
    }
}
