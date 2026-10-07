//! BENCH VO board: always-on bench mode. No camera, no WiFi, no modes.
//!
//! The localization map is baked into flash at build time
//! (`VO_MAP_TXT=<path-to-map.txt> cargo run --bin vo_replay`) and parsed
//! into PSRAM once at boot. The modem is never started and the camera is
//! never touched, so power draw is minimal/comparable during the timed run.
//! To run the map -> localize lifecycle instead, flash `main.rs` — switching
//! modes always means reflashing.
//!
//! Two tasks, one per core (see `run`):
//!   `link_task` (main task, core 0): owns SPI2 + the EKF. Polls DATA,
//!      fetches records, dispatches by magic:
//!      - IMU1 batch: split-rate fuse (nominal per sample, covariance every
//!        10). Runs continuously, including while `vo_task` computes.
//!      - TUM1 image: snapshot the EKF at the record's `t_us`, hand the gray
//!        frame to `vo_task`. The EKF keeps predicting live while VO runs.
//!      - when `vo_task` posts a fix: roll back to the `t_img` snapshot,
//!        `Ekf::correct_pose`, then re-fuse the buffered span — VO's ~1.3 s
//!        latency never enters the fix.
//!   `vo_task` (core 1, std::thread): blocks on fetched images, runs the
//!      BENCH_VI hybrid retrieval (see below), then match + PnP RANSAC,
//!      posts the result back. Owns the Embedder + all VO scratch and never
//!      touches SPI.
//!   Retrieval (BENCH_VI):
//!     branch 1 (strong EKF prior, `prior_from_ekf` Some): pick the top-3 map
//!       frames by attitude gate + prior-frustum projection, then brute-force
//!       Hamming-match the union of their in-frustum points. No calc8 embedding
//!       is computed on this path.
//!     branch 2 (weak/absent prior, None): calc8 embedding top-3 cosine frames,
//!       brute-force match each, keep the PnP with the most inliers.
//!   Between fixes the EKF alone drives the pose; `VO_IMU` lines show it live
//!   at ~5 Hz. Time gaps > 500 ms re-anchor the IMU clock.
//!
//! Wired link (VO = master, DATA = slave-HD segment mode, mode 0, 10 MHz):
//!   pins CS=1 CLK=14 IO0=21 IO1=47 IO2=41 IO3=42 (mirrors DATA 1:1).
//!   No ENQPI state. QIO (`RDDMA|0xA0`, addr+data 4-bit) with a 1-bit
//!   fallback: flip `USE_QIO`. The slave follows the master's command mask,
//!   so no slave-side change is needed to switch.
//!   Master transactions mirror `essl_spi.c` field-for-field (device
//!   command/address/dummy = 8/8/8, per-transaction `dummy_bits` = 8,
//!   `SPI_TRANS_VARIABLE_DUMMY`; RDDMA segs carry no addr phase; each chunk
//!   closes with CMD8 = INT0). `spi_master.h` IS in esp-idf-sys bindings, so
//!   no extra component is needed (unlike the slave-HD side).
//!
//! Shared regs, all u32 LE (see `data_board.rs`):
//!   0:READY(0xEE) 4:MAX 8:LEN 12:SEQ 16:CRC 20:READY_N 24:STATUS(0/1/2)
//!   28:CMD 32:XFER. VO discovers frames by XFER change, IDs by SEQ, and
//!   skips re-localizing SEQ == last-done (the duplicate is still drained so
//!   DATA advances instead of deadlocking on its INIT wait).
//!
//! Record layouts (all ints LE; `u32 n` prefix = bytes after it):
//!   TUM1 image: n | "TUM1" | u32 seq | u64 t_us | u16 w | u16 h | w*h gray
//!   IMU1 batch: n | "IMU1" | u32 seq | u64 t0_us | u16 nsamp | u16 dt_us
//!               | nsamp x 6xf32 (ax,ay,az,wx,wy,wz).
//! Commands the shared regs exactly like `data_board.rs` (see its docs).
//!
//! Serial log (USB-UART only; WiFi is off so the usbipd console caveat does
//! not apply beyond flashing):
//!   `VO_POSE <n> seq=<s> xfer=<x> mode=<prior|emb> kf=<k> surv=<s> inf=<i>
//!    cos1=<a> cos2=<b> feats=<f> matches=<m> inliers=<i> reproj=<px>
//!    roll=<d> pitch=<d> yaw=<d> center=(<x>,<y>,<z>) t=<dataset-s> spi=<ms>
//!    extract=<us> embed=<us> match=<us> pnp=<us> ekf t=<dataset-s>
//!    p=(<x>,<y>,<z>) estd=<m> corr=<res> imu=<k>`
//!   (applied at t_img via snapshot + rollback + buffered-IMU re-fuse)
//!   `VO_FAIL <n> ...` same tail, no corr; `VO_IMG seq=.. t=..` on image
//!   arrival; `VO_IMU t=<s> p=(..) estd=.. rpy=..` live during IMU-only
//!   stretches (~5 Hz); `VO_DROP seq=<s> <reason>` on transfer/CRC problems.
//!   Euler = ZYX deg; center = `-R^T t` (COLMAP world frame).
//!
//! Build/run: `VO_MAP_TXT=/abs/path/to/map.txt cargo run --bin vo_replay`
//! (same usbipd/COM5 runner as the other bins; the calc8 model blob still
//! flashes to 0x400000 for the Embedder). `VO_MAP_TXT` is required; the path
//! may be absolute or relative to `src/bin/`. Text costs ~2x flash vs binary,
//! so bench maps must be trimmed to fit factory (4 MB) — which the 100k-point
//! firmware cap requires anyway. Changing the map (or its path) rebuilds via
//! `rerun-if-env-changed` (see build.rs).
#![allow(dead_code)]

#[path = "../camera.rs"]
mod camera;
#[path = "../semantic.rs"]
mod semantic;

use std::net::Ipv4Addr;
use std::time::{Duration, Instant};

use esp_idf_hal::cpu::Core;
use esp_idf_hal::delay::FreeRtos;
use esp_idf_hal::task::thread::ThreadSpawnConfiguration;
use esp_idf_svc::sys::EspError;
use vo_box_lite::fast;
use vo_box_lite::ekf;
use vo_box_lite::localize;
use vo_box_lite::localize::MapPoint;
use vo_box_lite::matcher;
use vo_box_lite::pyramid;
use vo_box_lite::ransac;

// ---- map (baked at build time; see VO_MAP_TXT above) ----
/// Full text of the localization map (`CAMERA` + `FRAME`/`POINT` groups as
/// written by scripts/write_map.py), embedded in flash, parsed to PSRAM once.
const MAP_TXT: &str = include_str!(env!("VO_MAP_TXT"));
const CAMERA_MODEL_SIMPLE_RADIAL: &str = "SIMPLE_RADIAL";
// SoftAP creds for semantic.rs::run() (the embedding streamer; never called
// in bench mode — kept only so the shared module resolves).
const AP_SSID: &str = "vo-box";
const AP_PASS: &str = "vobox1234";
const AP_IP_FALLBACK: Ipv4Addr = Ipv4Addr::new(192, 168, 71, 1);
const TCP_PORT: u16 = 5000;
const EMBEDDING_DIM: usize = 1064;
const MAX_MAP_FRAMES: usize = 1024;
const MAX_MAP_POINTS: usize = 100_000;

// ---- BENCH_VI pose-prior retrieval (plan branch 1) ----
/// Strong prior: keyframes within this attitude gate and whose points project
/// into the prior frustum; top-K nearest by camera-center distance.
const MAX_KF_ANGLE: f32 = core::f32::consts::FRAC_PI_6; // 30 deg
const PRIOR_TOPK: usize = 3;
/// Prior is "strong" iff mean position/attitude std is under these (else the
/// embedding top-K fallback, plan branch 2).
const PRIOR_POS_STD_MAX: f32 = 0.5; // m
const PRIOR_ATT_STD_MAX: f32 = 0.05; // rad (~2.9 deg)
/// Candidate-point scratch bound for `localize_prior`.
const MAX_PRIOR_CANDIDATES: usize = 8192;

// ---- replay (SPI master + localize) ----
const CAM_W: usize = 640;
const CAM_H: usize = 480;
const CHUNK_BYTES: usize = 4096; // must match data_board.rs
// Record = 4 B length prefix + 20 B header (magic/seq/t_us/w/h) + VGA gray.
const MAX_RECORD: usize = 4 + 20 + CAM_W * CAM_H;
const VO_TASK_STACK: usize = 16 * 1024;
const VO_TASK_PRIO: u8 = 5;
/// Emit a live `VO_IMU` line every N fused samples (~5 Hz at 200 Hz IMU).
const IMU_LOG_EVERY: usize = 40;
const TUM1: &[u8; 4] = b"TUM1";
const IMU1: &[u8; 4] = b"IMU1";
const INIT_MAGIC: [u8; 4] = *b"INIT";
// EKF fusion tuning (TUNE(BENCH_VI): refine vs EuRoC groundtruth alignment).
const R_POS_VAR: f32 = 0.01; // (0.1 m)^2 VO-fix position variance
const R_ATT_VAR: f32 = 0.002; // (~2.6 deg)^2 VO-fix attitude variance
const GAP_RESYNC_US: u64 = 500_000; // IMU time gap -> re-anchor, skip predict
// TODO(BENCH_VI): fill the EuRoC cam0<-IMU extrinsic; identity treats the
// camera and IMU frames as one, so the EKF state is the camera body frame.

const QSPI_CS: i32 = 1;
const QSPI_CLK: i32 = 14;
const QSPI_IO0: i32 = 21;
const QSPI_IO1: i32 = 47;
const QSPI_IO2: i32 = 41;
const QSPI_IO3: i32 = 42;
const SPI_CLOCK_HZ: i32 = 10_000_000;
/// false = 1-bit bring-up fallback (slave follows the command mask; no reflash needed there).
const USE_QIO: bool = true;

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

// Slave-HD base opcodes (hal/spi_types.h) + QIO modifier (esp32s3 spi_ll.h:
// base|0xA0 = addr+data 4-bit; SEG_END/EN_QPI stay 1-bit, INTn keeps the mod
// like essl_spi.c's rddma_done does).
const OP_WRBUF: u16 = 0x01;
const OP_RDBUF: u16 = 0x02;
const OP_RDDMA: u16 = 0x04;
const OP_CMD8: u16 = 0x08; // INT0 = "rddma done", closes one chunk
const QIO_MOD: u16 = 0xA0;

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    if let Err(e) = run() {
        log::error!("vo_replay stopped: {e}");
        panic!("vo_replay: {e}");
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Bench mode: no WiFi (modem never started), no camera (never touched).
    // The map is baked into flash (VO_MAP_TXT) and parsed to PSRAM once.
    let t = Instant::now();
    let map = parse_map_txt(MAP_TXT).map_err(|e| format!("baked map: {e}"))?;
    log::info!(
        "baked map: {} frames / {} points, camera {:?} (parsed in {} ms)",
        map.frames.len(),
        map.frames.iter().map(|f| f.points.len()).sum::<usize>(),
        map.params,
        t.elapsed().as_millis()
    );
    // Embedder first: its conv2 SRAM pin needs internal RAM before the DMA
    // buffer and the second task stack take theirs.
    let loc = Localizer::new().map_err(|e| format!("localize init: {e}"))?;
    log::info!("embedder + localize scratch ready");

    // ---- SPI master init + DMA frame buffer ----
    let spi = init_spi_master()?;
    log::info!(
        "SPI2 master CS={QSPI_CS} CLK={QSPI_CLK} IO0..3={QSPI_IO0},{QSPI_IO1},{QSPI_IO2},{QSPI_IO3} @{}MHz {}",
        SPI_CLOCK_HZ / 1_000_000,
        if USE_QIO { "QIO" } else { "1-bit" }
    );
    let mut dma = DmaBuf::alloc(MAX_RECORD + 4)?;

    // ---- two tasks: VO on core 1, SPI + EKF (this, main) task on core 0 ----
    let (frame_tx, frame_rx) = std::sync::mpsc::channel::<Frame>();
    let (fix_tx, fix_rx) = std::sync::mpsc::channel::<VoMsg>();
    let mut conf = ThreadSpawnConfiguration::default();
    conf.name = Some(c"vo_task");
    conf.stack_size = VO_TASK_STACK;
    conf.priority = VO_TASK_PRIO;
    conf.pin_to_core = Some(Core::Core1);
    let prev_conf = ThreadSpawnConfiguration::get();
    conf.set()?;
    let loc = SendLocalizer(loc);
    let spawned = std::thread::Builder::new()
        .stack_size(VO_TASK_STACK)
        .spawn(move || vo_task(loc, map, frame_rx, fix_tx));
    // Restore the pthread spawn default: the core-1 pin is vo_task's alone.
    if let Some(prev) = prev_conf {
        let _ = prev.set();
    }
    spawned?;
    log::info!("vo_task spawned on core 1 (prio {VO_TASK_PRIO}, {VO_TASK_STACK} B stack)");

    link_loop(spi, &mut dma, &frame_tx, &fix_rx);
}

fn check(result: esp_idf_sys::esp_err_t) -> Result<(), EspError> {
    EspError::convert(result)
}

// ---------------------------------------------------------------- baked map

struct LocalMap {
    params: [f32; 4], // SIMPLE_RADIAL f, cx, cy, k1
    frames: Vec<LocalMapFrame>,
}

struct LocalMapFrame {
    embedding: [u8; EMBEDDING_DIM],
    points: Vec<MapPoint>,
    /// Camera center in the map frame (`# POSE`; zeros if absent).
    pos: [f32; 3],
    /// World->camera quaternion `[w,x,y,z]` (`# POSE`).
    quat: [f32; 4],
    has_pose: bool,
}

/// Give the shared selector/matcher access to a map frame's pose + points.
impl vo_box_lite::localize::PosedFrame for LocalMapFrame {
    fn has_pose(&self) -> bool {
        self.has_pose
    }
    fn pos(&self) -> [f32; 3] {
        self.pos
    }
    fn quat(&self) -> [f32; 4] {
        self.quat
    }
    fn points(&self) -> &[MapPoint] {
        &self.points
    }
}

fn hex_val(c: u8) -> Result<u8, String> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(format!("bad hex char '{}'", c as char)),
    }
}

/// Decode even-length hex into `out`; errs on length/char mismatch.
fn decode_hex(s: &str, out: &mut [u8]) -> Result<(), String> {
    let b = s.as_bytes();
    if b.len() != out.len() * 2 {
        return Err(format!("hex len {} != {} B", b.len(), out.len()));
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = hex_val(b[2 * i])? << 4 | hex_val(b[2 * i + 1])?;
    }
    Ok(())
}

/// Parse baked map.txt (same grammar as scripts/receive_map.py load_map_txt):
/// `#` comments, `CAMERA SIMPLE_RADIAL f cx cy k1`, then per frame
/// `FRAME <stem> <2128-hex embedding>` + `POINT x y z <64-hex>` lines.
fn parse_map_txt(txt: &str) -> Result<LocalMap, String> {
    let mut params: Option<[f32; 4]> = None;
    let mut frames: Vec<LocalMapFrame> = Vec::new();
    let mut total_points = 0usize;
    for (ln, line) in txt.lines().enumerate() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        // `# POSE <stem> qx qy qz qw cx cy cz` (world->camera q + center C).
        // A comment for older parsers, but the prior path needs it.
        if t[0] == "#" && t.len() == 10 && t[1] == "POSE" {
            let cur = frames.last_mut().ok_or(format!("line {ln}: POSE before FRAME"))?;
            cur.quat = [
                t[6].parse::<f32>().map_err(|_| format!("line {ln}: bad POSE qw"))?,
                t[3].parse::<f32>().map_err(|_| format!("line {ln}: bad POSE qx"))?,
                t[4].parse::<f32>().map_err(|_| format!("line {ln}: bad POSE qy"))?,
                t[5].parse::<f32>().map_err(|_| format!("line {ln}: bad POSE qz"))?,
            ];
            for (i, v) in cur.pos.iter_mut().enumerate() {
                *v = t[7 + i].parse::<f32>().map_err(|_| format!("line {ln}: bad POSE C"))?;
            }
            cur.has_pose = true;
            continue;
        }
        if t[0].starts_with('#') {
            continue;
        }
        match t[0] {
            "CAMERA" => {
                if t.len() != 6 || t[1] != CAMERA_MODEL_SIMPLE_RADIAL {
                    return Err(format!("line {ln}: bad CAMERA line"));
                }
                let mut p = [0f32; 4];
                for (i, v) in p.iter_mut().enumerate() {
                    *v = t[2 + i].parse::<f32>().map_err(|_| format!("line {ln}: bad camera param"))?;
                }
                params = Some(p);
            }
            "FRAME" => {
                if t.len() != 3 {
                    return Err(format!("line {ln}: bad FRAME line"));
                }
                if frames.len() >= MAX_MAP_FRAMES {
                    return Err(format!("frame count > {MAX_MAP_FRAMES}"));
                }
                let mut embedding = [0u8; EMBEDDING_DIM];
                decode_hex(t[2], &mut embedding).map_err(|e| format!("line {ln}: {e}"))?;
                frames.push(LocalMapFrame {
                    embedding,
                    points: Vec::new(),
                    pos: [0.0; 3],
                    quat: [1.0, 0.0, 0.0, 0.0],
                    has_pose: false,
                });
            }
            "POINT" => {
                if t.len() != 5 {
                    return Err(format!("line {ln}: bad POINT line"));
                }
                let cur = frames.last_mut().ok_or(format!("line {ln}: POINT before FRAME"))?;
                total_points += 1;
                if total_points > MAX_MAP_POINTS {
                    return Err(format!("point count > {MAX_MAP_POINTS}"));
                }
                let mut xyz = [0f32; 3];
                for (i, v) in xyz.iter_mut().enumerate() {
                    *v = t[1 + i].parse::<f32>().map_err(|_| format!("line {ln}: bad xyz"))?;
                }
                let mut raw = [0u8; 32];
                decode_hex(t[4], &mut raw).map_err(|e| format!("line {ln}: {e}"))?;
                let mut desc = [0u32; 8];
                for (w, d) in desc.iter_mut().enumerate() {
                    *d = u32::from_le_bytes(raw[w * 4..w * 4 + 4].try_into().unwrap());
                }
                cur.points.push(MapPoint { xyz, desc });
            }
            _ => return Err(format!("line {ln}: unknown record '{}'", t[0])),
        }
    }
    let params = params.ok_or("no CAMERA line")?;
    if frames.is_empty() {
        return Err("no FRAME lines".into());
    }
    Ok(LocalMap { params, frames })
}

// ---------------------------------------------------------------- SPI master

type SpiHandle = esp_idf_sys::spi_device_handle_t;

fn mode_flags() -> u32 {
    if USE_QIO {
        esp_idf_sys::SPI_TRANS_MODE_QIO | esp_idf_sys::SPI_TRANS_MODE_DIOQIO_ADDR
    } else {
        0
    }
}

fn op(base: u16) -> u16 {
    base | if USE_QIO { QIO_MOD } else { 0 }
}

fn init_spi_master() -> Result<SpiHandle, EspError> {
    use esp_idf_sys::*;
    let mut bus: spi_bus_config_t = Default::default();
    bus.__bindgen_anon_1.mosi_io_num = QSPI_IO0;
    bus.__bindgen_anon_2.miso_io_num = QSPI_IO1;
    bus.__bindgen_anon_3.quadwp_io_num = QSPI_IO2;
    bus.__bindgen_anon_4.quadhd_io_num = QSPI_IO3;
    bus.sclk_io_num = QSPI_CLK;
    bus.max_transfer_sz = (CHUNK_BYTES + 64) as i32;
    bus.flags = 0;
    bus.intr_flags = 0;
    check(unsafe {
        spi_bus_initialize(
            spi_host_device_t_SPI2_HOST,
            &bus,
            spi_common_dma_t_SPI_DMA_CH_AUTO,
        )
    })?;

    let mut dev: spi_device_interface_config_t = Default::default();
    dev.command_bits = 8;
    dev.address_bits = 8;
    dev.dummy_bits = 8;
    dev.mode = 0;
    dev.clock_speed_hz = SPI_CLOCK_HZ;
    dev.spics_io_num = QSPI_CS;
    dev.queue_size = 4;
    dev.flags = SPI_DEVICE_HALFDUPLEX;
    let mut handle: SpiHandle = std::ptr::null_mut();
    check(unsafe { spi_bus_add_device(spi_host_device_t_SPI2_HOST, &dev, &mut handle) })?;
    Ok(handle)
}

/// RDBUF: read `out.len()` bytes from the slave's shared regs at `addr`.
fn spi_rdbuf(spi: SpiHandle, out: &mut [u8], addr: i32) -> Result<(), EspError> {
    use esp_idf_sys::*;
    let mut t: spi_transaction_ext_t = Default::default();
    t.base.cmd = op(OP_RDBUF);
    t.base.addr = addr as u64;
    t.base.rxlength = out.len() * 8;
    t.base.flags = mode_flags() | SPI_TRANS_VARIABLE_DUMMY;
    t.base.__bindgen_anon_2.rx_buffer = out.as_mut_ptr() as *mut core::ffi::c_void;
    t.dummy_bits = 8;
    check(unsafe { spi_device_transmit(spi, &mut t.base as *mut spi_transaction_t) })
}

/// WRBUF: write `data` to the slave's shared regs at `addr`.
fn spi_wrbuf(spi: SpiHandle, data: &[u8], addr: i32) -> Result<(), EspError> {
    use esp_idf_sys::*;
    let mut t: spi_transaction_ext_t = Default::default();
    t.base.cmd = op(OP_WRBUF);
    t.base.addr = addr as u64;
    t.base.length = data.len() * 8;
    t.base.flags = mode_flags() | SPI_TRANS_VARIABLE_DUMMY;
    t.base.__bindgen_anon_1.tx_buffer = data.as_ptr() as *const core::ffi::c_void;
    t.dummy_bits = 8;
    check(unsafe { spi_device_transmit(spi, &mut t.base as *mut spi_transaction_t) })
}

/// RDDMA: read one `out.len()`-byte segment from the slave's TX queue.
fn spi_rddma_seg(spi: SpiHandle, out: &mut [u8]) -> Result<(), EspError> {
    use esp_idf_sys::*;
    let mut t: spi_transaction_ext_t = Default::default();
    t.base.cmd = op(OP_RDDMA);
    t.base.rxlength = out.len() * 8;
    t.base.flags = mode_flags() | SPI_TRANS_VARIABLE_DUMMY;
    t.base.__bindgen_anon_2.rx_buffer = out.as_mut_ptr() as *mut core::ffi::c_void;
    t.dummy_bits = 8;
    check(unsafe { spi_device_transmit(spi, &mut t.base as *mut spi_transaction_t) })
}

/// CMD8: close one chunk (INT0, same encoding as essl_spi_rddma_done).
fn spi_cmd8(spi: SpiHandle) -> Result<(), EspError> {
    use esp_idf_sys::*;
    let mut t: spi_transaction_t = Default::default();
    t.cmd = op(OP_CMD8);
    t.flags = mode_flags();
    check(unsafe { spi_device_transmit(spi, &mut t as *mut spi_transaction_t) })
}

fn rdbuf_u32(spi: SpiHandle, addr: i32) -> Result<u32, EspError> {
    let mut b = [0u8; 4];
    spi_rdbuf(spi, &mut b, addr)?;
    Ok(u32::from_le_bytes(b))
}

/// Two consecutive equal reads; None = torn/failed (caller retries).
fn stable_u32(spi: SpiHandle, addr: i32) -> Option<u32> {
    let a = rdbuf_u32(spi, addr).ok()?;
    FreeRtos::delay_ms(2);
    let b = rdbuf_u32(spi, addr).ok()?;
    (a == b).then_some(a)
}

/// Single reg snapshot (no double-read): the post-transfer CRC backstops a
/// torn read, and a mismatched XFER/SEQ just costs one VO_DROP + re-present.
/// Fast enough to hold IMU-batch rate; the stable_* variants stay for the
/// INIT/STREAMING handshake below.
fn snapshot_once(spi: SpiHandle) -> Option<(u32, u32, u32, u32, u32)> {
    Some((
        rdbuf_u32(spi, REG_STATUS).ok()?,
        rdbuf_u32(spi, REG_XFER).ok()?,
        rdbuf_u32(spi, REG_LEN).ok()?,
        rdbuf_u32(spi, REG_SEQ).ok()?,
        rdbuf_u32(spi, REG_CRC).ok()?,
    ))
}

/// Poll `f` every `step_ms` until `deadline`; None on timeout.
fn poll_until<T>(deadline: Instant, step_ms: u32, mut f: impl FnMut() -> Option<T>) -> Option<T> {
    while Instant::now() < deadline {
        if let Some(v) = f() {
            return Some(v);
        }
        FreeRtos::delay_ms(step_ms);
    }
    f()
}

// ---------------------------------------------------------------- record fetch

/// Pull one parked record: INIT handshake, then k gated RDDMA+CMD8 chunks.
/// Returns the record length on success.
fn fetch_record(spi: SpiHandle, dma: &mut DmaBuf, len: u32) -> Result<usize, &'static str> {
    let len = len as usize;
    // 1) claim it
    spi_wrbuf(spi, &INIT_MAGIC, REG_CMD).map_err(|_| "WRBUF INIT failed")?;
    // 2) DATA clears CMD (its wait_init ack)
    poll_until(Instant::now() + Duration::from_secs(2), 10, || {
        stable_u32(spi, REG_CMD).filter(|&c| c == 0)
    })
    .ok_or("CMD never cleared (DATA didn't see INIT)")?;
    // 3) DATA enters STREAMING
    poll_until(Instant::now() + Duration::from_secs(2), 10, || {
        stable_u32(spi, REG_STATUS).filter(|&s| s == STATUS_STREAMING)
    })
    .ok_or("STATUS never went STREAMING")?;
    // 4) chunks, each gated on READY_N > consumed (double-read, like the slave side)
    let k = len.div_ceil(CHUNK_BYTES);
    let buf = dma.as_slice_mut();
    for i in 0..k {
        let want = (i + 1) as u32;
        poll_until(Instant::now() + Duration::from_secs(5), 5, || {
            stable_u32(spi, REG_READY_N).filter(|&r| r >= want)
        })
        .ok_or("READY_N stalled (DATA chunk timeout)")?;
        let off = i * CHUNK_BYTES;
        let take = (len - off).min(CHUNK_BYTES);
        spi_rddma_seg(spi, &mut buf[off..off + take]).map_err(|_| "RDDMA failed")?;
        spi_cmd8(spi).map_err(|_| "CMD8 failed")?;
    }
    Ok(len)
}

// -------------------------------------------------------------- link (core 0)

/// One fetched image handed to `vo_task`.
struct Frame {
    seq: u32,
    xfer: u32,
    t_us: u64,
    spi_ms: u64,
    gray: Vec<u8>,
    /// Camera pose prior from the EKF; Some = strong (prior branch), None =
    /// weak/absent (embedding top-K fallback).
    prior: Option<localize::PosePrior>,
}

/// One VO result handed back to the link task for fusion + logging.
struct VoMsg {
    seq: u32,
    xfer: u32,
    t_us: u64,
    spi_ms: u64,
    extract_us: u64,
    embed_us: u64,
    match_us: u64,
    pnp_us: u64,
    cos1: f32,
    cos2: f32,
    nfeat: usize,
    matches: usize,
    best_inliers: usize,
    pnp: Option<ransac::PnpResult>,
    /// 1 = pose-prior branch, 0 = embedding fallback.
    used_prior: bool,
    nkf: usize,
    survivors: usize,
    infrustum: usize,
}

impl VoMsg {
    /// No PnP attempt was made (embed failure / empty map).
    fn failed(seq: u32, xfer: u32, t_us: u64, spi_ms: u64, nfeat: usize) -> Self {
        Self {
            seq,
            xfer,
            t_us,
            spi_ms,
            extract_us: 0,
            embed_us: 0,
            match_us: 0,
            pnp_us: 0,
            cos1: 0.0,
            cos2: 0.0,
            nfeat,
            matches: 0,
            best_inliers: 0,
            pnp: None,
            used_prior: false,
            nkf: 0,
            survivors: 0,
            infrustum: 0,
        }
    }
}

/// The TFLM `Embedder` holds raw pointers, so `Localizer` is not `Send` by
/// default. It is built on the main task (before the DMA buffer, so the conv2
/// SRAM pin succeeds), moved into `vo_task`, and used only there afterwards —
/// no pointer is ever shared across threads.
struct SendLocalizer(Localizer);
unsafe impl Send for SendLocalizer {}

/// SPI + EKF owner. Never blocks on VO, so inertial data keeps flowing while
/// `vo_task` computes.
fn link_loop(
    spi: SpiHandle,
    dma: &mut DmaBuf,
    frame_tx: &std::sync::mpsc::Sender<Frame>,
    fix_rx: &std::sync::mpsc::Receiver<VoMsg>,
) -> ! {
    log::info!("link: waiting for DATA READY (0xEE at reg 0)");
    let mut n = 0u64;
    let mut n_imu = 0u64;
    let mut imu_since_log = 0usize;
    let mut last_seen_xfer: Option<u32> = None;
    let mut last_done_seq: Option<u32> = None;
    // EKF starts at origin/identity with a wide P0; the first VO fix snaps it
    // into the map frame. IMU predicts live, VO corrects.
    let mut ekf = ekf::Ekf::new(
        [0.0; 3],
        [1.0, 0.0, 0.0, 0.0],
        ekf::EkfNoise::default(),
        [10.0, 1.0, 0.5, 0.1, 0.01],
    );
    let mut ekf_time_us: Option<u64> = None;
    // Snapshot taken at each image: (state, clock, t_img). IMU fused after it
    // is also appended to imu_buf, so a late fix rewinds and replays exactly.
    let mut snap: Option<(ekf::Ekf, Option<u64>, u64)> = None;
    let mut imu_buf: Vec<ImuSample> = Vec::new();
    // Set once the first VO fix lands: gates the strong/weak prior choice.
    let mut ekf_initialized = false;
    log::info!("ekf init: origin/identity, wide P0; first VO fix snaps it in");
    loop {
        // A finished fix (arrived while IMU kept flowing) is applied first:
        // rewind to t_img, correct, replay the buffered span.
        while let Ok(msg) = fix_rx.try_recv() {
            apply_fix(
                &mut ekf,
                &mut ekf_time_us,
                &mut snap,
                &mut imu_buf,
                &mut n,
                n_imu,
                &mut ekf_initialized,
                &msg,
            );
            imu_since_log = 0;
        }
        // DATA alive? Single read; a torn READY just retries next spin.
        match rdbuf_u32(spi, REG_READY) {
            Ok(READY_FLAG) => {}
            _ => {
                FreeRtos::delay_ms(20);
                continue;
            }
        }
        let Some((status, xfer, len, seq, crc)) = snapshot_once(spi) else {
            FreeRtos::delay_ms(2);
            continue;
        };
        if status != STATUS_READY {
            // BUSY (sender filling) or STREAMING (a transfer we didn't start —
            // DATA times out on its own and re-presents; wait for READY).
            FreeRtos::delay_ms(if status == STATUS_BUSY { 5 } else { 20 });
            continue;
        }
        if len < 24 || len as usize > MAX_RECORD {
            log::warn!("VO_DROP seq={seq} bad LEN {len}");
            FreeRtos::delay_ms(50);
            continue;
        }
        if last_seen_xfer == Some(xfer) {
            FreeRtos::delay_ms(2);
            continue;
        }
        last_seen_xfer = Some(xfer);
        let t_spi = Instant::now();
        let dup = last_done_seq == Some(seq);
        match fetch_record(spi, dma, len) {
            Ok(got) if got == len as usize => {}
            _ => {
                log::warn!("VO_DROP seq={seq} xfer={xfer} transfer failed (DATA re-presents)");
                continue;
            }
        }
        let spi_ms = t_spi.elapsed().as_millis() as u64;
        // Validate: len prefix + known magic + CRC vs the snapshot, then
        // dispatch. Header layouts are type-specific (see module docs).
        let buf = dma.as_slice_mut();
        let rec = &buf[..len as usize];
        let prefix_ok =
            u32::from_le_bytes(rec[0..4].try_into().unwrap()) as usize == len as usize - 4;
        let is_img = &rec[4..8] == TUM1;
        let is_imu = &rec[4..8] == IMU1;
        if !prefix_ok || (!is_img && !is_imu) || crc32(rec) != crc {
            log::warn!("VO_DROP seq={seq} xfer={xfer} validation failed (torn snapshot?)");
            continue;
        }
        if dup {
            // Already consumed this SEQ (DATA re-presented after its own
            // timeout); drained above, skip so poses stay 1:1.
            continue;
        }
        if is_imu {
            match parse_imu_samples(rec) {
                Ok(samples) => {
                    let fed = fuse_imu_samples(&mut ekf, &mut ekf_time_us, &samples);
                    n_imu += fed as u64;
                    if snap.is_some() {
                        imu_buf.extend(samples);
                    }
                    last_done_seq = Some(seq);
                    imu_since_log += fed;
                    if imu_since_log >= IMU_LOG_EVERY {
                        imu_since_log = 0;
                        log_imu(&ekf, ekf_time_us, n_imu);
                    }
                }
                Err(e) => log::warn!("VO_DROP seq={seq} xfer={xfer} bad IMU1 ({e})"),
            }
            continue;
        }
        // ---- image: timestamped TUM1 (magic, seq, t_us, w, h, gray)
        if rec.len() != 24 + CAM_W * CAM_H
            || u16::from_le_bytes(rec[20..22].try_into().unwrap()) as usize != CAM_W
            || u16::from_le_bytes(rec[22..24].try_into().unwrap()) as usize != CAM_H
        {
            log::warn!("VO_DROP seq={seq} xfer={xfer} bad TUM1 dims");
            continue;
        }
        let t_img = u64::from_le_bytes(rec[12..20].try_into().unwrap());
        if snap.is_some() {
            log::warn!("VO_IMG seq={seq} previous fix still pending; replacing snapshot");
        }
        snap = Some((ekf.clone(), ekf_time_us, t_img));
        imu_buf.clear();
        last_done_seq = Some(seq);
        let prior = prior_from_ekf(&ekf, ekf_initialized);
        let gray = rec[24..].to_vec();
        log::info!(
            "VO_IMG seq={seq} xfer={xfer} t={} spi={spi_ms}ms prior={} -> vo_task",
            fmt_us(t_img),
            if prior.is_some() { "strong" } else { "weak" }
        );
        if frame_tx
            .send(Frame { seq, xfer, t_us: t_img, spi_ms, gray, prior })
            .is_err()
        {
            log::error!("VO_IMG vo_task gone; image dropped");
        }
    }
}

/// Strong prior = first fix landed and the EKF's mean position/attitude std is
/// under threshold; otherwise None (embedding fallback).
fn prior_from_ekf(ekf: &ekf::Ekf, initialized: bool) -> Option<localize::PosePrior> {
    if !initialized {
        return None;
    }
    let ps = ekf.pos_std();
    let as_ = ekf.att_std();
    let strong = (ps[0] + ps[1] + ps[2]) / 3.0 < PRIOR_POS_STD_MAX
        && (as_[0] + as_[1] + as_[2]) / 3.0 < PRIOR_ATT_STD_MAX;
    if !strong {
        return None;
    }
    Some(localize::PosePrior {
        pos: ekf.p,
        quat: ekf::quat_from_mat(ekf.body_to_world()),
    })
}

/// Apply one finished VO result: rewind to the t_img snapshot, fuse the fix,
/// then replay every IMU sample that arrived in the meantime.
#[allow(clippy::too_many_arguments)]
fn apply_fix(
    ekf: &mut ekf::Ekf,
    ekf_time_us: &mut Option<u64>,
    snap: &mut Option<(ekf::Ekf, Option<u64>, u64)>,
    imu_buf: &mut Vec<ImuSample>,
    n: &mut u64,
    n_imu: u64,
    initialized: &mut bool,
    msg: &VoMsg,
) {
    let Some((s_ekf, s_t, t_img)) = snap.take() else {
        log::warn!("VO_DROP seq={} fix with no snapshot (stale)", msg.seq);
        return;
    };
    *ekf = s_ekf;
    *ekf_time_us = s_t;
    *n += 1;
    let n_fix = *n;
    // PnP gives world->cam [R|t]; the EKF state is body->world, so transpose
    // for the fix (identity cam<-IMU extrinsic: TODO(BENCH_VI)).
    let corr = msg.pnp.map(|p| {
        ekf.correct_pose(
            camera_center(&p.r, &p.t),
            ekf::quat_from_mat(transpose3(&p.r)),
            R_POS_VAR,
            R_ATT_VAR,
        )
    });
    if matches!(corr, Some(Some(_))) {
        *initialized = true;
    }
    fuse_imu_samples(ekf, ekf_time_us, imu_buf);
    imu_buf.clear();
    match msg.pnp {
        Some(p) => {
            let corr_s = match corr.flatten() {
                Some(r) => format!("{r:.2}"),
                None => "REJ".to_string(),
            };
            let (roll, pitch, yaw) = zyx_deg(&p.r);
            let c = camera_center(&p.r, &p.t);
            log::info!(
                "VO_POSE {n_fix} seq={} xfer={} mode={} kf={} surv={} inf={} cos1={:.3} cos2={:.3} feats={} matches={} inliers={} reproj={:.2}px roll={roll:.1} pitch={pitch:.1} yaw={yaw:.1} center=({:.2},{:.2},{:.2}) t={} spi={}ms extract={} embed={} match={} pnp={} us {} corr={corr_s}",
                msg.seq, msg.xfer,
                if msg.used_prior { "prior" } else { "emb" },
                msg.nkf, msg.survivors, msg.infrustum,
                msg.cos1, msg.cos2, msg.nfeat, msg.matches,
                p.inlier_count, p.mean_reproj_error_px, c[0], c[1], c[2], fmt_us(t_img),
                msg.spi_ms, msg.extract_us, msg.embed_us, msg.match_us, msg.pnp_us,
                ekf_summary(ekf, *ekf_time_us, n_imu),
            );
        }
        None => log::info!(
            "VO_FAIL {n_fix} seq={} xfer={} mode={} kf={} surv={} inf={} cos1={:.3} cos2={:.3} feats={} matches={} best={} inl t={} spi={}ms extract={} embed={} match={} pnp={} us {}",
            msg.seq, msg.xfer,
            if msg.used_prior { "prior" } else { "emb" },
            msg.nkf, msg.survivors, msg.infrustum,
            msg.cos1, msg.cos2, msg.nfeat, msg.matches,
            msg.best_inliers, fmt_us(t_img), msg.spi_ms, msg.extract_us, msg.embed_us,
            msg.match_us, msg.pnp_us, ekf_summary(ekf, *ekf_time_us, n_imu),
        ),
    }
}

/// Live EKF line (~5 Hz, `imu` = samples fused so far).
fn log_imu(e: &ekf::Ekf, time_us: Option<u64>, n_imu: u64) {
    let (roll, pitch, yaw) = zyx_deg(&transpose3(&e.body_to_world()));
    log::info!(
        "VO_IMU {} roll={roll:.1} pitch={pitch:.1} yaw={yaw:.1}",
        ekf_summary(e, time_us, n_imu)
    );
}

fn fmt_us(t: u64) -> String {
    format!("{}.{:02}", t / 1_000_000, (t % 1_000_000) / 10_000)
}

// --------------------------------------------------------- vo task (core 1)

/// Scratch for the pose-prior branch, allocated once in `vo_task`.
struct PriorBufs {
    cands: Vec<localize::Candidate>,
    cand_query: Vec<u32>,
    cand_dist: Vec<u32>,
    best_idx: Vec<u32>,
    best_dist: Vec<u32>,
    second_dist: Vec<u32>,
    corrs: Vec<ransac::Correspondence>,
    mask: Vec<bool>,
}

impl PriorBufs {
    fn new() -> Self {
        let nf = pyramid::MAX_FEATURES;
        PriorBufs {
            cands: vec![localize::Candidate::default(); MAX_PRIOR_CANDIDATES],
            cand_query: vec![0u32; MAX_PRIOR_CANDIDATES],
            cand_dist: vec![u32::MAX; MAX_PRIOR_CANDIDATES],
            best_idx: vec![0u32; nf],
            best_dist: vec![u32::MAX; nf],
            second_dist: vec![u32::MAX; nf],
            corrs: vec![
                ransac::Correspondence { world: [0.0; 3], xn: 0.0, yn: 0.0 };
                nf
            ],
            mask: vec![false; nf],
        }
    }
}

fn pnp_score(s: &localize::LocalizeStats) -> usize {
    s.pnp.map_or(s.pnp_best.inlier_count, |p| p.inlier_count)
}

/// Plan branch 1: strong prior -> attitude/frustum keyframes + brute match.
fn vo_prior(
    loc: &mut Localizer,
    map: &LocalMap,
    nfeat: usize,
    prior: &localize::PosePrior,
    cam: &ransac::Camera,
    opts: &ransac::PnpOptions,
    bufs: &mut PriorBufs,
) -> (Option<localize::LocalizeStats>, localize::KfSelect) {
    let mut ids = [0usize; localize::MAX_KEYFRAMES];
    let sel = localize::select_keyframes(
        prior, &map.frames, cam, CAM_W, CAM_H, MAX_KF_ANGLE, PRIOR_TOPK, &mut ids,
    );
    if sel.n == 0 {
        return (None, sel);
    }
    let mut ws = localize::PriorScratch {
        cands: &mut bufs.cands,
        cand_query: &mut bufs.cand_query,
        cand_dist: &mut bufs.cand_dist,
        best_idx: &mut bufs.best_idx,
        best_dist: &mut bufs.best_dist,
        second_dist: &mut bufs.second_dist,
        corrs: &mut bufs.corrs,
        mask: &mut bufs.mask,
    };
    let st = localize::localize_prior(
        &loc.feats[..nfeat],
        &map.frames,
        &ids[..sel.n],
        prior,
        cam,
        CAM_W,
        CAM_H,
        opts,
        &mut loc.rng,
        &mut ws,
        now_us,
    );
    (Some(st), sel)
}

/// Plan branch 2: weak/absent prior -> embedding top-K frames, brute match,
/// keep the frame whose PnP has the most inliers.
fn vo_embed(
    loc: &mut Localizer,
    map: &LocalMap,
    nfeat: usize,
    cam: &ransac::Camera,
    opts: &ransac::PnpOptions,
) -> Option<(localize::LocalizeStats, f32, f32, u64)> {
    let t = Instant::now();
    if let Err(e) = loc.embedder.embed(&loc.frame, CAM_W, CAM_H, &mut loc.emb) {
        log::warn!("embed failed ({e})");
        return None;
    }
    let embed_us = t.elapsed().as_micros() as u64;
    let ranked = top_k_frames(map, &loc.emb, PRIOR_TOPK);
    let &(cos1, _) = ranked.first()?;
    let cos2 = ranked.get(1).map_or(0.0, |(s, _)| *s);
    let mut best: Option<localize::LocalizeStats> = None;
    for (_, frame) in &ranked {
        let st = loc.localize(nfeat, &frame.points, cam, opts);
        if best.as_ref().map_or(true, |b| pnp_score(&st) > pnp_score(b)) {
            best = Some(st);
        }
    }
    best.map(|st| (st, cos1, cos2, embed_us))
}

/// Owns the Embedder + all VO scratch; blocks on `rx`, never touches SPI.
fn vo_task(
    mut loc: SendLocalizer,
    map: LocalMap,
    rx: std::sync::mpsc::Receiver<Frame>,
    tx: std::sync::mpsc::Sender<VoMsg>,
) {
    let cam_model = ransac::Camera {
        fx: map.params[0],
        fy: map.params[0],
        cx: map.params[1],
        cy: map.params[2],
        k1: map.params[3],
    };
    let opts = ransac::PnpOptions::default();
    let mut prior_bufs = PriorBufs::new();
    log::info!("vo_task ready ({} map frames)", map.frames.len());
    while let Ok(f) = rx.recv() {
        let Frame { seq, xfer, t_us, spi_ms, gray, prior } = f;
        loc.0.frame.copy_from_slice(&gray);
        let t = Instant::now();
        let nfeat = pyramid::extract_pyramid(
            &loc.0.frame,
            CAM_W,
            CAM_H,
            pyramid::FAST_THRESHOLD,
            &mut loc.0.arena,
            &mut loc.0.work,
            &mut loc.0.vcol,
            &mut loc.0.corners,
            &mut loc.0.scores,
            &mut loc.0.rowidx,
            &mut loc.0.nms,
            &mut loc.0.cells,
            &mut loc.0.cand,
            &mut loc.0.feats,
            None,
        );
        let extract_us = t.elapsed().as_micros() as u64;
        // Branch 1 (strong prior) drives the whole match from the EKF; only
        // when it yields no keyframe do we pay for the calc8 embedding.
        let (stats, used_prior, nkf, survivors, infrustum, cos1, cos2, embed_us) =
            match prior {
                Some(p) => {
                    let (st, sel) = vo_prior(
                        &mut loc.0, &map, nfeat, &p, &cam_model, &opts, &mut prior_bufs,
                    );
                    match st {
                        Some(st) => (st, true, sel.n, sel.survivors, sel.infrustum, 0.0, 0.0, 0),
                        None => {
                            log::warn!(
                                "VO_IMG seq={seq} prior strong but no keyframe in frustum; embedding fallback"
                            );
                            match vo_embed(&mut loc.0, &map, nfeat, &cam_model, &opts) {
                                Some((st, c1, c2, e)) => {
                                    (st, false, 0, sel.survivors, 0, c1, c2, e)
                                }
                                None => {
                                    let _ = tx.send(VoMsg::failed(seq, xfer, t_us, spi_ms, nfeat));
                                    continue;
                                }
                            }
                        }
                    }
                }
                None => match vo_embed(&mut loc.0, &map, nfeat, &cam_model, &opts) {
                    Some((st, c1, c2, e)) => (st, false, 0, 0, 0, c1, c2, e),
                    None => {
                        let _ = tx.send(VoMsg::failed(seq, xfer, t_us, spi_ms, nfeat));
                        continue;
                    }
                },
            };
        let msg = VoMsg {
            seq,
            xfer,
            t_us,
            spi_ms,
            extract_us,
            embed_us,
            match_us: stats.match_us,
            pnp_us: stats.pnp_us,
            cos1,
            cos2,
            nfeat,
            matches: stats.matches,
            best_inliers: stats.pnp_best.inlier_count,
            pnp: stats.pnp,
            used_prior,
            nkf,
            survivors,
            infrustum,
        };
        if tx.send(msg).is_err() {
            log::warn!("VO_MSG link_task gone; vo_task stopping");
            break;
        }
    }
    log::warn!("VO_MSG frame channel closed; vo_task exiting");
}

// ---------------------------------------------------------------- EKF fusion

/// One IMU sample: dataset timestamp (µs), accel (m/s^2), gyro (rad/s).
type ImuSample = (u64, [f32; 3], [f32; 3]);

/// Split-rate fuse: nominal per sample, covariance every COV_EVERY samples
/// (batch-averaged refs). Clock: gaps re-anchor without integrating;
/// corrupt samples advance the clock but are skipped.
fn fuse_imu_samples(
    ekf: &mut ekf::Ekf,
    time_us: &mut Option<u64>,
    samples: &[ImuSample],
) -> usize {
    const COV_EVERY: usize = 10;
    let mut acc_a = [0.0; 3];
    let mut acc_w = [0.0; 3];
    let (mut n, mut dt_sum) = (0usize, 0.0f32);
    let mut fed = 0;
    for &(t, a, w) in samples {
        let dt = match *time_us {
            Some(last) if t > last && t - last <= GAP_RESYNC_US => (t - last) as f32 / 1e6,
            Some(last) => {
                flush_cov(ekf, &mut acc_a, &mut acc_w, &mut n, &mut dt_sum);
                log::warn!("ekf: IMU gap, re-anchoring clock (t={t} last={last})");
                *time_us = Some(t);
                continue;
            }
            None => {
                *time_us = Some(t);
                continue;
            }
        };
        if let Some((ac, wc)) = ekf.propagate_nominal(a, w, dt) {
            for k in 0..3 {
                acc_a[k] += ac[k];
                acc_w[k] += wc[k];
            }
            n += 1;
            dt_sum += dt;
            fed += 1;
            if n == COV_EVERY {
                flush_cov(ekf, &mut acc_a, &mut acc_w, &mut n, &mut dt_sum);
            }
        }
        *time_us = Some(t);
    }
    flush_cov(ekf, &mut acc_a, &mut acc_w, &mut n, &mut dt_sum);
    fed
}

/// One batch-averaged covariance step; no-op on an empty accumulator.
fn flush_cov(
    ekf: &mut ekf::Ekf,
    acc_a: &mut [f32; 3],
    acc_w: &mut [f32; 3],
    n: &mut usize,
    dt_sum: &mut f32,
) {
    if *n == 0 {
        return;
    }
    let inv = 1.0 / *n as f32;
    for k in 0..3 {
        acc_a[k] *= inv;
        acc_w[k] *= inv;
    }
    ekf.propagate_covariance(*acc_a, *acc_w, ekf.body_to_world(), *dt_sum);
    *acc_a = [0.0; 3];
    *acc_w = [0.0; 3];
    *n = 0;
    *dt_sum = 0.0;
}

/// Validate one IMU1 batch (rec includes the 4-byte length prefix) into
/// samples. Mirrors data_board.rs receive_record limits.
fn parse_imu_samples(rec: &[u8]) -> Result<Vec<ImuSample>, &'static str> {
    let t0 = u64::from_le_bytes(rec[12..20].try_into().map_err(|_| "bad IMU1")?);
    let nsamp = u16::from_le_bytes(rec[20..22].try_into().map_err(|_| "bad IMU1")?) as usize;
    let dt_us = u16::from_le_bytes(rec[22..24].try_into().map_err(|_| "bad IMU1")?) as u64;
    if nsamp == 0 || nsamp > 64 || dt_us < 100 {
        return Err("bad IMU1 header");
    }
    if rec.len() != 24 + 24 * nsamp {
        return Err("bad IMU1 length");
    }
    let mut out = Vec::with_capacity(nsamp);
    for i in 0..nsamp {
        let off = 24 + 24 * i;
        let f = |j: usize| {
            f32::from_le_bytes(rec[off + 4 * j..off + 4 * j + 4].try_into().unwrap())
        };
        out.push((t0 + i as u64 * dt_us, [f(0), f(1), f(2)], [f(3), f(4), f(5)]));
    }
    Ok(out)
}

/// Fused-state one-liner for VO_POSE / VO_FAIL: dataset time + position +
/// mean pos-std + samples fused. The scorer joins on t.
fn ekf_summary(e: &ekf::Ekf, time_us: Option<u64>, n_imu: u64) -> String {
    let s = e.pos_std();
    let t = match time_us {
        Some(u) => format!("{}.{:02}", u / 1_000_000, (u % 1_000_000) / 10_000),
        None => "none".to_string(),
    };
    format!(
        "ekf t={t} p=({:.2},{:.2},{:.2}) estd={:.2} imu={n_imu}",
        e.p[0], e.p[1], e.p[2], (s[0] + s[1] + s[2]) / 3.0
    )
}

fn transpose3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

fn cosine_similarity(a: &[u8], b: &[u8]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f32, *y as f32);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= f32::EPSILON { 0.0 } else { dot / denom }
}

fn top_k_frames<'a>(
    map: &'a LocalMap,
    query: &[u8; EMBEDDING_DIM],
    k: usize,
) -> Vec<(f32, &'a LocalMapFrame)> {
    let mut scored: Vec<(f32, &'a LocalMapFrame)> = map
        .frames
        .iter()
        .map(|f| (cosine_similarity(query, &f.embedding), f))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

struct Localizer {
    embedder: semantic::Embedder,
    arena: Vec<u8>,
    work: Vec<u8>,
    vcol: Vec<u16>,
    corners: Vec<fast::Corner>,
    scores: Vec<i32>,
    rowidx: Vec<usize>,
    nms: Vec<fast::Corner>,
    cells: Vec<u32>,
    cand: Vec<pyramid::Candidate>,
    feats: Vec<pyramid::Feature>,
    best_idx: Vec<u32>,
    best_dist: Vec<u32>,
    second_dist: Vec<u32>,
    point_query: Vec<u32>,
    point_dist: Vec<u32>,
    matches: Vec<matcher::Match>,
    corrs: Vec<ransac::Correspondence>,
    mask: Vec<bool>,
    frame: Vec<u8>,
    emb: [u8; EMBEDDING_DIM],
    rng: ransac::Xorshift64,
}

impl Localizer {
    fn new() -> Result<Localizer, String> {
        let (w, h) = (CAM_W, CAM_H);
        let nf = pyramid::MAX_FEATURES;
        let np = MAX_MAP_POINTS;
        Ok(Localizer {
            embedder: semantic::Embedder::init()?,
            arena: vec![0u8; pyramid::arena_bytes(w, h)],
            work: vec![0u8; w * h],
            vcol: vec![0u16; w],
            corners: vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX],
            scores: vec![0i32; pyramid::CORNERS_RAW_MAX],
            rowidx: vec![usize::MAX; h],
            nms: vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX],
            cells: vec![0u32; pyramid::bucket_cells(w, h) * pyramid::BUCKET_K],
            cand: vec![pyramid::Candidate::default(); pyramid::CAND_MAX],
            feats: vec![pyramid::Feature::default(); nf],
            best_idx: vec![0u32; nf],
            best_dist: vec![0u32; nf],
            second_dist: vec![0u32; nf],
            point_query: vec![0u32; np],
            point_dist: vec![0u32; np],
            matches: vec![matcher::Match::default(); nf],
            corrs: vec![
                ransac::Correspondence { world: [0.0; 3], xn: 0.0, yn: 0.0 };
                nf
            ],
            mask: vec![false; nf],
            frame: vec![0u8; w * h],
            emb: [0u8; EMBEDDING_DIM],
            rng: ransac::Xorshift64::new(now_us() | 1),
        })
    }

    fn localize(
        &mut self,
        nfeat: usize,
        points: &[MapPoint],
        cam: &ransac::Camera,
        opts: &ransac::PnpOptions,
    ) -> localize::LocalizeStats {
        let mut scratch = localize::LocalizeScratch {
            mb: matcher::MatchBuffers {
                best_idx: self.best_idx.as_mut_slice(),
                best_dist: self.best_dist.as_mut_slice(),
                second_dist: self.second_dist.as_mut_slice(),
                point_query: self.point_query.as_mut_slice(),
                point_dist: self.point_dist.as_mut_slice(),
            },
            matches: self.matches.as_mut_slice(),
            corrs: self.corrs.as_mut_slice(),
            mask: self.mask.as_mut_slice(),
        };
        localize::localize_frame(
            &self.feats[..nfeat],
            points,
            cam,
            opts,
            &mut self.rng,
            &mut scratch,
            now_us,
        )
    }
}

fn now_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

fn zyx_deg(r_w2c: &[[f32; 3]; 3]) -> (f32, f32, f32) {
    let r = [
        [r_w2c[0][0], r_w2c[1][0], r_w2c[2][0]],
        [r_w2c[0][1], r_w2c[1][1], r_w2c[2][1]],
        [r_w2c[0][2], r_w2c[1][2], r_w2c[2][2]],
    ];
    let pitch = (-r[2][0]).clamp(-1.0, 1.0).asin();
    let roll = r[2][1].atan2(r[2][2]);
    let yaw = r[1][0].atan2(r[0][0]);
    (roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees())
}

fn camera_center(r: &[[f32; 3]; 3], t: &[f32; 3]) -> [f32; 3] {
    [
        -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
        -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
        -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
    ]
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

// DMA-capable frame staging (one full TUM1 record; mirrors data_board.rs).
struct DmaBuf {
    ptr: *mut u8,
    len: usize,
}

const MALLOC_CAP_8BIT: u32 = 1 << 2;
const MALLOC_CAP_DMA: u32 = 1 << 3;

extern "C" {
    fn heap_caps_malloc(size: usize, caps: u32) -> *mut core::ffi::c_void;
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
