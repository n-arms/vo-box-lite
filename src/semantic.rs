//! Semantic embedding task: run the int8 CALC encoder (esp-tflite-micro + esp-nn)
//! on each 640x480 grayscale frame, streaming the frame + raw descriptor over
//! SoftAP TCP (EMB1). Input is 4x4-downscaled; model blob is mmap'd from flash.

use core::ffi::{c_char, c_void};
use core::ptr;
use std::io::Write as _;
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::slice;
use std::time::{Duration, Instant};

use crate::camera;
use esp_idf_hal::delay::FreeRtos;
use esp_idf_hal::peripherals::Peripherals;
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys;
use esp_idf_svc::wifi::{
    AccessPointConfiguration, AuthMethod, BlockingWifi, Configuration, EspWifi,
};
use vo_box_lite::downscale;

/// Partition label + custom data subtype from partitions.csv.
const MODEL_LABEL: &[u8] = b"model\0";
const MODEL_SUBTYPE: u32 = 0x40;

/// CALC encoder input (H x W, grayscale) and descriptor size.
const INPUT_W: usize = 160;
const INPUT_H: usize = 120;
const EMB_DIM: usize = 1064;

/// Camera capture size (grayscale VGA); 4x4-downscaled to the model input.
const SRC_W: usize = 640;
const SRC_H: usize = 480;

/// TFLite-Micro working arena (PSRAM). CALC's peak activation is ~325 KB.
const ARENA_BYTES: usize = 1024 * 1024;

// ---- EMB1 wire format (semantic task -> laptop, see receive_embeddings.py) ----
/// Frame+embedding record magic.
const MAGIC_EMB: &[u8; 4] = b"EMB1";
/// Wire pixel-format id: 3 = grayscale (our own enum, not the driver's).
const FMT_GRAYSCALE: u8 = 3;
/// Fixed header after the u32 length: magic(4) + fmt(1) + w(2) + h(2) +
/// ndim(2) + emb_scale(4) + emb_zp(4).
const EMB_HEADER_BYTES: usize = 19;

/// One-shot calc8 encoder ready to embed frames: mmap'd model partition +
/// interpreter over a PSRAM arena. The interpreter is a process singleton, so
/// only one Embedder may exist at a time.
pub struct Embedder {
    in_ptr: *mut u8,
    out_ptr: *const u8,
    scale: f32,
    zp: i32,
    _arena: Vec<u8>,
    _mmap_handle: sys::esp_partition_mmap_handle_t,
}

impl Embedder {
    /// Mmap the `model` partition, build the interpreter and enable per-op
    /// profiling. Err (instead of idling) so the caller can fall back.
    pub fn init() -> Result<Embedder, String> {
        let part = unsafe {
            sys::esp_partition_find_first(
                sys::esp_partition_type_t_ESP_PARTITION_TYPE_DATA,
                MODEL_SUBTYPE,
                MODEL_LABEL.as_ptr() as *const c_char,
            )
        };
        if part.is_null() {
            return Err("no 'model' partition (flash the .tflite via cargo_run.sh)".into());
        }
        let part_size = unsafe { (*part).size as usize };

        let mut model_ptr: *const c_void = ptr::null();
        let mut mmap_handle: sys::esp_partition_mmap_handle_t = 0;
        let e = unsafe {
            sys::esp_partition_mmap(
                part,
                0,
                part_size,
                sys::esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
                &mut model_ptr,
                &mut mmap_handle,
            )
        };
        if e != 0 || model_ptr.is_null() {
            return Err(format!("model mmap failed ({e})"));
        }
        log::info!("semantic: model partition {part_size} B mapped at {model_ptr:p}");

        let mut arena = vec![0u8; ARENA_BYTES];
        let rc = unsafe {
            sys::tflite::semantic_model_load(
                model_ptr as *const u8,
                part_size,
                arena.as_mut_ptr(),
                arena.len(),
            )
        };
        if rc != 0 {
            return Err(format!("model load failed rc={rc}: {}", unsafe { last_error() }));
        }

        let in_size = unsafe { sys::tflite::semantic_input_size() };
        let in_ptr = unsafe { sys::tflite::semantic_input_data() };
        let n_out = unsafe { sys::tflite::semantic_output_count() };
        let out_size = unsafe { sys::tflite::semantic_output_size(0) };
        let out_ptr = unsafe { sys::tflite::semantic_output_data(0) };
        let scale = unsafe { sys::tflite::semantic_output_scale(0) };
        let zp = unsafe { sys::tflite::semantic_output_zero_point(0) };
        if in_ptr.is_null()
            || in_size < INPUT_W * INPUT_H
            || n_out < 1
            || out_ptr.is_null()
            || out_size < EMB_DIM
        {
            return Err(format!(
                "unexpected model I/O (input {in_size} B, {n_out} outputs, output {out_size} B)"
            ));
        }
        log::info!(
            "semantic: model ready — input {in_size} B, output {out_size} B (scale {scale}, zp {zp}), arena {ARENA_BYTES} B"
        );
        // Per-op µs breakdown, one line per TFLite op after each invoke.
        unsafe { sys::tflite::semantic_profile_enable(1) };
        Ok(Embedder { in_ptr, out_ptr, scale, zp, _arena: arena, _mmap_handle: mmap_handle })
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn zero_point(&self) -> i32 {
        self.zp
    }

    /// Downscale `gray` (`src_w` x `src_h`, 4:1 to the 160x120 model input),
    /// run one inference and copy the `EMB_DIM` descriptor into `out`.
    pub fn embed(
        &mut self,
        gray: &[u8],
        src_w: usize,
        src_h: usize,
        out: &mut [u8; EMB_DIM],
    ) -> Result<(), String> {
        if gray.len() < src_w * src_h {
            return Err(format!("frame too short ({} B < {src_w}x{src_h})", gray.len()));
        }
        if src_w / 4 != INPUT_W || src_h / 4 != INPUT_H {
            return Err(format!("frame {src_w}x{src_h} does not downscale to {INPUT_W}x{INPUT_H}"));
        }
        let input = unsafe { slice::from_raw_parts_mut(self.in_ptr, INPUT_W * INPUT_H) };
        if !downscale::downscale_4x4(gray, src_w, src_h, input) {
            return Err("downscale_4x4 failed".into());
        }
        let rc = unsafe { sys::tflite::semantic_invoke() };
        if rc != 0 {
            return Err(format!("invoke failed rc={rc}: {}", unsafe { last_error() }));
        }
        let emb = unsafe { slice::from_raw_parts(self.out_ptr, EMB_DIM) };
        out.copy_from_slice(emb);
        Ok(())
    }
}

pub fn run() -> ! {
    log::info!(
        "semantic: calc8 int8 encoder on {}x{} grayscale (model from the 'model' flash partition)",
        INPUT_W,
        INPUT_H
    );

    // `model` is a raw data partition (no filesystem); Embedder maps it in
    // place from flash and builds the interpreter (no PSRAM model copy).
    let mut embedder = match Embedder::init() {
        Ok(e) => e,
        Err(e) => {
            log::error!("semantic: {e} — idling");
            idle();
        }
    };

    // ---- Camera: OV3660 VGA grayscale, captured every tick ----
    let cam = match camera::Camera::init(&camera::CameraConfig {
        frame_size: camera::FrameSize::Vga,
        pixel_format: camera::PixelFormat::Grayscale,
        ..camera::CameraConfig::with_pins(camera::CameraPins::FREENOVE_ESP32S3_WROOM)
    }) {
        Ok(cam) => {
            // Manual exposure to cut motion blur; AGC auto capped at 64x.
            // set_gain_ceiling writes the raw gain registers (the driver enum is broken).
            let aec = cam.set_exposure_ctrl(false);
            let agc = cam.set_gain_ctrl(true);
            let aecv = cam.set_aec_value(45);
            let ceil = cam.set_gain_ceiling(64);
            log::info!(
                "semantic: camera exposure: manual aec=45, gain auto, ceiling 64x (rets {aec:?}/{agc:?}/{aecv:?}/{ceil:?})"
            );
            cam
        }
        Err(e) => {
            log::error!("semantic: camera init failed ({e}) — idling");
            idle();
        }
    };

    // ---- SoftAP + TCP listener: the laptop connects and reads EMB1 records ----
    let peripherals = Peripherals::take().expect("Peripherals::take");
    let sysloop = EspSystemEventLoop::take().expect("EspSystemEventLoop::take");
    let nvs = EspDefaultNvsPartition::take().expect("EspDefaultNvsPartition::take");
    let mut wifi = BlockingWifi::wrap(
        EspWifi::new(peripherals.modem, sysloop.clone(), Some(nvs)).expect("EspWifi::new"),
        sysloop,
    )
    .expect("BlockingWifi::wrap");
    wifi.set_configuration(&Configuration::AccessPoint(AccessPointConfiguration {
        ssid: crate::AP_SSID.try_into().unwrap(),
        password: crate::AP_PASS.try_into().unwrap(),
        auth_method: AuthMethod::WPA2Personal,
        ..AccessPointConfiguration::default()
    }))
    .expect("AP config");
    wifi.start().expect("AP start");
    // The AP netif gets its static IP shortly after start; poll it so the log
    // prints the real address (same pattern as map_mode in main.rs).
    let ap_ip = {
        let mut ip = None;
        for _ in 0..100 {
            let netif = wifi.wifi().ap_netif();
            if netif.is_up().unwrap_or(false) {
                ip = netif.get_ip_info().ok().map(|i| i.ip);
                break;
            }
            FreeRtos::delay_ms(100);
        }
        ip.unwrap_or(crate::AP_IP_FALLBACK)
    };
    log::info!(
        "semantic: SoftAP \"{}\" up — connect to {ap_ip}:{}",
        crate::AP_SSID,
        crate::TCP_PORT
    );
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, crate::TCP_PORT))
        .expect("semantic TCP bind");
    listener.set_nonblocking(true).expect("listener set_nonblocking");
    log::info!("semantic: listening on 0.0.0.0:{}", crate::TCP_PORT);

    // Raw grayscale frame copy, kept only while a laptop is connected.
    let mut pix = vec![0u8; SRC_W * SRC_H];
    let mut tx: Vec<u8> = Vec::with_capacity(EMB_HEADER_BYTES + pix.len() + EMB_DIM);
    let mut client: Option<TcpStream> = None;

    let mut n = 0u64;
    let mut last_loop = Instant::now();
    loop {
        let loop_t = Instant::now();
        // True inter-frame period (includes the previous frame's log + accept).
        let period_us = loop_t.duration_since(last_loop).as_micros().max(1);
        last_loop = loop_t;
        // One laptop at a time: accept a new one when none is connected, and
        // drop a dead/slow one on send failure.
        if client.is_none() {
            match listener.accept() {
                Ok((s, peer)) => {
                    let _ = s.set_nonblocking(false);
                    let _ = s.set_nodelay(true);
                    let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
                    log::info!("semantic: laptop connected: {peer}");
                    client = Some(s);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => log::warn!("semantic: accept failed ({e})"),
            }
        }

        let t = Instant::now();
        let frame = match cam.capture() {
            Some(f) => f,
            None => {
                log::warn!("semantic: capture failed — skipping frame");
                FreeRtos::delay_ms(100);
                continue;
            }
        };
        let cap_us = t.elapsed().as_micros();
        if frame.width() != SRC_W
            || frame.height() != SRC_H
            || frame.data().len() != pix.len()
        {
            log::error!(
                "semantic: unexpected frame {}x{} ({} B; want {SRC_W}x{SRC_H} gray) — skipping",
                frame.width(),
                frame.height(),
                frame.data().len()
            );
            continue;
        }
        // Run calc8 (4x4-downscale into the model input + invoke) and keep the
        // gray bytes for the wire; the fb must be dropped before invoking with
        // fb_count == 1, so grab both first.
        let t = Instant::now();
        let mut emb = [0u8; EMB_DIM];
        let r = embedder.embed(frame.data(), SRC_W, SRC_H, &mut emb);
        if client.is_some() {
            pix.copy_from_slice(frame.data());
        }
        let infer_us = t.elapsed().as_micros();
        drop(frame);
        n += 1;
        if let Err(e) = r {
            log::error!(
                "semantic: invoke {n} failed: {e} (cap {cap_us} us, infer {infer_us} us)"
            );
            continue;
        }
        unsafe { sys::tflite::semantic_profile_log() };

        let t = Instant::now();
        if client.is_some() {
            build_emb_frame(&mut tx, &pix, &emb, embedder.scale(), embedder.zero_point());
        }
        let build_us = t.elapsed().as_micros();
        let t = Instant::now();
        let mut streamed = false;
        if let Some(s) = client.as_mut() {
            match s.write_all(&tx) {
                Ok(()) => streamed = true,
                Err(e) => {
                    log::warn!("semantic: send failed ({e}) — dropping client");
                    client = None;
                }
            }
        }
        let send_us = t.elapsed().as_micros();

        // `core` is all on-device compute (capture + downscale + inference);
        // `send` is the network write. No pacing: max throughput.
        let core_us = cap_us + infer_us;
        let total_us = loop_t.elapsed().as_micros().max(1);
        log::info!(
            "semantic {n}: emb[{EMB_DIM}]{} — core {core_us} us (cap {cap_us} + infer {infer_us}) | build {build_us} | send {send_us} | total {total_us} us | period {period_us} us ({:.2} fps)",
            if streamed { " (streamed)" } else { "" },
            1_000_000.0 / period_us as f32,
        );
        // Let IDLE0 run: the loop is otherwise a CPU-bound spin and trips the
        // task watchdog (inference alone is seconds here).
        FreeRtos::delay_ms(1);
    }
}

/// Assemble one EMB1 record into `buf`: u32 LE len | "EMB1" | u8 fmt | u16 w |
/// u16 h | u16 ndim | f32 emb_scale | i32 emb_zp | w*h gray bytes | ndim u8.
fn build_emb_frame(buf: &mut Vec<u8>, gray: &[u8], emb: &[u8], scale: f32, zp: i32) {
    // Payload after the u32 length: header(19) + pixels + embedding bytes.
    let payload = EMB_HEADER_BYTES + gray.len() + emb.len();
    buf.clear();
    buf.extend_from_slice(&(payload as u32).to_le_bytes());
    buf.extend_from_slice(MAGIC_EMB);
    buf.push(FMT_GRAYSCALE);
    buf.extend_from_slice(&(SRC_W as u16).to_le_bytes());
    buf.extend_from_slice(&(SRC_H as u16).to_le_bytes());
    buf.extend_from_slice(&(emb.len() as u16).to_le_bytes());
    buf.extend_from_slice(&scale.to_le_bytes());
    buf.extend_from_slice(&zp.to_le_bytes());
    buf.extend_from_slice(gray);
    buf.extend_from_slice(emb);
}

fn idle() -> ! {
    loop {
        FreeRtos::delay_ms(5000);
    }
}

unsafe fn last_error() -> String {
    let p = sys::tflite::semantic_last_error();
    if p.is_null() {
        "?".into()
    } else {
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
