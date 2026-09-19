//! Semantic bring-up task: run the int8 yolov8n TFLite model (esp-tflite-micro
//! + esp-nn kernels) over a 640x640x3 PSRAM framebuffer in a loop.
//!
//! The model blob lives in the `model` flash partition (see partitions.csv) and
//! is memory-mapped, so only the TFLite-Micro arena is PSRAM. Each pass a 640x480
//! RGB565 OV3660 frame is letterboxed (114-gray) into the model's input tensor
//! in place. The task brings up the SoftAP + TCP server and streams each frame
//! plus its bounding boxes as SEM1 records (see scripts/receive_semantic.py).

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

/// Partition label + custom data subtype from partitions.csv.
const MODEL_LABEL: &[u8] = b"model\0";
const MODEL_SUBTYPE: u32 = 0x40;

const INPUT_W: usize = 640;
const INPUT_H: usize = 640;
const INPUT_C: usize = 3;

/// Camera capture size (VGA RGB565), letterboxed into the square model input.
const SRC_W: usize = 640;
const SRC_H: usize = 480;
/// 8-bit letterbox gray (yolov8 convention), centered above and below the frame.
const PAD_GRAY: u8 = 114;

/// TFLite-Micro working arena (PSRAM). yolov8n's peak activation working set is
/// a few MB at 640x640 int8.
const ARENA_BYTES: usize = 6 * 1024 * 1024;

// ---- Model outputs (models/yolov8n.tflite) ----
// Two int8 tensors: boxes [1, 4, anchors] (cx,cy,w,h in letterbox px) and class
// scores [1, 80, anchors]. Separate outputs = separate scales, so scores keep
// their 0..1 range (a single shared Concat scale was box-dominated and useless).
const NUM_CLASSES: usize = 80;
const BOX_CHANNELS: usize = 4;
/// Head anchors: 640/8, 640/16, 640/32 grids -> 80*80 + 40*40 + 20*20.
const ANCHORS: usize = 8400;
/// Minimum class score kept as a detection (a probability, 0..1).
const CONF_THRESHOLD: f32 = 0.25;
/// Greedy per-class NMS IoU threshold.
const NMS_IOU: f32 = 0.45;
/// Max detections produced per frame (also the decode scratch size).
const MAX_DETECTIONS: usize = 64;

// ---- SEM1 wire format (semantic task -> laptop, see scripts/receive_semantic.py) ----
/// Frame+detections record magic.
const MAGIC_SEM: &[u8; 4] = b"SEM1";
/// Wire pixel-format id: 2 = RGB565 (our own enum, not the driver's).
const FMT_RGB565: u8 = 2;
/// SEM1 header before the pixels: len(4) + magic(4) + fmt(1) + w(2) + h(2) + ndet(2).
const SEM_HEADER_BYTES: usize = 15;
/// Per-detection wire bytes: class u8 + 5 x f32.
const SEM_DET_BYTES: usize = 21;

/// One detection: COCO class id + box in SOURCE (640x480) pixels, top-left
/// `x`/`y` and size `w`/`h`.
#[derive(Clone, Copy)]
pub struct Detection {
    pub class_id: u8,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

impl Detection {
    const EMPTY: Detection = Detection {
        class_id: 0,
        x: 0.0,
        y: 0.0,
        w: 0.0,
        h: 0.0,
        score: 0.0,
    };
}

pub fn run() -> ! {
    log::info!(
        "semantic: yolov8n int8 via esp-nn on a {}x{}x{} buffer (model from the 'model' flash partition)",
        INPUT_W,
        INPUT_H,
        INPUT_C
    );

    // `model` is a raw data partition (no filesystem); map it so the flatbuffer
    // is read in place from flash (no PSRAM copy, matches TFLite-Micro's const
    // model contract).
    let part = unsafe {
        sys::esp_partition_find_first(
            sys::esp_partition_type_t_ESP_PARTITION_TYPE_DATA,
            MODEL_SUBTYPE,
            MODEL_LABEL.as_ptr() as *const c_char,
        )
    };
    if part.is_null() {
        log::error!("semantic: no 'model' partition (flash the .tflite via cargo_run.sh) — idling");
        idle();
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
        log::error!("semantic: model mmap failed ({e}) — idling");
        idle();
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
        log::error!(
            "semantic: model load failed rc={rc}: {} — idling",
            unsafe { last_error() }
        );
        idle();
    }

    let in_size = unsafe { sys::tflite::semantic_input_size() };
    let n_out = unsafe { sys::tflite::semantic_output_count() };
    let in_ptr = unsafe { sys::tflite::semantic_input_data() };
    // Identify the two outputs by element count (order is converter-defined).
    let mut box_idx = usize::MAX;
    let mut cls_idx = usize::MAX;
    for i in 0..n_out {
        let sz = unsafe { sys::tflite::semantic_output_size(i) };
        if sz == BOX_CHANNELS * ANCHORS {
            box_idx = i;
        } else if sz == NUM_CLASSES * ANCHORS {
            cls_idx = i;
        }
    }
    if box_idx == usize::MAX || cls_idx == usize::MAX {
        log::error!(
            "semantic: unexpected outputs ({n_out}) — need boxes {} B, scores {} B — idling",
            BOX_CHANNELS * ANCHORS,
            NUM_CLASSES * ANCHORS
        );
        idle();
    }
    let box_ptr = unsafe { sys::tflite::semantic_output_data(box_idx) } as *const i8;
    let cls_ptr = unsafe { sys::tflite::semantic_output_data(cls_idx) } as *const i8;
    let box_scale = unsafe { sys::tflite::semantic_output_scale(box_idx) };
    let box_zp = unsafe { sys::tflite::semantic_output_zero_point(box_idx) };
    let cls_scale = unsafe { sys::tflite::semantic_output_scale(cls_idx) };
    let cls_zp = unsafe { sys::tflite::semantic_output_zero_point(cls_idx) };
    log::info!(
        "semantic: model ready — input {in_size} B, {n_out} outputs (boxes@{box_idx} scale {box_scale}, scores@{cls_idx} scale {cls_scale}), arena {ARENA_BYTES} B"
    );

    // ---- Camera: OV3660 VGA RGB565, captured every pass ----
    let cam = match camera::Camera::init(&camera::CameraConfig {
        frame_size: camera::FrameSize::Vga,
        pixel_format: camera::PixelFormat::Rgb565,
        ..camera::CameraConfig::with_pins(camera::CameraPins::FREENOVE_ESP32S3_WROOM)
    }) {
        Ok(cam) => cam,
        Err(e) => {
            log::error!("semantic: camera init failed ({e}) — idling");
            idle();
        }
    };

    // ---- SoftAP + TCP listener: the laptop connects and reads SEM1 records ----
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

    // Decode scratch, reused every frame (kept off the stack; the main task
    // stack is small).
    let mut dets = vec![Detection::EMPTY; MAX_DETECTIONS];
    // Raw RGB565 frame copy, kept only while a laptop is connected.
    let mut pix = vec![0u8; SRC_W * SRC_H * 2];
    let mut tx: Vec<u8> =
        Vec::with_capacity(SEM_HEADER_BYTES + pix.len() + MAX_DETECTIONS * SEM_DET_BYTES);
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
        if frame.width() != SRC_W || frame.height() != SRC_H || frame.data().len() != pix.len() {
            log::error!(
                "semantic: unexpected frame {}x{} ({} B; want {SRC_W}x{SRC_H} RGB565) — skipping",
                frame.width(),
                frame.height(),
                frame.data().len()
            );
            continue;
        }
        // Letterbox into the int8 input in place, and keep the RGB565 bytes for
        // the wire (the fb must be dropped before invoke with fb_count == 1).
        let t = Instant::now();
        unsafe { fill_input(in_ptr, frame.data()) };
        if client.is_some() {
            pix.copy_from_slice(frame.data());
        }
        let prep_us = t.elapsed().as_micros();
        drop(frame);

        let t = Instant::now();
        let rc = unsafe { sys::tflite::semantic_invoke() };
        let infer_us = t.elapsed().as_micros();
        n += 1;
        if rc != 0 {
            log::error!(
                "semantic: invoke {n} failed rc={rc}: {} (cap {cap_us} us, prep {prep_us} us, infer {infer_us} us)",
                unsafe { last_error() }
            );
            continue;
        }
        let t = Instant::now();
        let boxes = unsafe { slice::from_raw_parts(box_ptr, BOX_CHANNELS * ANCHORS) };
        let classes = unsafe { slice::from_raw_parts(cls_ptr, NUM_CLASSES * ANCHORS) };
        let nd = decode_detections(boxes, classes, box_scale, box_zp, cls_scale, cls_zp, &mut dets);
        let decode_us = t.elapsed().as_micros();

        let t = Instant::now();
        if client.is_some() {
            build_sem_frame(&mut tx, &pix, &dets[..nd]);
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

        // `core` is all on-device compute (capture + letterbox/copy + inference
        // + decode); `send` is the network write. No pacing: max throughput.
        let core_us = cap_us + prep_us + infer_us + decode_us;
        let total_us = loop_t.elapsed().as_micros().max(1);
        let mut list = String::new();
        for d in &dets[..nd] {
            use core::fmt::Write;
            let _ = write!(
                list,
                " [c{} {:.0},{:.0} {:.0}x{:.0} {:.2}]",
                d.class_id, d.x, d.y, d.w, d.h, d.score
            );
        }
        log::info!(
            "semantic {n}: {nd} dets{} — core {core_us} us (cap {cap_us} + prep {prep_us} + infer \
             {infer_us} + decode {decode_us}) | build {build_us} | send {send_us} | total {total_us} \
             us | period {period_us} us ({:.2} fps){list}",
            if streamed { " (streamed)" } else { "" },
            1_000_000.0 / period_us as f32,
        );
    }
}

/// Assemble one SEM1 record into `buf`: u32 LE len | "SEM1" | u8 fmt | u16 w |
/// u16 h | u16 ndet | w*h*2 RGB565 bytes | ndet x {u8 class, f32 x,y,w,h,score}.
fn build_sem_frame(buf: &mut Vec<u8>, rgb565: &[u8], dets: &[Detection]) {
    let nd = dets.len();
    // Payload after the u32 length: magic(4) + fmt(1) + w(2) + h(2) + ndet(2)
    // + pixels + nd * 21.
    let payload = 11 + rgb565.len() + nd * SEM_DET_BYTES;
    buf.clear();
    buf.extend_from_slice(&(payload as u32).to_le_bytes());
    buf.extend_from_slice(MAGIC_SEM);
    buf.push(FMT_RGB565);
    buf.extend_from_slice(&(SRC_W as u16).to_le_bytes());
    buf.extend_from_slice(&(SRC_H as u16).to_le_bytes());
    buf.extend_from_slice(&(nd as u16).to_le_bytes());
    buf.extend_from_slice(rgb565);
    for d in dets {
        buf.push(d.class_id);
        buf.extend_from_slice(&d.x.to_le_bytes());
        buf.extend_from_slice(&d.y.to_le_bytes());
        buf.extend_from_slice(&d.w.to_le_bytes());
        buf.extend_from_slice(&d.h.to_le_bytes());
        buf.extend_from_slice(&d.score.to_le_bytes());
    }
}

/// Copy a 640x480 RGB565 frame into the int8 640x640 input tensor, centered with
/// 80 rows of 114-gray letterbox above and below.
///
/// The input is quantized scale 1/255, zero_point -128, so an 8-bit pixel `p`
/// becomes `p ^ 0x80` (= p - 128) and 114-gray becomes 0x8E.
unsafe fn fill_input(in_ptr: *mut i8, src: &[u8]) {
    let dst = in_ptr as *mut u8;
    let row = INPUT_W * INPUT_C;
    let top = ((INPUT_H - SRC_H) / 2) * row;
    let pad = PAD_GRAY ^ 0x80;
    // memset the whole tensor to the pad color, then overwrite the frame band.
    ptr::write_bytes(dst, pad, INPUT_W * INPUT_H * INPUT_C);
    for y in 0..SRC_H {
        let s = src.as_ptr().add(y * SRC_W * 2);
        let d = dst.add(top + y * row);
        for x in 0..SRC_W {
            let v = (*s.add(x * 2) as u16) | ((*s.add(x * 2 + 1) as u16) << 8);
            let r5 = ((v >> 11) & 0x1f) as u8;
            let g6 = ((v >> 5) & 0x3f) as u8;
            let b5 = (v & 0x1f) as u8;
            *d.add(x * 3) = (((r5 << 3) | (r5 >> 2)) ^ 0x80) as u8;
            *d.add(x * 3 + 1) = (((g6 << 2) | (g6 >> 4)) ^ 0x80) as u8;
            *d.add(x * 3 + 2) = (((b5 << 3) | (b5 >> 2)) ^ 0x80) as u8;
        }
    }
}

fn dequant(q: i8, scale: f32, zp: i32) -> f32 {
    (q as i32 - zp) as f32 * scale
}

/// Argmax of the 80 class rows of anchor `j`, on the raw int8 values (the class
/// scale is > 0, so it preserves order); returns (class, raw score).
fn argmax_class(classes: &[i8], anchors: usize, j: usize) -> (u8, i8) {
    let mut best_c = 0u8;
    let mut best_q = i8::MIN;
    for c in 0..NUM_CLASSES {
        let q = classes[c * anchors + j];
        if q > best_q {
            best_q = q;
            best_c = c as u8;
        }
    }
    (best_c, best_q)
}

/// Decode the two int8 head tensors and NMS them. Returns the number of
/// detections written to `dets`.
///
/// The input was letterboxed 640x480 -> 640x640 with only a top/bottom row pad
/// (see `fill_input`), so undo the pad and the coords are already source-scale.
fn decode_detections(
    boxes: &[i8],
    classes: &[i8],
    box_scale: f32,
    box_zp: i32,
    cls_scale: f32,
    cls_zp: i32,
    dets: &mut [Detection],
) -> usize {
    if boxes.len() < BOX_CHANNELS * ANCHORS || dets.is_empty() {
        return 0;
    }
    let pad_y = ((INPUT_H - SRC_H) / 2) as f32;
    let mut n = 0usize;
    for j in 0..ANCHORS {
        let (class_id, q) = argmax_class(classes, ANCHORS, j);
        let score = dequant(q, cls_scale, cls_zp);
        if score < CONF_THRESHOLD {
            continue;
        }
        if n == dets.len() {
            break;
        }
        let cx = dequant(boxes[j], box_scale, box_zp);
        let cy = dequant(boxes[ANCHORS + j], box_scale, box_zp) - pad_y;
        let w = dequant(boxes[2 * ANCHORS + j], box_scale, box_zp).clamp(0.0, SRC_W as f32);
        let h = dequant(boxes[3 * ANCHORS + j], box_scale, box_zp).clamp(0.0, SRC_H as f32);
        dets[n] = Detection {
            class_id,
            x: (cx - w * 0.5).clamp(0.0, SRC_W as f32),
            y: (cy - h * 0.5).clamp(0.0, SRC_H as f32),
            w,
            h,
            score,
        };
        n += 1;
    }
    nms(dets, n)
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
    let iy = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
    let inter = ix.max(0.0) * iy.max(0.0);
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// In-place greedy NMS (score-descending, per class). Returns the kept count.
fn nms(dets: &mut [Detection], n: usize) -> usize {
    for i in 1..n {
        let d = dets[i];
        let mut k = i;
        while k > 0 && dets[k - 1].score < d.score {
            dets[k] = dets[k - 1];
            k -= 1;
        }
        dets[k] = d;
    }
    let mut kept = 0usize;
    for i in 0..n {
        if (0..kept).all(|j| dets[j].class_id != dets[i].class_id || iou(&dets[j], &dets[i]) <= NMS_IOU) {
            dets[kept] = dets[i];
            kept += 1;
        }
    }
    kept
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
