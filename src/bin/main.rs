//! SoftAP + TCP server: on connect the laptop sends STRT (map run: streams VOX2
//! frames, ends with VOXD) or MAPU (upload a built map + intrinsics; MCU stores
//! it and idles in localize mode). See scripts/receive_map.py for the protocol.

// The old map/localize entry point is kept but unused while the semantic task
// (src/semantic.rs) is the startup task.
#![allow(dead_code)]

#[path = "../camera.rs"]
mod camera;
#[path = "../semantic.rs"]
mod semantic;

use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use esp_idf_hal::delay::FreeRtos;
use esp_idf_hal::peripherals::Peripherals;
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys::EspError;
use esp_idf_svc::wifi::{
    AccessPointConfiguration, AuthMethod, BlockingWifi, Configuration, EspWifi,
};
use vo_box_lite::pyramid;
use vo_box_lite::rbrief;

/// SoftAP credentials the laptop joins with (WPA2 passphrase must be >= 8 chars).
const AP_SSID: &str = "vo-box";
const AP_PASS: &str = "vobox1234"; // TODO: real passphrase
const TCP_PORT: u16 = 5000;
/// IDF v5.5 default AP IP; fallback if the netif-up poll times out.
const AP_IP_FALLBACK: Ipv4Addr = Ipv4Addr::new(192, 168, 71, 1);
/// Command magic laptop -> MCU, sent right after connect: `u32 LE n | "STRT"`
/// [+ payload]. The MCU streams nothing until STRT arrives.
const MAGIC_START: &[u8; 4] = b"STRT";
/// STRT payload defaults (the command can override; see read_command):
/// 300 s run, one frame per 1000 ms (0 = max rate), 60 s STRT timeout.
const DEFAULT_DURATION_S: u64 = 300;
const DEFAULT_INTERVAL_MS: u64 = 1000;
const START_TIMEOUT_S: u64 = 60;
/// Record magic for the final "done mapping" record.
const MAGIC_DONE: &[u8; 4] = b"VOXD";
/// Laptop -> MCU map upload: `u32 LE n | "MAPU" | u8 model | 4 x f32 params |
/// u32 n_points | n_points x {f32 x,y,z, 32 B desc}` (n = full record length).
const MAGIC_MAP_UPLOAD: &[u8; 4] = b"MAPU";
/// MCU -> laptop map-upload ack: `u32 LE n` (= 8) | b"MAPK" | u32 LE n_points.
const MAGIC_MAP_OK: &[u8; 4] = b"MAPK";
/// Camera model ids accepted in a MAPU header (COLMAP `SIMPLE_RADIAL` only).
const CAMERA_MODEL_SIMPLE_RADIAL: u8 = 1;
/// Sanity cap on uploaded points (44 B each) to bound the PSRAM allocation.
const MAX_MAP_POINTS: usize = 100_000;
/// Record magic for frame+features (VOX2).
const MAGIC: &[u8; 4] = b"VOX2";
/// esp32-camera PIXFORMAT_GRAYSCALE (the only format we configure).
const FMT_GRAYSCALE: u8 = 3;
/// Sensor resolution: VGA 640x480 grayscale, streamed raw. No on-device
/// downscale/pyramid — the laptop runs the same Rust extractor.
const CAM_W: usize = 640;
const CAM_H: usize = 480;

/// Entry point: run the map task (SoftAP + raw VOX2 frame stream).
fn main() {
    map_mode().expect("map task exited with an error");
}

/// Map task: SoftAP + VOX2 frame stream / map upload.
#[allow(dead_code)]
fn map_mode() -> Result<(), EspError> {
    // Required once: links the esp-idf runtime patches (esp-idf-template#71).
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    log::info!("=== vo-box-lite: frame stream (VGA 640x480 raw; laptop extracts) ===");

    // ---- SoftAP: WIFI_MODE_AP, no STA ----
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
    wifi.start()?; // blocking: returns once the AP is started

    // The AP netif gets its static IP (192.168.71.1 on IDF v5.5) shortly after
    // start; poll it so the log prints the real address.
    let ap_ip = {
        let mut ip = None;
        for _ in 0..100 {
            let netif = wifi.wifi().ap_netif();
            if netif.is_up()? {
                ip = Some(netif.get_ip_info()?.ip);
                break;
            }
            FreeRtos::delay_ms(100);
        }
        ip.unwrap_or(AP_IP_FALLBACK)
    };
    log::info!("SoftAP \"{AP_SSID}\" up — connect to {ap_ip}:{TCP_PORT} from the laptop");

    // ---- Camera: optional, the TCP server still comes up without it ----
    let camera = match camera::Camera::init(&camera::CameraConfig {
        frame_size: camera::FrameSize::Vga, // 640x480 gray (CAM_W x CAM_H)
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
                "camera exposure: manual aec=45, gain auto, ceiling 64x (rets {aec:?}/{agc:?}/{aecv:?}/{ceil:?})"
            );
            log::info!("camera ready");
            Some(cam)
        }
        Err(e) => {
            log::error!("camera init failed ({e}); no frames will stream");
            None
        }
    };

    // ---- TCP server (frame stream: capture + raw VOX2 upload) ----
    let listener =
        TcpListener::bind((Ipv4Addr::UNSPECIFIED, TCP_PORT)).expect("TCP bind 0.0.0.0:5000 failed");
    log::info!("listening on 0.0.0.0:{TCP_PORT}");

    feature_server(listener, camera.as_ref())
}

/// Serve one connection = one command: STRT streams paced VOX2 frames until the
/// run deadline then sends VOXD; MAPU stores the uploaded map and acks. Either
/// way the connection is dropped and the next one accepted.
fn feature_server(listener: TcpListener, camera: Option<&camera::Camera>) -> ! {
    // Pipeline buffers are sized for the VGA frame dims and reused every frame
    // (no per-frame allocation).
    let mut pipe: Option<MapPipeline> = None;
    // Last uploaded map; the localize task (not written) will consume it.
    let mut map: Option<LocalMap> = None;
    loop {
        let (mut stream, peer) = match listener.accept() {
            Ok(conn) => conn,
            Err(e) => {
                log::error!("accept failed: {e}");
                FreeRtos::delay_ms(500);
                continue;
            }
        };
        log::info!("laptop connected: {peer} — waiting for its command");
        match map.as_ref() {
            Some(m) => log::info!("localize map loaded ({} points)", m.points.len()),
            None => log::info!("no localize map loaded"),
        }
        // Disable Nagle (pairs with the enlarged lwIP send buffer in sdkconfig).
        if let Err(e) = stream.set_nodelay(true) {
            log::warn!("set_nodelay failed ({e})");
        }
        if pipe.is_none() {
            pipe = Some(MapPipeline::new(CAM_W, CAM_H));
        }
        let p = pipe.as_mut().unwrap();

        // The MCU streams nothing until a command arrives: STRT starts a run,
        // MAPU stores a map. Clean disconnect -> Ok(None); timeout/garbage -> Err.
        let (dur_s, interval_ms) = match read_command(&mut stream) {
            Ok(Some(Command::Start { duration_s, interval_ms })) => (duration_s, interval_ms),
            Ok(Some(Command::MapUpload(m))) => {
                let n_points = m.points.len() as u32;
                log::info!("map upload: {n_points} points, model {}, params {:?}",
                           m.model, m.params);
                map = Some(m);
                build_map_ok_record(&mut p.tx, n_points);
                if let Err(e) = send_record(&mut stream, &p.tx) {
                    log::warn!("map ack send failed ({e})");
                } else {
                    log::info!("map stored — localize mode (idle until the next STRT)");
                }
                continue;
            }
            Ok(None) => {
                log::info!("client {peer} disconnected before sending a command");
                continue;
            }
            Err(e) => {
                log::warn!("waiting for a command from {peer} failed ({e}); dropping");
                continue;
            }
        };
        // Command phase used non-blocking reads; stream writes must block
        // again (a WouldBlock on a 200 KB VOX2 frame would look like a drop).
        if let Err(e) = stream.set_nonblocking(false) {
            log::warn!("set_nonblocking(false) failed ({e}); dropping client");
            continue;
        }

        log::info!(
            "starting a {dur_s}s frame run for {peer} (frame every {interval_ms} ms; laptop extracts)"
        );
        let run_start = Instant::now();
        let deadline = run_start + Duration::from_secs(dur_s);
        let mut sent = 0u64;
        let mut interrupted = false; // laptop dropped mid-run
        while Instant::now() < deadline {
            let frame_start = Instant::now();
            let cam = match camera {
                Some(cam) => cam,
                None => {
                    FreeRtos::delay_ms(200); // camera down: idle, keep the connection
                    continue;
                }
            };

            let mut tm = match map_frame(cam, p) {
                Ok(t) => t,
                Err(e) => {
                    log::warn!("map_frame failed ({e})");
                    FreeRtos::delay_ms(200);
                    continue;
                }
            };
            // Send the assembled VOX2 record (raw frame, 0 features).
            let t_send = Instant::now();
            let send_max_ms = match send_record(&mut stream, &p.tx) {
                Ok(v) => v,
                Err(_) => {
                    interrupted = true;
                    break;
                }
            };
            tm.send_us = t_send.elapsed().as_micros() as u64;
            sent += 1;

            // Pace to one frame per `interval_ms` (capture + upload
            // time counts towards the interval; 0 = as fast as possible).
            let wait_ms = interval_ms.saturating_sub(frame_start.elapsed().as_millis() as u64);
            if wait_ms > 0 {
                FreeRtos::delay_ms(wait_ms as u32);
            }
            tm.pace_us = wait_ms * 1000;

            // Debug breakdown (every frame at a paced run, every 20th at max rate).
            if interval_ms > 0 || sent % 20 == 0 {
                let total_us = frame_start.elapsed().as_micros() as u64;
                log::info!(
                    "frame {sent} @ {}s: {}x{} — total {total_us} us | capture {} | build {} | send {} (max {}ms) | pace {}",
                    run_start.elapsed().as_secs(),
                    p.w,
                    p.h,
                    tm.capture_us,
                    tm.build_us,
                    tm.send_us,
                    send_max_ms,
                    tm.pace_us,
                );
            }
        }

        if !interrupted {
            // Deadline reached: tell the laptop the run is done, then drop the
            // connection (the laptop extracts features + runs COLMAP).
            build_done_record(&mut p.tx, sent, 0);
            if let Err(e) = send_record(&mut stream, &p.tx) {
                log::warn!("done record send failed ({e})");
            } else {
                log::info!("map run complete: {sent} frames -> {peer}; VOXD done record sent");
            }
        } else {
            log::info!("client {peer} disconnected mid-run after {sent} frames; re-accepting");
        }
    }
}

/// One command on a fresh laptop connection (length-prefixed record).
enum Command {
    Start { duration_s: u64, interval_ms: u64 },
    MapUpload(LocalMap),
}

/// Uploaded map: COLMAP-refined intrinsics + 3D points with rBRIEF descriptors.
/// Held in PSRAM until the next upload / reboot.
struct LocalMap {
    model: u8,
    /// SIMPLE_RADIAL params: f, cx, cy, k1.
    params: [f32; 4],
    points: Vec<LocalMapPoint>,
}

#[allow(dead_code)] // consumed by the localize task (not written yet)
struct LocalMapPoint {
    xyz: [f32; 3],
    desc: rbrief::Descriptor,
}

/// Read the laptop's command (STRT = map run, MAPU = map upload). Ok(None) on a
/// clean disconnect, Err on timeout / bad record. MAPU leaves the stream
/// blocking; STRT leaves it non-blocking.
fn read_command(stream: &mut TcpStream) -> std::io::Result<Option<Command>> {
    let deadline = Instant::now() + Duration::from_secs(START_TIMEOUT_S);
    stream.set_nonblocking(true)?;
    let mut len_buf = [0u8; 4];
    match read_exact_poll(stream, &mut len_buf, deadline) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n = u32::from_le_bytes(len_buf) as usize;
    let mut magic = [0u8; 4];
    if n < 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad command length {n}"),
        ));
    }
    read_exact_poll(stream, &mut magic, deadline)?;
    if &magic == MAGIC_START {
        if !(4..=12).contains(&n) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad STRT length {n}"),
            ));
        }
        let mut body = [0u8; 12];
        if n > 4 {
            read_exact_poll(stream, &mut body[4..n], deadline)?;
        }
        // Optional payload: u32 duration_s, u32 interval_ms (0 = max rate).
        let mut dur_s = DEFAULT_DURATION_S;
        let mut interval_ms = DEFAULT_INTERVAL_MS;
        if n >= 8 {
            dur_s = u32::from_le_bytes(body[4..8].try_into().unwrap()) as u64;
            if dur_s == 0 {
                dur_s = DEFAULT_DURATION_S;
            }
        }
        if n >= 12 {
            interval_ms = u32::from_le_bytes(body[8..12].try_into().unwrap()) as u64;
        }
        return Ok(Some(Command::Start { duration_s: dur_s, interval_ms }));
    }
    if &magic == MAGIC_MAP_UPLOAD {
        stream.set_nonblocking(false)?; // large body: read it blocking
        return Ok(Some(Command::MapUpload(read_map_upload(stream, n)?)));
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("unknown command {magic:?}"),
    ))
}

/// Parse a MAPU body (magic already read; `n` = full record length): u8 model,
/// 4 x f32 params, u32 n_points, then n_points x {f32 x,y,z, 32 B descriptor}.
fn read_map_upload(stream: &mut TcpStream, n: usize) -> std::io::Result<LocalMap> {
    const HEADER: usize = 1 + 4 * 4 + 4;
    if n < 4 + HEADER {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("map upload record too short ({n} bytes)"),
        ));
    }
    let mut hdr = [0u8; HEADER];
    stream.read_exact(&mut hdr)?;
    let model = hdr[0];
    if model != CAMERA_MODEL_SIMPLE_RADIAL {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported camera model id {model}"),
        ));
    }
    let mut params = [0f32; 4];
    for (i, p) in params.iter_mut().enumerate() {
        *p = f32::from_le_bytes(hdr[1 + i * 4..5 + i * 4].try_into().unwrap());
    }
    let n_points = u32::from_le_bytes(hdr[17..21].try_into().unwrap()) as usize;
    let body_bytes = n - 4 - HEADER;
    if n_points > MAX_MAP_POINTS || body_bytes != n_points * 44 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("map point count {n_points} != {body_bytes} B payload"),
        ));
    }
    let mut points = Vec::with_capacity(n_points);
    let mut buf = [0u8; 44];
    for _ in 0..n_points {
        stream.read_exact(&mut buf)?;
        let mut xyz = [0f32; 3];
        for (i, v) in xyz.iter_mut().enumerate() {
            *v = f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let mut desc = [0u32; 8];
        for (w, d) in desc.iter_mut().enumerate() {
            *d = u32::from_le_bytes(buf[12 + w * 4..16 + w * 4].try_into().unwrap());
        }
        points.push(LocalMapPoint { xyz, desc });
    }
    Ok(LocalMap { model, params, points })
}

/// Read exactly `buf.len()` bytes, tolerating WouldBlock (non-blocking socket)
/// by polling with small delays until `deadline`. Err(UnexpectedEof) on a
/// clean close, Err(TimedOut) past the deadline.
fn read_exact_poll(
    stream: &mut TcpStream,
    buf: &mut [u8],
    deadline: Instant,
) -> std::io::Result<()> {
    let mut got = 0;
    while got < buf.len() {
        use std::io::Read;
        match stream.read(&mut buf[got..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "client closed",
                ))
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "timed out waiting for the laptop command",
                    ));
                }
                FreeRtos::delay_ms(20);
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Per-frame stage timings (µs): `capture` and `build` (VOX2 assembly);
/// `send`/`pace` come from the caller. No device pyramid or downscale runs.
#[derive(Clone, Copy, Default)]
struct FrameTimings {
    capture_us: u64,
    downscale_us: u64,
    build_us: u64,
    send_us: u64,
    pace_us: u64,
}

/// Capture one VGA frame, copy it into `p.frame`, and assemble the featureless
/// VOX2 record into `p.tx`. Err on capture/format surprises (dims changed).
fn map_frame(cam: &camera::Camera, p: &mut MapPipeline) -> Result<FrameTimings, &'static str> {
    let t_cap = Instant::now();
    let fb = cam.capture().ok_or("capture() returned no frame")?;
    let mut tm = FrameTimings {
        capture_us: t_cap.elapsed().as_micros() as u64,
        ..FrameTimings::default()
    };
    let (w, h) = (fb.width(), fb.height());
    if (w, h) != (p.cam_w, p.cam_h) {
        return Err("camera dims changed (pipeline sized for another frame)");
    }
    if fb.data().len() != w * h {
        return Err("frame length != w*h (format not grayscale?)");
    }
    // Copy the fb out (fb_count == 1: holding it stops the DMA), then return it
    // so the camera keeps capturing while we build/send.
    p.frame.copy_from_slice(fb.data());
    drop(fb);
    // build_record measures its own assembly time and embeds it (plus capture)
    // in the record footer.
    tm.build_us = build_record(&mut p.tx, p.w, p.h, &p.frame, &tm);
    Ok(tm)
}

/// Size of the VOX2 timing footer (appended after the last feature
/// descriptor): 3 stage u32s + per pyramid level {6 phase u32s + u16
/// corners}. See build_record + receive_frames.py for the byte layout.
const VOX2_TIMING_FOOTER_BYTES: usize = 3 * 4 + pyramid::LEVELS * (6 * 4 + 2);

/// Assemble a featureless VOX2 record (raw frame + timing footer) for the map
/// run; the laptop extracts features. Timings ride the WiFi stream (the console
/// UART wedges at station join); returns the build time in µs.
fn build_record(buf: &mut Vec<u8>, w: usize, h: usize, frame: &[u8], tm: &FrameTimings) -> u64 {
    let t0 = Instant::now();
    // Payload after the u32 length: magic(4)+fmt(1)+w(2)+h(2)+nfeat(2)+pixels+footer.
    let payload = 11 + w * h + VOX2_TIMING_FOOTER_BYTES;

    buf.clear();
    buf.extend_from_slice(&(payload as u32).to_le_bytes());
    buf.extend_from_slice(MAGIC);
    buf.push(FMT_GRAYSCALE);
    buf.extend_from_slice(&(w as u16).to_le_bytes());
    buf.extend_from_slice(&(h as u16).to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // nfeat: the laptop extracts
    buf.extend_from_slice(frame);
    // Footer: build_us excludes this append; no device pyramid ran, so the
    // per-level phase fields are zero (order must match VOX2_FOOTER_FMT).
    let build_us = t0.elapsed().as_micros() as u64;
    buf.extend_from_slice(&(tm.capture_us as u32).to_le_bytes());
    buf.extend_from_slice(&(tm.downscale_us as u32).to_le_bytes());
    buf.extend_from_slice(&(build_us as u32).to_le_bytes());
    for _ in 0..pyramid::LEVELS {
        for _ in 0..6 {
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
        buf.extend_from_slice(&0u16.to_le_bytes());
    }
    build_us
}

/// Assemble the VOXD "done mapping" record into `buf`: length | magic |
/// frames | total features (u32 LE each). Sent once at the end of every map
/// run so the laptop can cross-check its own receive counts and start COLMAP.
fn build_done_record(buf: &mut Vec<u8>, frames: u64, features: u64) {
    // Payload after the u32 length: magic(4) + frames(4) + features(4).
    let payload = 12;
    buf.clear();
    buf.extend_from_slice(&(payload as u32).to_le_bytes());
    buf.extend_from_slice(MAGIC_DONE);
    buf.extend_from_slice(&(frames as u32).to_le_bytes());
    buf.extend_from_slice(&(features as u32).to_le_bytes());
}

/// Assemble the MAPK map-upload ack into `buf`: length | magic | n_points.
fn build_map_ok_record(buf: &mut Vec<u8>, n_points: u32) {
    let payload = 8;
    buf.clear();
    buf.extend_from_slice(&(payload as u32).to_le_bytes());
    buf.extend_from_slice(MAGIC_MAP_OK);
    buf.extend_from_slice(&n_points.to_le_bytes());
}

/// Write `record`, returning the longest single write() stall in ms (a
/// multi-second stall = TCP retransmit, not bandwidth).
fn send_record(stream: &mut TcpStream, record: &[u8]) -> std::io::Result<u64> {
    let mut off = 0usize;
    let mut max_ms = 0u64;
    while off < record.len() {
        let t0 = Instant::now();
        let n = stream.write(&record[off..])?;
        max_ms = max_ms.max(t0.elapsed().as_millis() as u64);
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "write returned 0",
            ));
        }
        off += n;
    }
    Ok(max_ms)
}

/// Frame-recycled buffers for the frame stream. Sizes derive from the camera
/// (VGA 640x480); allocate once (PSRAM), reuse every frame.
struct MapPipeline {
    /// Expected camera (sensor) frame dims — VGA 640x480.
    cam_w: usize,
    cam_h: usize,
    /// Level-0 dims == camera dims (no downscale).
    w: usize,
    h: usize,
    /// Frame copy == the upload payload.
    frame: Vec<u8>,
    /// Assembled VOX2 record for the last frame.
    tx: Vec<u8>,
}

impl MapPipeline {
    /// Build the pipeline for camera dims `cam_w` x `cam_h` (streamed at 1:1).
    fn new(cam_w: usize, cam_h: usize) -> Self {
        assert!(cam_w > 0 && cam_h > 0, "bad camera dims {cam_w}x{cam_h}");
        let (w, h) = (cam_w, cam_h);
        let nframe = w * h;
        MapPipeline {
            cam_w,
            cam_h,
            w,
            h,
            frame: vec![0u8; nframe],
            // Record: length(4) + magic(4) + fmt(1) + w(2) + h(2) + nfeat(2)
            // + frame + footer.
            tx: Vec::with_capacity(15 + nframe + VOX2_TIMING_FOOTER_BYTES),
        }
    }
}
