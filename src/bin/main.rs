//! SoftAP + TCP map-task server: the MCU idles until the laptop sends an STRT
//! command, then streams VOX2 frame+feature records at the requested pace for
//! the requested duration and ends with a VOXD "done" record (see receive_map).

#[path = "../camera.rs"]
mod camera;

use std::io::Write;
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use esp_idf_hal::delay::FreeRtos;
use esp_idf_hal::peripherals::Peripherals;
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys::EspError;
use esp_idf_svc::wifi::{
    AccessPointConfiguration, AuthMethod, BlockingWifi, Configuration, EspWifi,
};
use vo_box_lite::downscale;
use vo_box_lite::fast;
use vo_box_lite::pyramid::{self, Feature};

/// SoftAP credentials the laptop joins with (WPA2 passphrase must be >= 8 chars).
const AP_SSID: &str = "vo-box";
const AP_PASS: &str = "vobox1234"; // TODO: real passphrase
const TCP_PORT: u16 = 5000;
/// IDF v5.5 default AP IP; fallback if the netif-up poll times out.
const AP_IP_FALLBACK: Ipv4Addr = Ipv4Addr::new(192, 168, 71, 1);
/// Command magic laptop -> MCU, sent right after connect: `u32 LE n | "STRT"`
/// [+ payload]. The MCU streams nothing until STRT arrives.
const MAGIC_START: &[u8; 4] = b"STRT";
/// STRT payload defaults (the command can override; see read_start_command):
/// 300 s run, one frame per 1000 ms (0 = max rate), 60 s STRT timeout.
const DEFAULT_DURATION_S: u64 = 300;
const DEFAULT_INTERVAL_MS: u64 = 1000;
const START_TIMEOUT_S: u64 = 60;
/// Record magic for the final "done mapping" record.
const MAGIC_DONE: &[u8; 4] = b"VOXD";
/// Record magic for frame+features (VOX2).
const MAGIC: &[u8; 4] = b"VOX2";
/// esp32-camera PIXFORMAT_GRAYSCALE (the only format we configure).
const FMT_GRAYSCALE: u8 = 3;
/// Sensor resolution: the largest grayscale the OV3660 supports. The engine
/// downsamples 4x4 to `level-0` dims, so the pyramid/upload budget is small
/// while the capture keeps full sensor detail (see map_frame).
const CAM_W: usize = 2048;
const CAM_H: usize = 1536;

/// Monotonic µs clock for the perf logs (`Instant`/esp_timer, anchored at boot
/// since only deltas matter). A bare fn pointer so it can ride on the no_std
/// [`pyramid::PyramidProfile`].
static BOOT_TIME: OnceLock<Instant> = OnceLock::new();
fn now_us() -> u64 {
    let boot = *BOOT_TIME.get_or_init(Instant::now);
    Instant::now().duration_since(boot).as_micros() as u64
}

fn main() -> Result<(), EspError> {
    // Required once: links the esp-idf runtime patches (esp-idf-template#71).
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    log::info!("=== vo-box-lite: feature stream (QXGA 2048x1536 -> 4x4 -> 512x384 pyramid) ===");

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
        frame_size: camera::FrameSize::Qxga, // 2048x1536 gray (CAM_W x CAM_H)
        ..camera::CameraConfig::with_pins(camera::CameraPins::FREENOVE_ESP32S3_WROOM)
    }) {
        Ok(cam) => {
            log::info!("camera ready");
            Some(cam)
        }
        Err(e) => {
            log::error!("camera init failed ({e}); no frames will stream");
            None
        }
    };

    // ---- TCP server (feature stream: capture + downsample + pyramid + upload) ----
    let listener =
        TcpListener::bind((Ipv4Addr::UNSPECIFIED, TCP_PORT)).expect("TCP bind 0.0.0.0:5000 failed");
    log::info!("listening on 0.0.0.0:{TCP_PORT}");

    feature_server(listener, camera.as_ref())
}

/// Serve one connection = one map run: wait for the laptop's STRT command,
/// stream paced VOX2 frames until the run deadline, then send the VOXD done
/// record and drop the connection (re-accept for the next run).
fn feature_server(listener: TcpListener, camera: Option<&camera::Camera>) -> ! {
    // Pipeline buffers are sized for the downsampled level-0 dims and reused
    // every frame (no per-frame allocation; see pyramid::extract_pyramid).
    let mut pipe: Option<MapPipeline> = None;
    loop {
        let (mut stream, peer) = match listener.accept() {
            Ok(conn) => conn,
            Err(e) => {
                log::error!("accept failed: {e}");
                FreeRtos::delay_ms(500);
                continue;
            }
        };
        log::info!("laptop connected: {peer} — waiting for its STRT command");
        // Disable Nagle (pairs with the enlarged lwIP send buffer in sdkconfig).
        if let Err(e) = stream.set_nodelay(true) {
            log::warn!("set_nodelay failed ({e})");
        }
        if pipe.is_none() {
            pipe = Some(MapPipeline::new(CAM_W, CAM_H));
        }
        let p = pipe.as_mut().unwrap();

        // Idle until the laptop kicks the map task off (streams nothing
        // before that). Clean disconnect -> Ok(None); timeout/garbage -> Err.
        let (dur_s, interval_ms) = match read_start_command(&mut stream) {
            Ok(Some(params)) => params,
            Ok(None) => {
                log::info!("client {peer} disconnected before sending STRT");
                continue;
            }
            Err(e) => {
                log::warn!("waiting for STRT from {peer} failed ({e}); dropping");
                continue;
            }
        };
        // Command phase used non-blocking reads; stream writes must block
        // again (a WouldBlock on a 200 KB VOX2 frame would look like a drop).
        if let Err(e) = stream.set_nonblocking(false) {
            log::warn!("set_nonblocking(false) failed ({e}); dropping client");
            continue;
        }

        log::info!("starting a {dur_s}s map run for {peer} (frame every {interval_ms} ms)");
        let run_start = Instant::now();
        let deadline = run_start + Duration::from_secs(dur_s);
        let mut sent = 0u64;
        let mut feats_sent = 0u64;
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

            // One profile per frame: pyramid::extract_pyramid fills the
            // per-level phase timers (fast/score/nms/blur/rbrief/downscale).
            let mut prof = pyramid::PyramidProfile::new(now_us);
            let mut tm = match map_frame(cam, p, &mut prof) {
                Ok(t) => t,
                Err(e) => {
                    log::warn!("map_frame failed ({e})");
                    FreeRtos::delay_ms(200);
                    continue;
                }
            };
            // Send the assembled VOX2 record (downsampled frame + features).
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
            feats_sent += p.last_n as u64;

            // Pace to one frame per `interval_ms` (capture + pyramid + upload
            // time counts towards the interval; 0 = as fast as possible).
            let wait_ms = interval_ms.saturating_sub(frame_start.elapsed().as_millis() as u64);
            if wait_ms > 0 {
                FreeRtos::delay_ms(wait_ms as u32);
            }
            tm.pace_us = wait_ms * 1000;

            // Debug breakdown (same cadence as before: every frame at a paced
            // run, every 20th at max rate). All values are µs on one clock.
            if interval_ms > 0 || sent % 20 == 0 {
                let p_fast = prof.fast_us.iter().sum::<u64>();
                let p_score = prof.score_us.iter().sum::<u64>();
                let p_nms = prof.nms_us.iter().sum::<u64>();
                let p_blur = prof.blur_us.iter().sum::<u64>();
                let p_rbrief = prof.rbrief_us.iter().sum::<u64>();
                // rBRIEF phase split (CCOUNT cycles -> us at the 160 MHz clock).
                let p_ang = prof.rbrief_angle_cyc.iter().sum::<u64>() / 160;
                let p_smp = prof.rbrief_sample_cyc.iter().sum::<u64>() / 160;
                let p_ds = prof.downscale_us.iter().sum::<u64>();
                let p_pyr = p_fast + p_score + p_nms + p_blur + p_rbrief + p_ds;
                let total_us = frame_start.elapsed().as_micros() as u64;
                log::info!(
                    "frame {sent} @ {}s: {} feats — total {total_us} us | capture {} | ds4 {} | \
                     pyramid {p_pyr} (fast {p_fast} + score {p_score} + nms {p_nms} + blur {p_blur} \
                     + rbrief {p_rbrief} [angle {p_ang} + sample {p_smp}] + ds65 {p_ds}) | build {} | send {} (max {}ms) | pace {}",
                    run_start.elapsed().as_secs(),
                    p.last_n,
                    tm.capture_us,
                    tm.downscale_us,
                    tm.build_us,
                    tm.send_us,
                    send_max_ms,
                    tm.pace_us,
                );
                // Per-level pyramid cost (all phases summed) + survivors.
                let mut per_level = String::from("  per level: ");
                for l in 0..pyramid::LEVELS {
                    let ltot = prof.fast_us[l]
                        + prof.score_us[l]
                        + prof.nms_us[l]
                        + prof.blur_us[l]
                        + prof.rbrief_us[l]
                        + prof.downscale_us[l];
                    per_level.push_str(&format!("L{l} {ltot}us/{}feats, ", prof.corners[l]));
                }
                log::info!("{per_level}");
            }
        }

        if !interrupted {
            // Deadline reached: tell the laptop the run is done, then drop the
            // connection (it runs COLMAP on the frames it received).
            build_done_record(&mut p.tx, sent, feats_sent);
            if let Err(e) = send_record(&mut stream, &p.tx) {
                log::warn!("done record send failed ({e})");
            } else {
                log::info!("map run complete: {sent} frames / {feats_sent} features -> {peer}; VOXD done record sent");
            }
        } else {
            log::info!("client {peer} disconnected mid-run after {sent} frames; re-accepting");
        }
    }
}

/// Read the laptop's start command (length-prefixed records, laptop -> MCU).
/// Returns Ok(Some((duration_s, interval_ms))) on STRT, Ok(None) if the
/// client disconnects before sending anything, Err on timeout / bad record.
fn read_start_command(stream: &mut TcpStream) -> std::io::Result<Option<(u64, u64)>> {
    let deadline = Instant::now() + Duration::from_secs(START_TIMEOUT_S);
    stream.set_nonblocking(true)?;
    loop {
        let mut len_buf = [0u8; 4];
        match read_exact_poll(stream, &mut len_buf, deadline) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let n = u32::from_le_bytes(len_buf) as usize;
        if !(4..=12).contains(&n) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad command record length {n}"),
            ));
        }
        let mut body = [0u8; 12];
        read_exact_poll(stream, &mut body[..n], deadline)?;
        if &body[..4] != MAGIC_START {
            log::warn!("ignoring unknown command {:?}", &body[..4]);
            continue; // keep waiting for a valid STRT
        }
        // Payload (optional): u32 duration_s, u32 interval_ms (see const docs).
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
        return Ok(Some((dur_s, interval_ms)));
    }
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

/// Per-frame stage timings (µs, same clock as pyramid): `capture`, `downscale`
/// (4x4 sensor -> level-0) and `build` (VOX2 assembly); `send`/`pace` come from
/// the caller, the pyramid phases ride on [`pyramid::PyramidProfile`].
#[derive(Clone, Copy, Default)]
struct FrameTimings {
    capture_us: u64,
    downscale_us: u64,
    build_us: u64,
    send_us: u64,
    pace_us: u64,
}

/// Capture one QXGA frame, 4x4-downsample into level-0 `p.frame`, run the
/// pyramid (filling `prof`) and assemble the VOX2 record into `p.tx`. Err on
/// capture/format surprises (dims changed).
fn map_frame(
    cam: &camera::Camera,
    p: &mut MapPipeline,
    prof: &mut pyramid::PyramidProfile,
) -> Result<FrameTimings, &'static str> {
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
    // Capture is full QXGA; the pyramid/laptop see the 4x4 INTER_AREA mean. The
    // fb is read straight from the driver and returned right away (no full-size
    // copy) so it can keep capturing while we process.
    let t_ds = Instant::now();
    if !downscale::downscale_4x4(fb.data(), w, h, &mut p.frame) {
        return Err("downscale_4x4 failed (buffer sizes?)");
    }
    tm.downscale_us = t_ds.elapsed().as_micros() as u64;
    drop(fb);

    let n = pyramid::extract_pyramid(
        &p.frame,
        p.w,
        p.h,
        pyramid::FAST_THRESHOLD,
        &mut p.arena,
        &mut p.work,
        &mut p.vcol[p.vcol_off..],
        &mut p.corners,
        &mut p.scores,
        &mut p.rowidx,
        &mut p.nms,
        &mut p.feats,
        Some(prof),
    );
    p.last_n = n;
    // Only the first `n` entries are valid (p.feats keeps its full capacity).
    // build_record measures its own assembly time and embeds it (plus tm's
    // capture/4x4-downscale and prof's pyramid phases) in the record footer.
    tm.build_us = build_record(&mut p.tx, p.w, p.h, &p.frame, &p.feats[..n], &tm, prof);
    Ok(tm)
}

/// Size of the VOX2 timing footer (appended after the last feature
/// descriptor): 3 stage u32s + per pyramid level {6 phase u32s + u16
/// corners}. See build_record + receive_frames.py for the byte layout.
const VOX2_TIMING_FOOTER_BYTES: usize = 3 * 4 + pyramid::LEVELS * (6 * 4 + 2);

/// Assemble one VOX2 record: length | magic | fmt | w | h | nfeat | pixels |
/// {level,x,y,desc} per feature | timing footer. Timings ride the WiFi stream
/// (the console UART wedges at station join); returns the build time in µs.
fn build_record(
    buf: &mut Vec<u8>,
    w: usize,
    h: usize,
    frame: &[u8],
    feats: &[Feature],
    tm: &FrameTimings,
    prof: &pyramid::PyramidProfile,
) -> u64 {
    let t0 = Instant::now();
    let nfeat = feats.len();
    // Payload after the u32 length: magic(4) + fmt(1) + w(2) + h(2) + nfeat(2)
    // + w*h pixels + nfeat * (level 1 + x 4 + y 4 + desc 32) + timing footer.
    let payload = 11 + w * h + nfeat * 41 + VOX2_TIMING_FOOTER_BYTES;

    buf.clear();
    buf.extend_from_slice(&(payload as u32).to_le_bytes());
    buf.extend_from_slice(MAGIC);
    buf.push(FMT_GRAYSCALE);
    buf.extend_from_slice(&(w as u16).to_le_bytes());
    buf.extend_from_slice(&(h as u16).to_le_bytes());
    buf.extend_from_slice(&(nfeat as u16).to_le_bytes());
    buf.extend_from_slice(frame);
    for f in feats {
        buf.push(f.level);
        buf.extend_from_slice(&f.x.to_le_bytes());
        buf.extend_from_slice(&f.y.to_le_bytes());
        for word in f.desc {
            buf.extend_from_slice(&word.to_le_bytes());
        }
    }
    // Footer (after the features): build_us excludes this append; capture/
    // downscale came from map_frame, pyramid phases from extract_pyramid. Phase
    // order must match receive_frames.py's VOX2_FOOTER_FMT ("6IH").
    let build_us = t0.elapsed().as_micros() as u64;
    buf.extend_from_slice(&(tm.capture_us as u32).to_le_bytes());
    buf.extend_from_slice(&(tm.downscale_us as u32).to_le_bytes());
    buf.extend_from_slice(&(build_us as u32).to_le_bytes());
    for l in 0..pyramid::LEVELS {
        buf.extend_from_slice(&(prof.fast_us[l] as u32).to_le_bytes());
        buf.extend_from_slice(&(prof.score_us[l] as u32).to_le_bytes());
        buf.extend_from_slice(&(prof.nms_us[l] as u32).to_le_bytes());
        buf.extend_from_slice(&(prof.blur_us[l] as u32).to_le_bytes());
        buf.extend_from_slice(&(prof.rbrief_us[l] as u32).to_le_bytes());
        buf.extend_from_slice(&(prof.downscale_us[l] as u32).to_le_bytes());
        buf.extend_from_slice(&(prof.corners[l] as u16).to_le_bytes());
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

/// Frame-recycled buffers for the feature stream. Sizes are derived from the
/// level-0 dims (the 4x4-downsampled camera frame); the QXGA fb itself is
/// driver-owned and never copied. Allocate once (PSRAM), reuse every frame.
struct MapPipeline {
    /// Expected camera (sensor) frame dims — QXGA 2048x1536.
    cam_w: usize,
    cam_h: usize,
    /// Processed level-0 dims after the 4x4 downsample (512x384 at QXGA).
    w: usize,
    h: usize,
    /// 4x-downsampled frame == level 0 of the pyramid == the upload payload.
    frame: Vec<u8>,
    /// Downscaled-level storage (pyramid::extract_pyramid writes levels 1..6
    /// here, forward, never re-reading a region once its level is processed).
    arena: Vec<u8>,
    /// Blur destination / downscale h-pass scratch.
    work: Vec<u8>,
    /// Box-blur running column sums (one u16/col). Over-allocated + sliced at
    /// `vcol_off` so the EE SIMD blur gets its 16-byte-aligned base.
    vcol: Vec<u16>,
    /// u16 offset of the aligned start inside `vcol`.
    vcol_off: usize,
    /// Per-level RAW FAST corner scratch (candidates before NMS).
    corners: Vec<fast::Corner>,
    /// Per-corner FAST scores (>= corners.len() i32s).
    scores: Vec<i32>,
    /// NMS row cursors: one usize per level-0 row.
    rowidx: Vec<usize>,
    /// NMS survivor store (features are described from here).
    nms: Vec<fast::Corner>,
    /// Feature store (first `n` of the last extraction are valid).
    feats: Vec<Feature>,
    /// Assembled VOX2 record for the last frame.
    tx: Vec<u8>,
    /// Features in the last extraction (p.feats[..last_n]).
    last_n: usize,
}

impl MapPipeline {
    /// Build the pipeline for camera dims `cam_w` x `cam_h` (level-0 dims are
    /// those divided by 4 via downscale_4x4_size).
    fn new(cam_w: usize, cam_h: usize) -> Self {
        assert!(cam_w > 0 && cam_h > 0, "bad camera dims {cam_w}x{cam_h}");
        let w = downscale::downscale_4x4_size(cam_w);
        let h = downscale::downscale_4x4_size(cam_h);
        assert!(w > 0 && h > 0, "camera dims too small to 4x4 downsample");
        let nframe = w * h;
        let nfeat = pyramid::MAX_FEATURES;
        // Over-allocate vcol so the blur slice lands on a 16-byte boundary
        // (align_of::<Vec<u16>>() is 2); misaligned just falls back to scalar.
        let vcol = vec![0u16; w + 8];
        let vcol_base = vcol.as_ptr() as usize;
        let vcol_off = ((16 - (vcol_base & 15)) & 15) / 2;
        MapPipeline {
            cam_w,
            cam_h,
            w,
            h,
            frame: vec![0u8; nframe],
            arena: vec![0u8; pyramid::arena_bytes(w, h)],
            work: vec![0u8; nframe],
            vcol,
            vcol_off,
            corners: vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX],
            scores: vec![0i32; pyramid::CORNERS_RAW_MAX],
            rowidx: vec![usize::MAX; h], // one per level-0 row (largest level)
            nms: vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX],
            feats: vec![Feature::default(); nfeat],
            // Record: length(4) + magic(4) + fmt(1) + w(2) + h(2) + nfeat(2)
            // + frame + nfeat*41. Slightly over capacity is fine.
            tx: Vec::with_capacity(15 + nframe + nfeat * 41),
            last_n: 0,
        }
    }
}
