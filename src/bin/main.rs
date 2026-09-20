//! SoftAP + TCP server: on connect the laptop sends STRT (map run: streams VOX2
//! frames, ends with VOXD) or MAP2 (upload a built map: per-frame calc8
//! embedding + its triangulated points; MCU stores it and idles in localize
//! mode). See scripts/receive_map.py for the protocol.

// The old map/localize entry point is kept but unused while the semantic task
// (src/semantic.rs) is the startup task.
#![allow(dead_code)]

#[path = "../camera.rs"]
mod camera;
#[path = "../semantic.rs"]
mod semantic;

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use esp_idf_hal::delay::FreeRtos;
use esp_idf_hal::peripherals::Peripherals;
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys::EspError;
use esp_idf_svc::wifi::{
    AccessPointConfiguration, AuthMethod, BlockingWifi, Configuration, EspWifi,
};
use vo_box_lite::fast;
use vo_box_lite::localize;
use vo_box_lite::localize::MapPoint;
use vo_box_lite::matcher;
use vo_box_lite::pyramid;
use vo_box_lite::ransac;

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
/// Laptop -> MCU map upload (v2): `u32 LE n | "MAP2" | u8 model | 4 x f32
/// params | u32 n_frames | n_frames x {u8 embedding[1064] | u32 n_points |
/// n_points x {f32 x,y,z, 32 B desc}}` (n = full record length).
const MAGIC_MAP_UPLOAD: &[u8; 4] = b"MAP2";
/// MCU -> laptop map-upload ack: `u32 LE n` (= 12) | b"MAPK" | u32 LE n_frames
/// | u32 LE n_points.
const MAGIC_MAP_OK: &[u8; 4] = b"MAPK";
/// Camera model ids accepted in a MAP2 header (COLMAP `SIMPLE_RADIAL` only).
const CAMERA_MODEL_SIMPLE_RADIAL: u8 = 1;
/// calc8 place-recognition descriptor width (bytes) per map frame.
const EMBEDDING_DIM: usize = 1064;
/// Sanity cap on uploaded frames (1064 B embedding + its points each).
const MAX_MAP_FRAMES: usize = 1024;
/// Sanity cap on uploaded points total (44 B each), across all frames.
const MAX_MAP_POINTS: usize = 100_000;
/// Diagnostic: also match each query against the union of ALL map frames, which
/// isolates embedding retrieval from descriptor/geometry problems. Off by
/// default; flip on when validating a new map's retrieval (costs an extra PnP).
const LOCALIZE_DEBUG_ALL_FRAMES: bool = false;
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
/// run deadline then sends VOXD; MAP2 stores the uploaded map and acks. Either
/// way the connection is dropped and the next one accepted.
fn feature_server(listener: TcpListener, camera: Option<&camera::Camera>) -> ! {
    // Pipeline buffers are sized for the VGA frame dims and reused every frame
    // (no per-frame allocation).
    let mut pipe: Option<MapPipeline> = None;
    // Localize-mode pipeline + the last uploaded map it runs against.
    let mut localizer: Option<Localizer> = None;
    let mut map: Option<LocalMap> = None;
    // Connection accepted by the localize loop, handled on the next iteration.
    let mut pending: Option<(TcpStream, SocketAddr)> = None;
    loop {
        let (mut stream, peer) = match pending.take() {
            Some(conn) => conn,
            None => match listener.accept() {
                Ok(conn) => conn,
                Err(e) => {
                    log::error!("accept failed: {e}");
                    FreeRtos::delay_ms(500);
                    continue;
                }
            },
        };
        log::info!("laptop connected: {peer} — waiting for its command");
        match map.as_ref() {
            Some(m) => log::info!(
                "localize map loaded ({} frames, {} points)",
                m.frames.len(),
                m.frames.iter().map(|f| f.points.len()).sum::<usize>()
            ),
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
        // MAP2 stores a map. Clean disconnect -> Ok(None); timeout/garbage -> Err.
        let (dur_s, interval_ms) = match read_command(&mut stream) {
            Ok(Some(Command::Start { duration_s, interval_ms })) => (duration_s, interval_ms),
            Ok(Some(Command::MapUpload(m))) => {
                let n_frames = m.frames.len() as u32;
                let n_points: u32 =
                    m.frames.iter().map(|f| f.points.len()).sum::<usize>() as u32;
                log::info!(
                    "map upload: {n_frames} frames / {n_points} points, model {}, params {:?}",
                    m.model, m.params
                );
                map = Some(m);
                build_map_ok_record(&mut p.tx, n_frames, n_points);
                if let Err(e) = send_record(&mut stream, &p.tx) {
                    log::warn!("map ack send failed ({e})");
                } else {
                    log::info!("map stored — entering localize mode");
                }
                drop(stream);

                // Lazy-init the localizer (calc8 embedder + reused buffers) on
                // the first upload; a failed init leaves the map inert.
                if localizer.is_none() && camera.is_some() {
                    match Localizer::new() {
                        Ok(l) => localizer = Some(l),
                        Err(e) => {
                            log::error!("localize init failed ({e}); map stored but inert")
                        }
                    }
                }
                match (camera, localizer.as_mut()) {
                    (Some(cam), Some(loc)) => {
                        let r = run_localize(&listener, cam, map.as_ref().unwrap(), loc);
                        // run_localize returns with the listener blocking again;
                        // force it in case it bailed out early.
                        let _ = listener.set_nonblocking(false);
                        match r {
                            Ok(conn) => pending = Some(conn),
                            Err(e) => log::warn!("localize accept failed ({e})"),
                        }
                    }
                    _ => {
                        log::warn!("no camera/model — idling until the next command")
                    }
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

/// Uploaded map: COLMAP-refined intrinsics + per-frame calc8 embedding with the
/// frame's triangulated points. Held in PSRAM until the next upload / reboot.
struct LocalMap {
    model: u8,
    /// SIMPLE_RADIAL params: f, cx, cy, k1.
    params: [f32; 4],
    frames: Vec<LocalMapFrame>,
}

struct LocalMapFrame {
    /// calc8 place-recognition descriptor for this frame.
    embedding: [u8; EMBEDDING_DIM],
    /// This frame's triangulated map points (COLMAP xyz + rBRIEF descriptor).
    points: Vec<MapPoint>,
}

/// Cosine similarity of two uint8 descriptors; they are unnormalized, so divide
/// by the norms. Returns 0 for an all-zero vector.
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

/// The K map frames most cosine-similar to `query`, best first, with scores.
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

/// Localize-mode pipeline: the calc8 embedder plus all pyramid / matching / PnP
/// buffers, allocated once and reused every frame.
struct Localizer {
    embedder: semantic::Embedder,
    arena: Vec<u8>,
    work: Vec<u8>,
    vcol: Vec<u16>,
    corners: Vec<fast::Corner>,
    scores: Vec<i32>,
    rowidx: Vec<usize>,
    nms: Vec<fast::Corner>,
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
    /// Query calc8 descriptor (kept off the small main-task stack).
    emb: [u8; EMBEDDING_DIM],
    rng: ransac::Xorshift64,
}

impl Localizer {
    /// Allocate the VGA pipeline + load calc8. Err if the model is missing.
    fn new() -> Result<Localizer, String> {
        let (w, h) = (CAM_W, CAM_H);
        let nf = pyramid::MAX_FEATURES;
        let np = MAX_MAP_POINTS; // one frame can hold at most the global cap
        Ok(Localizer {
            embedder: semantic::Embedder::init()?,
            arena: vec![0u8; pyramid::arena_bytes(w, h)],
            work: vec![0u8; w * h],
            vcol: vec![0u16; w],
            corners: vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX],
            scores: vec![0i32; pyramid::CORNERS_RAW_MAX],
            rowidx: vec![usize::MAX; h],
            nms: vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX],
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

    /// Match + PnP the current query features against `points`, reusing buffers.
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

/// Localize loop: capture VGA -> pyramid FAST+rBRIEF -> calc8 embedding -> top-1
/// map frame by cosine similarity -> match + PnP RANSAC against that frame's
/// points. Returns the next laptop connection (which preempts localizing).
fn run_localize(
    listener: &TcpListener,
    cam: &camera::Camera,
    map: &LocalMap,
    loc: &mut Localizer,
) -> std::io::Result<(TcpStream, SocketAddr)> {
    let cam_model = ransac::Camera {
        fx: map.params[0],
        fy: map.params[0],
        cx: map.params[1],
        cy: map.params[2],
        k1: map.params[3],
    };
    let opts = ransac::PnpOptions::default();
    // Non-blocking so a new laptop command can interrupt the localize loop.
    listener.set_nonblocking(true)?;
    log::info!(
        "localize: {} map frames / {} points; running pose estimation",
        map.frames.len(),
        map.frames.iter().map(|f| f.points.len()).sum::<usize>()
    );
    // Union of every frame's points (diagnostic fallback, see the const above).
    let all_points: Vec<MapPoint> = if LOCALIZE_DEBUG_ALL_FRAMES {
        map.frames.iter().flat_map(|f| f.points.iter().copied()).collect()
    } else {
        Vec::new()
    };

    let mut n = 0u64;
    loop {
        match listener.accept() {
            Ok(conn) => {
                listener.set_nonblocking(false)?;
                log::info!("localize: laptop reconnected — leaving localize mode");
                return Ok(conn);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => log::warn!("localize: accept failed ({e})"),
        }

        let t_cap = Instant::now();
        let fb = match cam.capture() {
            Some(f) => f,
            None => {
                FreeRtos::delay_ms(20);
                continue;
            }
        };
        let (w, h) = (fb.width(), fb.height());
        if (w, h) != (CAM_W, CAM_H) || fb.data().len() != w * h {
            log::warn!("localize: unexpected frame {w}x{h} ({} B)", fb.data().len());
            drop(fb);
            FreeRtos::delay_ms(50);
            continue;
        }
        loc.frame.copy_from_slice(fb.data());
        let cap_us = t_cap.elapsed().as_micros() as u64;
        drop(fb);

        // 1) 7-level pyramid FAST-12 + rBRIEF on the full 640x480 frame.
        let t = Instant::now();
        let nfeat = pyramid::extract_pyramid(
            &loc.frame,
            CAM_W,
            CAM_H,
            pyramid::FAST_THRESHOLD,
            &mut loc.arena,
            &mut loc.work,
            &mut loc.vcol,
            &mut loc.corners,
            &mut loc.scores,
            &mut loc.rowidx,
            &mut loc.nms,
            &mut loc.feats,
            None,
        );
        let extract_us = t.elapsed().as_micros() as u64;

        // 2) calc8 place-recognition descriptor of the 160x120 downscale.
        let t = Instant::now();
        if let Err(e) = loc.embedder.embed(&loc.frame, CAM_W, CAM_H, &mut loc.emb) {
            log::error!("localize: embed failed ({e})");
            FreeRtos::delay_ms(50);
            continue;
        }
        let embed_us = t.elapsed().as_micros() as u64;

        // 3) Top-1 map frame by embedding cosine similarity (top-2 for the
        // margin, a retrieval-confidence diagnostic).
        let ranked = top_k_frames(map, &loc.emb, 2);
        let Some(&(cos1, frame)) = ranked.first() else {
            FreeRtos::delay_ms(50);
            continue;
        };
        let cos2 = ranked.get(1).map_or(0.0, |(s, _)| *s);

        // 4) Match + PnP RANSAC against just that frame's points.
        let stats = loc.localize(nfeat, &frame.points, &cam_model, &opts);
        let all_stats = LOCALIZE_DEBUG_ALL_FRAMES
            .then(|| loc.localize(nfeat, &all_points, &cam_model, &opts));
        n += 1;

        match stats.pnp {
            Some(p) => {
                let (roll, pitch, yaw) = zyx_deg(&p.r);
                let c = camera_center(&p.r, &p.t);
                log::info!(
                    "localize {n}: cos1 {cos1:.3} cos2 {cos2:.3} | feats {nfeat} | matches {} | inliers {} | reproj {:.2}px | roll {roll:.1} pitch {pitch:.1} yaw {yaw:.1} deg | center ({:.2}, {:.2}, {:.2}) | cap {cap_us} extract {extract_us} embed {embed_us} match {} pnp {} us",
                    stats.matches,
                    p.inlier_count,
                    p.mean_reproj_error_px,
                    c[0], c[1], c[2],
                    stats.match_us,
                    stats.pnp_us,
                );
            }
            None => log::info!(
                "localize {n}: cos1 {cos1:.3} cos2 {cos2:.3} | feats {nfeat} | matches {} | pnp failed (best {} inl, {:.2}px) | cap {cap_us} extract {extract_us} embed {embed_us} match {} pnp {} us",
                stats.matches,
                stats.pnp_best.inlier_count,
                stats.pnp_best.mean_reproj_error_px,
                stats.match_us,
                stats.pnp_us,
            ),
        }
        if let Some(a) = all_stats {
            match a.pnp {
                Some(p) => log::info!(
                    "  diag all-frames: matches {} | inliers {} | reproj {:.2}px | match {} pnp {} us",
                    a.matches, p.inlier_count, p.mean_reproj_error_px, a.match_us, a.pnp_us
                ),
                None => log::info!(
                    "  diag all-frames: matches {} | pnp failed (best {} inl, {:.2}px) | match {} pnp {} us",
                    a.matches,
                    a.pnp_best.inlier_count,
                    a.pnp_best.mean_reproj_error_px,
                    a.match_us,
                    a.pnp_us
                ),
            }
        }
        FreeRtos::delay_ms(1);
    }
}

/// Wall-clock microsecond clock for the localize phase timers.
fn now_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// ZYX Euler angles (degrees) of the camera-to-world rotation R_c2w = R_w2c^T:
/// R_c2w = Rz(yaw) * Ry(pitch) * Rx(roll).
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

/// Camera center in world coordinates: C = -R_w2c^T * t_w2c.
fn camera_center(r: &[[f32; 3]; 3], t: &[f32; 3]) -> [f32; 3] {
    [
        -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
        -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
        -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
    ]
}

/// Read the laptop's command (STRT = map run, MAP2 = map upload). Ok(None) on a
/// clean disconnect, Err on timeout / bad record. MAP2 leaves the stream
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

/// Parse a MAP2 body (magic already read; `n` = full record length): u8 model,
/// 4 x f32 params, u32 n_frames, then per frame a 1064 B embedding + u32
/// n_points + n_points x {f32 x,y,z, 32 B descriptor}.
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
    let n_frames = u32::from_le_bytes(hdr[17..21].try_into().unwrap()) as usize;
    let body_bytes = n - 4 - HEADER;
    if n_frames > MAX_MAP_FRAMES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("map frame count {n_frames} > {MAX_MAP_FRAMES}"),
        ));
    }
    let mut frames = Vec::with_capacity(n_frames);
    let mut total_points = 0usize;
    let mut consumed = 0usize;
    for _ in 0..n_frames {
        let mut embedding = [0u8; EMBEDDING_DIM];
        stream.read_exact(&mut embedding)?;
        let mut nb = [0u8; 4];
        stream.read_exact(&mut nb)?;
        let np = u32::from_le_bytes(nb) as usize;
        total_points += np;
        if total_points > MAX_MAP_POINTS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("map point count {total_points} > {MAX_MAP_POINTS}"),
            ));
        }
        let mut points = Vec::with_capacity(np);
        let mut buf = [0u8; 44];
        for _ in 0..np {
            stream.read_exact(&mut buf)?;
            let mut xyz = [0f32; 3];
            for (i, v) in xyz.iter_mut().enumerate() {
                *v = f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
            }
            let mut desc = [0u32; 8];
            for (w, d) in desc.iter_mut().enumerate() {
                *d = u32::from_le_bytes(buf[12 + w * 4..16 + w * 4].try_into().unwrap());
            }
            points.push(MapPoint { xyz, desc });
        }
        consumed += EMBEDDING_DIM + 4 + np * 44;
        frames.push(LocalMapFrame { embedding, points });
    }
    if consumed != body_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("map payload {consumed} B != {body_bytes} B declared"),
        ));
    }
    Ok(LocalMap { model, params, frames })
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

/// Assemble the MAPK map-upload ack into `buf`: length | magic | n_frames |
/// n_points.
fn build_map_ok_record(buf: &mut Vec<u8>, n_frames: u32, n_points: u32) {
    let payload = 12;
    buf.clear();
    buf.extend_from_slice(&(payload as u32).to_le_bytes());
    buf.extend_from_slice(MAGIC_MAP_OK);
    buf.extend_from_slice(&n_frames.to_le_bytes());
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
