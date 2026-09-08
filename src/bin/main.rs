//! SoftAP + TCP feature server — the S3-side "localization engine" stream:
//! capture the largest grayscale frame the OV3660 can output (QXGA 2048x1536),
//! 4x4 INTER_AREA downsample it (`vo_box_lite::downscale::downscale_4x4`) to a
//! manageable 512x384, run the 7-level pyramid extractor on that (FAST-12 +
//! 5x5 box blur + rBRIEF at 1x..1.2^6x, `vo_box_lite::pyramid`), then stream
//! the downsampled frame + all features as one VOX2 record per frame. Wire
//! format (length-prefixed records): see scripts/receive_frames.py.

#[path = "../camera.rs"]
mod camera;

use std::io::Write;
use std::net::{Ipv4Addr, TcpListener, TcpStream};

use esp_idf_hal::delay::FreeRtos;
use esp_idf_hal::peripherals::Peripherals;
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys::EspError;
use esp_idf_svc::wifi::{AccessPointConfiguration, AuthMethod, BlockingWifi, Configuration, EspWifi};
use vo_box_lite::downscale;
use vo_box_lite::fast;
use vo_box_lite::pyramid::{self, Feature};

/// SoftAP credentials the laptop joins with (WPA2 passphrase must be >= 8 chars).
const AP_SSID: &str = "vo-box";
const AP_PASS: &str = "vobox1234"; // TODO: real passphrase
const TCP_PORT: u16 = 5000;
/// IDF v5.5 default AP IP; fallback if the netif-up poll times out.
const AP_IP_FALLBACK: Ipv4Addr = Ipv4Addr::new(192, 168, 71, 1);
/// Record magic for frame+features (VOX2).
const MAGIC: &[u8; 4] = b"VOX2";
/// esp32-camera PIXFORMAT_GRAYSCALE (the only format we configure).
const FMT_GRAYSCALE: u8 = 3;
/// Sensor resolution: the largest grayscale the OV3660 supports. The engine
/// downsamples 4x4 to `level-0` dims, so the pyramid/upload budget is small
/// while the capture keeps full sensor detail (see map_frame).
const CAM_W: usize = 2048;
const CAM_H: usize = 1536;

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
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, TCP_PORT))
        .expect("TCP bind 0.0.0.0:5000 failed");
    log::info!("listening on 0.0.0.0:{TCP_PORT}");

    feature_server(listener, camera.as_ref())
}

/// One persistent connection at a time; per frame: capture (QXGA) -> 4x4
/// downsample -> pyramid -> send VOX2 (frame + features). Re-accepts when the
/// laptop drops.
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
        log::info!("laptop connected: {peer}");

        let mut sent = 0u64;
        loop {
            let cam = match camera {
                Some(cam) => cam,
                None => {
                    FreeRtos::delay_ms(200); // camera down: idle, keep the connection
                    continue;
                }
            };
            if pipe.is_none() {
                pipe = Some(MapPipeline::new(CAM_W, CAM_H));
            }
            let p = pipe.as_mut().unwrap();

            if let Err(e) = map_frame(cam, p) {
                log::warn!("map_frame failed ({e})");
                FreeRtos::delay_ms(200);
                continue;
            }
            // Send the assembled VOX2 record (downsampled frame + features).
            if let Err(e) = send_record(&mut stream, &p.tx) {
                log::info!("client {peer} disconnected ({e}); re-accepting");
                break;
            }
            sent += 1;
            if sent % 20 == 0 {
                log::info!("streamed {sent} frames to {peer} ({} features/frame)", p.last_n);
            }
        }
    }
}

/// Capture one QXGA frame, 4x4-downsample it into the pipeline's level-0
/// buffer (the upload payload), run the pyramid over it and assemble the VOX2
/// record into `p.tx`. Err only on capture/format surprises (dims changed).
fn map_frame(cam: &camera::Camera, p: &mut MapPipeline) -> Result<(), &'static str> {
    let fb = cam.capture().ok_or("capture() returned no frame")?;
    let (w, h) = (fb.width(), fb.height());
    if (w, h) != (p.cam_w, p.cam_h) {
        return Err("camera dims changed (pipeline sized for another frame)");
    }
    if fb.data().len() != w * h {
        return Err("frame length != w*h (format not grayscale?)");
    }
    // Full sensor detail is captured at QXGA; the frame the pyramid (and the
    // laptop) sees is the 4x4 INTER_AREA mean down to level-0 dims. The fb is
    // read straight from the driver (no full-size copy) and returned right
    // away so the driver can keep capturing while we process.
    if !downscale::downscale_4x4(fb.data(), w, h, &mut p.frame) {
        return Err("downscale_4x4 failed (buffer sizes?)");
    }
    drop(fb);

    let n = pyramid::extract_pyramid(
        &p.frame,
        p.w,
        p.h,
        pyramid::FAST_THRESHOLD,
        &mut p.arena,
        &mut p.work,
        &mut p.vcol,
        &mut p.corners,
        &mut p.scores,
        &mut p.rowidx,
        &mut p.nms,
        &mut p.feats,
    );
    p.last_n = n;
    // Only the first `n` entries are valid (p.feats keeps its full capacity).
    build_record(&mut p.tx, p.w, p.h, &p.frame, &p.feats[..n]);
    Ok(())
}

/// Assemble one VOX2 record into `buf`: length | magic | fmt | w | h | nfeat
/// | raw pixels | per-feature {level, x, y, descriptor} (see receive_frames.py).
fn build_record(buf: &mut Vec<u8>, w: usize, h: usize, frame: &[u8], feats: &[Feature]) {
    let nfeat = feats.len();
    // Payload after the u32 length: magic(4) + fmt(1) + w(2) + h(2) + nfeat(2)
    // + w*h pixels + nfeat * (level 1 + x 4 + y 4 + desc 32).
    let payload = 11 + w * h + nfeat * 41;

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
}

fn send_record(stream: &mut TcpStream, record: &[u8]) -> std::io::Result<()> {
    stream.write_all(record)
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
    /// Box-blur running column sums (one u16 per column; hot, per output row).
    vcol: Vec<u16>,
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
        MapPipeline {
            cam_w,
            cam_h,
            w,
            h,
            frame: vec![0u8; nframe],
            arena: vec![0u8; pyramid::arena_bytes(w, h)],
            work: vec![0u8; nframe],
            vcol: vec![0u16; w],
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
