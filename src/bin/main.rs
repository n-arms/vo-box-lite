//! SoftAP + TCP frame-stream server: esp-idf-svc SoftAP → camera (PSRAM fb) →
//! TCP server on 0.0.0.0:5000, one persistent laptop connection until it drops.
//! Wire format (VOX1 length-prefixed records): see scripts/receive_frames.py.

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

/// SoftAP credentials the laptop joins with (WPA2 passphrase must be >= 8 chars).
const AP_SSID: &str = "vo-box";
const AP_PASS: &str = "vobox1234"; // TODO: real passphrase
const TCP_PORT: u16 = 5000;
/// IDF v5.5 default AP IP; fallback if the netif-up poll times out.
const AP_IP_FALLBACK: Ipv4Addr = Ipv4Addr::new(192, 168, 71, 1);

fn main() -> Result<(), EspError> {
    // Required once: links the esp-idf runtime patches (esp-idf-template#71).
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    log::info!("=== vo-box-lite uplink: SoftAP + TCP frame server ===");

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

    // ---- Camera: optional, TCP still works without it ----
    let camera = match camera::Camera::init(&camera::CameraConfig::with_pins(
        camera::CameraPins::FREENOVE_ESP32S3_WROOM,
    )) {
        Ok(cam) => {
            log::info!("camera ready");
            Some(cam)
        }
        Err(e) => {
            log::error!("camera init failed ({e}); no frames will stream");
            None
        }
    };

    // ---- TCP server ----
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, TCP_PORT))
        .expect("TCP bind 0.0.0.0:5000 failed");
    log::info!("listening on 0.0.0.0:{TCP_PORT}");

    serve(listener, camera.as_ref())
}

/// Serve one persistent connection at a time until it drops, then re-accept.
fn serve(listener: TcpListener, camera: Option<&camera::Camera>) -> ! {
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
            // capture() blocks until the next frame is ready, pacing the stream.
            let frame = match camera {
                Some(cam) => match cam.capture() {
                    Some(f) => f,
                    None => {
                        log::warn!("capture() returned no frame");
                        FreeRtos::delay_ms(200);
                        continue;
                    }
                },
                None => {
                    FreeRtos::delay_ms(200); // camera down: idle, keep the connection
                    continue;
                }
            };

            if let Err(e) = send_frame(&mut stream, &frame) {
                log::info!("client {peer} disconnected ({e}); re-accepting");
                break;
            }
            sent += 1;
            if sent % 30 == 0 {
                log::info!("streamed {sent} frames to {peer}");
            }
        }
    }
}

/// Write one frame record (wire format documented in scripts/receive_frames.py).
fn send_frame(stream: &mut TcpStream, frame: &camera::Frame<'_>) -> std::io::Result<()> {
    let data = frame.data();
    // Header: len(4) | "VOX1"(4) | format(1) | width(2) | height(2)
    let mut hdr = [0u8; 13];
    hdr[0..4].copy_from_slice(&((9 + data.len()) as u32).to_le_bytes());
    hdr[4..8].copy_from_slice(b"VOX1");
    hdr[8] = frame.format().map(|f| f.as_raw() as u8).unwrap_or(0);
    hdr[9..11].copy_from_slice(&(frame.width() as u16).to_le_bytes());
    hdr[11..13].copy_from_slice(&(frame.height() as u16).to_le_bytes());

    stream.write_all(&hdr)?;
    stream.write_all(data)
}
