//! Safe wrapper around the `esp32-camera` C component (OV3660), FFI-bound by
//! esp-idf-sys's `extra_components` (raw bindings: `esp_idf_sys::camera`, imported
//! as `c`). App-layer code, pulled into main.rs via `#[path = "../camera.rs"]`.

#![allow(dead_code)] // compiled-but-unused until the app logic in main.rs drives it

use core::marker::PhantomData;

use esp_idf_sys::camera as c;
use esp_idf_sys::EspError;

/// PID reported by an OV3660 sensor (c.f. `camera_pid_t_OV3660_PID` in the bindings).
pub const OV3660_PID: u16 = 0x3660;

// ---------------------------------------------------------------------------
// Pixel format
// ---------------------------------------------------------------------------

/// Pixel formats understood by the driver (`pixformat_t`). OV3660 supports
/// RGB565 / YUV422 / GRAYSCALE / JPEG (its built-in compression) / RGB888 / RAW.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PixelFormat {
    Rgb565,
    Yuv422,
    Yuv420,
    Grayscale,
    Jpeg,
    Rgb888,
    Raw,
    Rgb444,
    Rgb555,
    Raw8,
}

impl PixelFormat {
    /// Raw `pixformat_t` value handed to the C driver.
    pub fn as_raw(self) -> u32 {
        match self {
            PixelFormat::Rgb565 => c::pixformat_t_PIXFORMAT_RGB565,
            PixelFormat::Yuv422 => c::pixformat_t_PIXFORMAT_YUV422,
            PixelFormat::Yuv420 => c::pixformat_t_PIXFORMAT_YUV420,
            PixelFormat::Grayscale => c::pixformat_t_PIXFORMAT_GRAYSCALE,
            PixelFormat::Jpeg => c::pixformat_t_PIXFORMAT_JPEG,
            PixelFormat::Rgb888 => c::pixformat_t_PIXFORMAT_RGB888,
            PixelFormat::Raw => c::pixformat_t_PIXFORMAT_RAW,
            PixelFormat::Rgb444 => c::pixformat_t_PIXFORMAT_RGB444,
            PixelFormat::Rgb555 => c::pixformat_t_PIXFORMAT_RGB555,
            PixelFormat::Raw8 => c::pixformat_t_PIXFORMAT_RAW8,
        }
    }

    /// Inverse of [`PixelFormat::as_raw`]. Values are matched against the bindgen
    /// constants so they cannot drift from the C header.
    pub fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            _ if raw == c::pixformat_t_PIXFORMAT_RGB565 => Some(PixelFormat::Rgb565),
            _ if raw == c::pixformat_t_PIXFORMAT_YUV422 => Some(PixelFormat::Yuv422),
            _ if raw == c::pixformat_t_PIXFORMAT_YUV420 => Some(PixelFormat::Yuv420),
            _ if raw == c::pixformat_t_PIXFORMAT_GRAYSCALE => Some(PixelFormat::Grayscale),
            _ if raw == c::pixformat_t_PIXFORMAT_JPEG => Some(PixelFormat::Jpeg),
            _ if raw == c::pixformat_t_PIXFORMAT_RGB888 => Some(PixelFormat::Rgb888),
            _ if raw == c::pixformat_t_PIXFORMAT_RAW => Some(PixelFormat::Raw),
            _ if raw == c::pixformat_t_PIXFORMAT_RGB444 => Some(PixelFormat::Rgb444),
            _ if raw == c::pixformat_t_PIXFORMAT_RGB555 => Some(PixelFormat::Rgb555),
            _ if raw == c::pixformat_t_PIXFORMAT_RAW8 => Some(PixelFormat::Raw8),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame size
// ---------------------------------------------------------------------------

/// Output resolutions understood by the driver (`framesize_t`).
/// OV3660's maximum is [`FrameSize::Qxga`] (2048x1536).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FrameSize {
    R96x96,
    Qqvga,
    R128x128,
    Qcif,
    Hqvga,
    R240x240,
    Qvga,
    R320x320,
    Cif,
    Hvga,
    Vga,
    Svga,
    Xga,
    Hd,
    Sxga,
    Uxga,
    Fhd,
    PHd,
    P3mp,
    Qxga,
    Qhd,
    Wqxga,
    PFhd,
    Qsxga,
    Mp5,
}

impl FrameSize {
    /// Raw `framesize_t` value handed to the C driver.
    pub fn as_raw(self) -> u32 {
        match self {
            FrameSize::R96x96 => c::framesize_t_FRAMESIZE_96X96,
            FrameSize::Qqvga => c::framesize_t_FRAMESIZE_QQVGA,
            FrameSize::R128x128 => c::framesize_t_FRAMESIZE_128X128,
            FrameSize::Qcif => c::framesize_t_FRAMESIZE_QCIF,
            FrameSize::Hqvga => c::framesize_t_FRAMESIZE_HQVGA,
            FrameSize::R240x240 => c::framesize_t_FRAMESIZE_240X240,
            FrameSize::Qvga => c::framesize_t_FRAMESIZE_QVGA,
            FrameSize::R320x320 => c::framesize_t_FRAMESIZE_320X320,
            FrameSize::Cif => c::framesize_t_FRAMESIZE_CIF,
            FrameSize::Hvga => c::framesize_t_FRAMESIZE_HVGA,
            FrameSize::Vga => c::framesize_t_FRAMESIZE_VGA,
            FrameSize::Svga => c::framesize_t_FRAMESIZE_SVGA,
            FrameSize::Xga => c::framesize_t_FRAMESIZE_XGA,
            FrameSize::Hd => c::framesize_t_FRAMESIZE_HD,
            FrameSize::Sxga => c::framesize_t_FRAMESIZE_SXGA,
            FrameSize::Uxga => c::framesize_t_FRAMESIZE_UXGA,
            FrameSize::Fhd => c::framesize_t_FRAMESIZE_FHD,
            FrameSize::PHd => c::framesize_t_FRAMESIZE_P_HD,
            FrameSize::P3mp => c::framesize_t_FRAMESIZE_P_3MP,
            FrameSize::Qxga => c::framesize_t_FRAMESIZE_QXGA,
            FrameSize::Qhd => c::framesize_t_FRAMESIZE_QHD,
            FrameSize::Wqxga => c::framesize_t_FRAMESIZE_WQXGA,
            FrameSize::PFhd => c::framesize_t_FRAMESIZE_P_FHD,
            FrameSize::Qsxga => c::framesize_t_FRAMESIZE_QSXGA,
            FrameSize::Mp5 => c::framesize_t_FRAMESIZE_5MP,
        }
    }

    /// Inverse of [`FrameSize::as_raw`] (`FRAMESIZE_INVALID` maps to `None`).
    pub fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            _ if raw == c::framesize_t_FRAMESIZE_96X96 => Some(FrameSize::R96x96),
            _ if raw == c::framesize_t_FRAMESIZE_QQVGA => Some(FrameSize::Qqvga),
            _ if raw == c::framesize_t_FRAMESIZE_128X128 => Some(FrameSize::R128x128),
            _ if raw == c::framesize_t_FRAMESIZE_QCIF => Some(FrameSize::Qcif),
            _ if raw == c::framesize_t_FRAMESIZE_HQVGA => Some(FrameSize::Hqvga),
            _ if raw == c::framesize_t_FRAMESIZE_240X240 => Some(FrameSize::R240x240),
            _ if raw == c::framesize_t_FRAMESIZE_QVGA => Some(FrameSize::Qvga),
            _ if raw == c::framesize_t_FRAMESIZE_320X320 => Some(FrameSize::R320x320),
            _ if raw == c::framesize_t_FRAMESIZE_CIF => Some(FrameSize::Cif),
            _ if raw == c::framesize_t_FRAMESIZE_HVGA => Some(FrameSize::Hvga),
            _ if raw == c::framesize_t_FRAMESIZE_VGA => Some(FrameSize::Vga),
            _ if raw == c::framesize_t_FRAMESIZE_SVGA => Some(FrameSize::Svga),
            _ if raw == c::framesize_t_FRAMESIZE_XGA => Some(FrameSize::Xga),
            _ if raw == c::framesize_t_FRAMESIZE_HD => Some(FrameSize::Hd),
            _ if raw == c::framesize_t_FRAMESIZE_SXGA => Some(FrameSize::Sxga),
            _ if raw == c::framesize_t_FRAMESIZE_UXGA => Some(FrameSize::Uxga),
            _ if raw == c::framesize_t_FRAMESIZE_FHD => Some(FrameSize::Fhd),
            _ if raw == c::framesize_t_FRAMESIZE_P_HD => Some(FrameSize::PHd),
            _ if raw == c::framesize_t_FRAMESIZE_P_3MP => Some(FrameSize::P3mp),
            _ if raw == c::framesize_t_FRAMESIZE_QXGA => Some(FrameSize::Qxga),
            _ if raw == c::framesize_t_FRAMESIZE_QHD => Some(FrameSize::Qhd),
            _ if raw == c::framesize_t_FRAMESIZE_WQXGA => Some(FrameSize::Wqxga),
            _ if raw == c::framesize_t_FRAMESIZE_P_FHD => Some(FrameSize::PFhd),
            _ if raw == c::framesize_t_FRAMESIZE_QSXGA => Some(FrameSize::Qsxga),
            _ if raw == c::framesize_t_FRAMESIZE_5MP => Some(FrameSize::Mp5),
            _ => None,
        }
    }

    /// Nominal pixel dimensions (from the driver's `resolution[]` table values).
    pub fn dimensions(self) -> (u16, u16) {
        match self {
            FrameSize::R96x96 => (96, 96),
            FrameSize::Qqvga => (160, 120),
            FrameSize::R128x128 => (128, 128),
            FrameSize::Qcif => (176, 144),
            FrameSize::Hqvga => (240, 176),
            FrameSize::R240x240 => (240, 240),
            FrameSize::Qvga => (320, 240),
            FrameSize::R320x320 => (320, 320),
            FrameSize::Cif => (400, 296),
            FrameSize::Hvga => (480, 320),
            FrameSize::Vga => (640, 480),
            FrameSize::Svga => (800, 600),
            FrameSize::Xga => (1024, 768),
            FrameSize::Hd => (1280, 720),
            FrameSize::Sxga => (1280, 1024),
            FrameSize::Uxga => (1600, 1200),
            FrameSize::Fhd => (1920, 1080),
            FrameSize::PHd => (720, 1280),
            FrameSize::P3mp => (864, 1536),
            FrameSize::Qxga => (2048, 1536),
            FrameSize::Qhd => (2560, 1440),
            FrameSize::Wqxga => (2560, 1600),
            FrameSize::PFhd => (1080, 1920),
            FrameSize::Qsxga => (2560, 1920),
            FrameSize::Mp5 => (2592, 1944),
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer handling knobs
// ---------------------------------------------------------------------------

/// Where the frame buffer lives (`camera_fb_location_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FbLocation {
    /// External PSRAM — required for anything beyond CIF/small JPEG, and the
    /// recommended choice for raw formats. Needs `CONFIG_SPIRAM` in the sdkconfig.
    Psram,
    /// Internal DRAM — small buffers only (e.g. QVGA), no PSRAM needed.
    Dram,
}

impl FbLocation {
    pub fn as_raw(self) -> u32 {
        match self {
            FbLocation::Psram => c::camera_fb_location_t_CAMERA_FB_IN_PSRAM,
            FbLocation::Dram => c::camera_fb_location_t_CAMERA_FB_IN_DRAM,
        }
    }
}

/// When buffers are filled (`camera_grab_mode_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GrabMode {
    /// Fill a buffer as soon as it is free. With `fb_count == 1` this blocks in
    /// `capture()` until the *next* frame finishes — what a VO loop wants.
    WhenEmpty,
    /// Keep the queue topped up with the latest frames (multi-buffer/JPEG mode).
    Latest,
}

impl GrabMode {
    pub fn as_raw(self) -> u32 {
        match self {
            GrabMode::WhenEmpty => c::camera_grab_mode_t_CAMERA_GRAB_WHEN_EMPTY,
            GrabMode::Latest => c::camera_grab_mode_t_CAMERA_GRAB_LATEST,
        }
    }
}

// ---------------------------------------------------------------------------
// Pin map
// ---------------------------------------------------------------------------

/// Camera parallel-interface + SCCB pin map. `-1` = not connected (e.g. `pwdn`/
/// `reset` when hard-wired; the driver software-resets the sensor over SCCB).
/// Dev board preset: [`CameraPins::FREENOVE_ESP32S3_WROOM`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraPins {
    pub pwdn: i32,
    pub reset: i32,
    pub xclk: i32,
    pub sccb_sda: i32,
    pub sccb_scl: i32,
    pub d7: i32,
    pub d6: i32,
    pub d5: i32,
    pub d4: i32,
    pub d3: i32,
    pub d2: i32,
    pub d1: i32,
    pub d0: i32,
    pub vsync: i32,
    pub href: i32,
    pub pclk: i32,
}

impl CameraPins {
    /// Everything disconnected (`-1`) — start from this and override the pins your
    /// board actually wires up.
    pub const ALL_DISABLED: CameraPins = CameraPins {
        pwdn: -1,
        reset: -1,
        xclk: -1,
        sccb_sda: -1,
        sccb_scl: -1,
        d7: -1,
        d6: -1,
        d5: -1,
        d4: -1,
        d3: -1,
        d2: -1,
        d1: -1,
        d0: -1,
        vsync: -1,
        href: -1,
        pclk: -1,
    };

    /// Freenove ESP32-S3 WROOM (N8R8/N16R8): from `camera_pins.h`
    /// (`CAMERA_MODEL_ESP32S3_EYE`); D0..D7 = Y2..Y9, PWDN/RESET unwired (-1).
    pub const FREENOVE_ESP32S3_WROOM: CameraPins = CameraPins {
        pwdn: -1,
        reset: -1,
        xclk: 15,
        sccb_sda: 4,
        sccb_scl: 5,
        d7: 16, // Y9
        d6: 17, // Y8
        d5: 18, // Y7
        d4: 12, // Y6
        d3: 10, // Y5
        d2: 8,  // Y4
        d1: 9,  // Y3
        d0: 11, // Y2
        vsync: 6,
        href: 7,
        pclk: 13,
    };
}

impl Default for CameraPins {
    fn default() -> Self {
        Self::ALL_DISABLED
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Everything needed to initialize the camera driver (`camera_config_t`).
#[derive(Debug, Clone, Copy)]
pub struct CameraConfig {
    pub pins: CameraPins,
    /// XCLK frequency in Hz (20 MHz is the driver default and what the OV3660
    /// register sets are tuned for).
    pub xclk_freq_hz: u32,
    pub pixel_format: PixelFormat,
    pub frame_size: FrameSize,
    /// JPEG quality 0-63 (lower = better). Only used when `pixel_format` is JPEG.
    pub jpeg_quality: u8,
    /// Number of frame buffers: 1 = single-shot (waits for VSYNC, then returns the
    /// frame). >1 runs continuous mode — intended for JPEG only.
    pub fb_count: usize,
    pub fb_location: FbLocation,
    pub grab_mode: GrabMode,
}

impl Default for CameraConfig {
    /// Grayscale VGA 640x480, single fb, grab-as-soon-as-free. Pins default to
    /// all-disabled — set via [`CameraConfig::with_pins`]. Fb in PSRAM (board has
    /// 8MB OPI PSRAM, enabled in `sdkconfig.defaults`).
    fn default() -> Self {
        CameraConfig {
            pins: CameraPins::ALL_DISABLED,
            xclk_freq_hz: 20_000_000,
            pixel_format: PixelFormat::Grayscale,
            frame_size: FrameSize::Vga,
            jpeg_quality: 12,
            fb_count: 1,
            fb_location: FbLocation::Psram,
            grab_mode: GrabMode::WhenEmpty,
        }
    }
}

impl CameraConfig {
    /// [`CameraConfig::default`] with the pin map filled in.
    pub fn with_pins(pins: CameraPins) -> Self {
        CameraConfig {
            pins,
            ..CameraConfig::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// Handle to the driver (a process-wide singleton: init succeeds once at a time;
/// re-init is possible after the previous [`Camera`] is dropped).
#[derive(Debug)]
pub struct Camera {
    _private: (),
}

/// What the driver actually found on the SCCB bus after init (read from the sensor
/// handle). PID 0x3660 = OV3660.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensorInfo {
    pub pid: u16,
    pub ver: u8,
    /// e.g. "OV3660" (points into the driver's static sensor table).
    pub name: &'static str,
}

impl Camera {
    /// Initialize + power the sensor and start capture. Fails (via [`EspError`])
    /// if the SCCB probe finds no supported sensor (wrong pins / not powered) or
    /// the frame buffers can't be allocated (see [`CameraConfig::default`]).
    pub fn init(config: &CameraConfig) -> Result<Self, EspError> {
        let raw = ffi_config(config);
        // Safety: `raw` mirrors camera_config_t exactly and outlives the call.
        EspError::convert(unsafe { c::esp_camera_init(&raw) })?;

        let cam = Camera { _private: () };
        if let Some(sensor) = cam.sensor() {
            log::info!(
                "camera: detected {} (PID 0x{:04x}, VER 0x{:02x})",
                sensor.name,
                sensor.pid,
                sensor.ver
            );
            if sensor.pid != OV3660_PID {
                log::warn!("camera: expected OV3660 (PID 0x{OV3660_PID:04x}) but driver detected PID 0x{:04x}", sensor.pid);
            }
        } else {
            log::warn!("camera: initialized but no sensor handle available");
        }
        Ok(cam)
    }

    /// Detected sensor (name/PID/VER) or `None` if the driver has no sensor state.
    pub fn sensor(&self) -> Option<SensorInfo> {
        // Safety: esp_camera_sensor_get returns a pointer into driver state that stays
        // valid until deinit; we only read the id + name fields.
        unsafe {
            let s = c::esp_camera_sensor_get();
            if s.is_null() {
                return None;
            }
            let id = (*s).id; // sensor_id_t copy: MIDH, MIDL, PID (u16), VER
            let info = c::esp_camera_sensor_get_info(&id as *const _ as *mut _);
            let name = if info.is_null() {
                "unknown"
            } else {
                // Static string in the driver's camera_sensor[] table.
                match core::ffi::CStr::from_ptr((*info).name).to_str() {
                    Ok(s) => s,
                    Err(_) => "unknown",
                }
            };
            Some(SensorInfo {
                pid: id.PID,
                ver: id.VER,
                name,
            })
        }
    }

    /// Convenience: is the attached sensor an OV3660?
    pub fn is_ov3660(&self) -> Option<bool> {
        self.sensor().map(|s| s.pid == OV3660_PID)
    }

    /// Block until a frame is ready and hand out a view over it; `None` on failure.
    /// With `fb_count == 1`, drop the previous [`Frame`] before the next
    /// `capture()` — it returns the driver buffer on drop.
    pub fn capture(&self) -> Option<Frame<'_>> {
        // Safety: fb pointer stays valid until esp_camera_fb_return (our Frame::drop).
        let fb = unsafe { c::esp_camera_fb_get() };
        if fb.is_null() {
            return None;
        }
        Some(Frame {
            raw: fb,
            _camera: PhantomData,
        })
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        // Safety: releases the singleton camera driver; no outstanding Frame may exist
        // (they borrow &self, so the borrow checker prevents that).
        let err = unsafe { c::esp_camera_deinit() };
        if err != 0 {
            log::warn!("camera: deinit returned 0x{err:08x}");
        }
    }
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

/// A captured frame; owns the driver's frame buffer until dropped. With
/// `fb_count == 1` the contents are stable until the next `capture()`.
#[derive(Debug)]
pub struct Frame<'a> {
    raw: *mut c::camera_fb_t,
    _camera: PhantomData<&'a Camera>,
}

impl<'a> Frame<'a> {
    /// Raw pixel bytes. Interpretation depends on [`Frame::format`]:
    /// grayscale = 1 byte/pixel (row-major, `width` * `height` = `len`),
    /// RGB565/YUV422 = 2 bytes/pixel, JPEG = compressed stream of `len` bytes.
    pub fn data(&self) -> &'a [u8] {
        // Safety: buf/len describe the driver-owned buffer, valid until fb_return.
        unsafe { core::slice::from_raw_parts((*self.raw).buf, (*self.raw).len) }
    }

    pub fn width(&self) -> usize {
        unsafe { (*self.raw).width }
    }

    pub fn height(&self) -> usize {
        unsafe { (*self.raw).height }
    }

    pub fn format(&self) -> Option<PixelFormat> {
        PixelFormat::from_raw(unsafe { (*self.raw).format })
    }

    /// Timestamp (µs since boot) of the frame's first DMA buffer.
    pub fn timestamp_us(&self) -> u64 {
        // timeval: tv_sec (i64) + tv_usec (i32), as bound by bindgen.
        unsafe {
            ((*self.raw).timestamp.tv_sec as i64 * 1_000_000 + (*self.raw).timestamp.tv_usec as i64)
                as u64
        }
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        // Safety: hands the buffer back to the driver for reuse.
        unsafe { c::esp_camera_fb_return(self.raw) };
    }
}

// ---------------------------------------------------------------------------
// CameraConfig -> camera_config_t
// ---------------------------------------------------------------------------

/// Translate the Rust-side config into the raw bindgen struct (same layout as C).
fn ffi_config(config: &CameraConfig) -> c::camera_config_t {
    use c::camera_config_t__bindgen_ty_1 as Sda; // union { pin_sccb_sda | pin_sscb_sda }
    use c::camera_config_t__bindgen_ty_2 as Scl; // union { pin_sccb_scl | pin_sscb_scl }

    let p = &config.pins;
    let mut raw: c::camera_config_t = Default::default(); // zeroed

    raw.pin_pwdn = p.pwdn;
    raw.pin_reset = p.reset;
    raw.pin_xclk = p.xclk;
    // Union init with exactly one field is safe.
    raw.__bindgen_anon_1 = Sda {
        pin_sccb_sda: p.sccb_sda,
    };
    raw.__bindgen_anon_2 = Scl {
        pin_sccb_scl: p.sccb_scl,
    };
    raw.pin_d7 = p.d7;
    raw.pin_d6 = p.d6;
    raw.pin_d5 = p.d5;
    raw.pin_d4 = p.d4;
    raw.pin_d3 = p.d3;
    raw.pin_d2 = p.d2;
    raw.pin_d1 = p.d1;
    raw.pin_d0 = p.d0;
    raw.pin_vsync = p.vsync;
    raw.pin_href = p.href;
    raw.pin_pclk = p.pclk;

    raw.xclk_freq_hz = config.xclk_freq_hz as i32;
    // LEDC timer/channel for the XCLK clock — fixed like the driver examples.
    raw.ledc_timer = c::ledc_timer_t_LEDC_TIMER_0;
    raw.ledc_channel = c::ledc_channel_t_LEDC_CHANNEL_0;
    raw.pixel_format = config.pixel_format.as_raw();
    raw.frame_size = config.frame_size.as_raw();
    raw.jpeg_quality = config.jpeg_quality as i32;
    raw.fb_count = config.fb_count;
    raw.fb_location = config.fb_location.as_raw();
    raw.grab_mode = config.grab_mode.as_raw();
    // -1 = drive SCCB over the GPIOs above (don't reuse a configured I2C bus).
    raw.sccb_i2c_port = -1;

    raw
}
