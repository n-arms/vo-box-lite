//! Host-side feature extractor: runs the same Rust modules as the ESP32-S3
//! (pyramid + FAST-12 + blur + rBRIEF) so laptop map descriptors match the
//! device's localize queries. Built with `rustc +stable` (see receive_map.py).

#![allow(dead_code)] // the included modules expose more than this harness uses

#[path = "../src/blur.rs"]
mod blur;
#[path = "../src/downscale.rs"]
mod downscale;
#[path = "../src/fast.rs"]
mod fast;
#[path = "../src/pyramid.rs"]
mod pyramid;
#[path = "../src/rbrief.rs"]
mod rbrief;

use std::path::PathBuf;
use std::process::exit;
use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: extract_host <raw_dir> <out_dir> <w> <h> [threshold]");
        exit(2);
    }
    let raw_dir = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let w: usize = args[3].parse().expect("bad width");
    let h: usize = args[4].parse().expect("bad height");
    let thr: i32 = args
        .get(5)
        .map(|s| s.parse().expect("bad threshold"))
        .unwrap_or(pyramid::FAST_THRESHOLD);
    if w < 7 || h < 7 {
        eprintln!("frame too small: {w}x{h}");
        exit(2);
    }
    fs::create_dir_all(&out_dir).expect("create out_dir");

    // Caller-owned scratch, allocated once and reused across frames (the lib is
    // alloc-free; this bin is a thin std shell around it).
    let mut arena = vec![0u8; pyramid::arena_bytes(w, h)];
    let mut work = vec![0u8; w * h];
    let mut vcol = vec![0u16; w];
    let mut corners = vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX];
    let mut scores = vec![0i32; pyramid::CORNERS_RAW_MAX];
    let mut rowidx = vec![usize::MAX; h];
    let mut nms = vec![fast::Corner { x: 0, y: 0 }; pyramid::CORNERS_RAW_MAX];
    let mut out = vec![pyramid::Feature::default(); pyramid::MAX_FEATURES];

    let mut raws: Vec<PathBuf> = fs::read_dir(&raw_dir)
        .unwrap_or_else(|e| {
            eprintln!("cannot read {}: {e}", raw_dir.display());
            exit(1);
        })
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "bit").unwrap_or(false))
        .collect();
    raws.sort();

    let mut total = 0usize;
    for path in &raws {
        let img = match fs::read(path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  !! {}: {e}", path.display());
                continue;
            }
        };
        if img.len() < w * h {
            eprintln!(
                "  !! {}: {} B < {}x{} — skipping",
                path.display(),
                img.len(),
                w,
                h
            );
            continue;
        }
        let n = pyramid::extract_pyramid(
            &img,
            w,
            h,
            thr,
            &mut arena,
            &mut work,
            &mut vcol,
            &mut corners,
            &mut scores,
            &mut rowidx,
            &mut nms,
            &mut out,
            None,
        );
        let csv = format_csv(&out[..n]);
        let stem = path.file_stem().unwrap().to_string_lossy();
        if let Err(e) = fs::write(out_dir.join(format!("{stem}.csv")), csv) {
            eprintln!("  !! writing {stem}.csv: {e}");
            continue;
        }
        total += n;
        println!("{stem}: {n} features");
    }
    println!(
        "extracted {total} features from {} frame(s) @ {w}x{h}, threshold {thr}",
        raws.len()
    );
}

/// slam-exp grey-features format: `x.xx,y.yy,<64 hex>` per row. Descriptor words
/// are serialized little-endian, exactly like the firmware's VOX2 record.
fn format_csv(feats: &[pyramid::Feature]) -> String {
    let mut s = String::with_capacity(feats.len() * 72);
    for f in feats {
        s.push_str(&format!("{:.2},{:.2},", f.x, f.y));
        for word in f.desc {
            for b in word.to_le_bytes() {
                s.push_str(&format!("{b:02x}"));
            }
        }
        s.push('\n');
    }
    s
}
