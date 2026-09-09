//! Core algorithms for the S3 VO pipeline (image preprocessing + feature
//! matching + pose estimation): `no_std` + allocation-free, lifted only for
//! host tests (`cfg(test)`).

#![cfg_attr(not(test), no_std)]
// The ESP32-S3 EE/PIE SIMD inline asm (fast.rs::ee) needs the experimental-arch
// asm gate; x86 host test builds (rustc +stable --test src/fast.rs) never see it.
#![cfg_attr(target_arch = "xtensa", feature(asm_experimental_arch))]

pub mod blur;
pub mod downscale;
pub mod fast;
pub mod pyramid;
pub mod ranac;
pub mod rbrief;
