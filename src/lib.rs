//! Core algorithms for the S3 VO pipeline: feature matching + pose estimation.
//!
//! Everything is `no_std` and allocation-free. `no_std` is lifted only for
//! host-side unit tests (`cfg(test)`), so `cargo test --lib` works with a std
//! toolchain.

#![cfg_attr(not(test), no_std)]

pub mod ranac;
