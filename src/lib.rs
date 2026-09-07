//! Core algorithms for the S3 VO pipeline (image preprocessing + feature
//! matching + pose estimation): `no_std` + allocation-free, lifted only for
//! host tests (`cfg(test)`).

#![cfg_attr(not(test), no_std)]

pub mod blur;
pub mod fast;
pub mod ranac;
