fn main() {
    embuild::espidf::sysenv::output();
    // The bench binary bakes VO_MAP_TXT into flash via include_str!(env!): a
    // changed path alone would otherwise look like a no-op to cargo.
    println!("cargo:rerun-if-env-changed=VO_MAP_TXT");
}
