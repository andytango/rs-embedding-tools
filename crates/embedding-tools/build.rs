fn main() {
    // Only build napi when the napi feature is enabled
    #[cfg(feature = "napi")]
    napi_build::setup();
}
