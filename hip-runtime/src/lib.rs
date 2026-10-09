pub mod blas;
pub mod device;
pub mod error;
pub mod memory;
pub mod module;
pub mod rng;
pub mod toolchain;

/// Puts the HIP runtime settings this crate depends on into the process environment.
///
/// They are read once, when the runtime initializes on the first HIP API call, so every
/// entry point that can be the first call ([`device::HipDevice::device_count`],
/// [`device::HipDevice::new`], [`device::runtime_version`]) calls this first.
///
/// `GPU_RESOURCE_CACHE_SIZE`: AMD's runtime keeps freed device allocations in an internal
/// resource cache (about 1 GB by default). On Windows that memory stays charged to the
/// process - Task Manager keeps showing it - after every tensor and the stream-ordered
/// pool have been released, so a model that was unloaded still looks resident. Measured
/// on an RX 6600 (ROCm 7.2): 1176 MB stayed charged after a full release with the cache,
/// 184 MB (the context) without it, with identical prefill/decode times. Set
/// `CANDLE_ROCM_RESOURCE_CACHE_MB` to keep a cache of that size, or set
/// `GPU_RESOURCE_CACHE_SIZE` yourself to bypass this.
pub fn configure_runtime_env() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("GPU_RESOURCE_CACHE_SIZE").is_none() {
            let mb = std::env::var("CANDLE_ROCM_RESOURCE_CACHE_MB")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(0);
            #[allow(unused_unsafe)]
            unsafe {
                std::env::set_var("GPU_RESOURCE_CACHE_SIZE", mb.to_string());
            }
        }
    });
}
