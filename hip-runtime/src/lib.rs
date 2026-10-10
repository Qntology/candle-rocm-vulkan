pub mod blas;
pub mod device;
pub mod error;
pub mod memory;
pub mod module;
pub mod rng;
pub mod toolchain;
pub mod track;

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
