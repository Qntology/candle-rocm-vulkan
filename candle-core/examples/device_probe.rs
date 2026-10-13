//! Prints which accelerator devices this build can open (used by CI to decide whether GPU runtime
//! tests can run at all, e.g. on virtualized GitHub-hosted macOS runners). Never fails: the
//! result lines are `DEVICE_PROBE <backend> OK|FAIL <detail>`.
use candle_core::{Device, Tensor};

fn report(name: &str, dev: candle_core::Result<Device>) {
    match dev.and_then(|d| {
        // A tiny matmul proves the device can run kernels, not only enumerate.
        let a = Tensor::arange(0f32, 6., &d)?.reshape((2, 3))?;
        let b = a.t()?.contiguous()?;
        let c = a.matmul(&b)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        Ok((d, c))
    }) {
        Ok((d, c)) => println!("DEVICE_PROBE {name} OK {:?} matmul={:?}", d.location(), c),
        Err(e) => println!(
            "DEVICE_PROBE {name} FAIL {}",
            e.to_string().replace('\n', " ")
        ),
    }
}

fn main() {
    report("cpu", Ok(Device::Cpu));
    #[cfg(feature = "metal")]
    report("metal", Device::new_metal(0));
    #[cfg(feature = "vulkan")]
    report("vulkan", Device::new_vulkan(0));
}
