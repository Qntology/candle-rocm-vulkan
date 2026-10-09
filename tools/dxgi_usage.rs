//! What Windows (WDDM) charges to *this process* on the GPU, via
//! `IDXGIAdapter3::QueryVideoMemoryInfo`. This is the number Task Manager adds up,
//! independent of what the HIP / Vulkan runtime believes it has freed.
#![allow(dead_code)]

#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessVram {
    /// Dedicated (VRAM) bytes currently used by this process.
    pub local_usage: u64,
    pub local_budget: u64,
    /// Shared (system RAM mapped for the GPU) bytes currently used by this process.
    pub nonlocal_usage: u64,
    pub nonlocal_budget: u64,
}

#[cfg(windows)]
pub fn process_vram() -> Option<ProcessVram> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::*;
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best: Option<(IDXGIAdapter1, usize)> = None;
        let mut i = 0u32;
        while let Ok(a) = factory.EnumAdapters1(i) {
            i += 1;
            let Ok(desc) = a.GetDesc1() else { continue };
            if desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }
            let ded = desc.DedicatedVideoMemory;
            if best.as_ref().map(|b| ded > b.1).unwrap_or(true) {
                best = Some((a, ded));
            }
        }
        let (a, _) = best?;
        let a3: IDXGIAdapter3 = a.cast().ok()?;
        let mut l = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut n = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut l).ok()?;
        a3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut n).ok()?;
        Some(ProcessVram {
            local_usage: l.CurrentUsage,
            local_budget: l.Budget,
            nonlocal_usage: n.CurrentUsage,
            nonlocal_budget: n.Budget,
        })
    }
}

#[cfg(not(windows))]
pub fn process_vram() -> Option<ProcessVram> {
    None
}
