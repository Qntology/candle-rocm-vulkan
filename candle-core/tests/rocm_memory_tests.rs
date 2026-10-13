//! ROCm device and memory lifecycle (checklist §3-2, I-1 leak / VRAM sawtooth regressions), the
//! ROCm counterpart of `cuda_memory_tests.rs`:
//! - `Device::new_rocm(0)` hands out the one process-wide device of the ordinal,
//! - tensors dropped on other threads free their memory without a failed `hipFree(Async)`
//!   (`hip_runtime::memory::free_failures`),
//! - `trim_memory_pool(0)` returns the idle memory of the stream-ordered pool (`hipMallocAsync`)
//!   and `mem_info` stays consistent,
//! - without a stream-ordered pool (`hipDeviceAttributeMemoryPoolsSupported` can be 0 on Windows
//!   HIP, or `CANDLE_ROCM_ASYNC_ALLOC=0`) `pool_idle_bytes` is 0 and `trim_memory_pool` only
//!   synchronizes,
//! - the pool defaults: `CANDLE_ROCM_TRIM_ON_SYNC` off, release threshold
//!   (`CANDLE_ROCM_POOL_KEEP_MB`) 256 MiB; the overrides are checked in a child process.
//!
//! Every test prints its measurements on `[rocm-mem]` lines:
//! `cargo test -p candle-core --features rocm,flash-attn,turbo-quant --test rocm_memory_tests -- --test-threads=1 --nocapture`
//!
//! Process-global state: the device cache behind `Device::new_rocm`, the default memory pool of
//! the device (its release threshold is one per-device attribute for the whole process), the GEMM
//! backend switch of the shared device and the process-wide free failure counter. Every test
//! holds `SERIAL` and puts back what it changes with drop guards (also when it panics), so the
//! file is correct with `--test-threads=1` (the intended mode) and with the parallel runner. The
//! checks of the defaults skip when a `CANDLE_ROCM_*` pool variable is set in the environment.
#![cfg(feature = "rocm")]

use candle_core::rocm_backend::hip_runtime::error::error_string;
use candle_core::rocm_backend::hip_runtime::memory as hipmem;
use candle_core::rocm_backend::hip_sys::hip_runtime as sys;
use candle_core::rocm_backend::{GemmBackend, RocmDevice, WrapErr};
use candle_core::{DType, Device, Error, Result, Tensor};
use std::ffi::c_void;
use std::sync::{Arc, Barrier, Mutex, MutexGuard, Once};
use std::time::Instant;

const MIB: usize = 1 << 20;

/// Release threshold the backend gives the default pool when `CANDLE_ROCM_POOL_KEEP_MB` (and
/// `CANDLE_GPU_POOL_KEEP_MB`) is unset (`pool_keep_bytes` in `rocm_backend/device.rs`).
const DEFAULT_KEEP_BYTES: u64 = 256 << 20;

/// `hipDeviceAttributeMemoryPoolsSupported` of `hipDeviceAttribute_t` in the ROCm 7.2
/// `hip_runtime_api.h` (the entry right after `hipDeviceAttributeWarpSize` = 87). hip-sys only
/// binds the attributes the backend uses.
const HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: i32 = 88;

/// Variables that change the pool policy of a device when it is created.
const POOL_ENV_VARS: [&str; 5] = [
    "CANDLE_ROCM_ASYNC_ALLOC",
    "CANDLE_ROCM_TRIM_ON_SYNC",
    "CANDLE_ROCM_POOL_KEEP_MB",
    "CANDLE_GPU_POOL_KEEP_MB",
    "CANDLE_ROCM_POOL_RELEASE_ZERO",
];

/// Set by `pool_env_overrides_in_a_child_process` for the child run of `child_pool_env_overrides`.
const CHILD_ENV: &str = "CANDLE_ROCM_MEMTEST_CHILD";

/// The tests share the process-wide device and look at its memory pool: one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

fn device() -> Result<Device> {
    let dev = Device::new_rocm(0)?;
    print_banner(&dev);
    Ok(dev)
}

fn rocm(dev: &Device) -> Result<RocmDevice> {
    Ok(dev.as_rocm_device()?.clone())
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB as f64
}

fn delta_mib(after: usize, before: usize) -> f64 {
    (after as f64 - before as f64) / MIB as f64
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn hip_check(code: i32, what: &str) -> Result<()> {
    if code == sys::HIP_SUCCESS {
        return Ok(());
    }
    unsafe { sys::hipGetLastError() };
    Err(Error::Msg(format!(
        "{what}: HIP error {code} ({})",
        error_string(code)
    )))
}

/// `hipDeviceAttributeMemoryPoolsSupported` of the device, or the HIP error code of the query.
fn memory_pools_supported_attr(ordinal: usize) -> std::result::Result<i32, i32> {
    let mut v = 0i32;
    let code = unsafe {
        sys::hipDeviceGetAttribute(
            &mut v,
            HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            ordinal as i32,
        )
    };
    if code == sys::HIP_SUCCESS {
        Ok(v)
    } else {
        unsafe { sys::hipGetLastError() };
        Err(code)
    }
}

/// Device current on the calling thread (`hipGetDevice`).
fn current_device() -> Result<i32> {
    let mut d = -1i32;
    hip_check(unsafe { sys::hipGetDevice(&mut d) }, "hipGetDevice")?;
    Ok(d)
}

/// `hipMemPoolAttrReleaseThreshold` of the default pool of the device.
fn pool_release_threshold(ordinal: usize) -> Result<u64> {
    let mut pool: sys::hipMemPool_t = std::ptr::null_mut();
    hip_check(
        unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, ordinal as i32) },
        "hipDeviceGetDefaultMemPool",
    )?;
    let mut v = 0u64;
    hip_check(
        unsafe {
            sys::hipMemPoolGetAttribute(
                pool,
                sys::HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut v as *mut u64 as *mut c_void,
            )
        },
        "hipMemPoolGetAttribute(ReleaseThreshold)",
    )?;
    Ok(v)
}

#[derive(Debug, Clone, Copy)]
struct PoolUsage {
    reserved: u64,
    used: u64,
}

impl PoolUsage {
    fn idle(&self) -> u64 {
        self.reserved.saturating_sub(self.used)
    }
}

impl std::fmt::Display for PoolUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reserved {:.1} MiB, used {:.1} MiB, idle {:.1} MiB",
            mib(self.reserved),
            mib(self.used),
            mib(self.idle())
        )
    }
}

/// Reserved / used bytes of the default pool, `None` when the backend does not allocate from it.
fn pool_usage(r: &RocmDevice) -> Result<Option<PoolUsage>> {
    if !r.uses_stream_ordered_alloc() {
        return Ok(None);
    }
    r.set_current()?;
    let (reserved, used) = hipmem::default_pool_usage(r.ordinal() as i32).w()?;
    Ok(Some(PoolUsage { reserved, used }))
}

/// `(free, total)` of `Device::mem_info`, checked for consistency.
fn mem(dev: &Device, what: &str) -> Result<(usize, usize)> {
    let (free, total) = dev
        .mem_info()?
        .ok_or_else(|| Error::Msg("the ROCm device reports no mem_info".into()))?;
    assert!(total > 0, "{what}: mem_info total is 0");
    assert!(
        free > 0 && free <= total,
        "{what}: mem_info free {free} total {total}"
    );
    Ok((free, total))
}

fn print_banner(dev: &Device) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let Ok(r) = dev.as_rocm_device() else { return };
        let name = r.name().unwrap_or_else(|e| format!("<{e}>"));
        let env: Vec<String> = POOL_ENV_VARS
            .iter()
            .map(|k| format!("{k}={}", std::env::var(k).unwrap_or_else(|_| "<unset>".into())))
            .collect();
        println!("[rocm-mem] device 0: {name}; {}", r.runtime_info());
        println!(
            "[rocm-mem] hipDeviceAttributeMemoryPoolsSupported = {:?}, backend allocates from the stream-ordered pool (hipMallocAsync) = {}",
            memory_pools_supported_attr(r.ordinal()),
            r.uses_stream_ordered_alloc()
        );
        println!("[rocm-mem] env: {}", env.join(", "));
    });
}

/// Puts the release threshold of the default pool back when dropped (also on panic).
struct ThresholdGuard {
    dev: Device,
    bytes: u64,
}

impl Drop for ThresholdGuard {
    fn drop(&mut self) {
        if let Err(e) = self.dev.set_memory_pool_release_threshold(self.bytes) {
            eprintln!("[rocm-mem] could not restore the pool release threshold: {e}");
        }
    }
}

/// Puts the GEMM backend of the shared device back when dropped (also on panic).
struct GemmBackendGuard {
    dev: RocmDevice,
    backend: GemmBackend,
}

impl Drop for GemmBackendGuard {
    fn drop(&mut self) {
        self.dev.set_gemm_backend(self.backend);
    }
}

/// Queues matmuls on the device without waiting for them. The input is filled on the device: a
/// CPU `randn` of the (unoptimized) test build takes about 1 s, the GPU drains a small queue in
/// that time and the wait of `trim_memory_pool` is no longer observable (measured on an RX 6600:
/// 1536^2 randn + 16 matmuls queued in 1178 ms, 1.36 ms left for synchronize_all). The host also
/// runs at most about one GEMM ahead of the GPU there (3072^2: 8 ms left to wait after 16
/// GEMMs), so the size of one GEMM sets how much wait is left to observe: 4096^2 is ~19 ms.
fn queue_busy_work(dev: &Device) -> Result<Tensor> {
    const N: usize = 4096;
    let a = Tensor::ones((N, N), DType::F32, dev)?;
    let mut acc = a.clone();
    for _ in 0..8 {
        acc = (acc.matmul(&a)? * (1.0 / N as f64))?;
    }
    Ok(acc)
}

/// §3-2 "두 번 new_rocm → 같은 디바이스".
#[test]
fn new_rocm_returns_the_shared_device() -> Result<()> {
    let _g = serial();
    let a = device()?;
    let (free_a, _) = mem(&a, "after the first new_rocm")?;
    let b = device()?;
    let (free_b, _) = mem(&b, "after the second new_rocm")?;
    assert!(a.same_device(&b));
    assert_eq!(a.location(), b.location());
    let (ra, rb) = (rocm(&a)?, rocm(&b)?);
    assert_eq!((ra.ordinal(), rb.ordinal()), (0, 0));

    // `same_device` only compares ordinals. Identity goes through state of the shared inner
    // device: a second instance would keep its own GEMM backend switch.
    let orig = ra.gemm_backend();
    let _restore = GemmBackendGuard {
        dev: ra.clone(),
        backend: orig,
    };
    let other = match orig {
        GemmBackend::RocBlas => GemmBackend::Hip,
        GemmBackend::Hip => GemmBackend::RocBlas,
    };
    ra.set_gemm_backend(other);
    let seen_by_b = rb.gemm_backend();
    let seen_by_thread = std::thread::spawn(|| -> Result<GemmBackend> {
        Ok(rocm(&Device::new_rocm(0)?)?.gemm_backend())
    })
    .join()
    .unwrap()?;
    ra.set_gemm_backend(orig);
    println!(
        "[rocm-mem] [3-2 same device] gemm backend {} -> {} on handle a: handle b sees {}, new_rocm(0) on another thread sees {}",
        orig.name(),
        other.name(),
        seen_by_b.name(),
        seen_by_thread.name()
    );
    assert_eq!(
        seen_by_b, other,
        "the second new_rocm(0) returned a separate device instance"
    );
    assert_eq!(
        seen_by_thread, other,
        "new_rocm(0) on another thread returned a separate device instance"
    );

    // Moving a tensor between the two handles is a no-op (same tensor, no copy).
    let x = Tensor::arange(0f32, 16., &a)?;
    let y = x.to_device(&b)?;
    assert_eq!(x.id(), y.id());
    assert_eq!(
        y.to_vec1::<f32>()?,
        (0..16).map(|i| i as f32).collect::<Vec<_>>()
    );
    println!(
        "[rocm-mem] [3-2 same device] free after the 1st new_rocm {:.1} MiB, after the 2nd {:.1} MiB ({:+.1} MiB)",
        mib(free_a as u64),
        mib(free_b as u64),
        delta_mib(free_b, free_a)
    );
    Ok(())
}

/// §3-2 "다른 스레드에서 해제해도 실패 카운터 0".
#[test]
fn drop_on_other_threads_keeps_the_free_failure_counter_at_zero() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let r = rocm(&dev)?;
    dev.synchronize_all()?;
    let failures_before = hipmem::free_failures();
    let pool_before = pool_usage(&r)?;
    let (free_before, _) = mem(&dev, "before the allocations")?;

    let bytes = 16 * MIB;
    let big = Tensor::zeros(bytes / 4, DType::F32, &dev)?;
    // 64 x 256 KiB, dropped by four threads at once.
    let small: Vec<Tensor> = (0..64)
        .map(|_| Tensor::zeros(64 * 1024, DType::F32, &dev))
        .collect::<Result<_>>()?;
    // A tensor whose last reference goes away on another thread.
    let shared = Tensor::zeros(bytes / 4, DType::F32, &dev)?;
    let shared_clone = shared.clone();
    dev.synchronize_all()?;
    let pool_during = pool_usage(&r)?;
    let (free_during, _) = mem(&dev, "with the tensors")?;

    // 1. A fresh thread that never bound a device.
    let fresh_thread_device = std::thread::spawn(move || -> Result<i32> {
        let current = current_device()?;
        drop(big);
        Ok(current)
    })
    .join()
    .unwrap()?;

    // 2. Four threads dropping at the same time.
    let mut chunks: Vec<Vec<Tensor>> = (0..4).map(|_| Vec::new()).collect();
    for (i, t) in small.into_iter().enumerate() {
        chunks[i % 4].push(t);
    }
    let barrier = Arc::new(Barrier::new(chunks.len()));
    let handles: Vec<_> = chunks
        .into_iter()
        .map(|chunk| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                drop(chunk);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    // 3. The last reference of a shared tensor.
    drop(shared);
    std::thread::spawn(move || drop(shared_clone))
        .join()
        .unwrap();

    // 4. Allocated on another thread, dropped here.
    let d = dev.clone();
    let from_thread = std::thread::spawn(move || Tensor::zeros(bytes / 4, DType::F32, &d))
        .join()
        .unwrap()?;
    drop(from_thread);

    dev.synchronize_all()?;
    let failures_after = hipmem::free_failures();
    let pool_after = pool_usage(&r)?;
    let (free_after, _) = mem(&dev, "after the drops")?;
    println!(
        "[rocm-mem] [3-2 cross-thread free] free failures: before {failures_before}, after {failures_after}; current device of a fresh thread: {fresh_thread_device}"
    );
    println!(
        "[rocm-mem] [3-2 cross-thread free] driver free memory: before {:.1} MiB, with 48 MiB of tensors {:.1} MiB ({:+.1}), after the drops {:.1} MiB ({:+.1})",
        mib(free_before as u64),
        mib(free_during as u64),
        delta_mib(free_during, free_before),
        mib(free_after as u64),
        delta_mib(free_after, free_before)
    );
    assert_eq!(
        failures_after, failures_before,
        "hipFree/hipFreeAsync failed in DeviceBuffer::drop for a tensor dropped on another thread"
    );
    assert_eq!(
        failures_after, 0,
        "free failures recorded earlier in this process"
    );
    match (pool_before, pool_during, pool_after) {
        (Some(before), Some(during), Some(after)) => {
            println!(
                "[rocm-mem] [3-2 cross-thread free] pool: before [{before}], with the tensors [{during}], after the drops [{after}]"
            );
            assert!(
                during.used >= before.used + 3 * bytes as u64,
                "the 48 MiB of tensors are not in the pool's used memory"
            );
            assert!(
                after.used <= before.used + MIB as u64,
                "pool usage did not come back after the cross-thread drops ({} -> {})",
                before.used,
                after.used
            );
        }
        _ => println!(
            "[rocm-mem] [3-2 cross-thread free] branch: no stream-ordered pool (hipMalloc/hipFree), pool usage check skipped"
        ),
    }
    Ok(())
}

/// §3-2 "`trim_memory_pool(0)` 후 예약량 감소, `mem_info` 정상" and "작업 종료 후
/// `trim_memory_pool(0)`".
#[test]
fn trim_memory_pool_returns_idle_memory_and_mem_info_stays_sane() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let r = rocm(&dev)?;
    let bytes = 64 * MIB;
    if !r.uses_stream_ordered_alloc() {
        println!(
            "[rocm-mem] [3-2 trim] branch: NO stream-ordered pool, nothing is reserved to trim (see memory_pools_supported_attribute_and_the_no_pool_branch)"
        );
        assert_eq!(dev.pool_idle_bytes()?, 0);
        let (_, total) = mem(&dev, "no pool, before trim")?;
        dev.trim_memory_pool(0)?;
        let (_, total_after) = mem(&dev, "no pool, after trim")?;
        assert_eq!(total_after, total);
        assert_eq!(dev.pool_idle_bytes()?, 0);
        return Ok(());
    }
    println!("[rocm-mem] [3-2 trim] branch: stream-ordered pool");

    // Keep everything reserved across synchronizations, then trim explicitly.
    let threshold = pool_release_threshold(r.ordinal())?;
    let _restore = ThresholdGuard {
        dev: dev.clone(),
        bytes: threshold,
    };
    dev.set_memory_pool_release_threshold(u64::MAX)?;
    assert_eq!(pool_release_threshold(r.ordinal())?, u64::MAX);
    let t = Tensor::zeros(bytes / 4, DType::F32, &dev)?;
    dev.synchronize()?;
    drop(t);
    dev.synchronize()?;

    let idle_before = dev.pool_idle_bytes()?;
    let pool_before = pool_usage(&r)?.unwrap();
    let (free_before, total) = mem(&dev, "before trim")?;
    let effective = dev.effective_free_memory()?.unwrap();
    println!(
        "[rocm-mem] [3-2 trim] before trim(0): pool [{pool_before}], pool_idle_bytes {:.1} MiB, mem_info free {:.1} / {:.1} MiB, effective free {:.1} MiB",
        mib(idle_before as u64),
        mib(free_before as u64),
        mib(total as u64),
        mib(effective as u64)
    );
    assert!(
        idle_before >= bytes,
        "the pool should keep the freed 64 MiB"
    );
    assert_eq!(idle_before as u64, pool_before.idle());
    assert!(
        effective >= free_before,
        "effective free memory below the driver's free memory"
    );

    let t0 = Instant::now();
    dev.trim_memory_pool(0)?;
    let trim_ms = ms(t0);
    let idle_after = dev.pool_idle_bytes()?;
    let pool_after = pool_usage(&r)?.unwrap();
    let (free_after, total_after) = mem(&dev, "after trim")?;
    println!(
        "[rocm-mem] [3-2 trim] after trim(0) ({trim_ms:.2} ms): pool [{pool_after}], pool_idle_bytes {:.1} MiB, mem_info free {:.1} MiB ({:+.1} MiB)",
        mib(idle_after as u64),
        mib(free_after as u64),
        delta_mib(free_after, free_before)
    );
    assert!(
        pool_after.reserved + bytes as u64 <= pool_before.reserved,
        "trim_memory_pool(0) did not lower the reserved memory ({} -> {})",
        pool_before.reserved,
        pool_after.reserved
    );
    assert!(
        idle_after + bytes <= idle_before,
        "trim_memory_pool(0) did not release the idle memory ({idle_before} -> {idle_after})"
    );
    assert_eq!(
        total_after, total,
        "mem_info total changed across trim_memory_pool"
    );
    let (_, raw_total) = candle_core::rocm::mem_info(0)?;
    assert_eq!(
        raw_total, total,
        "rocm::mem_info(0) and Device::mem_info disagree on the total"
    );
    println!(
        "[rocm-mem] [3-2 trim] end-of-work state after trim(0): idle {:.1} MiB, reserved {:.1} MiB",
        mib(pool_after.idle()),
        mib(pool_after.reserved)
    );
    Ok(())
}

/// §3-2 "[!] Windows HIP에서 `hipDeviceAttributeMemoryPoolsSupported`가 0일 수 있음 → 이때
/// `pool_idle_bytes=0`, trim은 동기화만 하는지". The test takes the branch of this device and
/// prints it.
#[test]
fn memory_pools_supported_attribute_and_the_no_pool_branch() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let r = rocm(&dev)?;
    let attr = memory_pools_supported_attr(r.ordinal());
    let stream_ordered = r.uses_stream_ordered_alloc();
    let async_env = std::env::var("CANDLE_ROCM_ASYNC_ALLOC").ok();
    let async_disabled = async_env
        .as_deref()
        .is_some_and(|v| v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"));
    println!(
        "[rocm-mem] [3-2 pools] hipDeviceAttributeMemoryPoolsSupported = {attr:?}, CANDLE_ROCM_ASYNC_ALLOC = {async_env:?}, backend uses the stream-ordered pool = {stream_ordered}"
    );

    if stream_ordered {
        println!("[rocm-mem] [3-2 pools] branch: POOL (the hipMallocAsync probe succeeded)");
        // The backend decides with a hipMallocAsync probe (hip_runtime::memory::
        // stream_ordered_alloc_supported), not with the attribute.
        assert_ne!(
            attr,
            Ok(0),
            "hipDeviceAttributeMemoryPoolsSupported is 0 but the backend allocates with hipMallocAsync: pool_idle_bytes / trim_memory_pool query a pool the runtime reports as unsupported"
        );
        let usage = pool_usage(&r)?.unwrap();
        let threshold = pool_release_threshold(r.ordinal())?;
        println!(
            "[rocm-mem] [3-2 pools] default pool: [{usage}], release threshold {:.1} MiB",
            mib(threshold)
        );
        assert_eq!(
            dev.pool_idle_bytes()? as u64,
            pool_usage(&r)?.unwrap().idle()
        );
        return Ok(());
    }

    let reason = if attr == Ok(0) {
        "the runtime reports no memory pool support"
    } else if async_disabled {
        "CANDLE_ROCM_ASYNC_ALLOC disables it"
    } else {
        "the hipMallocAsync probe failed"
    };
    println!(
        "[rocm-mem] [3-2 pools] branch: NO POOL ({reason}): pool_idle_bytes must stay 0 and trim_memory_pool must only synchronize"
    );
    if attr == Ok(1) && !async_disabled {
        println!(
            "[rocm-mem] [3-2 pools] note: the runtime reports pool support but the backend does not use the pool"
        );
    }

    let bytes = 64 * MIB;
    assert_eq!(dev.pool_idle_bytes()?, 0);
    let (free0, total) = mem(&dev, "no pool, start")?;
    let t = Tensor::zeros(bytes / 4, DType::F32, &dev)?;
    dev.synchronize()?;
    let (free1, _) = mem(&dev, "no pool, with 64 MiB")?;
    assert_eq!(dev.pool_idle_bytes()?, 0);
    drop(t);
    dev.synchronize()?;
    let (free2, _) = mem(&dev, "no pool, after the drop")?;
    assert_eq!(dev.pool_idle_bytes()?, 0, "pool_idle_bytes without a pool");
    let effective = dev.effective_free_memory()?.unwrap();
    // The release threshold is a no-op without a pool.
    dev.set_memory_pool_release_threshold(u64::MAX)?;
    let t0 = Instant::now();
    dev.trim_memory_pool(0)?;
    let trim_ms = ms(t0);
    let (free3, total3) = mem(&dev, "no pool, after trim")?;
    assert_eq!(
        dev.pool_idle_bytes()?,
        0,
        "pool_idle_bytes after trim without a pool"
    );
    assert_eq!(total3, total);
    println!(
        "[rocm-mem] [3-2 pools] free: start {:.1} MiB, with 64 MiB {:.1} ({:+.1}), after the drop {:.1} ({:+.1}), after trim(0) {:.1} ({:+.1}); effective free {:.1} MiB; trim(0) took {trim_ms:.2} ms (synchronize only)",
        mib(free0 as u64),
        mib(free1 as u64),
        delta_mib(free1, free0),
        mib(free2 as u64),
        delta_mib(free2, free0),
        mib(free3 as u64),
        delta_mib(free3, free0),
        mib(effective as u64)
    );
    Ok(())
}

/// `trim_memory_pool` starts with `hipDeviceSynchronize` in both branches: after it returns, the
/// queue is empty. This is the observable part of "trim은 동기화만" in the no-pool branch.
#[test]
fn trim_memory_pool_waits_for_queued_work() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let r = rocm(&dev)?;
    let branch = if r.uses_stream_ordered_alloc() {
        "pool: synchronize + hipMemPoolTrimTo"
    } else {
        "no pool: synchronize only"
    };
    // Warm-up: loads the kernels and the GEMM handle outside the measurements.
    match queue_busy_work(&dev) {
        Ok(t) => {
            dev.synchronize_all()?;
            drop(t);
        }
        Err(e) => {
            println!("[rocm-mem] [3-2 trim waits] skipped, the busy work needs kernels: {e}");
            return Ok(());
        }
    }

    // Reference: the same work waited for by synchronize_all.
    let t0 = Instant::now();
    let acc = queue_busy_work(&dev)?;
    let ref_queue_ms = ms(t0);
    let t1 = Instant::now();
    dev.synchronize_all()?;
    let ref_wait_ms = ms(t1);
    drop(acc);
    println!(
        "[rocm-mem] [3-2 trim waits] branch {branch}; reference: queuing {ref_queue_ms:.2} ms, synchronize_all waited {ref_wait_ms:.2} ms"
    );

    let mut rows = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        let acc = queue_busy_work(&dev)?;
        let queue_ms = ms(t0);
        let t1 = Instant::now();
        dev.trim_memory_pool(0)?;
        let trim_ms = ms(t1);
        let t2 = Instant::now();
        dev.synchronize_all()?;
        let sync_ms = ms(t2);
        drop(acc);
        println!(
            "[rocm-mem] [3-2 trim waits] queuing {queue_ms:.2} ms, trim_memory_pool(0) {trim_ms:.2} ms, synchronize_all right after {sync_ms:.3} ms"
        );
        rows.push((trim_ms, sync_ms));
    }
    if ref_wait_ms < 5.0 {
        println!(
            "[rocm-mem] [3-2 trim waits] the work completes while it is queued ({ref_wait_ms:.2} ms left to wait): the wait of trim is not observable, not asserted"
        );
        return Ok(());
    }
    let bound = (ref_wait_ms * 0.25).max(2.0);
    for (trim_ms, sync_ms) in rows {
        assert!(
            sync_ms <= bound,
            "work was still queued after trim_memory_pool(0) returned: synchronize_all waited {sync_ms:.2} ms (trim took {trim_ms:.2} ms, bound {bound:.2} ms)"
        );
    }
    Ok(())
}

/// §3-2 "기본값 변경 확인: `CANDLE_ROCM_TRIM_ON_SYNC` 기본 off, 풀 임계값 256 MiB
/// (`CANDLE_ROCM_POOL_KEEP_MB`)".
#[test]
fn pool_defaults_trim_on_sync_off_and_keep_256_mib() -> Result<()> {
    let _g = serial();
    let set: Vec<&str> = POOL_ENV_VARS
        .iter()
        .copied()
        .filter(|k| std::env::var_os(k).is_some())
        .collect();
    if !set.is_empty() {
        println!("[rocm-mem] [3-2 defaults] skipped: {set:?} set in the environment, the defaults are not in effect");
        return Ok(());
    }
    let dev = device()?;
    let r = rocm(&dev)?;
    if !r.uses_stream_ordered_alloc() {
        println!(
            "[rocm-mem] [3-2 defaults] branch: NO stream-ordered pool, the release threshold and trim-on-sync have nothing to act on"
        );
        return Ok(());
    }
    let threshold = pool_release_threshold(r.ordinal())?;
    let _restore = ThresholdGuard {
        dev: dev.clone(),
        bytes: threshold,
    };
    println!(
        "[rocm-mem] [3-2 defaults] release threshold of the default pool: {:.1} MiB (expected 256)",
        mib(threshold)
    );
    assert_eq!(
        threshold, DEFAULT_KEEP_BYTES,
        "default CANDLE_ROCM_POOL_KEEP_MB is not 256 MiB"
    );

    dev.trim_memory_pool(0)?;
    let base = pool_usage(&r)?.unwrap();

    // TRIM_ON_SYNC off: a freed 64 MiB block (below the threshold) stays reserved across
    // `Device::synchronize`, the call that trims when trim-on-sync is on.
    let bytes = 64 * MIB;
    let t = Tensor::zeros(bytes / 4, DType::F32, &dev)?;
    dev.synchronize()?;
    drop(t);
    dev.synchronize()?;
    let after_small = pool_usage(&r)?.unwrap();
    println!(
        "[rocm-mem] [3-2 defaults] 64 MiB freed then Device::synchronize: pool [{after_small}] (start [{base}])"
    );
    assert!(
        after_small.idle() >= base.idle() + bytes as u64,
        "Device::synchronize released the freed 64 MiB: trim-on-sync is on by default"
    );

    // Above the threshold: 6 x 64 MiB freed, then synchronize. HIP may release down to the
    // threshold at the synchronization; trim-on-sync (hipMemPoolTrimTo(0)) would empty the pool.
    let ts: Vec<Tensor> = (0..6)
        .map(|_| Tensor::zeros(bytes / 4, DType::F32, &dev))
        .collect::<Result<_>>()?;
    dev.synchronize()?;
    let peak = pool_usage(&r)?.unwrap();
    drop(ts);
    dev.synchronize()?;
    let after_big = pool_usage(&r)?.unwrap();
    let released = if after_big.reserved + (bytes as u64) / 2 < peak.reserved {
        "released down toward the threshold"
    } else {
        "kept everything"
    };
    println!(
        "[rocm-mem] [3-2 defaults] 384 MiB live: pool [{peak}]; freed then Device::synchronize: pool [{after_big}] -> HIP {released} (threshold {:.0} MiB)",
        mib(threshold)
    );
    assert!(
        after_big.reserved >= 2 * bytes as u64,
        "the pool kept only {:.1} MiB after synchronize with a 256 MiB threshold",
        mib(after_big.reserved)
    );

    // End of work: the explicit trim returns what the threshold kept.
    dev.trim_memory_pool(0)?;
    let end = pool_usage(&r)?.unwrap();
    println!("[rocm-mem] [3-2 defaults] end-of-work trim_memory_pool(0): pool [{end}]");
    assert!(end.reserved + bytes as u64 <= after_big.reserved);
    Ok(())
}

/// §3-2 defaults, the other side: the variables are read when the device is created, which
/// happens once per process, so the overrides run in a child process of this test binary.
#[test]
fn pool_env_overrides_in_a_child_process() -> Result<()> {
    let _g = serial();
    let exe = std::env::current_exe()?;
    let out = std::process::Command::new(exe)
        .args([
            "child_pool_env_overrides",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .env("CANDLE_ROCM_POOL_KEEP_MB", "64")
        .env("CANDLE_ROCM_TRIM_ON_SYNC", "1")
        .env_remove("CANDLE_GPU_POOL_KEEP_MB")
        .env_remove("CANDLE_ROCM_POOL_RELEASE_ZERO")
        .env_remove("CANDLE_ROCM_ASYNC_ALLOC")
        .output()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stdout.lines().filter(|l| l.contains("[rocm-mem]")) {
        println!("{line}");
    }
    if !out.status.success() {
        println!("--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}");
    }
    assert!(
        out.status.success(),
        "the child process failed: {}",
        out.status
    );
    assert!(
        stdout.contains("1 passed"),
        "the child test did not run:\n{stdout}"
    );
    Ok(())
}

/// Child side of `pool_env_overrides_in_a_child_process` (`CANDLE_ROCM_POOL_KEEP_MB=64`,
/// `CANDLE_ROCM_TRIM_ON_SYNC=1`).
#[test]
#[ignore = "run by pool_env_overrides_in_a_child_process"]
fn child_pool_env_overrides() -> Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        println!("[rocm-mem] child_pool_env_overrides: not a child run, skipped");
        return Ok(());
    }
    let dev = device()?;
    let r = rocm(&dev)?;
    if !r.uses_stream_ordered_alloc() {
        println!("[rocm-mem] [3-2 env child] branch: NO stream-ordered pool, the overrides have nothing to act on");
        return Ok(());
    }
    let threshold = pool_release_threshold(r.ordinal())?;
    println!(
        "[rocm-mem] [3-2 env child] CANDLE_ROCM_POOL_KEEP_MB=64 -> release threshold {:.1} MiB",
        mib(threshold)
    );
    assert_eq!(
        threshold,
        64 << 20,
        "CANDLE_ROCM_POOL_KEEP_MB=64 not applied"
    );

    // 32 MiB is below the 64 MiB threshold: only trim-on-sync releases it at synchronize.
    let bytes = 32 * MIB;
    let t = Tensor::zeros(bytes / 4, DType::F32, &dev)?;
    dev.synchronize_all()?;
    let during = pool_usage(&r)?.unwrap();
    drop(t);
    dev.synchronize()?;
    let after = pool_usage(&r)?.unwrap();
    println!(
        "[rocm-mem] [3-2 env child] CANDLE_ROCM_TRIM_ON_SYNC=1: 32 MiB live [{during}] -> freed then Device::synchronize [{after}]"
    );
    assert!(
        after.idle() < bytes as u64,
        "CANDLE_ROCM_TRIM_ON_SYNC=1: Device::synchronize kept the freed 32 MiB"
    );
    Ok(())
}

#[test]
fn mem_info_and_binding_on_a_fresh_thread() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let r = rocm(&dev)?;
    let d = dev.clone();
    let (current, free, total, effective, idle) = std::thread::spawn(move || -> Result<_> {
        let current = current_device()?;
        let (free, total) = d.mem_info()?.expect("rocm reports its memory");
        let effective = d.effective_free_memory()?.unwrap();
        let idle = d.pool_idle_bytes()?;
        Ok((current, free, total, effective, idle))
    })
    .join()
    .unwrap()?;
    let device_total = r.total_memory()?;
    println!(
        "[rocm-mem] [3-2 mem_info] fresh thread (current device {current}): free {:.1} MiB / total {:.1} MiB, effective {:.1} MiB, pool idle {:.1} MiB; hipDeviceTotalMem {:.1} MiB",
        mib(free as u64),
        mib(total as u64),
        mib(effective as u64),
        mib(idle as u64),
        mib(device_total as u64)
    );
    assert!(
        total > 0 && free > 0 && free <= total,
        "free {free} total {total}"
    );
    // `effective` comes from a second hipMemGetInfo: allow other processes to move `free` a bit.
    assert!(
        effective + 16 * MIB >= free,
        "effective {effective} < free {free}"
    );

    let d = dev.clone();
    let bound = std::thread::spawn(move || -> Result<i32> {
        d.bind_to_thread()?;
        current_device()
    })
    .join()
    .unwrap()?;
    assert_eq!(bound, 0, "bind_to_thread did not make device 0 current");

    // The other backends answer without a GPU query.
    assert_eq!(Device::Cpu.mem_info()?, None);
    assert_eq!(Device::Cpu.effective_free_memory()?, None);
    assert_eq!(Device::Cpu.pool_idle_bytes()?, 0);
    assert_eq!(Device::Cpu.take_deferred_error(), None);
    Device::Cpu.synchronize_all()?;
    Device::Cpu.trim_memory_pool(0)?;
    Device::Cpu.release_workspaces()?;
    // ROCm has no deferred error state.
    assert_eq!(dev.take_deferred_error(), None);
    Ok(())
}

/// §3-2 "멀티 GPU면: `RocmStorage` Drop에서 `hipSetDevice` 바인딩 검토": a device 0 tensor dropped
/// on a thread where device 1 is current. Skipped with a single GPU.
#[test]
fn drop_while_another_device_is_current() -> Result<()> {
    let _g = serial();
    let n = candle_core::rocm::device_count();
    if n < 2 {
        println!("[rocm-mem] [3-2 multi-GPU] {n} device(s): skipped (needs 2)");
        return Ok(());
    }
    let dev = device()?;
    let r = rocm(&dev)?;
    dev.synchronize_all()?;
    let failures_before = hipmem::free_failures();
    let pool_before = pool_usage(&r)?;
    let t = Tensor::zeros(4 * MIB, DType::F32, &dev)?;
    dev.synchronize_all()?;
    let current = std::thread::spawn(move || -> Result<i32> {
        hip_check(unsafe { sys::hipSetDevice(1) }, "hipSetDevice(1)")?;
        drop(t);
        current_device()
    })
    .join()
    .unwrap()?;
    dev.synchronize_all()?;
    let failures_after = hipmem::free_failures();
    let pool_after = pool_usage(&r)?;
    println!(
        "[rocm-mem] [3-2 multi-GPU] 16 MiB device 0 tensor dropped while device {current} is current: free failures {failures_before} -> {failures_after}, device 0 pool {pool_before:?} -> {pool_after:?}"
    );
    assert_eq!(
        failures_after, failures_before,
        "freeing a device 0 buffer failed while device 1 was current (DeviceBuffer::drop does not bind its device)"
    );
    if let (Some(before), Some(after)) = (pool_before, pool_after) {
        assert!(
            after.used <= before.used + MIB as u64,
            "the device 0 pool did not get the memory back ({} -> {})",
            before.used,
            after.used
        );
    }
    Ok(())
}
