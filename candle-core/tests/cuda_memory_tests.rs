//! CUDA device and memory lifecycle: the process-wide device of `Device::new_cuda`, drops and
//! synchronization from other threads, the stream-ordered memory pool, pinned host copies on the
//! copy stream and the quantized matmul workspaces.
#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::{result as drv, sys};
use candle_core::cuda_backend::WrapErr;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{CudaDevice, CudaStorage, DType, Device, Error, Module, Result, Storage, Tensor};
use std::sync::Mutex;

const MIB: usize = 1 << 20;

/// The tests share the process-wide device and look at its memory pool: one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

fn device() -> Result<Device> {
    Device::new_cuda(0)
}

fn current_context() -> Option<sys::CUcontext> {
    drv::ctx::get_current().expect("cuCtxGetCurrent")
}

fn with_cuda_storage<R>(t: &Tensor, f: impl FnOnce(&CudaStorage) -> R) -> R {
    let (storage, _layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cuda(s) => f(s),
        _ => panic!("not a cuda tensor"),
    }
}

/// Queues a few hundred milliseconds of matmuls on the device without waiting for them.
fn queue_busy_work(dev: &Device) -> Result<Tensor> {
    let a = Tensor::randn(0f32, 1., (1536, 1536), dev)?;
    let mut acc = a.clone();
    for _ in 0..16 {
        acc = (acc.matmul(&a)? * 1e-2)?;
    }
    Ok(acc)
}

fn event_complete(ev: &candle_core::cuda_backend::cudarc::driver::CudaEvent) -> bool {
    unsafe { drv::event::query(ev.cu_event()) }.is_ok()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn new_cuda_returns_the_shared_device() -> Result<()> {
    let _g = serial();
    let a = device()?;
    let b = device()?;
    assert!(a.same_device(&b));
    assert_eq!(a.as_cuda_device()?.id(), b.as_cuda_device()?.id());

    // Moving a tensor between two handles is a no-op (same tensor, no copy).
    let x = Tensor::arange(0f32, 16., &a)?;
    let y = x.to_device(&b)?;
    assert_eq!(x.id(), y.id());

    // Another thread gets the same device too.
    let id = std::thread::spawn(|| device().map(|d| d.as_cuda_device().map(|c| c.id())))
        .join()
        .unwrap()??;
    assert_eq!(id, a.as_cuda_device()?.id());

    // `new_cuda_with_stream` stays an independent device.
    let c = Device::new_cuda_with_stream(0)?;
    assert!(!a.same_device(&c));
    let z = x.to_device(&c)?.to_vec1::<f32>()?;
    assert_eq!(z, x.to_vec1::<f32>()?);
    drop(c);
    // The shared device keeps working after the independent one is gone.
    assert_eq!(x.sum_all()?.to_scalar::<f32>()?, 120.);
    assert_eq!(a.take_deferred_error(), None);
    Ok(())
}

#[test]
fn drop_on_a_thread_without_context() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let cuda = dev.as_cuda_device()?.clone();
    dev.synchronize()?;
    let _ = dev.take_deferred_error();
    let before = cuda.pool_stats()?.map(|s| s.used);

    let bytes = 16 * MIB;
    let t = Tensor::ones(bytes / 4, DType::F32, &dev)?;
    dev.synchronize()?;
    let during = cuda.pool_stats()?.map(|s| s.used);

    std::thread::spawn(move || -> Result<()> {
        if current_context().is_some() {
            return Err(Error::Msg("a fresh thread already has a CUDA context".into()));
        }
        drop(t);
        Ok(())
    })
    .join()
    .unwrap()?;

    // Checked before any synchronize: synchronize drains (and logs) deferred errors.
    assert_eq!(dev.take_deferred_error(), None);
    dev.synchronize()?;
    if let (Some(before), Some(during)) = (before, during) {
        let after = cuda.pool_stats()?.unwrap().used;
        eprintln!("pool used: before {before}, with the tensor {during}, after the drop {after}");
        assert!(during >= before + bytes as u64);
        assert!(
            after <= before + MIB as u64,
            "pool usage did not come back after the cross-thread drop ({before} -> {after})"
        );
    } else {
        eprintln!("no stream-ordered memory pool on this GPU: usage check skipped");
    }
    Ok(())
}

#[test]
fn synchronize_waits_for_work_queued_by_other_threads() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    dev.synchronize()?;

    let d = dev.clone();
    let event = std::thread::spawn(move || -> Result<_> {
        let acc = queue_busy_work(&d)?;
        let event = d.as_cuda_device()?.cuda_stream().record_event(None).w()?;
        drop(acc);
        Ok(event)
    })
    .join()
    .unwrap()?;

    let d = dev.clone();
    let (pending_before, complete_after) = std::thread::spawn(move || -> Result<_> {
        let pending_before = !event_complete(&event);
        d.synchronize()?;
        Ok((pending_before, event_complete(&event)))
    })
    .join()
    .unwrap()?;
    eprintln!("work of the other thread still pending before synchronize: {pending_before}");
    assert!(complete_after, "synchronize on thread B did not wait for thread A's work");

    // Same with synchronize_all on yet another thread.
    let d = dev.clone();
    let event = std::thread::spawn(move || -> Result<_> {
        let acc = queue_busy_work(&d)?;
        let event = d.as_cuda_device()?.cuda_stream().record_event(None).w()?;
        drop(acc);
        Ok(event)
    })
    .join()
    .unwrap()?;
    let d = dev.clone();
    let complete = std::thread::spawn(move || -> Result<_> {
        d.synchronize_all()?;
        Ok(event_complete(&event))
    })
    .join()
    .unwrap()?;
    assert!(complete, "synchronize_all did not wait for the other thread's work");
    assert_eq!(dev.take_deferred_error(), None);
    Ok(())
}

#[test]
fn trim_memory_pool_returns_idle_memory() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let cuda = dev.as_cuda_device()?.clone();
    let Some(stats) = cuda.pool_stats()? else {
        eprintln!("no stream-ordered memory pool on this GPU: skipped");
        return Ok(());
    };
    let threshold = stats.release_threshold;
    eprintln!("default release threshold: {} MiB", threshold >> 20);

    // Keep everything reserved across synchronizations, then trim explicitly.
    dev.set_memory_pool_release_threshold(u64::MAX)?;
    assert_eq!(cuda.pool_stats()?.unwrap().release_threshold, u64::MAX);
    let bytes = 64 * MIB;
    let t = Tensor::zeros(bytes / 4, DType::F32, &dev)?;
    dev.synchronize()?;
    drop(t);
    dev.synchronize()?;

    let idle_before = dev.pool_idle_bytes()?;
    let (free_before, total) = dev.mem_info()?.unwrap();
    let effective = dev.effective_free_memory()?.unwrap();
    eprintln!(
        "before trim: idle {} MiB, free {} MiB / {} MiB, effective {} MiB",
        idle_before / MIB,
        free_before / MIB,
        total / MIB,
        effective / MIB
    );
    assert!(idle_before >= bytes, "the pool should keep the freed 64 MiB");
    assert!(effective >= free_before);
    let reserved_before = cuda.pool_stats()?.unwrap().reserved;

    dev.trim_memory_pool(0)?;
    let idle_after = dev.pool_idle_bytes()?;
    let (free_after, _) = dev.mem_info()?.unwrap();
    eprintln!(
        "after trim: idle {} MiB, free {} MiB",
        idle_after / MIB,
        free_after / MIB
    );
    assert!(
        idle_after + bytes <= idle_before,
        "trim_memory_pool(0) did not release the idle memory ({idle_before} -> {idle_after})"
    );
    let reserved_after = cuda.pool_stats()?.unwrap().reserved;
    assert!(reserved_after + bytes as u64 <= reserved_before);

    dev.set_memory_pool_release_threshold(threshold)?;
    assert_eq!(cuda.pool_stats()?.unwrap().release_threshold, threshold);
    assert_eq!(dev.take_deferred_error(), None);
    Ok(())
}

#[test]
fn mem_info_and_binding_on_a_fresh_thread() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let d = dev.clone();
    let (free, total, err) = std::thread::spawn(move || -> Result<_> {
        assert!(current_context().is_none(), "fresh thread with a context");
        let (free, total) = d.mem_info()?.expect("cuda reports its memory");
        let effective = d.effective_free_memory()?.unwrap();
        assert!(effective >= free);
        let _ = d.pool_idle_bytes()?;
        Ok((free, total, d.take_deferred_error()))
    })
    .join()
    .unwrap()?;
    assert!(total > 0 && free > 0 && free <= total, "free {free} total {total}");
    assert_eq!(err, None);

    let d = dev.clone();
    std::thread::spawn(move || -> Result<()> {
        assert!(current_context().is_none());
        d.bind_to_thread()?;
        let want = d.as_cuda_device()?.cuda_context().cu_ctx();
        assert_eq!(current_context(), Some(want));
        Ok(())
    })
    .join()
    .unwrap()?;

    // The other backends answer without a GPU query.
    assert_eq!(Device::Cpu.mem_info()?, None);
    assert_eq!(Device::Cpu.effective_free_memory()?, None);
    assert_eq!(Device::Cpu.pool_idle_bytes()?, 0);
    assert_eq!(Device::Cpu.take_deferred_error(), None);
    Device::Cpu.synchronize_all()?;
    Device::Cpu.trim_memory_pool(0)?;
    Device::Cpu.release_workspaces()?;
    Ok(())
}

#[test]
fn pinned_upload_download_round_trip_and_ordering() -> Result<()> {
    let _g = serial();
    let dev = device()?;
    let cuda: CudaDevice = dev.as_cuda_device()?.clone();
    let copy = cuda.copy_stream()?;

    let n = 1 << 20;
    let bytes = n * 4;
    let pattern: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 - 500.).collect();
    let mut host = cuda.alloc_pinned(bytes)?;
    assert_eq!(host.len(), bytes);
    host.as_mut_slice().copy_from_slice(&f32_bytes(&pattern));

    // Upload ordering: the copy must land after the compute work queued before it (here a slow
    // queue of matmuls followed by an in-place fill of the destination with 7s).
    let dst = Tensor::zeros(n, DType::F32, &dev)?;
    let busy = queue_busy_work(&dev)?;
    let sevens = (Tensor::ones(n, DType::F32, &dev)? * 7.)?;
    dst.slice_set(&sevens, 0, 0)?;
    let up = with_cuda_storage(&dst, |s| copy.upload(&host, 0, s, 0, bytes))?;
    eprintln!("upload complete right after queuing: {}", up.is_complete());
    up.wait_on_compute(&cuda)?;
    let got = dst.to_vec1::<f32>()?;
    assert_eq!(got, pattern, "upload not ordered after the compute writes");
    drop(busy);

    // Download ordering: the copy must read the result of compute queued before it.
    let busy = queue_busy_work(&dev)?;
    let doubled = (&dst * 2.)?;
    let mut out = cuda.alloc_pinned(bytes)?;
    let down = with_cuda_storage(&doubled, |s| copy.download(s, 0, &mut out, 0, bytes))?;
    down.wait_on_compute(&cuda)?;
    down.synchronize()?;
    assert!(down.is_complete());
    let got = bytes_f32(out.as_slice());
    let want: Vec<f32> = pattern.iter().map(|v| v * 2.).collect();
    assert_eq!(got, want, "download not ordered after the compute that produced it");
    drop(busy);

    // Partial ranges at byte offsets, and reuse of the staging buffer.
    let off = 4096;
    let len = 64 * 1024;
    host.as_mut_slice()[off..off + len].fill(0);
    let up = with_cuda_storage(&dst, |s| copy.upload(&host, off, s, off, len))?;
    up.wait_on_compute(&cuda)?;
    let got = dst.to_vec1::<f32>()?;
    for (i, (g, p)) in got.iter().zip(pattern.iter()).enumerate() {
        let in_range = i * 4 >= off && i * 4 < off + len;
        assert_eq!(*g, if in_range { 0. } else { *p }, "element {i}");
    }
    // An unwaited event is waited by its drop, which orders later frees after the copy.
    let ev = with_cuda_storage(&dst, |s| copy.download(s, 0, &mut out, 0, bytes))?;
    drop(ev);
    drop(dst);
    copy.synchronize()?;
    assert_eq!(bytes_f32(&out.as_slice()[off..off + 8]), vec![0., 0.]);

    // Out-of-range requests are rejected.
    let small = Tensor::zeros(16, DType::F32, &dev)?;
    assert!(with_cuda_storage(&small, |s| copy.upload(&host, 0, s, 0, 128)).is_err());
    assert!(with_cuda_storage(&small, |s| copy.upload(&host, bytes - 8, s, 0, 16)).is_err());
    assert!(with_cuda_storage(&small, |s| copy.download(s, 32, &mut out, 0, 64)).is_err());
    assert_eq!(dev.take_deferred_error(), None);
    Ok(())
}

#[test]
fn release_workspaces_after_quantized_matmul() -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute as A;
    let _g = serial();
    let dev = device()?;
    let cuda = dev.as_cuda_device()?.clone();
    dev.release_workspaces()?;
    assert_eq!(cuda.workspace_bytes(), 0);

    let (n, k) = (256, 512);
    let w = Tensor::randn(0f32, 1., (n, k), &dev)?;
    let qw = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
    let x_vec = Tensor::randn(0f32, 1., (4, k), &dev)?; // MMVQ path (<= 8 rows)
    let x_mat = Tensor::randn(0f32, 1., (64, k), &dev)?; // MMQ path

    let y_vec = qw.forward(&x_vec)?;
    let y_mat = qw.forward(&x_mat)?;
    dev.synchronize()?;
    let ws = cuda.workspace_bytes();
    let nsm = cuda.device_attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
    eprintln!("workspaces after the matmuls: {} KiB ({nsm} SMs)", ws / 1024);
    // The stream-k fixup is sized from the SM count (it used to be a fixed 16 MiB).
    assert!(ws >= nsm * 128 * 128 * 4, "fixup workspace missing");
    assert!(ws < nsm * 128 * 128 * 4 + 4 * MIB, "workspaces larger than needed: {ws}");

    let used_before = cuda.pool_stats()?.map(|s| s.used);
    dev.release_workspaces()?;
    assert_eq!(cuda.workspace_bytes(), 0);
    dev.synchronize()?;
    if let (Some(before), Some(after)) = (used_before, cuda.pool_stats()?.map(|s| s.used)) {
        eprintln!("pool used: {before} -> {after} after release_workspaces");
        assert!(after + (ws as u64) <= before + MIB as u64);
    }

    // The workspaces come back on demand and the results do not change.
    let y_vec2 = qw.forward(&x_vec)?;
    let y_mat2 = qw.forward(&x_mat)?;
    let d_vec = (y_vec - &y_vec2)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let d_mat = (y_mat.clone() - &y_mat2)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert_eq!(d_vec, 0.);
    assert_eq!(d_mat, 0.);
    assert!(cuda.workspace_bytes() > 0);

    // Sanity: the quantized matmul is close to the f32 one.
    let reference = x_mat.matmul(&w.t()?)?;
    let err = (y_mat - reference)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(err < 1.0, "q8_0 matmul error {err}");
    dev.release_workspaces()?;
    assert_eq!(dev.take_deferred_error(), None);
    Ok(())
}
