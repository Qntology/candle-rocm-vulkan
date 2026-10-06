//! Launch helpers shared by the ROCm storage implementation.
use crate::{DType, Layout};
use std::ffi::c_void;

/// Maximum rank of the by-value stride descriptors (`SI1`/`SI2`/`SI3` in `kernels/common.h`).
pub const MAX_DIMS: usize = 8;

/// Kernel parameters, each stored in its own 8-byte aligned buffer so that the pointers handed to
/// `hipModuleLaunchKernel` stay valid until the launch call returns.
#[derive(Default)]
pub struct LaunchArgs {
    params: Vec<Vec<u64>>,
}

impl LaunchArgs {
    pub fn new() -> Self {
        Self {
            params: Vec::with_capacity(12),
        }
    }

    pub fn usize(&mut self, v: usize) -> &mut Self {
        self.params.push(vec![v as u64]);
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.params.push(vec![v as u64]);
        self
    }

    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.params.push(vec![v.to_bits() as u64]);
        self
    }

    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.params.push(vec![v.to_bits()]);
        self
    }

    pub fn bits(&mut self, v: u64) -> &mut Self {
        self.params.push(vec![v]);
        self
    }

    pub fn ptr<T>(&mut self, p: *const T) -> &mut Self {
        self.params.push(vec![p as usize as u64]);
        self
    }

    pub fn words(&mut self, w: Vec<u64>) -> &mut Self {
        self.params.push(w);
        self
    }

    pub(crate) fn pointers(&mut self) -> Vec<*mut c_void> {
        self.params
            .iter_mut()
            .map(|p| p.as_mut_ptr() as *mut c_void)
            .collect()
    }
}

/// Strided layout description for `N` operands sharing the same dims.
/// `ndim == 0` means that every operand is contiguous.
#[derive(Debug, Clone)]
pub struct StridedInfo<const N: usize> {
    pub ndim: usize,
    pub dims: [usize; MAX_DIMS],
    pub strides: [[usize; MAX_DIMS]; N],
}

impl<const N: usize> StridedInfo<N> {
    pub fn contiguous() -> Self {
        Self {
            ndim: 0,
            dims: [0; MAX_DIMS],
            strides: [[0; MAX_DIMS]; N],
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.ndim == 0
    }

    /// Builds the descriptor, merging dimensions when possible. Returns `None` when the
    /// merged rank still exceeds `MAX_DIMS`.
    pub fn new(dims: &[usize], strides: [&[usize]; N]) -> Option<Self> {
        let (dims, strides) = coalesce(dims, &strides);
        let all_contiguous = (0..N).all(|o| {
            let mut acc = 1usize;
            for d in (0..dims.len()).rev() {
                if dims[d] > 1 && strides[o][d] != acc {
                    return false;
                }
                acc *= dims[d];
            }
            true
        });
        if all_contiguous {
            return Some(Self::contiguous());
        }
        if dims.len() > MAX_DIMS {
            return None;
        }
        let mut info = Self::contiguous();
        info.ndim = dims.len();
        info.dims[..dims.len()].copy_from_slice(&dims);
        for o in 0..N {
            info.strides[o][..dims.len()].copy_from_slice(&strides[o]);
        }
        Some(info)
    }

    pub fn from_layout(layout: &Layout) -> Option<StridedInfo<1>> {
        if layout.is_contiguous() {
            return Some(StridedInfo::<1>::contiguous());
        }
        StridedInfo::<1>::new(layout.dims(), [layout.stride()])
    }

    pub fn to_words(&self) -> Vec<u64> {
        let mut w = Vec::with_capacity(1 + MAX_DIMS * (N + 1));
        w.push(self.ndim as u64);
        w.extend(self.dims.iter().map(|&v| v as u64));
        for s in self.strides.iter() {
            w.extend(s.iter().map(|&v| v as u64));
        }
        w
    }
}

/// Removes size-1 dimensions and merges adjacent dimensions that are contiguous for every operand.
pub fn coalesce<const N: usize>(dims: &[usize], strides: &[&[usize]; N]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut out_dims: Vec<usize> = Vec::with_capacity(dims.len());
    let mut out_strides: Vec<Vec<usize>> = vec![Vec::with_capacity(dims.len()); N];
    for d in 0..dims.len() {
        if dims[d] == 1 {
            continue;
        }
        if let Some(&last) = out_dims.last() {
            let mergeable = (0..N).all(|o| {
                let prev = *out_strides[o].last().unwrap();
                prev == strides[o][d] * dims[d]
            });
            if mergeable {
                let n = out_dims.len();
                out_dims[n - 1] = last * dims[d];
                for o in 0..N {
                    let m = out_strides[o].len();
                    out_strides[o][m - 1] = strides[o][d];
                }
                continue;
            }
        }
        out_dims.push(dims[d]);
        for o in 0..N {
            out_strides[o].push(strides[o][d]);
        }
    }
    if out_dims.is_empty() {
        out_dims.push(1);
        for o in 0..N {
            out_strides[o].push(1);
        }
    }
    (out_dims, out_strides)
}

pub fn grid_1d(n: usize, block: u32) -> u32 {
    n.div_ceil(block as usize).clamp(1, 1 << 20) as u32
}

pub fn dtype_suffix(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::U8 => Some("u8"),
        DType::U32 => Some("u32"),
        DType::I16 => Some("i16"),
        DType::I32 => Some("i32"),
        DType::I64 => Some("i64"),
        DType::BF16 => Some("bf16"),
        DType::F16 => Some("f16"),
        DType::F32 => Some("f32"),
        DType::F64 => Some("f64"),
        DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => None,
    }
}

/// Suffix of the `cast_*` kernels, which also cover `F8E4M3`.
pub fn cast_suffix(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F8E4M3 => Some("f8e4m3"),
        d => dtype_suffix(d),
    }
}

pub fn is_float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
}

/// Bytes per element as stored on the device (the dummy sub-byte types use one byte each).
pub fn elem_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 => 1,
        d => d.size_in_bytes(),
    }
}

pub fn size_suffix(bytes: usize) -> Option<&'static str> {
    match bytes {
        1 => Some("b1"),
        2 => Some("b2"),
        4 => Some("b4"),
        8 => Some("b8"),
        _ => None,
    }
}

pub fn index_suffix(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::U8 => Some("u8"),
        DType::U32 => Some("u32"),
        DType::I64 => Some("i64"),
        _ => None,
    }
}
