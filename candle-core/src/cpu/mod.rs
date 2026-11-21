//! Traits and methods for CPU-backed Tensors
#![allow(clippy::needless_range_loop)]
pub mod erf;
pub mod kernels;

#[allow(unused)]
trait Cpu {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f32) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit);
}

#[allow(unused)]
trait CpuF16 {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;
    // How many outer loop steps to accumulate in Self::Unit before flushing to f32.
    // Set to a finite value to bound rounding error (16 for example).
    const FLUSH_INTERVAL: usize = usize::MAX;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit);
}

#[allow(unused)]
trait CpuBF16 {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const bf16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit);
}

use half::{bf16, f16};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
pub mod avx;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
pub use avx::{CurrentCpu, CurrentCpuBF16, CurrentCpuF16};

#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub mod simd128;
#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub use simd128::CurrentCpu;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub use neon::{CurrentCpu, CurrentCpuBF16, CurrentCpuF16};

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    let mut sum = CurrentCpu::zero_array();

    let mut i = 0;
    while i + CurrentCpu::STEP <= k {
        for j in 0..CurrentCpu::ARR {
            sum[j] = CurrentCpu::vec_fma(
                sum[j],
                CurrentCpu::load(a_row.add(i + j * CurrentCpu::EPR)),
                CurrentCpu::load(b_row.add(i + j * CurrentCpu::EPR)),
            );
        }
        i += CurrentCpu::STEP;
    }

    CurrentCpu::vec_reduce(sum, c);

    // leftovers
    while i < k {
        *c += *a_row.add(i) * (*b_row.add(i));
        i += 1;
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    for i in 0..k {
        *c += *a_row.add(i) * (*b_row.add(i));
    }
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
#[inline(always)]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    let np = k & !(CurrentCpu::STEP - 1);

    let mut sum = CurrentCpu::zero_array();
    let mut x = CurrentCpu::zero_array();

    for i in (0..np).step_by(CurrentCpu::STEP) {
        for j in 0..CurrentCpu::ARR {
            x[j] = CurrentCpu::load(row.add(i + j * CurrentCpu::EPR));
            sum[j] = CurrentCpu::vec_add(sum[j], x[j]);
        }
    }

    CurrentCpu::vec_reduce(sum, b);

    // leftovers
    for i in np..k {
        *b += *row.add(i)
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    *b = 0f32;
    for i in 0..k {
        *b += *row.add(i)
    }
}

#[cfg(any(target_feature = "avx2", target_feature = "neon"))]
pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    let mut sumf = 0.0f32;
    let mut sum = CurrentCpuF16::zero_array();
    let mut steps_since_flush = 0usize;
    let mut i = 0;
    while i + CurrentCpuF16::STEP <= k {
        for j in 0..CurrentCpuF16::ARR {
            sum[j] = CurrentCpuF16::vec_fma(
                sum[j],
                CurrentCpuF16::load(a_row.add(i + j * CurrentCpuF16::EPR)),
                CurrentCpuF16::load(b_row.add(i + j * CurrentCpuF16::EPR)),
            );
        }
        steps_since_flush += 1;
        if steps_since_flush == CurrentCpuF16::FLUSH_INTERVAL {
            let mut partial = 0.0f32;
            CurrentCpuF16::vec_reduce(sum, &mut partial);
            sumf += partial;
            sum = CurrentCpuF16::zero_array();
            steps_since_flush = 0;
        }

        i += CurrentCpuF16::STEP;
    }

    let mut partial = 0.0f32;
    CurrentCpuF16::vec_reduce(sum, &mut partial);
    sumf += partial;

    // leftovers
    while i < k {
        sumf += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        i += 1;
    }
    *c = sumf;
}

#[cfg(any(target_feature = "avx2", target_feature = "neon"))]
pub(crate) unsafe fn vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    let mut sum = CurrentCpuBF16::zero_array();

    let mut i = 0;
    while i + CurrentCpuBF16::STEP <= k {
        for j in 0..CurrentCpuBF16::ARR {
            sum[j] = CurrentCpuBF16::vec_fma(
                sum[j],
                CurrentCpuBF16::load(a_row.add(i + j * CurrentCpuBF16::EPR)),
                CurrentCpuBF16::load(b_row.add(i + j * CurrentCpuBF16::EPR)),
            );
        }
        i += CurrentCpuBF16::STEP;
    }

    CurrentCpuBF16::vec_reduce(sum, c);

    // leftovers
    while i < k {
        *c += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        i += 1;
    }
}

#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
#[inline(always)]
pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    // leftovers
    let mut sum = 0.0;
    for i in 0..k {
        sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
    }
    *c = sum;
}

#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
#[inline(always)]
pub(crate) unsafe fn vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    // leftovers
    let mut sum = 0.0;
    for i in 0..k {
        sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
    }
    *c = sum;
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
pub(crate) unsafe fn vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    let mut i = 0;
    while i + CurrentCpuF16::STEP <= k {
        for j in 0..CurrentCpuF16::ARR {
            CurrentCpuF16::vec_store(
                c.add(i + j * CurrentCpuF16::EPR),
                CurrentCpuF16::vec_add(
                    CurrentCpuF16::load(a_row.add(i + j * CurrentCpuF16::EPR)),
                    CurrentCpuF16::load(b_row.add(i + j * CurrentCpuF16::EPR)),
                ),
            );
        }
        i += CurrentCpuF16::STEP;
    }

    // leftovers
    for j in i..k {
        *c.add(j) = *a_row.add(j) + *b_row.add(j);
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    for i in 0..k {
        *c.add(i) = *a_row.add(i) + *b_row.add(i);
    }
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
pub(crate) unsafe fn vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    let mut i = 0;
    while i + CurrentCpuBF16::STEP <= k {
        for j in 0..CurrentCpuBF16::ARR {
            CurrentCpuBF16::vec_store(
                c.add(i + j * CurrentCpuBF16::EPR),
                CurrentCpuBF16::vec_add(
                    CurrentCpuBF16::load(a_row.add(i + j * CurrentCpuBF16::EPR)),
                    CurrentCpuBF16::load(b_row.add(i + j * CurrentCpuBF16::EPR)),
                ),
            );
        }
        i += CurrentCpuBF16::STEP;
    }

    // leftovers
    for j in i..k {
        *c.add(j) = *a_row.add(j) + *b_row.add(j);
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    for i in 0..k {
        *c.add(i) = *a_row.add(i) + *b_row.add(i);
    }
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    let sv = CurrentCpuF16::from_f32(scalar.to_f32());
    let mut i = 0;
    while i + CurrentCpuF16::STEP <= k {
        for j in 0..CurrentCpuF16::ARR {
            CurrentCpuF16::vec_store(
                ys.add(i + j * CurrentCpuF16::EPR),
                CurrentCpuF16::vec_add(CurrentCpuF16::load(xs.add(i + j * CurrentCpuF16::EPR)), sv),
            );
        }
        i += CurrentCpuF16::STEP;
    }
    for j in i..k {
        *ys.add(j) = *xs.add(j) + scalar;
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    for i in 0..k {
        *ys.add(i) = *xs.add(i) + scalar;
    }
}

#[cfg(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
))]
#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    let sv = CurrentCpuBF16::from_f32(scalar.to_f32());
    let mut i = 0;
    while i + CurrentCpuBF16::STEP <= k {
        for j in 0..CurrentCpuBF16::ARR {
            CurrentCpuBF16::vec_store(
                ys.add(i + j * CurrentCpuBF16::EPR),
                CurrentCpuBF16::vec_add(
                    CurrentCpuBF16::load(xs.add(i + j * CurrentCpuBF16::EPR)),
                    sv,
                ),
            );
        }
        i += CurrentCpuBF16::STEP;
    }
    for j in i..k {
        *ys.add(j) = *xs.add(j) + scalar;
    }
}

#[cfg(not(any(
    target_feature = "neon",
    target_feature = "avx2",
    target_feature = "simd128"
)))]
#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    for i in 0..k {
        *ys.add(i) = *xs.add(i) + scalar;
    }
}

// ---------------------------------------------------------------------------
// Runtime-dispatched SIMD FP32-accumulating dot & fmadd (F32 / F16 / BF16)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86_simd {
    use core::arch::x86_64::*;
    use half::{bf16, f16};

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn hsum256_ps(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let sum = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum);
        let sums = _mm_add_ps(sum, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s0);
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.add(i + 8)),
                _mm256_loadu_ps(b.add(i + 8)),
                s1,
            );
            i += 16;
        }
        if i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s0);
            i += 8;
        }
        let mut acc = hsum256_ps(_mm256_add_ps(s0, s1));
        while i < n {
            acc = (*a.add(i)).mul_add(*b.add(i), acc);
            i += 1;
        }
        acc
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn fmadd_f32_avx2(acc: *mut f32, v: *const f32, w: f32, n: usize) {
        let wv = _mm256_set1_ps(w);
        let mut i = 0;
        while i + 16 <= n {
            let a0 = _mm256_loadu_ps(acc.add(i));
            let a1 = _mm256_loadu_ps(acc.add(i + 8));
            _mm256_storeu_ps(acc.add(i), _mm256_fmadd_ps(_mm256_loadu_ps(v.add(i)), wv, a0));
            _mm256_storeu_ps(
                acc.add(i + 8),
                _mm256_fmadd_ps(_mm256_loadu_ps(v.add(i + 8)), wv, a1),
            );
            i += 16;
        }
        if i + 8 <= n {
            let a0 = _mm256_loadu_ps(acc.add(i));
            _mm256_storeu_ps(acc.add(i), _mm256_fmadd_ps(_mm256_loadu_ps(v.add(i)), wv, a0));
            i += 8;
        }
        while i < n {
            *acc.add(i) = (*v.add(i)).mul_add(w, *acc.add(i));
            i += 1;
        }
    }

    #[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
    pub(super) unsafe fn dot_f16_avx2(a: *const f16, b: *const f16, n: usize) -> f32 {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let a0 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(i) as *const __m128i));
            let b0 = _mm256_cvtph_ps(_mm_loadu_si128(b.add(i) as *const __m128i));
            let a1 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(i + 8) as *const __m128i));
            let b1 = _mm256_cvtph_ps(_mm_loadu_si128(b.add(i + 8) as *const __m128i));
            s0 = _mm256_fmadd_ps(a0, b0, s0);
            s1 = _mm256_fmadd_ps(a1, b1, s1);
            i += 16;
        }
        if i + 8 <= n {
            let a0 = _mm256_cvtph_ps(_mm_loadu_si128(a.add(i) as *const __m128i));
            let b0 = _mm256_cvtph_ps(_mm_loadu_si128(b.add(i) as *const __m128i));
            s0 = _mm256_fmadd_ps(a0, b0, s0);
            i += 8;
        }
        let mut acc = hsum256_ps(_mm256_add_ps(s0, s1));
        while i < n {
            acc = (*a.add(i)).to_f32().mul_add((*b.add(i)).to_f32(), acc);
            i += 1;
        }
        acc
    }

    #[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
    pub(super) unsafe fn fmadd_f16_avx2(acc: *mut f32, v: *const f16, w: f32, n: usize) {
        let wv = _mm256_set1_ps(w);
        let mut i = 0;
        while i + 8 <= n {
            let vf = _mm256_cvtph_ps(_mm_loadu_si128(v.add(i) as *const __m128i));
            let a0 = _mm256_loadu_ps(acc.add(i));
            _mm256_storeu_ps(acc.add(i), _mm256_fmadd_ps(vf, wv, a0));
            i += 8;
        }
        while i < n {
            *acc.add(i) = (*v.add(i)).to_f32().mul_add(w, *acc.add(i));
            i += 1;
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn dot_bf16_avx2(a: *const bf16, b: *const bf16, n: usize) -> f32 {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let a0 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(a.add(i) as *const __m128i)),
                16,
            ));
            let b0 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(b.add(i) as *const __m128i)),
                16,
            ));
            let a1 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(a.add(i + 8) as *const __m128i)),
                16,
            ));
            let b1 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(b.add(i + 8) as *const __m128i)),
                16,
            ));
            s0 = _mm256_fmadd_ps(a0, b0, s0);
            s1 = _mm256_fmadd_ps(a1, b1, s1);
            i += 16;
        }
        if i + 8 <= n {
            let a0 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(a.add(i) as *const __m128i)),
                16,
            ));
            let b0 = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(b.add(i) as *const __m128i)),
                16,
            ));
            s0 = _mm256_fmadd_ps(a0, b0, s0);
            i += 8;
        }
        let mut acc = hsum256_ps(_mm256_add_ps(s0, s1));
        while i < n {
            acc = (*a.add(i)).to_f32().mul_add((*b.add(i)).to_f32(), acc);
            i += 1;
        }
        acc
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn fmadd_bf16_avx2(acc: *mut f32, v: *const bf16, w: f32, n: usize) {
        let wv = _mm256_set1_ps(w);
        let mut i = 0;
        while i + 8 <= n {
            let vf = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(v.add(i) as *const __m128i)),
                16,
            ));
            let a0 = _mm256_loadu_ps(acc.add(i));
            _mm256_storeu_ps(acc.add(i), _mm256_fmadd_ps(vf, wv, a0));
            i += 8;
        }
        while i < n {
            *acc.add(i) = (*v.add(i)).to_f32().mul_add(w, *acc.add(i));
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_simd {
    use core::arch::aarch64::*;
    use half::{bf16, f16};

    /// Widening f16 -> f32 dot product accumulating strictly in f32 lanes (`Q6-F02`/`Q6-O02`).
    /// Never narrows to f16 accumulation even when `+fp16` is enabled (e.g. Apple Silicon).
    #[inline]
    pub(super) unsafe fn dot_f16_to_f32_neon(a: *const f16, b: *const f16, n: usize) -> f32 {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 8 <= n {
            let av = vld1q_u16(a.add(i) as *const u16);
            let bv = vld1q_u16(b.add(i) as *const u16);
            let a_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(av)));
            let b_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(bv)));
            let a_hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(av)));
            let b_hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(bv)));
            s0 = vfmaq_f32(s0, a_lo, b_lo);
            s1 = vfmaq_f32(s1, a_hi, b_hi);
            i += 8;
        }
        let mut sum = vaddvq_f32(vaddq_f32(s0, s1));
        while i < n {
            sum += (*a.add(i)).to_f32() * (*b.add(i)).to_f32();
            i += 1;
        }
        sum
    }

    #[inline]
    pub(super) unsafe fn fmadd_f16_to_f32_neon(acc: *mut f32, v: *const f16, w: f32, n: usize) {
        let wv = vdupq_n_f32(w);
        let mut i = 0usize;
        while i + 8 <= n {
            let vv = vld1q_u16(v.add(i) as *const u16);
            let v_lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vv)));
            let v_hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(vv)));
            let a_lo = vld1q_f32(acc.add(i));
            let a_hi = vld1q_f32(acc.add(i + 4));
            vst1q_f32(acc.add(i), vfmaq_f32(a_lo, v_lo, wv));
            vst1q_f32(acc.add(i + 4), vfmaq_f32(a_hi, v_hi, wv));
            i += 8;
        }
        while i < n {
            *acc.add(i) += (*v.add(i)).to_f32() * w;
            i += 1;
        }
    }

    #[inline]
    pub(super) unsafe fn dot_bf16_to_f32_neon(a: *const bf16, b: *const bf16, n: usize) -> f32 {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 8 <= n {
            let av = vld1q_u16(a.add(i) as *const u16);
            let bv = vld1q_u16(b.add(i) as *const u16);
            let a_lo = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(av)));
            let b_lo = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bv)));
            let a_hi = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(av)));
            let b_hi = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(bv)));
            s0 = vfmaq_f32(s0, a_lo, b_lo);
            s1 = vfmaq_f32(s1, a_hi, b_hi);
            i += 8;
        }
        let mut sum = vaddvq_f32(vaddq_f32(s0, s1));
        while i < n {
            sum += (*a.add(i)).to_f32() * (*b.add(i)).to_f32();
            i += 1;
        }
        sum
    }

    #[inline]
    pub(super) unsafe fn fmadd_bf16_to_f32_neon(acc: *mut f32, v: *const bf16, w: f32, n: usize) {
        let wv = vdupq_n_f32(w);
        let mut i = 0usize;
        while i + 8 <= n {
            let vv = vld1q_u16(v.add(i) as *const u16);
            let v_lo = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(vv)));
            let v_hi = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(vv)));
            let a_lo = vld1q_f32(acc.add(i));
            let a_hi = vld1q_f32(acc.add(i + 4));
            vst1q_f32(acc.add(i), vfmaq_f32(a_lo, v_lo, wv));
            vst1q_f32(acc.add(i + 4), vfmaq_f32(a_hi, v_hi, wv));
            i += 8;
        }
        while i < n {
            *acc.add(i) += (*v.add(i)).to_f32() * w;
            i += 1;
        }
    }
}

#[inline]
pub(crate) unsafe fn vec_dot_f32_simd(a: *const f32, b: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return x86_simd::dot_f32_avx2(a, b, n);
        }
    }
    let mut res = 0.0f32;
    vec_dot_f32(a, b, &mut res, n);
    res
}

#[inline]
pub(crate) fn vec_fmadd_f32_simd(acc: &mut [f32], v: &[f32], w: f32) {
    let n = acc.len().min(v.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { x86_simd::fmadd_f32_avx2(acc.as_mut_ptr(), v.as_ptr(), w, n) };
            return;
        }
    }
    for i in 0..n {
        acc[i] += v[i] * w;
    }
}

#[inline]
pub(crate) unsafe fn vec_dot_f16_to_f32_simd(a: *const f16, b: *const f16, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            return x86_simd::dot_f16_avx2(a, b, n);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return aarch64_simd::dot_f16_to_f32_neon(a, b, n);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut res = 0.0f32;
        vec_dot_f16(a, b, &mut res, n);
        res
    }
}

#[inline]
pub(crate) fn vec_fmadd_f16_to_f32_simd(acc: &mut [f32], v: &[f16], w: f32) {
    let n = acc.len().min(v.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            unsafe { x86_simd::fmadd_f16_avx2(acc.as_mut_ptr(), v.as_ptr(), w, n) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64_simd::fmadd_f16_to_f32_neon(acc.as_mut_ptr(), v.as_ptr(), w, n) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..n {
        acc[i] += v[i].to_f32() * w;
    }
}

#[inline]
pub(crate) unsafe fn vec_dot_bf16_to_f32_simd(a: *const bf16, b: *const bf16, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return x86_simd::dot_bf16_avx2(a, b, n);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return aarch64_simd::dot_bf16_to_f32_neon(a, b, n);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut res = 0.0f32;
        vec_dot_bf16(a, b, &mut res, n);
        res
    }
}

#[inline]
pub(crate) fn vec_fmadd_bf16_to_f32_simd(acc: &mut [f32], v: &[bf16], w: f32) {
    let n = acc.len().min(v.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { x86_simd::fmadd_bf16_avx2(acc.as_mut_ptr(), v.as_ptr(), w, n) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64_simd::fmadd_bf16_to_f32_neon(acc.as_mut_ptr(), v.as_ptr(), w, n) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..n {
        acc[i] += v[i].to_f32() * w;
    }
}
