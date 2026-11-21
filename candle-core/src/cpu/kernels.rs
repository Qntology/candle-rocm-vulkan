pub trait VecOps: num_traits::NumAssign + Copy {
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;

    /// Element-wise addition of two slices into a third.
    #[inline(always)]
    fn vec_add(lhs: &[Self], rhs: &[Self], res: &mut [Self]) {
        lhs.iter()
            .zip(rhs)
            .zip(res)
            .for_each(|((&a, &b), y)| *y = a + b)
    }

    /// Add a broadcast scalar to every element of a slice: `ys[i] = xs[i] + scalar`.
    #[inline(always)]
    fn scalar_add(scalar: Self, xs: &[Self], ys: &mut [Self]) {
        xs.iter().zip(ys).for_each(|(&x, y)| *y = x + scalar)
    }

    /// Dot-product of two vectors accumulated and returned in `f32` (never narrows half types).
    ///
    /// # Safety
    /// The length of `lhs` and `rhs` must be at least `len`.
    unsafe fn vec_dot_f32(lhs: *const Self, rhs: *const Self, len: usize) -> f32;

    /// Fused multiply-add of a `Self` vector into an `f32` accumulator: `acc[i] += v[i] * w`.
    fn vec_fmadd_to_f32(acc: &mut [f32], v: &[Self], w: f32);

    /// Dot-product of two vectors.
    ///
    /// # Safety
    ///
    /// The length of `lhs` and `rhs` have to be at least `len`. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        *res = Self::zero();
        for i in 0..len {
            *res += *lhs.add(i) * *rhs.add(i)
        }
    }

    /// Sum of all elements in a vector.
    ///
    /// # Safety
    ///
    /// The length of `xs` must be at least `len`. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_reduce_sum(xs: *const Self, res: *mut Self, len: usize) {
        *res = Self::zero();
        for i in 0..len {
            *res += *xs.add(i)
        }
    }

    /// Maximum element in a non-empty vector.
    ///
    /// # Safety
    ///
    /// The length of `xs` must be at least `len` and positive. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_reduce_max(xs: *const Self, res: *mut Self, len: usize) {
        *res = *xs;
        for i in 1..len {
            *res = (*res).max(*xs.add(i))
        }
    }

    /// Minimum element in a non-empty vector.
    ///
    /// # Safety
    ///
    /// The length of `xs` must be at least `len` and positive. `res` has to point to a valid
    /// element.
    #[inline(always)]
    unsafe fn vec_reduce_min(xs: *const Self, res: *mut Self, len: usize) {
        *res = *xs;
        for i in 1..len {
            *res = (*res).min(*xs.add(i))
        }
    }
}

impl VecOps for f32 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    fn vec_add(lhs: &[Self], rhs: &[Self], res: &mut [Self]) {
        #[cfg(feature = "mkl")]
        crate::mkl::vs_add(lhs, rhs, res);
        #[cfg(all(feature = "accelerate", not(feature = "mkl")))]
        crate::accelerate::vs_add(lhs, rhs, res);
        #[cfg(not(any(feature = "mkl", feature = "accelerate")))]
        lhs.iter()
            .zip(rhs)
            .zip(res)
            .for_each(|((&a, &b), y)| *y = a + b)
    }

    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        super::vec_dot_f32(lhs, rhs, res, len)
    }

    #[inline(always)]
    unsafe fn vec_dot_f32(lhs: *const Self, rhs: *const Self, len: usize) -> f32 {
        super::vec_dot_f32_simd(lhs, rhs, len)
    }

    #[inline(always)]
    fn vec_fmadd_to_f32(acc: &mut [f32], v: &[Self], w: f32) {
        super::vec_fmadd_f32_simd(acc, v, w)
    }

    #[inline(always)]
    unsafe fn vec_reduce_sum(xs: *const Self, res: *mut Self, len: usize) {
        super::vec_sum(xs, res, len)
    }
}

impl VecOps for half::f16 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    fn vec_add(lhs: &[Self], rhs: &[Self], res: &mut [Self]) {
        unsafe { super::vec_add_f16(lhs.as_ptr(), rhs.as_ptr(), res.as_mut_ptr(), lhs.len()) }
    }

    #[inline(always)]
    fn scalar_add(scalar: Self, xs: &[Self], ys: &mut [Self]) {
        unsafe { super::vec_scalar_add_f16(scalar, xs.as_ptr(), ys.as_mut_ptr(), xs.len()) }
    }

    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        let mut res_f32 = 0f32;
        super::vec_dot_f16(lhs, rhs, &mut res_f32, len);
        *res = half::f16::from_f32(res_f32);
    }

    #[inline(always)]
    unsafe fn vec_dot_f32(lhs: *const Self, rhs: *const Self, len: usize) -> f32 {
        super::vec_dot_f16_to_f32_simd(lhs, rhs, len)
    }

    #[inline(always)]
    fn vec_fmadd_to_f32(acc: &mut [f32], v: &[Self], w: f32) {
        super::vec_fmadd_f16_to_f32_simd(acc, v, w)
    }
}

impl VecOps for f64 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    #[inline(always)]
    unsafe fn vec_dot_f32(lhs: *const Self, rhs: *const Self, len: usize) -> f32 {
        let mut acc = 0.0f64;
        for i in 0..len {
            acc += *lhs.add(i) * *rhs.add(i);
        }
        acc as f32
    }

    #[inline(always)]
    fn vec_fmadd_to_f32(acc: &mut [f32], v: &[Self], w: f32) {
        for (a, &e) in acc.iter_mut().zip(v.iter()) {
            *a += (e as f32) * w;
        }
    }

    #[inline(always)]
    fn vec_add(lhs: &[f64], rhs: &[f64], res: &mut [f64]) {
        #[cfg(feature = "mkl")]
        crate::mkl::vd_add(lhs, rhs, res);
        #[cfg(all(feature = "accelerate", not(feature = "mkl")))]
        crate::accelerate::vd_add(lhs, rhs, res);
        #[cfg(not(any(feature = "mkl", feature = "accelerate")))]
        lhs.iter()
            .zip(rhs)
            .zip(res)
            .for_each(|((&a, &b), y)| *y = a + b)
    }
}
impl VecOps for half::bf16 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    #[inline(always)]
    fn vec_add(lhs: &[Self], rhs: &[Self], res: &mut [Self]) {
        unsafe { super::vec_add_bf16(lhs.as_ptr(), rhs.as_ptr(), res.as_mut_ptr(), lhs.len()) }
    }

    #[inline(always)]
    fn scalar_add(scalar: Self, xs: &[Self], ys: &mut [Self]) {
        unsafe { super::vec_scalar_add_bf16(scalar, xs.as_ptr(), ys.as_mut_ptr(), xs.len()) }
    }

    #[inline(always)]
    unsafe fn vec_dot(lhs: *const Self, rhs: *const Self, res: *mut Self, len: usize) {
        let mut res_f32 = 0f32;
        super::vec_dot_bf16(lhs, rhs, &mut res_f32, len);
        *res = half::bf16::from_f32(res_f32);
    }

    #[inline(always)]
    unsafe fn vec_dot_f32(lhs: *const Self, rhs: *const Self, len: usize) -> f32 {
        super::vec_dot_bf16_to_f32_simd(lhs, rhs, len)
    }

    #[inline(always)]
    fn vec_fmadd_to_f32(acc: &mut [f32], v: &[Self], w: f32) {
        super::vec_fmadd_bf16_to_f32_simd(acc, v, w)
    }
}
macro_rules! impl_int_vec_ops {
    ($ty:ty) => {
        impl VecOps for $ty {
            #[inline(always)]
            fn min(self, other: Self) -> Self {
                <Self as Ord>::min(self, other)
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                <Self as Ord>::max(self, other)
            }

            #[inline(always)]
            unsafe fn vec_dot_f32(lhs: *const Self, rhs: *const Self, len: usize) -> f32 {
                let mut sum = 0.0f64;
                for i in 0..len {
                    sum += (*lhs.add(i) as f64) * (*rhs.add(i) as f64);
                }
                sum as f32
            }

            #[inline(always)]
            fn vec_fmadd_to_f32(acc: &mut [f32], v: &[Self], w: f32) {
                for (a, &x) in acc.iter_mut().zip(v.iter()) {
                    *a += (x as f32) * w;
                }
            }
        }
    };
}

impl_int_vec_ops!(u8);
impl_int_vec_ops!(u32);
impl_int_vec_ops!(i16);
impl_int_vec_ops!(i32);
impl_int_vec_ops!(i64);

fn f8e4m3_to_f32_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for i in 0..256 {
            t[i] = float8::F8E4M3::from_bits(i as u8).to_f32();
        }
        t
    })
}

impl VecOps for float8::F8E4M3 {
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }

    #[inline(always)]
    unsafe fn vec_dot_f32(lhs: *const Self, rhs: *const Self, len: usize) -> f32 {
        let lut = f8e4m3_to_f32_lut();
        let mut sum = 0.0f32;
        for i in 0..len {
            let a = lut[(*lhs.add(i)).to_bits() as usize];
            let b = lut[(*rhs.add(i)).to_bits() as usize];
            sum += a * b;
        }
        sum
    }

    #[inline(always)]
    fn vec_fmadd_to_f32(acc: &mut [f32], v: &[Self], w: f32) {
        let lut = f8e4m3_to_f32_lut();
        for (a, &x) in acc.iter_mut().zip(v.iter()) {
            *a += lut[x.to_bits() as usize] * w;
        }
    }
}

#[inline(always)]
pub fn par_for_each(n_threads: usize, func: impl Fn(usize) + Send + Sync) {
    if n_threads == 1 {
        func(0)
    } else {
        rayon::scope(|s| {
            for thread_idx in 0..n_threads {
                let func = &func;
                s.spawn(move |_| func(thread_idx));
            }
        })
    }
}

#[inline(always)]
pub fn par_range(lo: usize, up: usize, n_threads: usize, func: impl Fn(usize) + Send + Sync) {
    if n_threads == 1 {
        for i in lo..up {
            func(i)
        }
    } else {
        rayon::scope(|s| {
            for thread_idx in 0..n_threads {
                let func = &func;
                s.spawn(move |_| {
                    for i in (lo + thread_idx..up).step_by(n_threads) {
                        func(i)
                    }
                });
            }
        })
    }
}
