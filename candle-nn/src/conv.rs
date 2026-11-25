//! Convolution Layers.
use crate::BatchNorm;
use candle::{conv::CudnnFwdAlgo, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv1dConfig,
}

impl Conv1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv1dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv1d_with_algo(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
            self.config.cudnn_fwd_algo,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose1dConfig {
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for ConvTranspose1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvTranspose1dConfig,
}

impl ConvTranspose1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: ConvTranspose1dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &ConvTranspose1dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for ConvTranspose1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv_transpose1d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv2dConfig,
}

impl Conv2d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv2dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn absorb_bn(&self, bn: &BatchNorm) -> Result<Self> {
        if let Some((w_bn, b_bn)) = bn.weight_and_bias() {
            let std_ = w_bn.div(&((bn.running_var() + bn.eps())?.sqrt()?))?;
            let weight = self
                .weight()
                .broadcast_mul(&(std_.reshape((self.weight().dims4()?.0, 1, 1, 1))?))?;
            let bias = match &self.bias {
                None => b_bn.sub(&(std_.mul(bn.running_mean())?))?,
                Some(bias) => b_bn.add(&(std_.mul(&bias.sub(bn.running_mean())?)?))?,
            };
            Ok(Self {
                weight,
                bias: Some(bias),
                config: self.config,
            })
        } else {
            candle::bail!("batch norm does not have weight_and_bias")
        }
    }
}

impl crate::Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv2d_with_algo(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
            self.config.cudnn_fwd_algo,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2dConfig {
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
    // TODO: support groups.
}

impl Default for ConvTranspose2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConvTranspose2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvTranspose2dConfig,
}

impl ConvTranspose2d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: ConvTranspose2dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &ConvTranspose2dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv_transpose2d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
            self.config.stride,
            self.config.dilation,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv1d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv1d::new(ws, Some(bs), cfg))
}

pub fn conv1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv1d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
        init_ws,
    )?;
    Ok(Conv1d::new(ws, None, cfg))
}

pub fn conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose1dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose1d> {
    let bound = 1. / (out_channels as f64 * kernel_size as f64).sqrt();
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels / cfg.groups, kernel_size),
        "weight",
        init,
    )?;
    let bs = vb.get_with_hints(out_channels, "bias", init)?;
    Ok(ConvTranspose1d::new(ws, Some(bs), cfg))
}

pub fn conv_transpose1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose1dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose1d> {
    let bound = 1. / (out_channels as f64 * kernel_size as f64).sqrt();
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels / cfg.groups, kernel_size),
        "weight",
        init,
    )?;
    Ok(ConvTranspose1d::new(ws, None, cfg))
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init_ws,
    )?;
    Ok(Conv2d::new(ws, None, cfg))
}

pub fn conv_transpose2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose2dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose2d> {
    let bound = 1. / (out_channels as f64).sqrt() / kernel_size as f64;
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels, kernel_size, kernel_size),
        "weight",
        init,
    )?;
    let bs = vb.get_with_hints(out_channels, "bias", init)?;
    Ok(ConvTranspose2d::new(ws, Some(bs), cfg))
}

pub fn conv_transpose2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose2dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose2d> {
    let bound = 1. / (out_channels as f64).sqrt() / kernel_size as f64;
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels, kernel_size, kernel_size),
        "weight",
        init,
    )?;
    Ok(ConvTranspose2d::new(ws, None, cfg))
}

/// Folds a 5D Conv3d patch-embed kernel `[C_out, C_in, T_p, K_h, K_w]` along the temporal axis
/// `dim = 2` for static-image inputs where the single frame is duplicated `T_p` times:
///
/// `W_static[co, ci, kh, kw] = sum_{t=0}^{T_p - 1} W[co, ci, t, kh, kw]`
///
/// Accumulated in FP64 with a single rounding back to the original dtype, cutting static-image
/// vision stem FLOPs and input bandwidth by `T_p`x (typically 2x for `T_p = 2` in Qwen2.5-VL /
/// Qwen3-VL / Qwen3.5) with zero accuracy loss.
pub fn fold_conv3d_temporal_weight(weight_5d: &Tensor) -> Result<Tensor> {
    let (c_out, c_in, t_p, k_h, k_w) = weight_5d.dims5()?;
    let orig_dtype = weight_5d.dtype();
    let dev = weight_5d.device().clone();
    let data = weight_5d
        .to_device(&candle::Device::Cpu)?
        .to_dtype(candle::DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let spatial = k_h * k_w;
    let mut folded = vec![0.0f64; c_out * c_in * spatial];
    for co in 0..c_out {
        for ci in 0..c_in {
            let dst = (co * c_in + ci) * spatial;
            for t in 0..t_p {
                let src = ((co * c_in + ci) * t_p + t) * spatial;
                for s in 0..spatial {
                    folded[dst + s] += data[src + s];
                }
            }
        }
    }
    Tensor::from_vec(folded, (c_out, c_in, k_h, k_w), &candle::Device::Cpu)?
        .to_dtype(orig_dtype)?
        .to_device(&dev)
}

/// Folds a flattened patch-embed projection weight `[D_out, C_in * T_p * P_h * P_w]` into
/// `[D_out, C_in * P_h * P_w]` for static images (exact FP64 sum over the `T_p` axis).
pub fn fold_conv3d_patch_linear_weight(
    weight_2d: &Tensor,
    c_in: usize,
    t_p: usize,
    p_h: usize,
    p_w: usize,
) -> Result<Tensor> {
    let (d_out, in_dim) = weight_2d.dims2()?;
    let expected = c_in * t_p * p_h * p_w;
    if in_dim != expected {
        candle::bail!(
            "fold_conv3d_patch_linear_weight: in_dim {in_dim} != c_in({c_in})*t_p({t_p})*p_h({p_h})*p_w({p_w})"
        )
    }
    let w5 = weight_2d.reshape((d_out, c_in, t_p, p_h, p_w))?;
    fold_conv3d_temporal_weight(&w5)?.reshape((d_out, c_in * p_h * p_w))
}

#[cfg(test)]
mod conv3d_fold_tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn fold_conv3d_temporal_weight_matches_duplicated_frame_exact() -> Result<()> {
        let dev = Device::Cpu;
        let (co, ci, tp, kh, kw) = (8usize, 3usize, 2usize, 4usize, 4usize);
        let w5 = Tensor::randn(0f32, 0.5, (co, ci, tp, kh, kw), &dev)?;
        let w2d = fold_conv3d_temporal_weight(&w5)?;
        assert_eq!(w2d.dims(), &[co, ci, kh, kw]);

        // Single static patch [1, ci, kh, kw] duplicated along T_p = 2:
        let x_single = Tensor::randn(0f32, 1.0, (1, ci, kh, kw), &dev)?;
        let x_flat_single = x_single.reshape((1, ci * kh * kw))?;
        let x_dup = Tensor::cat(&[&x_single, &x_single], 1)? // wait: match [ci, tp, kh, kw] layout
            ;
        let _ = x_dup;
        // Exact layout check: reshape x_single as [1, ci, 1, kh, kw], repeat along dim 2 -> [1, ci, tp, kh, kw]
        let x_t2 = x_single
            .unsqueeze(2)?
            .repeat((1, 1, tp, 1, 1))?
            .reshape((1, ci * tp * kh * kw))?;
        let w_full_2d = w5.reshape((co, ci * tp * kh * kw))?;
        let y_full = x_t2.matmul(&w_full_2d.t()?)?;

        let w_folded_lin = fold_conv3d_patch_linear_weight(&w_full_2d, ci, tp, kh, kw)?;
        let y_folded = x_flat_single.matmul(&w_folded_lin.t()?)?;

        let diff = (&y_full - &y_folded)?
            .abs()?
            .max_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "Conv3d temporal fold mismatch: {diff}");
        Ok(())
    }
}
