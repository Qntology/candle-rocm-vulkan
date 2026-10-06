#![allow(unused)]
use super::{GgmlDType, QuantizedType};
use crate::{CpuStorage, Error, Layout, Result, Shape, VulkanDevice, VulkanStorage};

pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
}

impl QVulkanStorage {
    pub fn zeros(_: &VulkanDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn dequantize(&self, _: usize) -> Result<VulkanStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn dequantize_f16(&self, _: usize) -> Result<VulkanStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize(&mut self, _: &VulkanStorage) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize_imatrix(&mut self, _: &VulkanStorage, _: &[f32], _: usize) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize_onto(&mut self, _: &CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn quantize_imatrix_onto(&mut self, _: &CpuStorage, _: &[f32], _: usize) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn embedding(
        &self,
        _: usize,
        _: usize,
        _: &VulkanStorage,
        _: &Layout,
    ) -> Result<VulkanStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn fwd(&self, _: &Shape, _: &VulkanStorage, _: &Layout) -> Result<(VulkanStorage, Shape)> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    pub fn indexed_moe_forward(
        &self,
        _: &Shape,
        _: &VulkanStorage,
        _: &Layout,
        _: &VulkanStorage,
        _: &Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}

pub fn fwd_cpu_weights(
    _: &dyn QuantizedType,
    _: &Shape,
    _: &VulkanStorage,
    _: &Layout,
) -> Result<(VulkanStorage, Shape)> {
    Err(Error::NotCompiledWithVulkanSupport)
}

pub fn load_quantized(_: &VulkanDevice, _: GgmlDType, _: &[u8]) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithVulkanSupport)
}
