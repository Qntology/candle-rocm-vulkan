#![allow(unused)]
use super::GgmlDType;
use crate::{CpuStorage, Error, Layout, Result, RocmDevice, RocmStorage, Shape};

pub struct QRocmStorage {
    dtype: GgmlDType,
    device: RocmDevice,
}

impl QRocmStorage {
    pub fn zeros(_: &RocmDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn dequantize(&self, _: usize) -> Result<RocmStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn dequantize_f16(&self, _: usize) -> Result<RocmStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize(&mut self, _: &RocmStorage) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_imatrix(&mut self, _: &RocmStorage, _: &[f32], _: usize) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_onto(&mut self, _: &CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_imatrix_onto(&mut self, _: &CpuStorage, _: &[f32], _: usize) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn embedding(&self, _: usize, _: usize, _: &RocmStorage, _: &Layout) -> Result<RocmStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn fwd(&self, _: &Shape, _: &RocmStorage, _: &Layout) -> Result<(RocmStorage, Shape)> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn indexed_moe_forward(
        &self,
        _: &Shape,
        _: &RocmStorage,
        _: &Layout,
        _: &RocmStorage,
        _: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        Err(Error::NotCompiledWithRocmSupport)
    }
}

pub fn load_quantized(_: &RocmDevice, _: GgmlDType, _: &[u8]) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithRocmSupport)
}
