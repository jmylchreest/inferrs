use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuDevice;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

#[derive(Debug, Clone)]
pub struct RocmDevice {
    ordinal: usize,
}

#[derive(Debug)]
pub(crate) struct RawRocmBuffer {
    ptr: *mut c_void,
    len_bytes: usize,
    ordinal: usize,
}

unsafe impl Send for RawRocmBuffer {}
unsafe impl Sync for RawRocmBuffer {}

impl Drop for RawRocmBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let Ok(runtime) = hiparc::HipRuntime::load() else {
            return;
        };
        let _ = runtime.init();
        let _ = runtime.set_device(self.ordinal as i32);
        let _ = runtime.free(self.ptr);
    }
}

#[derive(Debug, Clone)]
pub struct RocmStorage {
    buffer: Arc<RawRocmBuffer>,
    shadow: CpuStorage,
    device: RocmDevice,
}

fn map_hip_err(err: hiparc::Error) -> crate::Error {
    crate::Error::wrap(err)
}

unsafe fn cast_slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        slice.as_ptr() as *const u8,
        std::mem::size_of_val(slice),
    )
}

fn cpu_storage_as_bytes(storage: &CpuStorage) -> &[u8] {
    match storage {
        CpuStorage::U8(v) => v.as_slice(),
        CpuStorage::U32(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::I16(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::I32(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::I64(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::BF16(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::F16(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::F32(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::F64(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::F8E4M3(v) => unsafe { cast_slice_as_bytes(v.as_slice()) },
        CpuStorage::F6E2M3(v) => v.as_slice(),
        CpuStorage::F6E3M2(v) => v.as_slice(),
        CpuStorage::F4(v) => v.as_slice(),
        CpuStorage::F8E8M0(v) => v.as_slice(),
    }
}

fn upload_shadow(device: &RocmDevice, shadow: CpuStorage) -> Result<RocmStorage> {
    let runtime = hiparc::HipRuntime::load().map_err(map_hip_err)?;
    runtime.init().map_err(map_hip_err)?;
    runtime
        .set_device(device.ordinal as i32)
        .map_err(map_hip_err)?;
    let bytes = cpu_storage_as_bytes(&shadow);
    let ptr = if bytes.is_empty() {
        ptr::null_mut()
    } else {
        let ptr = runtime.malloc(bytes.len()).map_err(map_hip_err)?;
        runtime
            .memcpy(
                ptr,
                bytes.as_ptr() as *const c_void,
                bytes.len(),
                hiparc::HipMemcpyKind::HostToDevice,
            )
            .map_err(map_hip_err)?;
        ptr
    };
    Ok(RocmStorage {
        buffer: Arc::new(RawRocmBuffer {
            ptr,
            len_bytes: bytes.len(),
            ordinal: device.ordinal,
        }),
        shadow,
        device: device.clone(),
    })
}

impl RocmDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        Self::new(ordinal)
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(self.ordinal)
    }
}

impl RocmStorage {
    pub fn transfer_to_device(&self, dst: &RocmDevice) -> Result<Self> {
        dst.storage_from_cpu_storage(&self.shadow)
    }

    pub fn device_ptr(&self) -> *const u8 {
        self.buffer.ptr as *const u8
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.len_bytes
    }

    pub(crate) fn shadow(&self) -> &CpuStorage {
        &self.shadow
    }

    fn unary_cpu<F>(&self, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage) -> Result<CpuStorage>,
    {
        let shadow = f(&self.shadow)?;
        self.device.storage_from_cpu_storage_owned(shadow)
    }

    fn binary_cpu<F>(&self, rhs: &Self, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &CpuStorage) -> Result<CpuStorage>,
    {
        let shadow = f(&self.shadow, &rhs.shadow)?;
        self.device.storage_from_cpu_storage_owned(shadow)
    }

    fn mutate_cpu<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(&mut CpuStorage) -> Result<()>,
    {
        let mut shadow = self.shadow.clone();
        f(&mut shadow)?;
        *self = self.device.storage_from_cpu_storage_owned(shadow)?;
        Ok(())
    }
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        self.device.storage_from_cpu_storage(&self.shadow)
    }

    fn dtype(&self) -> DType {
        self.shadow.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.shadow.clone())
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.affine(layout, mul, add))
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.powf(layout, alpha))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.elu(layout, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.reduce_op(op, layout, s))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        self.binary_cpu(rhs, |lhs, rhs| lhs.cmp(op, rhs, lhs_layout, rhs_layout))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.to_dtype(layout, dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.unary_impl::<B>(layout))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        self.binary_cpu(rhs, |lhs, rhs| lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_layout: &Layout,
        f: &Self,
        f_layout: &Layout,
    ) -> Result<Self> {
        let shadow = self
            .shadow
            .where_cond(layout, &t.shadow, t_layout, &f.shadow, f_layout)?;
        self.device.storage_from_cpu_storage_owned(shadow)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.binary_cpu(kernel, |inp, k| inp.conv1d(l, k, kernel_l, params))
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.binary_cpu(kernel, |inp, k| inp.conv_transpose1d(l, k, kernel_l, params))
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.binary_cpu(kernel, |inp, k| inp.conv2d(l, k, kernel_l, params))
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.binary_cpu(kernel, |inp, k| inp.conv_transpose2d(l, k, kernel_l, params))
    }

    fn avg_pool2d(&self, layout: &Layout, kernel_size: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.avg_pool2d(layout, kernel_size, stride))
    }

    fn max_pool2d(&self, layout: &Layout, kernel_size: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.max_pool2d(layout, kernel_size, stride))
    }

    fn upsample_nearest1d(&self, layout: &Layout, target_size: usize) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.upsample_nearest1d(layout, target_size))
    }

    fn upsample_nearest2d(&self, layout: &Layout, target_h: usize, target_w: usize) -> Result<Self> {
        self.unary_cpu(|shadow| shadow.upsample_nearest2d(layout, target_h, target_w))
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scales_h: Option<f64>,
        scales_w: Option<f64>,
    ) -> Result<Self> {
        self.unary_cpu(|shadow| {
            shadow.upsample_bilinear2d(
                layout,
                target_h,
                target_w,
                align_corners,
                scales_h,
                scales_w,
            )
        })
    }

    fn gather(&self, layout: &Layout, indexes: &Self, indexes_layout: &Layout, dim: usize) -> Result<Self> {
        self.binary_cpu(indexes, |shadow, idx| shadow.gather(layout, idx, indexes_layout, dim))
    }

    fn scatter_set(
        &mut self,
        l1: &Layout,
        indexes: &Self,
        l2: &Layout,
        src: &Self,
        l3: &Layout,
        dim: usize,
    ) -> Result<()> {
        let indexes_shadow = indexes.shadow.clone();
        let src_shadow = src.shadow.clone();
        self.mutate_cpu(|shadow| shadow.scatter_set(l1, &indexes_shadow, l2, &src_shadow, l3, dim))
    }

    fn scatter_add_set(
        &mut self,
        l1: &Layout,
        indexes: &Self,
        l2: &Layout,
        src: &Self,
        l3: &Layout,
        dim: usize,
    ) -> Result<()> {
        let indexes_shadow = indexes.shadow.clone();
        let src_shadow = src.shadow.clone();
        self.mutate_cpu(|shadow| shadow.scatter_add_set(l1, &indexes_shadow, l2, &src_shadow, l3, dim))
    }

    fn index_select(&self, indexes: &Self, source_layout: &Layout, indexes_layout: &Layout, dim: usize) -> Result<Self> {
        self.binary_cpu(indexes, |shadow, idx| shadow.index_select(idx, source_layout, indexes_layout, dim))
    }

    fn index_add(
        &self,
        l1: &Layout,
        indexes: &Self,
        l2: &Layout,
        source: &Self,
        l3: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let shadow = self
            .shadow
            .index_add(l1, &indexes.shadow, l2, &source.shadow, l3, dim)?;
        self.device.storage_from_cpu_storage_owned(shadow)
    }

    fn matmul(&self, rhs: &Self, bmnk: (usize, usize, usize, usize), lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if self.shadow.dtype() == DType::BF16 {
            let lhs = self.shadow.to_dtype(lhs_l, DType::F32)?;
            let rhs = rhs.shadow.to_dtype(rhs_l, DType::F32)?;
            let lhs_l = Layout::contiguous(lhs_l.shape().clone());
            let rhs_l = Layout::contiguous(rhs_l.shape().clone());
            let out = lhs.matmul(&rhs, bmnk, &lhs_l, &rhs_l)?;

            let (_, m, n, _) = bmnk;
            let mut dims = lhs_l.dims().to_vec();
            let rank = dims.len();
            dims[rank - 2] = m;
            dims[rank - 1] = n;
            let out_l = Layout::contiguous(crate::Shape::from(dims));
            let out = out.to_dtype(&out_l, DType::BF16)?;
            self.device.storage_from_cpu_storage_owned(out)
        } else {
            self.binary_cpu(rhs, |lhs, rhs| lhs.matmul(rhs, bmnk, lhs_l, rhs_l))
        }
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let mut dst_shadow = dst.shadow.clone();
        self.shadow.copy_strided_src(&mut dst_shadow, dst_offset, src_l)?;
        let device = dst.device.clone();
        *dst = device.storage_from_cpu_storage_owned(dst_shadow)?;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let mut dst_shadow = dst.shadow.clone();
        self.shadow.copy2d(
            &mut dst_shadow,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )?;
        let device = dst.device.clone();
        *dst = device.storage_from_cpu_storage_owned(dst_shadow)?;
        Ok(())
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        self.mutate_cpu(|shadow| shadow.const_set(scalar, layout))
    }
}

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let runtime = hiparc::HipRuntime::load().map_err(map_hip_err)?;
        runtime.init().map_err(map_hip_err)?;
        runtime.set_device(ordinal as i32).map_err(map_hip_err)?;
        let count = runtime.device_count().map_err(map_hip_err)? as usize;
        if ordinal >= count {
            crate::bail!("ROCm device ordinal {ordinal} out of range, available devices: {count}")
        }
        Ok(Self { ordinal })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.ordinal == rhs.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(CpuDevice.zeros_impl(shape, dtype)?)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.zeros_impl(shape, dtype)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&T::to_cpu_storage(s))
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(s.clone())
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        upload_shadow(self, s)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(CpuDevice.rand_uniform(shape, dtype, min, max)?)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(CpuDevice.rand_normal(shape, dtype, mean, std)?)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        crate::bail!("cannot seed the ROCm fallback backend with set_seed")
    }

    fn get_current_seed(&self) -> Result<u64> {
        crate::bail!("cannot get the ROCm fallback backend seed")
    }

    fn synchronize(&self) -> Result<()> {
        let runtime = hiparc::HipRuntime::load().map_err(map_hip_err)?;
        runtime.init().map_err(map_hip_err)?;
        runtime
            .set_device(self.ordinal as i32)
            .map_err(map_hip_err)?;
        runtime.synchronize().map_err(map_hip_err)
    }
}
