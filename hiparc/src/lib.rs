use std::ffi::{c_char, c_void};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use libloading::Library;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    UnsupportedPlatform,
    LibraryNotFound { candidates: Vec<PathBuf> },
    SymbolLoad {
        library: PathBuf,
        symbol: &'static str,
        error: String,
    },
    Api {
        api: &'static str,
        code: i32,
    },
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(f, "HIP is not supported on this platform"),
            Self::LibraryNotFound { candidates } => {
                write!(f, "could not load HIP library from any of {candidates:?}")
            }
            Self::SymbolLoad {
                library,
                symbol,
                error,
            } => write!(
                f,
                "failed to load symbol {symbol} from {}: {error}",
                library.display()
            ),
            Self::Api { api, code } => write!(f, "{api} failed with HIP error code {code}"),
        }
    }
}

impl std::error::Error for Error {}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

type HipInitFn = unsafe extern "C" fn(u32) -> i32;
type HipGetDeviceCountFn = unsafe extern "C" fn(*mut i32) -> i32;
type HipSetDeviceFn = unsafe extern "C" fn(i32) -> i32;
type HipDeviceSynchronizeFn = unsafe extern "C" fn() -> i32;
type HipMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type HipFreeFn = unsafe extern "C" fn(*mut c_void) -> i32;
type HipMemcpyFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, HipMemcpyKind) -> i32;
type HipMemsetFn = unsafe extern "C" fn(*mut c_void, i32, usize) -> i32;
type HipStreamCreateFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type HipStreamDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
type HipStreamSynchronizeFn = unsafe extern "C" fn(*mut c_void) -> i32;
type HipModuleLoadDataFn = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
type HipModuleGetFunctionFn =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32;
type HipModuleLaunchKernelFn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

type HipblasCreateFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type HipblasDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
type HiprandCreateGeneratorFn = unsafe extern "C" fn(*mut *mut c_void, i32) -> i32;
type HiprandDestroyGeneratorFn = unsafe extern "C" fn(*mut c_void) -> i32;

pub struct HipRuntime {
    _lib: Library,
    _path: PathBuf,
    hip_init: HipInitFn,
    hip_get_device_count: HipGetDeviceCountFn,
    hip_set_device: HipSetDeviceFn,
    hip_device_synchronize: HipDeviceSynchronizeFn,
    hip_malloc: HipMallocFn,
    hip_free: HipFreeFn,
    hip_memcpy: HipMemcpyFn,
    hip_memset: HipMemsetFn,
    hip_stream_create: HipStreamCreateFn,
    hip_stream_destroy: HipStreamDestroyFn,
    hip_stream_synchronize: HipStreamSynchronizeFn,
    hip_module_load_data: HipModuleLoadDataFn,
    hip_module_get_function: HipModuleGetFunctionFn,
    hip_module_launch_kernel: HipModuleLaunchKernelFn,
}

impl HipRuntime {
    pub fn load() -> Result<Self> {
        let candidates = runtime_library_candidates();
        for candidate in &candidates {
            let lib = match unsafe { Library::new(candidate) } {
                Ok(lib) => lib,
                Err(_) => continue,
            };
            return Self::from_library(lib, candidate.clone());
        }
        Err(Error::LibraryNotFound { candidates })
    }

    pub fn probe_device() -> Result<i32> {
        let runtime = Self::load()?;
        runtime.init()?;
        runtime.device_count()
    }

    pub fn init(&self) -> Result<()> {
        map_status("hipInit", unsafe { (self.hip_init)(0) })
    }

    pub fn device_count(&self) -> Result<i32> {
        let mut count = 0;
        map_status(
            "hipGetDeviceCount",
            unsafe { (self.hip_get_device_count)(&mut count) },
        )?;
        Ok(count)
    }

    pub fn set_device(&self, ordinal: i32) -> Result<()> {
        map_status("hipSetDevice", unsafe { (self.hip_set_device)(ordinal) })
    }

    pub fn synchronize(&self) -> Result<()> {
        map_status(
            "hipDeviceSynchronize",
            unsafe { (self.hip_device_synchronize)() },
        )
    }

    pub fn malloc(&self, size: usize) -> Result<*mut c_void> {
        let mut ptr = std::ptr::null_mut();
        map_status("hipMalloc", unsafe { (self.hip_malloc)(&mut ptr, size) })?;
        Ok(ptr)
    }

    pub fn free(&self, ptr: *mut c_void) -> Result<()> {
        map_status("hipFree", unsafe { (self.hip_free)(ptr) })
    }

    pub fn memcpy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: HipMemcpyKind,
    ) -> Result<()> {
        map_status(
            "hipMemcpy",
            unsafe { (self.hip_memcpy)(dst, src, size, kind) },
        )
    }

    pub fn memset(&self, dst: *mut c_void, value: i32, size: usize) -> Result<()> {
        map_status("hipMemset", unsafe { (self.hip_memset)(dst, value, size) })
    }

    pub fn create_stream(&self) -> Result<*mut c_void> {
        let mut stream = std::ptr::null_mut();
        map_status(
            "hipStreamCreate",
            unsafe { (self.hip_stream_create)(&mut stream) },
        )?;
        Ok(stream)
    }

    pub fn destroy_stream(&self, stream: *mut c_void) -> Result<()> {
        map_status(
            "hipStreamDestroy",
            unsafe { (self.hip_stream_destroy)(stream) },
        )
    }

    pub fn synchronize_stream(&self, stream: *mut c_void) -> Result<()> {
        map_status(
            "hipStreamSynchronize",
            unsafe { (self.hip_stream_synchronize)(stream) },
        )
    }

    pub fn module_load_data(&self, data: *const c_void) -> Result<*mut c_void> {
        let mut module = std::ptr::null_mut();
        map_status(
            "hipModuleLoadData",
            unsafe { (self.hip_module_load_data)(&mut module, data) },
        )?;
        Ok(module)
    }

    pub fn module_get_function(&self, module: *mut c_void, name: *const c_char) -> Result<*mut c_void> {
        let mut function = std::ptr::null_mut();
        map_status(
            "hipModuleGetFunction",
            unsafe { (self.hip_module_get_function)(&mut function, module, name) },
        )?;
        Ok(function)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn module_launch_kernel(
        &self,
        function: *mut c_void,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> Result<()> {
        map_status(
            "hipModuleLaunchKernel",
            unsafe {
                (self.hip_module_launch_kernel)(
                    function,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    block_z,
                    shared_mem_bytes,
                    stream,
                    kernel_params,
                    extra,
                )
            },
        )
    }

    fn from_library(lib: Library, path: PathBuf) -> Result<Self> {
        Ok(Self {
            hip_init: load_symbol(&lib, &path, b"hipInit\0", "hipInit")?,
            hip_get_device_count: load_symbol(
                &lib,
                &path,
                b"hipGetDeviceCount\0",
                "hipGetDeviceCount",
            )?,
            hip_set_device: load_symbol(&lib, &path, b"hipSetDevice\0", "hipSetDevice")?,
            hip_device_synchronize: load_symbol(
                &lib,
                &path,
                b"hipDeviceSynchronize\0",
                "hipDeviceSynchronize",
            )?,
            hip_malloc: load_symbol(&lib, &path, b"hipMalloc\0", "hipMalloc")?,
            hip_free: load_symbol(&lib, &path, b"hipFree\0", "hipFree")?,
            hip_memcpy: load_symbol(&lib, &path, b"hipMemcpy\0", "hipMemcpy")?,
            hip_memset: load_symbol(&lib, &path, b"hipMemset\0", "hipMemset")?,
            hip_stream_create: load_symbol(
                &lib,
                &path,
                b"hipStreamCreate\0",
                "hipStreamCreate",
            )?,
            hip_stream_destroy: load_symbol(
                &lib,
                &path,
                b"hipStreamDestroy\0",
                "hipStreamDestroy",
            )?,
            hip_stream_synchronize: load_symbol(
                &lib,
                &path,
                b"hipStreamSynchronize\0",
                "hipStreamSynchronize",
            )?,
            hip_module_load_data: load_symbol(
                &lib,
                &path,
                b"hipModuleLoadData\0",
                "hipModuleLoadData",
            )?,
            hip_module_get_function: load_symbol(
                &lib,
                &path,
                b"hipModuleGetFunction\0",
                "hipModuleGetFunction",
            )?,
            hip_module_launch_kernel: load_symbol(
                &lib,
                &path,
                b"hipModuleLaunchKernel\0",
                "hipModuleLaunchKernel",
            )?,
            _lib: lib,
            _path: path,
        })
    }
}

pub struct HipBlas {
    _lib: Library,
    _path: PathBuf,
    hipblas_create: HipblasCreateFn,
    hipblas_destroy: HipblasDestroyFn,
}

impl HipBlas {
    pub fn load() -> Result<Self> {
        let candidates = secondary_library_candidates(&["libhipblas.so.2", "libhipblas.so"]);
        for candidate in &candidates {
            let lib = match unsafe { Library::new(candidate) } {
                Ok(lib) => lib,
                Err(_) => continue,
            };
            return Ok(Self {
                hipblas_create: load_symbol(&lib, candidate, b"hipblasCreate\0", "hipblasCreate")?,
                hipblas_destroy: load_symbol(
                    &lib,
                    candidate,
                    b"hipblasDestroy\0",
                    "hipblasDestroy",
                )?,
                _lib: lib,
                _path: candidate.clone(),
            });
        }
        Err(Error::LibraryNotFound { candidates })
    }

    pub fn create_handle(&self) -> Result<*mut c_void> {
        let mut handle = std::ptr::null_mut();
        map_status(
            "hipblasCreate",
            unsafe { (self.hipblas_create)(&mut handle) },
        )?;
        Ok(handle)
    }

    pub fn destroy_handle(&self, handle: *mut c_void) -> Result<()> {
        map_status("hipblasDestroy", unsafe { (self.hipblas_destroy)(handle) })
    }
}

pub struct HipRand {
    _lib: Library,
    _path: PathBuf,
    hiprand_create_generator: HiprandCreateGeneratorFn,
    hiprand_destroy_generator: HiprandDestroyGeneratorFn,
}

impl HipRand {
    pub fn load() -> Result<Self> {
        let candidates = secondary_library_candidates(&["libhiprand.so.1", "libhiprand.so"]);
        for candidate in &candidates {
            let lib = match unsafe { Library::new(candidate) } {
                Ok(lib) => lib,
                Err(_) => continue,
            };
            return Ok(Self {
                hiprand_create_generator: load_symbol(
                    &lib,
                    candidate,
                    b"hiprandCreateGenerator\0",
                    "hiprandCreateGenerator",
                )?,
                hiprand_destroy_generator: load_symbol(
                    &lib,
                    candidate,
                    b"hiprandDestroyGenerator\0",
                    "hiprandDestroyGenerator",
                )?,
                _lib: lib,
                _path: candidate.clone(),
            });
        }
        Err(Error::LibraryNotFound { candidates })
    }

    pub fn create_generator(&self, rng_type: i32) -> Result<*mut c_void> {
        let mut generator = std::ptr::null_mut();
        map_status(
            "hiprandCreateGenerator",
            unsafe { (self.hiprand_create_generator)(&mut generator, rng_type) },
        )?;
        Ok(generator)
    }

    pub fn destroy_generator(&self, generator: *mut c_void) -> Result<()> {
        map_status(
            "hiprandDestroyGenerator",
            unsafe { (self.hiprand_destroy_generator)(generator) },
        )
    }
}

fn map_status(api: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Api { api, code })
    }
}

fn load_symbol<T: Copy>(
    lib: &Library,
    library_path: &Path,
    symbol_bytes: &[u8],
    symbol_name: &'static str,
) -> Result<T> {
    let symbol = unsafe { lib.get::<T>(symbol_bytes) }.map_err(|error| Error::SymbolLoad {
        library: library_path.to_path_buf(),
        symbol: symbol_name,
        error: error.to_string(),
    })?;
    Ok(*symbol)
}

fn runtime_library_candidates() -> Vec<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        let mut candidates = secondary_library_candidates(&[
            "libamdhip64.so.7",
            "libamdhip64.so.6",
            "libamdhip64.so.5",
            "libamdhip64.so",
        ]);
        candidates.dedup();
        return candidates;
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        let mut candidates = Vec::new();
        for root_var in ["ROCM_PATH", "HIP_PATH"] {
            if let Ok(root) = std::env::var(root_var) {
                candidates.push(Path::new(&root).join("bin").join("amdhip64.dll"));
            }
        }
        candidates.push("amdhip64.dll".into());
        return candidates;
    }

    Vec::new()
}

fn secondary_library_candidates(_names: &[&str]) -> Vec<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        let mut candidates = Vec::new();
        for root_var in ["ROCM_PATH", "HIP_PATH"] {
            if let Ok(root) = std::env::var(root_var) {
                let lib_dir = Path::new(&root).join("lib");
                for name in _names {
                    candidates.push(lib_dir.join(name));
                }
            }
        }
        for name in _names {
            candidates.push(PathBuf::from(name));
        }
        return candidates;
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        let mut candidates = Vec::new();
        for root_var in ["ROCM_PATH", "HIP_PATH"] {
            if let Ok(root) = std::env::var(root_var) {
                let bin_dir = Path::new(&root).join("bin");
                for name in _names {
                    let dll = if let Some(stem) = name.strip_prefix("lib").and_then(|s| s.strip_suffix(".so")) {
                        format!("{stem}.dll")
                    } else {
                        (*name).to_string()
                    };
                    candidates.push(bin_dir.join(dll));
                }
            }
        }
        for name in _names {
            let dll = if let Some(stem) = name.strip_prefix("lib").and_then(|s| s.strip_suffix(".so")) {
                format!("{stem}.dll")
            } else {
                (*name).to_string()
            };
            candidates.push(PathBuf::from(dll));
        }
        return candidates;
    }

    Vec::new()
}
