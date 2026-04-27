//! Probe whether a ROCm/HIP device is available and functional.
//!
//! This plugin does not link against ROCm at compile time. It reuses the local
//! `hiparc` crate, which opens the HIP runtime (`libamdhip64.so` on Linux,
//! `amdhip64.dll` on Windows) at probe time and calls `hipInit` +
//! `hipGetDeviceCount`.
//!
//! Keeping the probe independent from `candle-core/cuda` matters because
//! Candle's CUDA feature currently pulls CUDA kernel build tooling
//! (`bindgen_cuda`, `nvcc`, `nvidia-smi`). A ROCm-only machine should still be
//! able to build and ship this backend-detection plugin.

/// Probe whether a ROCm/HIP device is available.
///
/// Returns 0 if at least one HIP device is usable, non-zero otherwise.
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    #[cfg(any(
        target_os = "linux",
        all(target_os = "windows", target_arch = "x86_64")
    ))]
    {
        match hiparc::HipRuntime::probe_device() {
            Ok(count) if count > 0 => 0,
            _ => 1,
        }
    }

    #[cfg(not(any(
        target_os = "linux",
        all(target_os = "windows", target_arch = "x86_64")
    )))]
    {
        1
    }
}
