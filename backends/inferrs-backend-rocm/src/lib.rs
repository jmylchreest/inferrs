//! Probe whether a ROCm/HIP device is available.
//!
//! This plugin loads the HIP runtime at probe time instead of linking Candle's
//! CUDA path, so it can build on ROCm-only hosts without `nvcc` or `nvidia-smi`.
///
/// Return 0 when a HIP device is available, non-zero otherwise.
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
