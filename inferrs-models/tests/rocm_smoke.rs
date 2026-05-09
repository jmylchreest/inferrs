#![cfg(feature = "rocm")]

use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Module, Result, Tensor,
};

fn rocm_device_or_skip() -> Option<Device> {
    match Device::new_rocm(0) {
        Ok(device) => Some(device),
        Err(err) => {
            eprintln!("skipping ROCm test: {err}");
            None
        }
    }
}

fn assert_close(lhs: &Tensor, rhs: &Tensor, tol: f32) -> Result<()> {
    let max_diff = (lhs - rhs)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_diff <= tol,
        "max diff {max_diff} exceeds tolerance {tol}\nlhs: {lhs}\nrhs: {rhs}"
    );
    Ok(())
}

fn round_tensor2(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    Ok(t.to_vec2::<f32>()?
        .into_iter()
        .map(|row| row.into_iter().map(|v| (v * b).round() / b).collect())
        .collect())
}

#[test]
fn rocm_tensor_roundtrip_and_matmul_smoke() -> Result<()> {
    let Some(rocm) = rocm_device_or_skip() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let lhs_cpu = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], (2, 3), &cpu)?;
    let rhs_cpu = Tensor::from_slice(&[7f32, 8., 9., 10., 11., 12.], (3, 2), &cpu)?;

    let lhs_rocm = lhs_cpu.to_device(&rocm)?;
    let rhs_rocm = rhs_cpu.to_device(&rocm)?;
    assert!(lhs_rocm.device().is_rocm());
    assert!(rhs_rocm.device().is_rocm());

    let roundtrip = lhs_rocm.to_device(&cpu)?;
    assert_eq!(roundtrip.to_vec2::<f32>()?, lhs_cpu.to_vec2::<f32>()?);

    let got = lhs_rocm.matmul(&rhs_rocm)?.to_device(&cpu)?;
    assert_eq!(
        got.to_vec2::<f32>()?,
        vec![vec![58.0, 64.0], vec![139.0, 154.0]]
    );

    Ok(())
}

#[test]
fn rocm_bf16_qmatmul_stays_on_rocm() -> Result<()> {
    let Some(rocm) = rocm_device_or_skip() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let weight_data = (0..(4 * 32))
        .map(|i| (i as f32 - 48.0) / 16.0)
        .collect::<Vec<_>>();
    let act_data = (0..(2 * 32))
        .map(|i| (i as f32 - 16.0) / 32.0)
        .collect::<Vec<_>>();

    let weight_cpu = Tensor::from_vec(weight_data, (4, 32), &cpu)?;
    let act_cpu = Tensor::from_vec(act_data, (2, 32), &cpu)?;

    let qtensor = QTensor::quantize_onto(&weight_cpu, GgmlDType::BF16, &rocm)?;
    assert!(qtensor.device().is_rocm());

    let matmul = QMatMul::from_qtensor(qtensor)?;
    let act_rocm = act_cpu.to_device(&rocm)?;
    let got = matmul.forward(&act_rocm)?;
    assert!(got.device().is_rocm());

    let weight_bf16 = weight_cpu.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let expected = act_cpu.matmul(&weight_bf16.t()?)?;
    let got = got.to_device(&cpu)?;
    assert_close(&got, &expected, 1e-4)?;

    Ok(())
}

#[test]
fn rocm_quantized_qmatmul_matches_dequantized_reference() -> Result<()> {
    let Some(rocm) = rocm_device_or_skip() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let weight_data = (0..(4 * 32))
        .map(|i| ((i % 19) as f32 - 9.0) / 8.0)
        .collect::<Vec<_>>();
    let act_data = (0..(3 * 32))
        .map(|i| ((i % 23) as f32 - 11.0) / 7.0)
        .collect::<Vec<_>>();

    let weight_cpu = Tensor::from_vec(weight_data, (4, 32), &cpu)?;
    let act_cpu = Tensor::from_vec(act_data, (3, 32), &cpu)?;

    let qtensor = QTensor::quantize_onto(&weight_cpu, GgmlDType::Q8_0, &rocm)?;
    assert!(qtensor.device().is_rocm());

    let expected_weight = qtensor.dequantize(&cpu)?;
    let expected = act_cpu.matmul(&expected_weight.t()?)?;

    let matmul = QMatMul::from_qtensor(qtensor)?;
    let act_rocm = act_cpu.to_device(&rocm)?;
    let got = matmul.forward(&act_rocm)?;
    assert!(got.device().is_rocm());

    let got = got.to_device(&cpu)?;
    assert_close(&got, &expected, 1e-4)?;

    Ok(())
}

#[test]
fn rocm_tensor_ops_match_cpu_expectations() -> Result<()> {
    let Some(device) = rocm_device_or_skip() else {
        return Ok(());
    };

    let data = &[[-3f32, 1., 4., -0.1, 0.5], [2.7, -1.8, -0.28, 1.8, 2.8]];
    let tensor = Tensor::new(data, &device)?;
    assert_eq!(
        round_tensor2(&tensor.gelu()?, 4)?,
        vec![
            vec![-0.0036, 0.8412, 3.9999, -0.046, 0.3457],
            vec![2.6911, -0.0647, -0.1091, 1.7353, 2.7933]
        ]
    );
    let t_f16 = tensor.to_dtype(DType::F16)?.gelu()?.to_dtype(DType::F32)?;
    let max_diff = (tensor.gelu()? - t_f16)?.flatten_all()?.max(0)?;
    assert!(max_diff.to_vec0::<f32>()? < 5e-3);

    let data1 = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor1 = Tensor::new(data1, &device)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, &device)?;
    let binary = (&tensor1 + (&tensor1 * &tensor1)? / (&tensor1 + &tensor2))?;
    assert_eq!(
        binary.to_vec2::<f32>()?,
        vec![
            vec![4.125, 1.1666666, 5.7777777, 1.1666666, 7.5],
            vec![3.0, 1.5, 10.5, 12.0, 3.0]
        ]
    );

    let ids = Tensor::new(&[[0u8, 1, 0, 1, 0], [1, 1, 1, 0, 0]], &device)?;
    let a = Tensor::new(&[[0f32, 1., 2., 3., 4.], [5., 6., 7., 8., 9.]], &device)?;
    let b = Tensor::new(
        &[[10f32, 11., 12., 13., 14.], [15., 16., 17., 18., 19.]],
        &device,
    )?;
    let where_out = ids.where_cond(&a, &b)?;
    assert_eq!(
        where_out.flatten_all()?.to_vec1::<f32>()?,
        vec![10., 1., 12., 3., 14., 5., 6., 7., 18., 19.]
    );

    let t = Tensor::arange(0f32, 24f32, &device)?.reshape((4, 2, 3))?;
    let offs = Tensor::new(&[100f32, 200f32], &device)?;
    let broadcast = t.broadcast_add(&offs.reshape((2, 1))?)?;
    assert_eq!(
        broadcast.to_vec3::<f32>()?,
        vec![
            vec![vec![100.0, 101.0, 102.0], vec![203.0, 204.0, 205.0]],
            vec![vec![106.0, 107.0, 108.0], vec![209.0, 210.0, 211.0]],
            vec![vec![112.0, 113.0, 114.0], vec![215.0, 216.0, 217.0]],
            vec![vec![118.0, 119.0, 120.0], vec![221.0, 222.0, 223.0]],
        ]
    );

    let ints = Tensor::new(&[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]], &device)?;
    assert_eq!(ints.sum_keepdim((0, 2, 1))?.to_vec3::<u32>()?, vec![vec![vec![51]]]);
    assert_eq!(ints.min_keepdim(2)?.to_vec3::<u32>()?, vec![vec![vec![1], vec![1]], vec![vec![1], vec![2]]]);
    assert_eq!(ints.max_keepdim(2)?.to_vec3::<u32>()?, vec![vec![vec![4], vec![9]], vec![vec![7], vec![8]]]);

    Ok(())
}

#[test]
fn rocm_indexing_and_scatter_ops_match_cpu_expectations() -> Result<()> {
    let Some(device) = rocm_device_or_skip() else {
        return Ok(());
    };

    let ids = Tensor::new(&[0u32, 2u32, 1u32], &device)?;
    let t = Tensor::arange(0f32, 12f32, &device)?.reshape((4, 3))?;
    let hs = t.index_select(&ids, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        vec![
            vec![0.0, 2.0, 1.0],
            vec![3.0, 5.0, 4.0],
            vec![6.0, 8.0, 7.0],
            vec![9.0, 11.0, 10.0]
        ]
    );

    let init = Tensor::ones((4, 2), DType::F32, &device)?;
    let hs = init.index_add(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        vec![
            vec![1.0, 4.0],
            vec![4.0, 10.0],
            vec![7.0, 16.0],
            vec![10.0, 22.0]
        ]
    );

    let gather_ids = Tensor::new(&[[0u32, 0u32], [2u32, 0u32], [1u32, 1u32], [0u32, 2u32]], &device)?;
    let hs = t.gather(&gather_ids, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        vec![
            vec![0.0, 0.0],
            vec![5.0, 3.0],
            vec![7.0, 7.0],
            vec![9.0, 11.0]
        ]
    );

    let scatter_ids = Tensor::new(&[[0u32, 1, 2], [3, 4, 0], [3, 3, 1], [2, 0, 4]], &device)?;
    let init = Tensor::ones((4, 5), DType::F32, &device)?;
    let hs = init.scatter_add(&scatter_ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        vec![
            vec![1.0, 2.0, 3.0, 1.0, 1.0],
            vec![6.0, 1.0, 1.0, 4.0, 5.0],
            vec![1.0, 9.0, 1.0, 14.0, 1.0],
            vec![11.0, 1.0, 10.0, 1.0, 12.0]
        ]
    );

    let init = Tensor::ones((6, 3), DType::F32, &device)?;
    let hs = init.scatter(&scatter_ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        vec![
            vec![0.0, 10.0, 5.0],
            vec![1.0, 1.0, 8.0],
            vec![9.0, 1.0, 2.0],
            vec![6.0, 7.0, 1.0],
            vec![1.0, 4.0, 11.0],
            vec![1.0, 1.0, 1.0]
        ]
    );

    let init = Tensor::ones((6, 3), DType::F32, &device)?;
    init.scatter_set(&scatter_ids, &t, 0)?;
    assert_eq!(
        init.to_vec2::<f32>()?,
        vec![
            vec![0.0, 10.0, 5.0],
            vec![1.0, 1.0, 8.0],
            vec![9.0, 1.0, 2.0],
            vec![6.0, 7.0, 1.0],
            vec![1.0, 4.0, 11.0],
            vec![1.0, 1.0, 1.0]
        ]
    );

    Ok(())
}
