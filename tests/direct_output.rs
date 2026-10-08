//! 直接输出契约：顺序、容量、失败前缀与 scratch 重用。
#![allow(clippy::panic_in_result_fn)] // Result 传播被测错误，断言验证输出契约。

use std::num::NonZeroU16;

use rectangle_decomposition::{
    PackedRectangles64, QuadLeaf64, SparseOptimalScratch64, SparseQuadError,
};

fn checkerboard() -> Vec<QuadLeaf64> {
    (0..64u8)
        .flat_map(|v| {
            (0..64u8).map(move |u| QuadLeaf64 {
                u,
                v,
                lod: 0,
                value: if (u + v) % 2 == 0 {
                    NonZeroU16::MIN
                } else {
                    NonZeroU16::MAX
                },
            })
        })
        .collect()
}

#[test]
fn callback_failure_retains_prefix_and_scratch_is_reusable() -> Result<(), SparseQuadError> {
    let leaves = checkerboard();
    let mut scratch = SparseOptimalScratch64::new();
    let expected = scratch.decompose_borrowed(&leaves)?.to_vec();
    for stop in [0, 63, 64, 4095] {
        let mut emitted = Vec::new();
        let result = scratch.decompose_into(&leaves, |rectangle| {
            if emitted.len() == stop {
                return Err(SparseQuadError::AllocationFailed);
            }
            emitted.push(rectangle);
            Ok(())
        });
        assert_eq!(result, Err(SparseQuadError::AllocationFailed));
        assert!(emitted.iter().eq(expected.iter().take(stop)));
        let mut recovered = Vec::new();
        assert_eq!(
            scratch.decompose_into(&leaves, |rectangle| {
                recovered.push(rectangle);
                Ok(())
            })?,
            4096
        );
        assert_eq!(recovered, expected);
        assert_eq!(scratch.decompose_borrowed(&leaves)?, &expected);
    }
    Ok(())
}

#[test]
fn invalid_input_never_calls_sink_and_clears_packed_output() -> Result<(), SparseQuadError> {
    let mut scratch = SparseOptimalScratch64::new();
    let mut output = PackedRectangles64::new();
    let mut full = [QuadLeaf64 {
        u: 0,
        v: 0,
        lod: 6,
        value: NonZeroU16::MAX,
    }];
    assert_eq!(scratch.decompose_packed(&full, &mut output)?, 1);
    assert_eq!(output.bounds(), &[u32::from_le_bytes([0, 64, 0, 64])]);
    assert_eq!(output.labels(), &[u16::MAX]);
    full.first_mut().ok_or(SparseQuadError::CapacityOverflow)?.u = 1;
    let mut calls = 0;
    assert_eq!(
        scratch.decompose_into(&full, |_| {
            calls += 1;
            Ok(())
        }),
        Err(SparseQuadError::Misaligned)
    );
    assert_eq!(calls, 0);
    assert_eq!(
        scratch.decompose_packed(&full, &mut output),
        Err(SparseQuadError::Misaligned)
    );
    assert!(output.is_empty());
    Ok(())
}

#[test]
fn packed_output_preserves_max_capacity_order_and_reuses_storage() -> Result<(), SparseQuadError> {
    let leaves = checkerboard();
    let mut scratch = SparseOptimalScratch64::new();
    let expected = scratch.decompose_borrowed(&leaves)?.to_vec();
    let mut output = PackedRectangles64::new();
    let bounds_address = output.bounds().as_ptr();
    let labels_address = output.labels().as_ptr();
    assert_eq!(scratch.decompose_packed(&leaves, &mut output)?, 4096);
    assert!(output.iter().eq(expected.iter().copied()));
    assert_eq!(output.bounds().len(), 4096);
    assert_eq!(output.labels().len(), 4096);
    assert_eq!(output.bounds().as_ptr(), bounds_address);
    assert_eq!(output.labels().as_ptr(), labels_address);
    output.clear();
    assert!(output.bounds().is_empty() && output.labels().is_empty());
    assert_eq!(scratch.decompose_packed(&[], &mut output)?, 0);
    assert_eq!(scratch.decompose_packed(&leaves, &mut output)?, 4096);
    assert!(output.iter().eq(expected.iter().copied()));
    Ok(())
}
