//! 原生标签切面与叶子自动栅格路径：SIMD 边界、错误前缀和 scratch 重用。
#![allow(clippy::panic_in_result_fn)] // Result 传播被测错误，断言验证输出和恢复契约。
#![allow(clippy::indexing_slicing)] // 所有坐标由固定 0..64 范围生成。

use std::num::NonZeroU16;

use rectangle_decomposition::{
    DenseLabels64, PackedRectangles64, QuadLeaf64, Rectangle, SparseOptimalScratch64,
    SparseQuadError, SparseQuadImage64,
};

fn pixels(labels: &DenseLabels64) -> Vec<QuadLeaf64> {
    labels
        .rows()
        .iter()
        .enumerate()
        .flat_map(|(v, row)| {
            row.iter().enumerate().filter_map(move |(u, &value)| {
                NonZeroU16::new(value).map(|value| QuadLeaf64 {
                    u: u8::try_from(u).unwrap_or_default(),
                    v: u8::try_from(v).unwrap_or_default(),
                    lod: 0,
                    value,
                })
            })
        })
        .collect()
}

fn labels_from_leaves(leaves: &[QuadLeaf64]) -> DenseLabels64 {
    let mut labels = DenseLabels64::new();
    for leaf in leaves {
        let side = 1u8 << leaf.lod;
        for v in leaf.v..leaf.v + side {
            for u in leaf.u..leaf.u + side {
                let pixel = &mut labels.rows_mut()[usize::from(v)][usize::from(u)];
                assert_eq!(*pixel, 0);
                *pixel = leaf.value.get();
            }
        }
    }
    labels
}

fn assert_coverage(rectangles: &[Rectangle], expected: &DenseLabels64) {
    let mut actual = DenseLabels64::new();
    for rectangle in rectangles {
        assert!(rectangle.x.start < rectangle.x.end && rectangle.x.end <= 64);
        assert!(rectangle.y.start < rectangle.y.end && rectangle.y.end <= 64);
        assert_ne!(rectangle.value, 0);
        for v in rectangle.y.start..rectangle.y.end {
            for u in rectangle.x.start..rectangle.x.end {
                let pixel = &mut actual.rows_mut()[usize::from(v)][usize::from(u)];
                assert_eq!(*pixel, 0);
                *pixel = rectangle.value;
            }
        }
    }
    assert_eq!(&actual, expected);
}

fn checkerboard() -> DenseLabels64 {
    let mut labels = DenseLabels64::new();
    for (v, row) in labels.rows_mut().iter_mut().enumerate() {
        for (u, value) in row.iter_mut().enumerate() {
            *value = if (u + v) % 2 == 0 { 1 } else { u16::MAX };
        }
    }
    labels
}

fn threshold_leaves(count: usize, mixed: bool) -> Vec<QuadLeaf64> {
    let mut leaves = Vec::with_capacity(count);
    if mixed {
        leaves.push(QuadLeaf64 {
            u: 0,
            v: 0,
            lod: 1,
            value: NonZeroU16::MAX,
        });
    }
    for v in 0..64u8 {
        for u in 0..64u8 {
            if leaves.len() == count {
                return leaves;
            }
            if !mixed || u >= 2 || v >= 2 {
                leaves.push(QuadLeaf64 {
                    u,
                    v,
                    lod: 0,
                    value: if (u + v) % 3 == 0 {
                        NonZeroU16::MAX
                    } else {
                        NonZeroU16::MIN
                    },
                });
            }
        }
    }
    leaves
}

#[test]
fn lane_boundaries_and_extreme_labels_match_small_leaf_event_path() -> Result<(), SparseQuadError> {
    let mut scratch = SparseOptimalScratch64::new();
    let mut packed = PackedRectangles64::new();
    for pattern in 0..4 {
        let mut labels = DenseLabels64::new();
        for v in 0..64usize {
            for u in 0..64usize {
                let boundary = [0, 31, 32, 63];
                labels.rows_mut()[v][u] = match pattern {
                    0 if boundary.contains(&u) && boundary.contains(&v) => u16::MAX,
                    1 if boundary.contains(&u) || boundary.contains(&v) => {
                        if u < 32 {
                            1
                        } else {
                            u16::MAX
                        }
                    }
                    2 if v == 31 || v == 32 => [0, 1, 32768, u16::MAX][u % 4],
                    3 if u == 31 || u == 32 || u == 63 => [0, u16::MAX, 1, 32768][v % 4],
                    _ => 0,
                };
            }
        }
        let leaves = pixels(&labels);
        assert!(leaves.len() < 512, "reference must use the event path");
        let expected = scratch.decompose(&leaves)?;
        assert_coverage(&expected, &labels);
        assert_eq!(scratch.decompose_labels_borrowed(&labels)?, &expected);
        assert_eq!(
            scratch.decompose_labels_packed(&labels, &mut packed)?,
            expected.len()
        );
        assert!(packed.iter().eq(expected.iter().copied()));
    }
    Ok(())
}

#[test]
fn empty_full_and_max_capacity_labels_reuse_every_output_path() -> Result<(), SparseQuadError> {
    let mut scratch = SparseOptimalScratch64::new();
    let mut packed = PackedRectangles64::new();
    let mut labels = checkerboard();
    let expected = scratch.decompose_labels_borrowed(&labels)?.to_vec();
    assert_eq!(expected.len(), 4096);
    assert_coverage(&expected, &labels);
    assert_eq!(scratch.decompose_labels_packed(&labels, &mut packed)?, 4096);
    assert!(packed.iter().eq(expected.iter().copied()));
    for label in [0, u16::MAX, 1, 0] {
        for row in labels.rows_mut() {
            row.fill(label);
        }
        let mut seen = 0;
        let count = scratch.decompose_labels_into(&labels, |rectangle| {
            assert_eq!(rectangle.value, label);
            assert_eq!(rectangle.x.start, 0);
            assert_eq!(rectangle.x.end, 64);
            assert_eq!(rectangle.y.start, 0);
            assert_eq!(rectangle.y.end, 64);
            seen += 1;
            Ok(())
        })?;
        assert_eq!(count, usize::from(label != 0));
        assert_eq!(seen, count);
        assert_eq!(scratch.decompose_labels_borrowed(&labels)?.len(), count);
        assert_eq!(
            scratch.decompose_labels_packed(&labels, &mut packed)?,
            count
        );
        if label == 0 {
            assert!(packed.is_empty());
        } else {
            assert_eq!(packed.bounds(), &[u32::from_le_bytes([0, 64, 0, 64])]);
            assert_eq!(packed.labels(), &[label]);
        }
    }
    labels.clear();
    assert_eq!(scratch.decompose_labels_borrowed(&labels)?, []);
    Ok(())
}

#[test]
fn native_sink_failure_preserves_exact_prefix_and_recovery() -> Result<(), SparseQuadError> {
    let labels = checkerboard();
    let mut scratch = SparseOptimalScratch64::new();
    let expected = scratch.decompose_labels_borrowed(&labels)?.to_vec();
    for stop in [0, 31, 32, 63, 64, 4095] {
        let mut emitted = Vec::new();
        let result = scratch.decompose_labels_into(&labels, |rectangle| {
            if emitted.len() == stop {
                return Err(SparseQuadError::AllocationFailed);
            }
            emitted.push(rectangle);
            Ok(())
        });
        assert_eq!(result, Err(SparseQuadError::AllocationFailed));
        assert_eq!(emitted.as_slice(), &expected[..stop]);
        assert_eq!(scratch.decompose_labels_borrowed(&labels)?, &expected);
        let mut recovered = Vec::new();
        assert_eq!(
            scratch.decompose_labels_into(&labels, |rectangle| {
                recovered.push(rectangle);
                Ok(())
            })?,
            expected.len()
        );
        assert_eq!(recovered, expected);
    }
    Ok(())
}

#[test]
fn automatic_raster_threshold_preserves_unit_and_mixed_lod_output() -> Result<(), SparseQuadError> {
    let mut scratch = SparseOptimalScratch64::new();
    let mut packed = PackedRectangles64::new();
    for count in [511, 512, 513] {
        for mixed in [false, true] {
            let mut leaves = threshold_leaves(count, mixed);
            assert_eq!(leaves.len(), count);
            let labels = labels_from_leaves(&leaves);
            let expected = scratch.decompose_labels_borrowed(&labels)?.to_vec();
            assert_coverage(&expected, &labels);
            for reverse in [false, true] {
                if reverse {
                    leaves.reverse();
                }
                let image = SparseQuadImage64::from_leaves(&leaves)?;
                assert_eq!(scratch.decompose_borrowed(&leaves)?, &expected);
                assert_eq!(scratch.decompose_quads_borrowed(&image)?, &expected);
                assert_eq!(
                    scratch.decompose_packed(&leaves, &mut packed)?,
                    expected.len()
                );
                assert!(packed.iter().eq(expected.iter().copied()));
            }
        }
    }
    Ok(())
}

#[test]
fn invalid_raster_inputs_never_emit_and_do_not_poison_reused_scratch() -> Result<(), SparseQuadError>
{
    let mut scratch = SparseOptimalScratch64::new();
    let mut packed = PackedRectangles64::new();
    for count in [511, 512, 513] {
        for mixed in [false, true] {
            let leaves = threshold_leaves(count, mixed);
            let labels = labels_from_leaves(&leaves);
            let expected = scratch.decompose_labels_borrowed(&labels)?.to_vec();
            for error in [
                SparseQuadError::Overlap,
                SparseQuadError::OutOfBounds,
                SparseQuadError::Misaligned,
                SparseQuadError::LodOutOfRange,
            ] {
                let mut invalid = leaves.clone();
                invalid[count - 1] = match error {
                    SparseQuadError::Overlap => leaves[0],
                    SparseQuadError::OutOfBounds => QuadLeaf64 {
                        u: 64,
                        v: 0,
                        lod: 0,
                        value: NonZeroU16::MAX,
                    },
                    SparseQuadError::Misaligned => QuadLeaf64 {
                        u: 63,
                        v: 32,
                        lod: 1,
                        value: NonZeroU16::MAX,
                    },
                    _ => QuadLeaf64 {
                        u: 0,
                        v: 0,
                        lod: 7,
                        value: NonZeroU16::MAX,
                    },
                };
                assert_eq!(scratch.decompose_borrowed(&invalid), Err(error));
                let mut calls = 0;
                assert_eq!(
                    scratch.decompose_into(&invalid, |_| {
                        calls += 1;
                        Ok(())
                    }),
                    Err(error)
                );
                assert_eq!(calls, 0);
                scratch.decompose_labels_packed(&labels, &mut packed)?;
                assert_eq!(scratch.decompose_packed(&invalid, &mut packed), Err(error));
                assert!(packed.is_empty());
                assert_eq!(scratch.decompose_borrowed(&leaves)?, &expected);
                assert_eq!(scratch.decompose_labels_borrowed(&labels)?, &expected);
            }
        }
    }
    Ok(())
}
