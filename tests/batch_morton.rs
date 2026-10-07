use std::num::NonZeroU16;

use rectangle_decomposition::{QuadLeaf64, SparseQuadError, SparseQuadImage64};

fn pixels(count: usize) -> Vec<QuadLeaf64> {
    (0u8..64)
        .flat_map(|v| {
            (0u8..64).map(move |u| QuadLeaf64 {
                u,
                v,
                lod: 0,
                value: NonZeroU16::MIN,
            })
        })
        .take(count)
        .collect()
}

fn leaf_mut(leaves: &mut [QuadLeaf64], index: usize) -> &mut QuadLeaf64 {
    leaves
        .get_mut(index)
        .unwrap_or_else(|| std::process::abort())
}

#[test]
fn constructor_handles_batch_threshold_and_tail_lengths() {
    for count in [
        0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 4095, 4096,
    ] {
        let leaves = pixels(count);
        for input in [leaves.clone(), leaves.into_iter().rev().collect()] {
            assert_eq!(
                SparseQuadImage64::from_leaves(&input).map(|image| image.len()),
                Ok(count)
            );
        }
    }
}

#[test]
fn first_shape_error_is_preserved_across_batch_boundaries() {
    for count in [63, 64, 65, 95, 96, 97, 127, 128, 129] {
        for first in [0, count / 2, count - 1] {
            let mut leaves = pixels(count);
            leaf_mut(&mut leaves, first).u = u8::MAX;
            if first + 1 < count {
                leaf_mut(&mut leaves, first + 1).lod = u8::MAX;
            }
            assert_eq!(
                SparseQuadImage64::from_leaves(&leaves).err(),
                Some(SparseQuadError::OutOfBounds)
            );

            leaf_mut(&mut leaves, first).lod = 7;
            assert_eq!(
                SparseQuadImage64::from_leaves(&leaves).err(),
                Some(SparseQuadError::LodOutOfRange)
            );
        }
    }
    for first in [0, 31, 32, 63, 64, 95, 96, 127] {
        let mut leaves = pixels(129);
        leaf_mut(&mut leaves, first).u = 1;
        leaf_mut(&mut leaves, first).lod = 1;
        leaf_mut(&mut leaves, first + 1).lod = 7;
        assert_eq!(
            SparseQuadImage64::from_leaves(&leaves).err(),
            Some(SparseQuadError::Misaligned)
        );
    }
}

#[test]
fn shape_errors_still_precede_overlap_errors() {
    let mut leaves = pixels(129);
    let first = *leaves.first().unwrap_or_else(|| std::process::abort());
    *leaf_mut(&mut leaves, 1) = first;
    leaf_mut(&mut leaves, 128).u = 64;
    assert_eq!(
        SparseQuadImage64::from_leaves(&leaves).err(),
        Some(SparseQuadError::OutOfBounds)
    );
    leaf_mut(&mut leaves, 128).u = 0;
    assert_eq!(
        SparseQuadImage64::from_leaves(&leaves).err(),
        Some(SparseQuadError::Overlap)
    );
}
