//! 压缩 quad 与独立单位像素展开对照，覆盖随机 LOD、标签和输入顺序。

use std::num::NonZeroU16;

use rectangle_decomposition::{
    DenseLabels64, PackedRectangles64, QuadLeaf64, Rectangle, SparseOptimalScratch64,
    SparseQuadError,
};
#[cfg(feature = "alloc")]
use rectangle_decomposition::{SparseLayerBuilder64, SparseQuadImage64};

const fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn append_tile(
    leaves: &mut Vec<QuadLeaf64>,
    state: &mut u64,
    u: u8,
    v: u8,
    lod: u8,
    split_rate: u64,
) {
    let sample = next_random(state);
    if lod > 0 && sample % 8 < split_rate {
        let child_lod = lod - 1;
        let half = 1u8 << child_lod;
        for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            append_tile(leaves, state, u + dx, v + dy, child_lod, split_rate);
        }
    } else if let Some(value) = NonZeroU16::new(u16::try_from(sample % 4).unwrap_or_default()) {
        leaves.push(QuadLeaf64 { u, v, lod, value });
    }
}

fn rasterize(leaves: &[QuadLeaf64]) -> [u16; 4096] {
    let mut pixels = [0; 4096];
    for leaf in leaves {
        let size = 1u8 << leaf.lod;
        for y in leaf.v..leaf.v + size {
            for x in leaf.u..leaf.u + size {
                let index = usize::from(y) * 64 + usize::from(x);
                if let Some(pixel) = pixels.get_mut(index) {
                    assert_eq!(*pixel, 0);
                    *pixel = leaf.value.get();
                }
            }
        }
    }
    pixels
}

fn assert_coverage(rectangles: &[Rectangle], expected: &[u16; 4096]) {
    let mut actual = [0; 4096];
    for rectangle in rectangles {
        assert!(rectangle.x.start < rectangle.x.end && rectangle.x.end <= 64);
        assert!(rectangle.y.start < rectangle.y.end && rectangle.y.end <= 64);
        for y in rectangle.y.start..rectangle.y.end {
            for x in rectangle.x.start..rectangle.x.end {
                let index = usize::from(y) * 64 + usize::from(x);
                if let Some(pixel) = actual.get_mut(index) {
                    assert_eq!(*pixel, 0);
                    *pixel = rectangle.value;
                }
            }
        }
    }
    assert_eq!(&actual, expected);
}

#[test]
#[allow(clippy::panic_in_result_fn)] // 断言检查不同表示的精确输出和覆盖。
fn compressed_events_match_unit_pixels_for_shuffled_quadtree_layers() -> Result<(), SparseQuadError>
{
    let mut state = 0x6a09_e667_f3bc_c909;
    let mut scratch = SparseOptimalScratch64::new();
    #[cfg(feature = "alloc")]
    let mut builder = SparseLayerBuilder64::new();
    let mut packed = PackedRectangles64::new();
    for case in 0..128 {
        let mut leaves = Vec::new();
        for (u, v) in [(0, 0), (32, 0), (0, 32), (32, 32)] {
            append_tile(&mut leaves, &mut state, u, v, 5, 3 + case % 5);
        }
        let expected_pixels = rasterize(&leaves);
        let pixels: Vec<_> = expected_pixels
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| {
                NonZeroU16::new(value).map(|value| QuadLeaf64 {
                    u: u8::try_from(index % 64).unwrap_or_default(),
                    v: u8::try_from(index / 64).unwrap_or_default(),
                    lod: 0,
                    value,
                })
            })
            .collect();
        let expected_rectangles = scratch.decompose_borrowed(&pixels)?.to_vec();
        assert_coverage(&expected_rectangles, &expected_pixels);
        let mut dense = DenseLabels64::new();
        for (row, values) in dense
            .rows_mut()
            .iter_mut()
            .zip(expected_pixels.as_chunks::<64>().0)
        {
            *row = *values;
        }
        assert_eq!(
            scratch.decompose_labels_borrowed(&dense)?,
            &expected_rectangles
        );
        assert_eq!(
            scratch.decompose_labels_packed(&dense, &mut packed)?,
            expected_rectangles.len()
        );
        assert!(packed.iter().eq(expected_rectangles.iter().copied()));
        for reverse in [false, true] {
            if reverse {
                leaves.reverse();
            }
            if !leaves.is_empty() {
                let shift =
                    usize::try_from(next_random(&mut state)).unwrap_or_default() % leaves.len();
                leaves.rotate_left(shift);
            }
            assert_eq!(scratch.decompose_borrowed(&leaves)?, &expected_rectangles);
            let mut direct = Vec::new();
            assert_eq!(
                scratch.decompose_into(&leaves, |rectangle| {
                    direct.push(rectangle);
                    Ok(())
                })?,
                expected_rectangles.len()
            );
            assert_eq!(direct, expected_rectangles);
            assert_eq!(
                scratch.decompose_packed(&leaves, &mut packed)?,
                expected_rectangles.len()
            );
            assert!(packed.iter().eq(expected_rectangles.iter().copied()));
            #[cfg(feature = "alloc")]
            {
                let image = SparseQuadImage64::from_leaves(&leaves)?;
                assert_eq!(
                    scratch.decompose_quads_borrowed(&image)?,
                    &expected_rectangles
                );
                assert_eq!(scratch.decompose(&leaves)?, expected_rectangles);
                builder.clear();
                for leaf in &leaves {
                    builder.push_square(leaf.u, leaf.v, leaf.lod, leaf.value)?;
                }
                assert_eq!(builder.finish(&mut scratch)?, expected_rectangles);
            }
        }
    }
    Ok(())
}
