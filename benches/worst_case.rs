use std::num::NonZeroU16;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rectangle_decomposition::{
    QuadLeaf64, SparseLayerBuilder64, SparseOptimalScratch64, SparseQuadImage64,
};

const EDGE: u8 = 64;
const EDGE_USIZE: usize = 64;
const PIXELS: usize = EDGE_USIZE * EDGE_USIZE;
const HOLES_7_8: usize = PIXELS / 8;

fn low_discrepancy_holes_7_8_leaves() -> Vec<QuadLeaf64> {
    let value = NonZeroU16::MIN;

    let holes = low_discrepancy_holes();
    let mut leaves = Vec::with_capacity(PIXELS - HOLES_7_8);
    for v in 0..EDGE {
        for u in 0..EDGE {
            let index = usize::from(v) * EDGE_USIZE + usize::from(u);
            if holes.get(index).copied().is_some_and(|hole| !hole) {
                leaves.push(QuadLeaf64 {
                    u,
                    v,
                    lod: 0,
                    value,
                });
            }
        }
    }
    leaves
}

fn low_discrepancy_holes() -> [bool; PIXELS] {
    let mut holes = [false; PIXELS];
    let mut hole_count = 0usize;
    let mut sample = 0u32;

    while hole_count < HOLES_7_8 {
        let u = radical_inverse_base2_64(sample);
        let v = radical_inverse_base3_64(sample);
        let index = usize::from(v) * EDGE_USIZE + usize::from(u);
        if let Some(hole) = holes.get_mut(index)
            && !*hole
        {
            *hole = true;
            hole_count += 1;
        }
        sample += 1;
    }

    holes
}

fn radical_inverse_base2_64(mut value: u32) -> u8 {
    let mut reversed = 0u8;
    for _ in 0..6 {
        reversed = (reversed << 1) | u8::from((value & 1) != 0);
        value >>= 1;
    }
    reversed
}

fn radical_inverse_base3_64(mut value: u32) -> u8 {
    let mut numerator = 0u32;
    let mut denominator = 1u32;
    while denominator < 729 {
        numerator = numerator * 3 + value % 3;
        denominator *= 3;
        value /= 3;
    }
    let coord = (numerator * u32::from(EDGE)) / denominator;
    u8::try_from(coord).unwrap_or(0)
}

fn bench_low_discrepancy_holes_7_8(c: &mut Criterion) {
    let leaves = low_discrepancy_holes_7_8_leaves();
    let mut scratch = SparseOptimalScratch64::new();

    c.bench_function("low_discrepancy_holes_7_8_64", |b| {
        b.iter(|| {
            let rect_count =
                require_ok(scratch.decompose_borrowed(black_box(leaves.as_slice()))).len();
            black_box(rect_count);
        });
    });
}

fn bench_prepare_image(c: &mut Criterion) {
    let leaves = low_discrepancy_holes_7_8_leaves();
    c.bench_function("prepare_image_64", |b| {
        b.iter(|| {
            black_box(require_ok(SparseQuadImage64::from_leaves(black_box(
                &leaves,
            ))))
        });
    });
}

fn require_ok<T, E: std::fmt::Debug>(result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("benchmark input or decomposition failed: {error:?}");
            std::process::exit(1);
        }
    }
}

fn unit_leaves(mut pixel_value: impl FnMut(u8, u8) -> Option<NonZeroU16>) -> Vec<QuadLeaf64> {
    let mut leaves = Vec::with_capacity(PIXELS);
    for v in 0..EDGE {
        for u in 0..EDGE {
            if let Some(value) = pixel_value(u, v) {
                leaves.push(QuadLeaf64 {
                    u,
                    v,
                    lod: 0,
                    value,
                });
            }
        }
    }
    leaves
}

fn regular_blocks() -> Vec<QuadLeaf64> {
    let mut leaves = Vec::with_capacity(64);
    for v in (0..EDGE).step_by(8) {
        for u in (0..EDGE).step_by(8) {
            let value = require_ok(NonZeroU16::try_from(1 + u16::from((u / 8 + v / 8) % 2)));
            leaves.push(QuadLeaf64 {
                u,
                v,
                lod: 3,
                value,
            });
        }
    }
    leaves
}

fn random_leaves(dense: bool) -> Vec<QuadLeaf64> {
    let mut state = 0x6a09_e667_f3bc_c909u64;
    unit_leaves(|_, _| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state & 7 != 0) == dense).then_some(NonZeroU16::MIN)
    })
}

fn many_small_color_groups() -> Vec<QuadLeaf64> {
    let mut leaves = Vec::with_capacity(32 * 21 * 4);
    for row in 0..32u8 {
        for column in 0..21u8 {
            let value = require_ok(NonZeroU16::try_from(
                u16::from(row) * 21 + u16::from(column) + 1,
            ));
            // Each independent four-pixel S has one horizontal chord.
            for (u, v) in [(1, 0), (2, 0), (0, 1), (1, 1)] {
                leaves.push(QuadLeaf64 {
                    u: column * 3 + u,
                    v: row * 2 + v,
                    lod: 0,
                    value,
                });
            }
        }
    }
    leaves
}

fn dense_boundary_chords() -> Vec<QuadLeaf64> {
    unit_leaves(|u, v| {
        let on_vertical_border = u == 0 || u == EDGE - 1;
        let on_horizontal_border = v == 0 || v == EDGE - 1;
        let occupied = match (on_vertical_border, on_horizontal_border) {
            (true, true) => false,
            (true, false) => v % 2 == 1,
            (false, true) => u % 2 == 1,
            (false, false) => true,
        };
        occupied.then_some(NonZeroU16::MIN)
    })
}

fn mixed_lod_bands() -> Vec<QuadLeaf64> {
    [
        (0, 0, 5, 1u16),
        (32, 0, 4, 1),
        (48, 0, 3, 1),
        (56, 0, 3, 1),
        (48, 8, 3, 1),
        (56, 8, 3, 1),
        (32, 16, 4, 1),
        (48, 16, 4, 1),
        (0, 32, 5, 1),
        (32, 32, 4, 2),
        (48, 32, 4, 2),
        (32, 48, 4, 3),
        (48, 48, 4, 3),
    ]
    .into_iter()
    .map(|(u, v, lod, value)| QuadLeaf64 {
        u,
        v,
        lod,
        value: require_ok(NonZeroU16::try_from(value)),
    })
    .collect()
}

fn representative_inputs() -> [(&'static str, Vec<QuadLeaf64>, Option<usize>); 11] {
    let second = require_ok(NonZeroU16::try_from(2));
    let mut reversed = unit_leaves(|_, _| Some(NonZeroU16::MIN));
    reversed.reverse();
    [
        (
            "large_lod",
            vec![QuadLeaf64 {
                u: 0,
                v: 0,
                lod: 6,
                value: NonZeroU16::MIN,
            }],
            Some(1),
        ),
        ("regular_blocks", regular_blocks(), Some(64)),
        ("mixed_lod_bands", mixed_lod_bands(), Some(4)),
        (
            "stripes",
            unit_leaves(|u, _| {
                Some(if (u / 2) % 2 == 0 {
                    NonZeroU16::MIN
                } else {
                    second
                })
            }),
            Some(32),
        ),
        (
            "two_color_checkerboard",
            unit_leaves(|u, v| {
                Some(if (u + v) % 2 == 0 {
                    NonZeroU16::MIN
                } else {
                    second
                })
            }),
            Some(PIXELS),
        ),
        (
            "alternating_holes_max_chords",
            unit_leaves(|u, v| (u % 2 != 0 || v % 2 != 0).then_some(NonZeroU16::MIN)),
            Some(1025),
        ),
        ("sparse_random", random_leaves(false), None),
        ("dense_random_holes", random_leaves(true), None),
        ("dense_boundary_chords", dense_boundary_chords(), Some(124)),
        (
            "many_small_color_groups",
            many_small_color_groups(),
            Some(32 * 21 * 2),
        ),
        ("reversed_unit_pixels", reversed, Some(1)),
    ]
}

fn bench_representative_inputs(c: &mut Criterion) {
    let mut group = c.benchmark_group("representative_64");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(1));

    for (name, leaves, expected_rectangles) in representative_inputs() {
        let image = require_ok(SparseQuadImage64::from_leaves(&leaves));
        let mut scratch = SparseOptimalScratch64::new();
        let mut builder = SparseLayerBuilder64::new();
        // Validate both paths and known simple optima before starting either timer.
        let leaf_count = require_ok(scratch.decompose_borrowed(&leaves)).len();
        let image_count = require_ok(scratch.decompose_quads_borrowed(&image)).len();
        assert_eq!(leaf_count, image_count, "entry points disagree for {name}");
        for &leaf in &leaves {
            require_ok(builder.push_square(leaf.u, leaf.v, leaf.lod, leaf.value));
        }
        assert_eq!(require_ok(builder.finish(&mut scratch)).len(), leaf_count);
        if let Some(expected) = expected_rectangles {
            assert_eq!(
                leaf_count, expected,
                "unexpected rectangle count for {name}"
            );
        }

        group.bench_function(BenchmarkId::new("borrowed_leaves", name), |b| {
            b.iter(|| {
                black_box(require_ok(scratch.decompose_borrowed(black_box(&leaves))).len());
            });
        });
        group.bench_function(BenchmarkId::new("prepared_image", name), |b| {
            b.iter(|| {
                black_box(require_ok(scratch.decompose_quads_borrowed(black_box(&image))).len());
            });
        });
        group.bench_function(BenchmarkId::new("prepare_image", name), |b| {
            b.iter(|| {
                black_box(require_ok(SparseQuadImage64::from_leaves(black_box(
                    &leaves,
                ))))
            });
        });
        group.bench_function(BenchmarkId::new("builder_layer", name), |b| {
            b.iter(|| {
                builder.clear();
                for &leaf in black_box(&leaves) {
                    require_ok(builder.push_square(leaf.u, leaf.v, leaf.lod, leaf.value));
                }
                let rectangles = require_ok(builder.finish(&mut scratch));
                black_box(rectangles.len());
            });
        });
    }
    group.finish();
}

fn bench_scratch_init(c: &mut Criterion) {
    c.bench_function("scratch_init_64", |b| {
        b.iter(|| {
            let mut scratch = SparseOptimalScratch64::new();
            black_box(&mut scratch);
        });
    });
}

criterion_group!(
    benches,
    bench_low_discrepancy_holes_7_8,
    bench_prepare_image,
    bench_scratch_init,
    bench_representative_inputs,
);
criterion_main!(benches);
