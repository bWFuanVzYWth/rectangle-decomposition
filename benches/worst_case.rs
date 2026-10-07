use std::num::NonZeroU16;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rectangle_decomposition::{
    DenseLabels64, QuadLeaf64, Rectangle, SparseLayerBuilder64, SparseOptimalScratch64,
    SparseQuadImage64,
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
            let rectangles = require_ok(scratch.decompose_borrowed(black_box(leaves.as_slice())));
            black_box((rectangles.len(), consume_rectangles(rectangles)));
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

fn rectangle_checksum(rectangle: Rectangle) -> u64 {
    let bounds = u32::from_le_bytes([
        rectangle.x.start,
        rectangle.x.end,
        rectangle.y.start,
        rectangle.y.end,
    ]);
    u64::from(bounds) | (u64::from(rectangle.value) << 32)
}

fn consume_rectangles(rectangles: &[Rectangle]) -> u64 {
    rectangles.iter().copied().fold(0, |sum, rectangle| {
        sum.wrapping_add(rectangle_checksum(rectangle))
    })
}

// 准备成本在计时外：真实上游可直接写这个连续标签布局。
fn dense_labels(leaves: &[QuadLeaf64]) -> DenseLabels64 {
    let mut labels = DenseLabels64::new();
    for leaf in leaves {
        let side = 1usize << leaf.lod;
        for row in labels
            .rows_mut()
            .iter_mut()
            .skip(usize::from(leaf.v))
            .take(side)
        {
            for pixel in row.iter_mut().skip(usize::from(leaf.u)).take(side) {
                assert_eq!(*pixel, 0, "benchmark leaves overlap");
                *pixel = leaf.value.get();
            }
        }
    }
    labels
}

fn validate_output_paths(
    name: &str,
    leaves: &[QuadLeaf64],
    image: &SparseQuadImage64,
    labels: &DenseLabels64,
    expected_count: Option<usize>,
    scratch: &mut SparseOptimalScratch64,
) -> usize {
    let expected = require_ok(scratch.decompose_borrowed(leaves)).to_vec();
    assert_eq!(
        require_ok(scratch.decompose_quads_borrowed(image)),
        &expected,
        "prepared image disagrees for {name}"
    );
    assert_eq!(
        require_ok(scratch.decompose_labels_borrowed(labels)),
        &expected,
        "native labels disagree for {name}"
    );
    let mut sink_index = 0;
    assert_eq!(
        require_ok(scratch.decompose_into(leaves, |rectangle| {
            assert_eq!(expected.get(sink_index), Some(&rectangle));
            sink_index += 1;
            Ok(())
        })),
        expected.len()
    );
    sink_index = 0;
    assert_eq!(
        require_ok(scratch.decompose_labels_into(labels, |rectangle| {
            assert_eq!(expected.get(sink_index), Some(&rectangle));
            sink_index += 1;
            Ok(())
        })),
        expected.len()
    );
    if let Some(count) = expected_count {
        assert_eq!(
            expected.len(),
            count,
            "unexpected rectangle count for {name}"
        );
    }
    expected.len()
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
        let labels = dense_labels(&leaves);
        let mut scratch = SparseOptimalScratch64::new();
        let mut builder = SparseLayerBuilder64::new();
        // 输出顺序与简单已知最优值在全部计时之前检查。
        let leaf_count = validate_output_paths(
            name,
            &leaves,
            &image,
            &labels,
            expected_rectangles,
            &mut scratch,
        );
        for &leaf in &leaves {
            require_ok(builder.push_square(leaf.u, leaf.v, leaf.lod, leaf.value));
        }
        assert_eq!(require_ok(builder.finish(&mut scratch)).len(), leaf_count);

        group.bench_function(BenchmarkId::new("borrowed_leaves", name), |b| {
            b.iter(|| {
                let rectangles = require_ok(scratch.decompose_borrowed(black_box(&leaves)));
                black_box((rectangles.len(), consume_rectangles(rectangles)));
            });
        });
        group.bench_function(BenchmarkId::new("prepared_image", name), |b| {
            b.iter(|| {
                let rectangles = require_ok(scratch.decompose_quads_borrowed(black_box(&image)));
                black_box((rectangles.len(), consume_rectangles(rectangles)));
            });
        });
        group.bench_function(BenchmarkId::new("native_labels", name), |b| {
            b.iter(|| {
                let rectangles = require_ok(scratch.decompose_labels_borrowed(black_box(&labels)));
                black_box((rectangles.len(), consume_rectangles(rectangles)));
            });
        });
        group.bench_function(BenchmarkId::new("direct_sink", name), |b| {
            b.iter(|| {
                let mut checksum = 0u64;
                let count = require_ok(scratch.decompose_into(black_box(&leaves), |rectangle| {
                    checksum = checksum.wrapping_add(rectangle_checksum(rectangle));
                    Ok(())
                }));
                black_box((count, checksum));
            });
        });
        group.bench_function(BenchmarkId::new("native_labels_sink", name), |b| {
            b.iter(|| {
                let mut checksum = 0u64;
                let count = require_ok(scratch.decompose_labels_into(
                    black_box(&labels),
                    |rectangle| {
                        checksum = checksum.wrapping_add(rectangle_checksum(rectangle));
                        Ok(())
                    },
                ));
                black_box((count, checksum));
            });
        });
        group.bench_function(BenchmarkId::new("prepare_image", name), |b| {
            b.iter(|| {
                black_box(require_ok(SparseQuadImage64::from_leaves(black_box(
                    &leaves,
                ))))
            });
        });
        group.bench_function(BenchmarkId::new("construct_and_decompose", name), |b| {
            b.iter(|| {
                let fresh_image = require_ok(SparseQuadImage64::from_leaves(black_box(&leaves)));
                let rectangles =
                    require_ok(scratch.decompose_quads_borrowed(black_box(&fresh_image)));
                black_box((rectangles.len(), consume_rectangles(rectangles)));
            });
        });
        group.bench_function(BenchmarkId::new("builder_layer", name), |b| {
            b.iter(|| {
                builder.clear();
                for &leaf in black_box(&leaves) {
                    require_ok(builder.push_square(leaf.u, leaf.v, leaf.lod, leaf.value));
                }
                let rectangles = require_ok(builder.finish(&mut scratch));
                black_box((rectangles.len(), consume_rectangles(&rectangles)));
            });
        });
    }
    group.finish();
}

fn for_chunk_planes<T>(interior: &[T; 2], boundary: &[T; 2], mut visit: impl FnMut(&T)) {
    let [first, last] = boundary;
    for _ in 0..3 {
        visit(first);
        for plane in interior.iter().cycle().take(63) {
            visit(plane);
        }
        visit(last);
    }
}

// 真实 64³ rock/air 棋盘格：每轴 63 个内层面与两个外边界面。
// 内层正负法线对应不同 label；两侧边界覆盖的棋盘相位相反。
fn voxel_checker_planes() -> ([Vec<QuadLeaf64>; 2], [Vec<QuadLeaf64>; 2]) {
    let second = require_ok(NonZeroU16::try_from(2));
    let interior = [false, true].map(|reverse| {
        unit_leaves(|u, v| {
            Some(if ((u + v) % 2 == 0) == reverse {
                second
            } else {
                NonZeroU16::MIN
            })
        })
    });
    let boundary = [false, true].map(|reverse| {
        unit_leaves(|u, v| (((u + v) % 2 == 0) != reverse).then_some(NonZeroU16::MIN))
    });
    (interior, boundary)
}

fn bench_voxel_checker_chunk(c: &mut Criterion) {
    let (interior, boundary) = voxel_checker_planes();
    let interior_labels = interior.each_ref().map(|leaves| dense_labels(leaves));
    let boundary_labels = boundary.each_ref().map(|leaves| dense_labels(leaves));
    let mut chunk_leaves = Vec::with_capacity(195);
    let mut chunk_labels = Vec::with_capacity(195);
    for_chunk_planes(&interior, &boundary, |leaves| {
        chunk_leaves.push(leaves.clone());
    });
    for_chunk_planes(&interior_labels, &boundary_labels, |labels| {
        chunk_labels.push(labels.clone());
    });
    let mut scratch = SparseOptimalScratch64::new();
    for (leaves, labels, count) in interior
        .iter()
        .zip(&interior_labels)
        .map(|(leaves, labels)| (leaves, labels, PIXELS))
        .chain(
            boundary
                .iter()
                .zip(&boundary_labels)
                .map(|(leaves, labels)| (leaves, labels, PIXELS / 2)),
        )
    {
        let image = require_ok(SparseQuadImage64::from_leaves(leaves));
        validate_output_paths(
            "voxel_checker_plane",
            leaves,
            &image,
            labels,
            Some(count),
            &mut scratch,
        );
    }
    let mut total = 0usize;
    for leaves in &chunk_leaves {
        total += require_ok(scratch.decompose_borrowed(leaves)).len();
    }
    assert_eq!(total, 786_432);

    // 195 份独立输入均在计时外准备，计时覆盖输入工作集与矩形 checksum 消费。
    // 这是分解吞吐，未包含可见面抽取、顶点去重、mesh 打包或 GPU 上传。
    let mut group = c.benchmark_group("voxel_checker_chunk_195_planes");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("borrowed_leaves", |b| {
        b.iter(|| {
            let mut count = 0usize;
            let mut checksum = 0u64;
            for leaves in &chunk_leaves {
                let rectangles = require_ok(scratch.decompose_borrowed(black_box(leaves)));
                count += rectangles.len();
                checksum = checksum.wrapping_add(consume_rectangles(rectangles));
            }
            black_box((count, checksum));
        });
    });
    group.bench_function("native_labels", |b| {
        b.iter(|| {
            let mut count = 0usize;
            let mut checksum = 0u64;
            for labels in &chunk_labels {
                let rectangles = require_ok(scratch.decompose_labels_borrowed(black_box(labels)));
                count += rectangles.len();
                checksum = checksum.wrapping_add(consume_rectangles(rectangles));
            }
            black_box((count, checksum));
        });
    });
    group.bench_function("direct_sink", |b| {
        b.iter(|| {
            let mut count = 0usize;
            let mut checksum = 0u64;
            for leaves in &chunk_leaves {
                count += require_ok(scratch.decompose_into(black_box(leaves), |rectangle| {
                    checksum = checksum.wrapping_add(rectangle_checksum(rectangle));
                    Ok(())
                }));
            }
            black_box((count, checksum));
        });
    });
    group.bench_function("native_labels_sink", |b| {
        b.iter(|| {
            let mut count = 0usize;
            let mut checksum = 0u64;
            for labels in &chunk_labels {
                count += require_ok(scratch.decompose_labels_into(
                    black_box(labels),
                    |rectangle| {
                        checksum = checksum.wrapping_add(rectangle_checksum(rectangle));
                        Ok(())
                    },
                ));
            }
            black_box((count, checksum));
        });
    });
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
    bench_voxel_checker_chunk,
);
criterion_main!(benches);
