use std::num::NonZeroU16;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rectangle_decomposition::{QuadLeaf64, SparseOptimalScratch64, SparseQuadImage64};

const EDGE: u8 = 64;
const EDGE_USIZE: usize = 64;
const PIXELS: usize = EDGE_USIZE * EDGE_USIZE;
const HOLES_7_8: usize = PIXELS / 8;

fn low_discrepancy_holes_7_8_leaves() -> Vec<QuadLeaf64> {
    let Some(value) = NonZeroU16::new(1) else {
        return Vec::new();
    };

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
            let rect_count = scratch
                .decompose_borrowed(black_box(leaves.as_slice()))
                .map_or(0, <[_]>::len);
            black_box(rect_count);
        });
    });
}

fn bench_prepare_image(c: &mut Criterion) {
    let leaves = low_discrepancy_holes_7_8_leaves();
    c.bench_function("prepare_image_64", |b| {
        b.iter(|| black_box(SparseQuadImage64::from_leaves(black_box(&leaves))));
    });
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
);
criterion_main!(benches);
