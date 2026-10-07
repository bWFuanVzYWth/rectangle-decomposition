use std::hint::black_box;
use std::num::NonZeroU16;
use std::time::Instant;
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn measure(mut run: impl FnMut() -> usize) -> f64 {
    let t = Instant::now();
    for _ in 0..8 {
        black_box(run());
    }
    t.elapsed().as_secs_f64() * 1e6 / 8.0
}
fn main() {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
fn run() {
    println!("input,seed,entry,before_us,after_us,paired_delta_median_pct");
    let mut old = before::SparseOptimalScratch64::new();
    let mut new = after::SparseOptimalScratch64::new();
    for kind in 0..3 {
        for seed in 0..64u64 {
            let mut state = 0x9e3779b97f4a7c15u64.wrapping_mul(seed + 1);
            let mut grid = after::DenseLabels64::new();
            let mut leaves = Vec::new();
            for y in 0..64u8 {
                for x in 0..64u8 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let label = match kind {
                        0 => u16::from(state & 7 != 0),
                        1 => u16::from(state & 1 != 0),
                        _ => 1 + (state & 1) as u16,
                    };
                    grid.rows_mut()[usize::from(y)][usize::from(x)] = label;
                    if let Some(value) = NonZeroU16::new(label) {
                        leaves.push(after::QuadLeaf64 {
                            u: x,
                            v: y,
                            lod: 0,
                            value,
                        });
                    }
                }
            }
            let old_leaves: Vec<_> = leaves
                .iter()
                .map(|l| before::QuadLeaf64 {
                    u: l.u,
                    v: l.v,
                    lod: l.lod,
                    value: l.value,
                })
                .collect();
            let expected = old
                .decompose_borrowed(&old_leaves)
                .unwrap()
                .iter()
                .map(|r| (r.value, r.x.start, r.x.end, r.y.start, r.y.end))
                .collect::<Vec<_>>();
            assert_eq!(
                new.decompose_borrowed(&leaves)
                    .unwrap()
                    .iter()
                    .map(|r| (r.value, r.x.start, r.x.end, r.y.start, r.y.end))
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                new.decompose_labels_borrowed(&grid)
                    .unwrap()
                    .iter()
                    .map(|r| (r.value, r.x.start, r.x.end, r.y.start, r.y.end))
                    .collect::<Vec<_>>(),
                expected
            );
            for mode in 0..2 {
                let mut a =
                    || black_box(old.decompose_borrowed(black_box(&old_leaves)).unwrap()).len();
                let mut b = || {
                    if mode == 0 {
                        black_box(new.decompose_borrowed(black_box(&leaves)).unwrap()).len()
                    } else {
                        black_box(new.decompose_labels_borrowed(black_box(&grid)).unwrap()).len()
                    }
                };
                for _ in 0..16 {
                    black_box(a());
                    black_box(b());
                }
                let mut av = Vec::new();
                let mut bv = Vec::new();
                let mut rv = Vec::new();
                for round in 0..21 {
                    let mut t = [0.0; 2];
                    for phase in 0..4 {
                        if (phase == 0 || phase == 3) != (round % 2 == 0) {
                            t[0] += measure(&mut a);
                        } else {
                            t[1] += measure(&mut b);
                        }
                    }
                    av.push(t[0] / 2.0);
                    bv.push(t[1] / 2.0);
                    rv.push((t[1] / t[0] - 1.0) * 100.0);
                }
                println!(
                    "{},{seed},{},{:.3},{:.3},{:.2}",
                    ["random_7_8", "random_1_2", "random_two_color"][kind],
                    ["leaves", "native_grid"][mode],
                    median(&mut av),
                    median(&mut bv),
                    median(&mut rv)
                );
            }
        }
    }
}
