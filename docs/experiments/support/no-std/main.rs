#[path = "../nextopt/cases.rs"]
mod cases;
use std::hint::black_box;
use std::time::Instant;
fn q(v: &mut [f64], n: usize) -> f64 {
    v.sort_by(f64::total_cmp);
    v[(v.len() - 1) * n / 4]
}
fn measure(mut run: impl FnMut() -> usize) -> f64 {
    let t = Instant::now();
    for _ in 0..64 {
        black_box(run());
    }
    t.elapsed().as_secs_f64() * 1e6 / 64.0
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
    println!("input,entry,before_us,after_us,delta_q25_pct,delta_median_pct,delta_q75_pct");
    for (name, leaves, known) in cases::inputs() {
        let old_leaves: Vec<_> = leaves
            .iter()
            .map(|l| before::QuadLeaf64 {
                u: l.u,
                v: l.v,
                lod: l.lod,
                value: l.value,
            })
            .collect();
        let mut old = before::SparseOptimalScratch64::new();
        let mut new = after::SparseOptimalScratch64::new();
        assert_eq!(std::mem::size_of_val(&old), std::mem::size_of_val(&new));
        let mut old_grid = before::DenseLabels64::new();
        let mut new_grid = after::DenseLabels64::new();
        for leaf in &leaves {
            let side = 1usize << leaf.lod;
            for y in usize::from(leaf.v)..usize::from(leaf.v) + side {
                for x in usize::from(leaf.u)..usize::from(leaf.u) + side {
                    old_grid.rows_mut()[y][x] = leaf.value.get();
                    new_grid.rows_mut()[y][x] = leaf.value.get();
                }
            }
        }
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
            new.decompose_labels_borrowed(&new_grid)
                .unwrap()
                .iter()
                .map(|r| (r.value, r.x.start, r.x.end, r.y.start, r.y.end))
                .collect::<Vec<_>>(),
            expected
        );
        if let Some(count) = known {
            assert_eq!(expected.len(), count);
        }
        for mode in 0..3 {
            let mut a = || {
                if mode == 0 {
                    black_box(old.decompose_borrowed(black_box(&old_leaves)).unwrap()).len()
                } else if mode == 1 {
                    black_box(old.decompose_labels_borrowed(black_box(&old_grid)).unwrap()).len()
                } else {
                    let mut sum = 0u64;
                    let count = old
                        .decompose_into(black_box(&old_leaves), |r| {
                            sum = sum.wrapping_add(
                                u64::from(r.value)
                                    + u64::from(u32::from_le_bytes([
                                        r.x.start, r.x.end, r.y.start, r.y.end,
                                    ])),
                            );
                            Ok(())
                        })
                        .unwrap();
                    black_box(sum);
                    count
                }
            };
            let mut b = || {
                if mode == 0 {
                    black_box(new.decompose_borrowed(black_box(&leaves)).unwrap()).len()
                } else if mode == 1 {
                    black_box(new.decompose_labels_borrowed(black_box(&new_grid)).unwrap()).len()
                } else {
                    let mut sum = 0u64;
                    let count = new
                        .decompose_into(black_box(&leaves), |r| {
                            sum = sum.wrapping_add(
                                u64::from(r.value)
                                    + u64::from(u32::from_le_bytes([
                                        r.x.start, r.x.end, r.y.start, r.y.end,
                                    ])),
                            );
                            Ok(())
                        })
                        .unwrap();
                    black_box(sum);
                    count
                }
            };
            assert_eq!(a(), b());
            for _ in 0..128 {
                black_box(a());
                black_box(b());
            }
            let mut av = Vec::new();
            let mut bv = Vec::new();
            let mut rv = Vec::new();
            for round in 0..41 {
                let mut ts = [0.0; 2];
                for phase in 0..4 {
                    if (phase == 0 || phase == 3) != (round % 2 == 0) {
                        ts[0] += measure(&mut a);
                    } else {
                        ts[1] += measure(&mut b);
                    }
                }
                av.push(ts[0] / 2.0);
                bv.push(ts[1] / 2.0);
                rv.push((ts[1] / ts[0] - 1.0) * 100.0);
            }
            println!(
                "{name},{},{:.3},{:.3},{:.2},{:.2},{:.2}",
                ["leaves", "native_grid", "leaves_sink"][mode],
                q(&mut av, 2),
                q(&mut bv, 2),
                q(&mut rv, 1),
                q(&mut rv, 2),
                q(&mut rv, 3)
            );
        }
    }
}
