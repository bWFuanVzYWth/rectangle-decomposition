mod cases;
use std::hint::black_box;
use std::time::Instant;

fn measure(mut run: impl FnMut() -> usize) -> f64 {
    let t = Instant::now();
    for _ in 0..64 {
        black_box(run());
    }
    t.elapsed().as_secs_f64() * 1e6 / 64.0
}
fn q(v: &mut [f64], n: usize) -> f64 {
    v.sort_by(f64::total_cmp);
    v[(v.len() - 1) * n / 4]
}
fn main() {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
fn raster(leaves: &[after::QuadLeaf64], grid: &mut after::DenseLabels64) {
    grid.clear();
    for leaf in leaves {
        let side = 1usize << leaf.lod;
        for row in &mut grid.rows_mut()[usize::from(leaf.v)..usize::from(leaf.v) + side] {
            row[usize::from(leaf.u)..usize::from(leaf.u) + side].fill(leaf.value.get());
        }
    }
}
fn run() {
    eprintln!(
        "scratch_before={} after={}",
        std::mem::size_of::<before::SparseOptimalScratch64>(),
        std::mem::size_of::<after::SparseOptimalScratch64>()
    );
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
        let mut grid = after::DenseLabels64::new();
        raster(&leaves, &mut grid);
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
        if let Some(count) = known {
            assert_eq!(expected.len(), count);
        }
        for mode in 0..4 {
            let mut old_emitted = Vec::with_capacity(old_leaves.len());
            let mut old_output = Vec::<(u32, u32)>::with_capacity(4096);
            let mut new_output = Vec::<(u32, u32)>::with_capacity(4096);
            let materials: Vec<u32> = (0..65536u32)
                .map(|label| label | ((label ^ 0x5a5a) << 16))
                .collect();
            let mut old_run = || {
                if mode == 2 {
                    old_emitted.clear();
                    for leaf in black_box(&old_leaves) {
                        old_emitted.push(*leaf);
                    }
                    black_box(old.decompose_borrowed(black_box(&old_emitted)).unwrap()).len()
                } else if mode == 3 {
                    old_output.clear();
                    for r in black_box(old.decompose_borrowed(black_box(&old_leaves)).unwrap()) {
                        old_output.push((
                            u32::from_le_bytes([r.x.start, r.x.end, r.y.start, r.y.end]),
                            materials[usize::from(r.value)],
                        ));
                    }
                    black_box(&old_output).len()
                } else {
                    black_box(old.decompose_borrowed(black_box(&old_leaves)).unwrap()).len()
                }
            };
            let mut new_run = || match mode {
                0 => black_box(new.decompose_borrowed(black_box(&leaves)).unwrap()).len(),
                1 => black_box(new.decompose_labels_borrowed(black_box(&grid)).unwrap()).len(),
                3 => {
                    new_output.clear();
                    new.decompose_into(black_box(&leaves), |r| {
                        new_output.push((
                            u32::from_le_bytes([r.x.start, r.x.end, r.y.start, r.y.end]),
                            materials[usize::from(r.value)],
                        ));
                        Ok(())
                    })
                    .unwrap();
                    black_box(&new_output).len()
                }
                _ => {
                    raster(black_box(&leaves), &mut grid);
                    black_box(new.decompose_labels_borrowed(black_box(&grid)).unwrap()).len()
                }
            };
            assert_eq!(old_run(), new_run());
            for _ in 0..128 {
                black_box(old_run());
                black_box(new_run());
            }
            let mut av = Vec::new();
            let mut bv = Vec::new();
            let mut rv = Vec::new();
            for round in 0..41 {
                let mut ts = [0.0; 2];
                for phase in 0..4 {
                    if (phase == 0 || phase == 3) != (round % 2 == 0) {
                        ts[0] += measure(&mut old_run);
                    } else {
                        ts[1] += measure(&mut new_run);
                    }
                }
                av.push(ts[0] / 2.0);
                bv.push(ts[1] / 2.0);
                rv.push((ts[1] / ts[0] - 1.0) * 100.0);
            }
            let mode = [
                "leaves",
                "native_grid",
                "producer_pipeline",
                "consume_pipeline",
            ][mode];
            println!(
                "{name},{mode},{:.3},{:.3},{:.2},{:.2},{:.2}",
                q(&mut av, 2),
                q(&mut bv, 2),
                q(&mut rv, 1),
                q(&mut rv, 2),
                q(&mut rv, 3)
            );
        }
    }
}
