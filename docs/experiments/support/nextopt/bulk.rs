use std::hint::black_box;
use std::num::NonZeroU16;
use std::time::Instant;

fn q(v: &mut [f64], n: usize) -> f64 {
    v.sort_by(f64::total_cmp);
    v[(v.len() - 1) * n / 4]
}
fn time(mut run: impl FnMut() -> usize) -> f64 {
    let start = Instant::now();
    black_box(run());
    start.elapsed().as_secs_f64() * 1e3
}
fn raster(leaves: &[after::QuadLeaf64], grid: &mut after::DenseLabels64) {
    grid.clear();
    for leaf in leaves {
        grid.rows_mut()[usize::from(leaf.v)][usize::from(leaf.u)] = leaf.value.get();
    }
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
    let mut templates = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for y in 0..64u8 {
        for x in 0..64u8 {
            let even = (x + y) % 2 == 0;
            templates[0].push(after::QuadLeaf64 {
                u: x,
                v: y,
                lod: 0,
                value: NonZeroU16::new(if even { 1 } else { 2 }).unwrap(),
            });
            templates[1].push(after::QuadLeaf64 {
                u: x,
                v: y,
                lod: 0,
                value: NonZeroU16::new(if even { 2 } else { 1 }).unwrap(),
            });
            templates[if even { 2 } else { 3 }].push(after::QuadLeaf64 {
                u: x,
                v: y,
                lod: 0,
                value: NonZeroU16::MIN,
            });
        }
    }
    // Every physical plane owns its input allocation, matching an upstream layer list.
    let inputs: Vec<_> = (0..195)
        .map(|i| {
            let layer = i % 65;
            templates[match layer {
                0 => 2,
                64 => 3,
                _ => (layer - 1) % 2,
            }]
            .clone()
        })
        .collect();
    let old_inputs: Vec<Vec<_>> = inputs
        .iter()
        .map(|ls| {
            ls.iter()
                .map(|l| before::QuadLeaf64 {
                    u: l.u,
                    v: l.v,
                    lod: l.lod,
                    value: l.value,
                })
                .collect()
        })
        .collect();
    let grids: Vec<_> = inputs
        .iter()
        .map(|ls| {
            let mut grid = after::DenseLabels64::new();
            raster(ls, &mut grid);
            grid
        })
        .collect();
    let mut old = before::SparseOptimalScratch64::new();
    let mut new = after::SparseOptimalScratch64::new();
    let expected = 786432;
    let materials = [0u32, 0x12345678, 0x98765432];
    for i in 0..195 {
        let a = old.decompose_borrowed(&old_inputs[i]).unwrap();
        let b = new.decompose_labels_borrowed(&grids[i]).unwrap();
        assert_eq!(
            a.iter()
                .map(|r| (r.value, r.x.start, r.x.end, r.y.start, r.y.end))
                .collect::<Vec<_>>(),
            b.iter()
                .map(|r| (r.value, r.x.start, r.x.end, r.y.start, r.y.end))
                .collect::<Vec<_>>()
        );
    }
    println!("input,entry,before_ms,after_ms,delta_q25_pct,delta_median_pct,delta_q75_pct");
    for mode in 0..4 {
        let mut old_output = Vec::with_capacity(expected);
        let mut new_output = Vec::with_capacity(expected);
        let mut old_emitted = Vec::with_capacity(4096);
        let mut new_grid = after::DenseLabels64::new();
        let mut a = || {
            old_output.clear();
            for _axis in 0..3 {
                for layer in 0..65 {
                    let i = _axis * 65 + layer;
                    let leaves = if mode == 3 {
                        old_emitted.clear();
                        for leaf in black_box(&old_inputs[i]) {
                            old_emitted.push(*leaf);
                        }
                        &old_emitted
                    } else {
                        &old_inputs[i]
                    };
                    for r in old.decompose_borrowed(black_box(leaves)).unwrap() {
                        old_output.push((
                            u32::from_le_bytes([r.x.start, r.x.end, r.y.start, r.y.end]),
                            materials[usize::from(r.value)],
                        ));
                    }
                }
            }
            black_box(&old_output).len()
        };
        let mut b = || {
            new_output.clear();
            for _axis in 0..3 {
                for layer in 0..65 {
                    let i = _axis * 65 + layer;
                    let mut sink = |r: after::Rectangle| {
                        new_output.push((
                            u32::from_le_bytes([r.x.start, r.x.end, r.y.start, r.y.end]),
                            materials[usize::from(r.value)],
                        ));
                        Ok(())
                    };
                    if mode == 0 {
                        for &r in new.decompose_borrowed(black_box(&inputs[i])).unwrap() {
                            sink(r).unwrap();
                        }
                    } else if mode == 1 {
                        new.decompose_into(black_box(&inputs[i]), sink).unwrap();
                    } else if mode == 2 {
                        new.decompose_labels_into(black_box(&grids[i]), sink)
                            .unwrap();
                    } else {
                        raster(black_box(&inputs[i]), &mut new_grid);
                        new.decompose_labels_into(black_box(&new_grid), sink)
                            .unwrap();
                    }
                }
            }
            black_box(&new_output).len()
        };
        assert_eq!(a(), expected);
        assert_eq!(b(), expected);
        // The consumer receives exactly the same ordered bounds/material records.
        drop(a);
        drop(b);
        assert_eq!(old_output, new_output);
        let mut a = || {
            old_output.clear();
            for _axis in 0..3 {
                for layer in 0..65 {
                    let i = _axis * 65 + layer;
                    let leaves = if mode == 3 {
                        old_emitted.clear();
                        old_emitted.extend_from_slice(black_box(&old_inputs[i]));
                        &old_emitted
                    } else {
                        &old_inputs[i]
                    };
                    for r in old.decompose_borrowed(black_box(leaves)).unwrap() {
                        old_output.push((
                            u32::from_le_bytes([r.x.start, r.x.end, r.y.start, r.y.end]),
                            materials[usize::from(r.value)],
                        ));
                    }
                }
            }
            black_box(&old_output).len()
        };
        let mut b = || {
            new_output.clear();
            for _axis in 0..3 {
                for layer in 0..65 {
                    let i = _axis * 65 + layer;
                    let mut sink = |r: after::Rectangle| {
                        new_output.push((
                            u32::from_le_bytes([r.x.start, r.x.end, r.y.start, r.y.end]),
                            materials[usize::from(r.value)],
                        ));
                        Ok(())
                    };
                    match mode {
                        0 => {
                            for &r in new.decompose_borrowed(black_box(&inputs[i])).unwrap() {
                                sink(r).unwrap();
                            }
                        }
                        1 => {
                            new.decompose_into(black_box(&inputs[i]), sink).unwrap();
                        }
                        2 => {
                            new.decompose_labels_into(black_box(&grids[i]), sink)
                                .unwrap();
                        }
                        _ => {
                            raster(black_box(&inputs[i]), &mut new_grid);
                            new.decompose_labels_into(black_box(&new_grid), sink)
                                .unwrap();
                        }
                    }
                }
            }
            black_box(&new_output).len()
        };
        for _ in 0..3 {
            black_box(a());
            black_box(b());
        }
        let mut av = Vec::new();
        let mut bv = Vec::new();
        let mut rv = Vec::new();
        for round in 0..41 {
            let mut t = [0.0; 2];
            for phase in 0..4 {
                if (phase == 0 || phase == 3) != (round % 2 == 0) {
                    t[0] += time(&mut a);
                } else {
                    t[1] += time(&mut b);
                }
            }
            av.push(t[0] / 2.0);
            bv.push(t[1] / 2.0);
            rv.push((t[1] / t[0] - 1.0) * 100.0);
        }
        println!(
            "checkerboard_chunk_195_planes,{}, {:.3},{:.3},{:.2},{:.2},{:.2}",
            [
                "leaves_consumer",
                "leaves_sink",
                "native_grid_sink",
                "producer_grid_sink"
            ][mode],
            q(&mut av, 2),
            q(&mut bv, 2),
            q(&mut rv, 1),
            q(&mut rv, 2),
            q(&mut rv, 3)
        );
    }
}
