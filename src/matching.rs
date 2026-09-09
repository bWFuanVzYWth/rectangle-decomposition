//! 稀疏 chord 匹配后端。
//!
//! 对每个颜色组的水平/垂直 chord 构建二分冲突图 CSR，再用
//! Hopcroft-Karp 求最大独立集。

#[cfg(feature = "profile")]
use std::time::{Duration, Instant};

use crate::types::ChordAccess;
#[cfg(test)]
use crate::types::EffectiveChord;
use crate::{copy, slice};

pub const IMAGE64_AXIS_LIMIT: usize = 64;
pub const IMAGE64_AXIS_LEN: usize = IMAGE64_AXIS_LIMIT + 1;
pub const IMAGE64_GRID_POINTS: usize = IMAGE64_AXIS_LEN * IMAGE64_AXIS_LEN;
pub const IMAGE64_MAX_CONFLICT_EDGES: usize = (IMAGE64_AXIS_LIMIT - 1) * (IMAGE64_AXIS_LIMIT - 1);

#[cfg(test)]
#[derive(Debug)]
pub struct MaximumIndependentSet {
    pub(crate) horizontal: Vec<usize>,
    pub(crate) vertical: Vec<usize>,
}

#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default)]
pub struct MatchingTimings {
    pub total: Duration,
    pub greedy: Duration,
    pub bfs: Duration,
    pub dfs_build: Duration,
    pub dfs_search: Duration,
    pub cover: Duration,
    pub collect: Duration,
}

#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default)]
pub struct MatchingCounts {
    pub greedy_matches: usize,
    pub phases: usize,
    pub augmentations: usize,
}

#[cfg(feature = "profile")]
#[derive(Debug)]
pub struct MatchingProfile {
    pub timings: MatchingTimings,
    pub counts: MatchingCounts,
}

#[derive(Clone, Copy, Debug)]
pub struct SparseEdge {
    pub(super) left: u16,
    pub(super) right: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct SparseAdjacencyRef<'a> {
    pub(super) offsets: &'a [usize],
    pub(super) edges: &'a [u16],
}

#[derive(Clone, Debug)]
pub struct ConflictFinalizeScratch {
    pub(super) right_keys: Vec<u32>,
    pub(super) next_offsets: Vec<usize>,
    pub(super) edge_buffer: [SparseEdge; IMAGE64_MAX_CONFLICT_EDGES],
    pub(super) edge_count: usize,
    pub(super) right_degrees: Vec<u8>,
    pub(super) adjacency_offsets: Vec<usize>,
    pub(super) adjacency_edges: Vec<u16>,
    pub(super) horizontal_grid: [u16; IMAGE64_GRID_POINTS],
    pub(super) horizontal_grid_marks: [u16; IMAGE64_GRID_POINTS],
    pub(super) horizontal_y_masks: [u64; IMAGE64_AXIS_LEN],
    pub(super) horizontal_x_marks: [u16; IMAGE64_AXIS_LEN],
    pub(super) grid_mark: u16,
}

impl Default for ConflictFinalizeScratch {
    fn default() -> Self {
        Self {
            right_keys: Vec::new(),
            next_offsets: Vec::new(),
            edge_buffer: [SparseEdge { left: 0, right: 0 }; IMAGE64_MAX_CONFLICT_EDGES],
            edge_count: 0,
            right_degrees: Vec::new(),
            adjacency_offsets: Vec::new(),
            adjacency_edges: Vec::new(),
            horizontal_grid: [u16::MAX; IMAGE64_GRID_POINTS],
            horizontal_grid_marks: [0; IMAGE64_GRID_POINTS],
            horizontal_y_masks: [0; IMAGE64_AXIS_LEN],
            horizontal_x_marks: [0; IMAGE64_AXIS_LEN],
            grid_mark: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConflictScratch {
    pub(super) finalize: ConflictFinalizeScratch,
}

/// HK 算法阶段的复用 scratch。
#[derive(Clone, Debug, Default)]
pub struct HkScratch {
    pub(super) pair_left: Vec<u16>,
    pub(super) pair_right: Vec<u16>,
    pub(super) distance: Vec<u16>,
    pub(super) queue: Vec<u16>,
    pub(super) next_edge: Vec<usize>,
    pub(super) unmatched_lefts: Vec<u16>,
    pub(super) touched_lefts: Vec<u16>,
    pub(super) reachable_left: Vec<bool>,
    pub(super) reachable_right: Vec<bool>,
    pub(super) transpose_offsets: Vec<usize>,
    pub(super) transpose_edges: Vec<u16>,
    pub(super) write_offsets: Vec<usize>,
    pub(super) right_order: Vec<u16>,
    pub(super) right_degree_counts: Vec<usize>,
    pub(super) left_degree_counts: Vec<usize>,
    pub(super) dfs_stack: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Default)]
pub struct MatchingScratch {
    pub(super) hk: HkScratch,
    pub(super) conflict: ConflictScratch,
    selected_horizontal: Vec<u16>,
    selected_vertical: Vec<u16>,
}

impl SparseAdjacencyRef<'_> {
    pub(super) fn neighbors(&self, left: usize) -> &[u16] {
        slice(
            self.edges,
            copy(self.offsets, left)..copy(self.offsets, left + 1),
        )
    }
}

pub const UNMATCHED_U16: u16 = u16::MAX;

impl MatchingScratch {
    /// 为 64x64 输入的 matching 阶段预分配容量。
    ///
    /// 资源契约：成功后 matching hot path 不需要扩容，除非未来上界被改大。
    pub(crate) fn preallocate_64(&mut self) -> Result<(), crate::SparseQuadError> {
        let max_chords = IMAGE64_AXIS_LIMIT * (IMAGE64_AXIS_LIMIT - 1);
        reserve_vec(&mut self.conflict.finalize.right_keys, max_chords)?;
        reserve_vec(&mut self.conflict.finalize.next_offsets, max_chords)?;
        reserve_vec(&mut self.conflict.finalize.right_degrees, max_chords)?;
        reserve_vec(
            &mut self.conflict.finalize.adjacency_offsets,
            max_chords + 1,
        )?;
        reserve_vec(
            &mut self.conflict.finalize.adjacency_edges,
            IMAGE64_MAX_CONFLICT_EDGES,
        )?;

        reserve_vec(&mut self.hk.pair_left, max_chords)?;
        reserve_vec(&mut self.hk.pair_right, max_chords)?;
        reserve_vec(&mut self.hk.distance, max_chords)?;
        reserve_vec(&mut self.hk.queue, max_chords)?;
        reserve_vec(&mut self.hk.next_edge, max_chords)?;
        reserve_vec(&mut self.hk.unmatched_lefts, max_chords)?;
        reserve_vec(&mut self.hk.touched_lefts, max_chords)?;
        reserve_vec(&mut self.hk.reachable_left, max_chords)?;
        reserve_vec(&mut self.hk.reachable_right, max_chords)?;
        reserve_vec(&mut self.hk.transpose_offsets, max_chords + 1)?;
        reserve_vec(&mut self.hk.transpose_edges, IMAGE64_MAX_CONFLICT_EDGES)?;
        reserve_vec(&mut self.hk.write_offsets, max_chords)?;
        reserve_vec(&mut self.hk.right_order, max_chords)?;
        reserve_vec(&mut self.hk.right_degree_counts, IMAGE64_AXIS_LIMIT)?;
        reserve_vec(&mut self.hk.left_degree_counts, IMAGE64_AXIS_LIMIT)?;
        reserve_vec(&mut self.hk.dfs_stack, max_chords)?;
        reserve_vec(&mut self.selected_horizontal, max_chords)?;
        reserve_vec(&mut self.selected_vertical, max_chords)?;
        Ok(())
    }

    #[cfg(test)]
    pub fn maximum_independent_set(
        &mut self,
        horizontal_edges: &[EffectiveChord],
        vertical_edges: &[EffectiveChord],
    ) -> MaximumIndependentSet {
        self.select_maximum_independent_set(horizontal_edges, vertical_edges);
        selected_from_scratch(self)
    }

    pub(crate) fn select_maximum_independent_set<H: ChordAccess, V: ChordAccess>(
        &mut self,
        horizontal_edges: &[H],
        vertical_edges: &[V],
    ) {
        self.selected_horizontal.clear();
        self.selected_vertical.clear();
        if horizontal_edges.is_empty() || vertical_edges.is_empty() {
            select_full_independent_set(
                horizontal_edges.len(),
                vertical_edges.len(),
                &mut self.selected_horizontal,
                &mut self.selected_vertical,
            );
            return;
        }

        let (sparse_adjacency, right_degrees) = crate::graph::build_sparse_conflict_graph_csr(
            horizontal_edges,
            vertical_edges,
            &mut self.conflict,
        );
        crate::hk::hopcroft_karp_sparse_csr_u16(
            &sparse_adjacency,
            vertical_edges.len(),
            right_degrees,
            &mut self.hk,
        );
        collect_independent_set(self);
    }

    #[cfg(feature = "profile")]
    pub(crate) fn maximum_independent_set_profile<H: ChordAccess, V: ChordAccess>(
        &mut self,
        horizontal_edges: &[H],
        vertical_edges: &[V],
    ) -> MatchingProfile {
        let start = Instant::now();
        self.selected_horizontal.clear();
        self.selected_vertical.clear();
        if horizontal_edges.is_empty() || vertical_edges.is_empty() {
            let collect_start = Instant::now();
            select_full_independent_set(
                horizontal_edges.len(),
                vertical_edges.len(),
                &mut self.selected_horizontal,
                &mut self.selected_vertical,
            );
            return MatchingProfile {
                timings: MatchingTimings {
                    total: start.elapsed(),
                    collect: collect_start.elapsed(),
                    ..MatchingTimings::default()
                },
                counts: MatchingCounts::default(),
            };
        }

        let (sparse_adjacency, right_degrees) = crate::graph::build_sparse_conflict_graph_csr(
            horizontal_edges,
            vertical_edges,
            &mut self.conflict,
        );
        let (mut timings, counts) = crate::hk::hopcroft_karp_sparse_csr_u16_profile(
            &sparse_adjacency,
            vertical_edges.len(),
            right_degrees,
            &mut self.hk,
        );
        let collect_start = Instant::now();
        collect_independent_set(self);
        timings.collect = collect_start.elapsed();
        timings.total = start.elapsed();
        MatchingProfile { timings, counts }
    }

    pub(crate) const fn selected_horizontal(&self) -> &[u16] {
        self.selected_horizontal.as_slice()
    }

    pub(crate) const fn selected_vertical(&self) -> &[u16] {
        self.selected_vertical.as_slice()
    }
}

fn select_full_independent_set(
    horizontal_len: usize,
    vertical_len: usize,
    horizontal_out: &mut Vec<u16>,
    vertical_out: &mut Vec<u16>,
) {
    for index in 0..horizontal_len {
        horizontal_out.push(crate::u16_index(index));
    }
    for index in 0..vertical_len {
        vertical_out.push(crate::u16_index(index));
    }
}

fn collect_independent_set(scratch: &mut MatchingScratch) {
    for (index, &reachable) in scratch.hk.reachable_left.iter().enumerate() {
        if reachable {
            scratch.selected_horizontal.push(crate::u16_index(index));
        }
    }
    for (index, &reachable) in scratch.hk.reachable_right.iter().enumerate() {
        if !reachable {
            scratch.selected_vertical.push(crate::u16_index(index));
        }
    }
}

#[cfg(test)]
fn selected_from_scratch(scratch: &MatchingScratch) -> MaximumIndependentSet {
    MaximumIndependentSet {
        horizontal: scratch
            .selected_horizontal
            .iter()
            .map(|&index| usize::from(index))
            .collect(),
        vertical: scratch
            .selected_vertical
            .iter()
            .map(|&index| usize::from(index))
            .collect(),
    }
}

fn reserve_vec<T>(items: &mut Vec<T>, capacity: usize) -> Result<(), crate::SparseQuadError> {
    if items.capacity() >= capacity {
        return Ok(());
    }
    items
        .try_reserve_exact(capacity - items.capacity())
        .map_err(|_| crate::SparseQuadError::AllocationFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Orientation;

    const fn h(x1: u8, y: u8, x2: u8) -> EffectiveChord {
        EffectiveChord {
            orientation: Orientation::Horizontal,
            x1,
            y1: y,
            x2,
            y2: y,
        }
    }

    const fn v(x: u8, y1: u8, y2: u8) -> EffectiveChord {
        EffectiveChord {
            orientation: Orientation::Vertical,
            x1: x,
            y1,
            x2: x,
            y2,
        }
    }

    #[test]
    fn csr_matching_matches_bruteforce_mis() {
        let horizontal = [h(0, 1, 4), h(2, 3, 6), h(1, 5, 5)];
        let vertical = [v(1, 0, 2), v(3, 1, 5), v(5, 2, 6)];
        let mut scratch = MatchingScratch::default();
        let selected = scratch.maximum_independent_set(&horizontal, &vertical);
        let selected_size = selected.horizontal.len() + selected.vertical.len();

        assert!(is_independent(&horizontal, &vertical, &selected));
        assert_eq!(selected_size, brute_force_mis_size(&horizontal, &vertical));
    }

    #[test]
    fn dense_complete_grid_selects_larger_side() {
        let horizontal = [h(0, 1, 4), h(0, 2, 4), h(0, 3, 4)];
        let vertical = [v(1, 0, 4), v(2, 0, 4), v(3, 0, 4), v(4, 0, 4)];
        let mut scratch = MatchingScratch::default();
        let selected = scratch.maximum_independent_set(&horizontal, &vertical);

        assert_eq!(selected.horizontal.len() + selected.vertical.len(), 4);
    }

    #[test]
    fn full_grid_uses_maximum_edge_capacity() {
        let mut horizontal = [h(0, 1, 64); IMAGE64_AXIS_LIMIT - 1];
        let mut vertical = [v(1, 0, 64); IMAGE64_AXIS_LIMIT - 1];
        for (chord, coord) in horizontal.iter_mut().zip(1u8..64) {
            *chord = h(0, coord, 64);
        }
        for (chord, coord) in vertical.iter_mut().zip(1u8..64) {
            *chord = v(coord, 0, 64);
        }
        let mut scratch = Box::new(MatchingScratch::default());
        scratch.select_maximum_independent_set(&horizontal, &vertical);
        assert_eq!(
            scratch.conflict.finalize.edge_count,
            IMAGE64_MAX_CONFLICT_EDGES
        );
    }

    fn is_independent(
        horizontal: &[EffectiveChord],
        vertical: &[EffectiveChord],
        selected: &MaximumIndependentSet,
    ) -> bool {
        for &h_index in &selected.horizontal {
            let Some(&h) = horizontal.get(h_index) else {
                return false;
            };
            for &v_index in &selected.vertical {
                let Some(&v) = vertical.get(v_index) else {
                    return false;
                };
                if intersects(h, v) {
                    return false;
                }
            }
        }
        true
    }

    fn brute_force_mis_size(horizontal: &[EffectiveChord], vertical: &[EffectiveChord]) -> usize {
        let total = horizontal.len() + vertical.len();
        if total >= usize::BITS as usize {
            return 0;
        }

        let mut best = 0usize;
        for mask in 0usize..(1usize << total) {
            if brute_force_independent(mask, horizontal, vertical) {
                best = best.max(mask.count_ones() as usize);
            }
        }
        best
    }

    fn brute_force_independent(
        mask: usize,
        horizontal: &[EffectiveChord],
        vertical: &[EffectiveChord],
    ) -> bool {
        for (h_index, &h) in horizontal.iter().enumerate() {
            if mask & (1usize << h_index) == 0 {
                continue;
            }
            for (v_index, &v) in vertical.iter().enumerate() {
                let bit = horizontal.len() + v_index;
                if mask & (1usize << bit) != 0 && intersects(h, v) {
                    return false;
                }
            }
        }
        true
    }

    fn intersects(horizontal: EffectiveChord, vertical: EffectiveChord) -> bool {
        let x_intersects = (horizontal.x1..=horizontal.x2).contains(&vertical.x1);
        let y_intersects = (vertical.y1..=vertical.y2).contains(&horizontal.y1);
        x_intersects && y_intersects
    }
}
