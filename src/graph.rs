//! 稀疏冲突图构建：将 chord 列表转为 CSR 邻接。

use crate::matching::{
    ConflictFinalizeScratch, ConflictScratch, IMAGE64_AXIS_LIMIT, IMAGE64_MAX_CONFLICT_EDGES,
    SparseAdjacencyRef, SparseEdge, UNMATCHED_U16,
};
use crate::types::ChordAccess;
use crate::{copy, get_mut, slice, u16_index, u32_index};

const IMAGE64_INTERNAL_MIN: usize = 1;
const IMAGE64_INTERNAL_MAX: usize = IMAGE64_AXIS_LIMIT - 1;

/// 利用按右顶点连续分组的 `edge_buffer`，在线性时间内构建有序左 CSR。
/// 每个左邻接表仍按 (右顶点度数, 右顶点编号) 排序。
pub fn build_sparse_adjacency_in_scratch(
    left_size: usize,
    right_size: usize,
    scratch: &mut ConflictFinalizeScratch,
) -> (SparseAdjacencyRef<'_>, &[u8]) {
    let edges = slice(&scratch.edge_buffer, 0..scratch.edge_count);
    let right_degrees = scratch.right_degrees.as_slice();
    debug_assert_eq!(right_degrees.len(), right_size);
    scratch.adjacency_offsets.clear();
    scratch.adjacency_offsets.resize(left_size + 1, 0);
    scratch.right_layout.clear();
    scratch.right_layout.resize(2 * right_size, 0);
    let layout = &mut scratch.right_layout;
    let offsets = &mut scratch.adjacency_offsets;
    let mut previous_right = UNMATCHED_U16;
    for (index, edge) in edges.iter().enumerate() {
        *get_mut(offsets, usize::from(edge.left) + 1) += 1;
        if edge.right != previous_right {
            *get_mut(layout, usize::from(edge.right)) = u32_index(index);
            previous_right = edge.right;
        }
    }
    for index in 1..offsets.len() {
        *get_mut(offsets, index) += copy(offsets, index - 1);
    }

    scratch.next_offsets.clear();
    scratch
        .next_offsets
        .extend_from_slice(slice(offsets, 0..left_size));
    scratch.adjacency_edges.clear();
    scratch.adjacency_edges.resize(edges.len(), 0);
    let compact_edges = &mut scratch.adjacency_edges;
    let mut degree_offsets = [0usize; IMAGE64_AXIS_LIMIT];
    for &degree in right_degrees {
        *get_mut(&mut degree_offsets, usize::from(degree)) += 1;
    }
    debug_assert_eq!(
        right_degrees
            .iter()
            .map(|&degree| usize::from(degree))
            .sum::<usize>(),
        edges.len()
    );

    let mut order_start = 0usize;
    for offset in &mut degree_offsets {
        let count = *offset;
        *offset = order_start;
        order_start += count;
    }
    for (right, &degree) in right_degrees.iter().enumerate() {
        let slot = get_mut(&mut degree_offsets, usize::from(degree));
        *get_mut(layout, right_size + *slot) = u32_index(right);
        *slot += 1;
    }

    // 稳定计数排序给出全局右顶点顺序；散布后，各左邻接表自然有序。
    for order in 0..right_size {
        let right = copy(layout, right_size + order) as usize;
        let start = copy(layout, right) as usize;
        let end = start + usize::from(copy(right_degrees, right));
        for edge in slice(edges, start..end) {
            debug_assert_eq!(usize::from(edge.right), right);
            let left = usize::from(edge.left);
            let slot = get_mut(&mut scratch.next_offsets, left);
            *get_mut(compact_edges, *slot) = edge.right;
            *slot += 1;
        }
    }

    (
        SparseAdjacencyRef {
            offsets: scratch.adjacency_offsets.as_slice(),
            edges: scratch.adjacency_edges.as_slice(),
        },
        scratch.right_degrees.as_slice(),
    )
}

pub fn build_sparse_conflict_graph_csr<'a, H: ChordAccess, V: ChordAccess>(
    horizontal_edges: &[H],
    vertical_edges: &[V],
    scratch: &'a mut ConflictScratch,
) -> (SparseAdjacencyRef<'a>, &'a [u8]) {
    build_sparse_conflict_graph_sweep(horizontal_edges, vertical_edges, &mut scratch.finalize);
    build_sparse_adjacency_in_scratch(
        horizontal_edges.len(),
        vertical_edges.len(),
        &mut scratch.finalize,
    )
}

/// x 事件桶和活跃 y 位图避免逐格展开水平 chord；总工作为 O(n + C + E)。
fn build_sparse_conflict_graph_sweep<H: ChordAccess, V: ChordAccess>(
    horizontal_edges: &[H],
    vertical_edges: &[V],
    scratch: &mut ConflictFinalizeScratch,
) {
    scratch.horizontal_start_heads.fill(UNMATCHED_U16);
    scratch.horizontal_end_heads.fill(UNMATCHED_U16);
    scratch.vertical_query_heads.fill(UNMATCHED_U16);
    scratch.right_layout.clear();
    scratch
        .right_layout
        .resize(horizontal_edges.len() + vertical_edges.len(), u32::MAX);
    scratch.edge_count = 0;
    scratch.right_degrees.clear();
    scratch.right_degrees.resize(vertical_edges.len(), 0);

    for (index, horizontal) in horizontal_edges.iter().enumerate() {
        let horizontal = horizontal.chord();
        debug_assert_eq!(horizontal.y1, horizontal.y2);
        let y = usize::from(horizontal.y1);
        debug_assert!((IMAGE64_INTERNAL_MIN..=IMAGE64_INTERNAL_MAX).contains(&y));
        let start = usize::from(horizontal.x1).max(IMAGE64_INTERNAL_MIN);
        let end = usize::from(horizontal.x2).min(IMAGE64_INTERNAL_MAX);
        if start > end {
            continue;
        }
        // 两个横向事件的 next 编号打包到一个 u32，后续 CSR 复用这块存储。
        let start_next = copy(&scratch.horizontal_start_heads, start);
        let end_next = copy(&scratch.horizontal_end_heads, end);
        *get_mut(&mut scratch.right_layout, index) =
            u32::from(start_next) | (u32::from(end_next) << 16);
        *get_mut(&mut scratch.horizontal_start_heads, start) = u16_index(index);
        *get_mut(&mut scratch.horizontal_end_heads, end) = u16_index(index);
    }

    for (index, vertical) in vertical_edges.iter().enumerate() {
        let vertical = vertical.chord();
        debug_assert_eq!(vertical.x1, vertical.x2);
        let x = usize::from(vertical.x1);
        debug_assert!((IMAGE64_INTERNAL_MIN..=IMAGE64_INTERNAL_MAX).contains(&x));
        *get_mut(&mut scratch.right_layout, horizontal_edges.len() + index) =
            u32::from(copy(&scratch.vertical_query_heads, x));
        *get_mut(&mut scratch.vertical_query_heads, x) = u16_index(index);
    }

    sweep_conflict_events(horizontal_edges, vertical_edges, scratch);
    debug_assert!(scratch.edge_count <= IMAGE64_MAX_CONFLICT_EDGES);
}

fn sweep_conflict_events<H: ChordAccess, V: ChordAccess>(
    horizontal_edges: &[H],
    vertical_edges: &[V],
    scratch: &mut ConflictFinalizeScratch,
) {
    let mut active_y = 0u64;
    for x in IMAGE64_INTERNAL_MIN..=IMAGE64_INTERNAL_MAX {
        // 闭端点：同一 x 的 start 先于 query，end 后于 query。
        let mut horizontal_index = copy(&scratch.horizontal_start_heads, x);
        while horizontal_index != UNMATCHED_U16 {
            let index = usize::from(horizontal_index);
            let y = usize::from(crate::get(horizontal_edges, index).chord().y1);
            debug_assert_eq!(active_y & (1u64 << y), 0);
            active_y |= 1u64 << y;
            *get_mut(&mut scratch.active_horizontal, y) = horizontal_index;
            horizontal_index = u16_index((copy(&scratch.right_layout, index) & 0xffff) as usize);
        }

        let mut vertical_index = copy(&scratch.vertical_query_heads, x);
        while vertical_index != UNMATCHED_U16 {
            let index = usize::from(vertical_index);
            let vertical = crate::get(vertical_edges, index).chord();
            let mut conflicts = active_y & internal_mask(vertical.y1, vertical.y2);
            // 每个右顶点只查询一次，边段连续；CSR 阶段重新定位各段并稳定排序。
            let start = scratch.edge_count;
            while conflicts != 0 {
                let y = usize::try_from(conflicts.trailing_zeros())
                    .unwrap_or_else(|_| std::process::abort());
                *get_mut(&mut scratch.edge_buffer, scratch.edge_count) = SparseEdge {
                    left: copy(&scratch.active_horizontal, y),
                    right: vertical_index,
                };
                scratch.edge_count += 1;
                conflicts &= conflicts - 1;
            }
            *get_mut(&mut scratch.right_degrees, index) =
                u8::try_from(scratch.edge_count - start).unwrap_or_else(|_| std::process::abort());
            vertical_index =
                u16_index(copy(&scratch.right_layout, horizontal_edges.len() + index) as usize);
        }

        horizontal_index = copy(&scratch.horizontal_end_heads, x);
        while horizontal_index != UNMATCHED_U16 {
            let index = usize::from(horizontal_index);
            let y = usize::from(crate::get(horizontal_edges, index).chord().y1);
            debug_assert_eq!(copy(&scratch.active_horizontal, y), horizontal_index);
            active_y &= !(1u64 << y);
            horizontal_index = u16_index((copy(&scratch.right_layout, index) >> 16) as usize);
        }
    }
    debug_assert_eq!(active_y, 0);
}

fn internal_mask(start: u8, end: u8) -> u64 {
    let start = usize::from(start).max(IMAGE64_INTERNAL_MIN);
    let end = usize::from(end).min(IMAGE64_INTERNAL_MAX);
    if start > end {
        return 0;
    }

    let start_mask = u64::MAX << start;
    let end_mask = if end >= 63 {
        u64::MAX
    } else {
        (1u64 << (end + 1)) - 1
    };
    start_mask & end_mask
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)] // 列表中的编号均由 0..63 构造。
mod tests {
    use super::*;
    use crate::matching::{HkScratch, IMAGE64_AXIS_LEN, MatchingScratch};
    use crate::types::{EffectiveChord, Orientation};

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
    fn sweep_reuse_ignores_stale_events_and_handles_empty_graphs() {
        let mut scratch = ConflictScratch::default();
        for (horizontal, vertical) in [
            (vec![h(1, 2, 6)], vec![v(3, 1, 5)]),
            (vec![h(4, 6, 6)], vec![v(3, 1, 5)]),
            (vec![], vec![v(63, 0, 64)]),
            (vec![h(0, 63, 64)], vec![]),
            (vec![], vec![]),
            (vec![h(1, 2, 6)], vec![v(3, 1, 5)]),
            (vec![h(1, 6, 6)], vec![v(3, 1, 5)]),
            (vec![h(1, 4, 6)], vec![v(3, 1, 5)]),
        ] {
            assert_matches_grid(&horizontal, &vertical, &mut scratch);
        }
    }

    #[test]
    fn closed_endpoints_clipping_and_input_permutations_match_grid() {
        let mut horizontal = [
            h(0, 1, 2),
            h(4, 1, 64),
            h(0, 2, 64),
            h(1, 3, 1),
            h(63, 63, 64),
            h(0, 5, 0),
            h(64, 6, 64),
            h(5, 7, 4),
        ];
        let mut vertical = [
            v(63, 63, 64),
            v(1, 0, 3),
            v(2, 1, 2),
            v(4, 1, 2),
            v(63, 0, 2),
            v(1, 0, 0),
            v(2, 64, 64),
            v(3, 7, 4),
        ];
        let mut scratch = ConflictScratch::default();
        for _ in 0..horizontal.len() {
            assert_matches_grid(&horizontal, &vertical, &mut scratch);
            assert_matching_selection(&horizontal, &vertical);
            horizontal.rotate_left(1);
            vertical.rotate_right(1);
        }
    }

    #[test]
    fn shuffled_nonoverlapping_segments_match_grid() {
        let mut random = 0x5a72_10c3_u32;
        let mut scratch = ConflictScratch::default();
        for _ in 0..64 {
            let mut horizontal = Vec::new();
            let mut vertical = Vec::new();
            for coordinate in 1u8..64 {
                let mut start = 0u8;
                while start < 64 {
                    let length = (next_random(&mut random) % 9) as u8;
                    let end = start.saturating_add(length).min(64);
                    if next_random(&mut random) & 1 != 0 {
                        horizontal.push(h(start, coordinate, end));
                    }
                    if next_random(&mut random) & 1 != 0 {
                        vertical.push(v(coordinate, start, end));
                    }
                    start = end.saturating_add(2);
                }
            }
            shuffle(&mut horizontal, &mut random);
            shuffle(&mut vertical, &mut random);
            assert_matches_grid(&horizontal, &vertical, &mut scratch);
            assert_matching_selection(&horizontal, &vertical);
        }
    }

    fn next_random(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    fn shuffle<T>(items: &mut [T], state: &mut u32) {
        for end in (1..items.len()).rev() {
            let other = (next_random(state) as usize) % (end + 1);
            items.swap(end, other);
        }
    }

    /// 旧逐格展开实现：右顶点按输入顺序，每段按相交 y 升序输出。
    #[allow(clippy::large_stack_arrays)]
    fn grid_edges(
        horizontal: &[EffectiveChord],
        vertical: &[EffectiveChord],
    ) -> (Vec<(u16, u16)>, Vec<u8>) {
        let mut grid = [UNMATCHED_U16; IMAGE64_AXIS_LEN * IMAGE64_AXIS_LEN];
        for (left, chord) in horizontal.iter().enumerate() {
            for x in usize::from(chord.x1).max(1)..=usize::from(chord.x2).min(63) {
                let slot = usize::from(chord.y1) * IMAGE64_AXIS_LEN + x;
                assert_eq!(grid[slot], UNMATCHED_U16);
                grid[slot] = u16_index(left);
            }
        }
        let mut edges = Vec::new();
        let mut degrees = vec![0; vertical.len()];
        for (right, chord) in vertical.iter().enumerate() {
            for y in usize::from(chord.y1).max(1)..=usize::from(chord.y2).min(63) {
                let left = grid[y * IMAGE64_AXIS_LEN + usize::from(chord.x1)];
                if left != UNMATCHED_U16 {
                    edges.push((left, u16_index(right)));
                    degrees[right] += 1;
                }
            }
        }
        (edges, degrees)
    }

    fn assert_matches_grid(
        horizontal: &[EffectiveChord],
        vertical: &[EffectiveChord],
        scratch: &mut ConflictScratch,
    ) {
        let (expected_edges, expected_degrees) = grid_edges(horizontal, vertical);
        let mut expected = vec![Vec::<u16>::new(); horizontal.len()];
        for &(left, right) in &expected_edges {
            expected[usize::from(left)].push(right);
        }
        for row in &mut expected {
            row.sort_unstable_by_key(|&right| (expected_degrees[usize::from(right)], right));
        }
        let (actual, degrees) = build_sparse_conflict_graph_csr(horizontal, vertical, scratch);
        assert_eq!(degrees, expected_degrees);
        for (left, row) in expected.iter().enumerate() {
            assert_eq!(actual.neighbors(left), row);
        }
        let mut actual_edges: Vec<_> = scratch.finalize.edge_buffer[..scratch.finalize.edge_count]
            .iter()
            .map(|edge| (edge.left, edge.right))
            .collect();
        actual_edges.sort_by_key(|&(_, right)| right);
        assert_eq!(actual_edges, expected_edges);
    }

    fn assert_matching_selection(horizontal: &[EffectiveChord], vertical: &[EffectiveChord]) {
        let (edges, degrees) = grid_edges(horizontal, vertical);
        if edges.is_empty() {
            return;
        }
        let mut legacy = ConflictScratch::default();
        legacy.finalize.edge_count = edges.len();
        legacy.finalize.right_degrees.extend_from_slice(&degrees);
        for (slot, (left, right)) in legacy.finalize.edge_buffer.iter_mut().zip(edges) {
            *slot = SparseEdge { left, right };
        }
        let (expected, right_degrees) = build_sparse_adjacency_in_scratch(
            horizontal.len(),
            vertical.len(),
            &mut legacy.finalize,
        );
        let mut expected_matching = HkScratch::default();
        crate::hk::hopcroft_karp_dw_sparse_csr_u16(
            &expected,
            vertical.len(),
            right_degrees,
            &mut expected_matching,
        );
        let mut actual = MatchingScratch::default();
        actual.select_maximum_independent_set(horizontal, vertical);
        assert_eq!(
            actual.hk.pair_left.as_slice(),
            expected_matching.pair_left.as_slice()
        );
        assert_eq!(
            actual.hk.pair_right.as_slice(),
            expected_matching.pair_right.as_slice()
        );
        assert_eq!(
            actual.hk.reachable_left.as_slice(),
            expected_matching.reachable_left.as_slice()
        );
        assert_eq!(
            actual.hk.reachable_right.as_slice(),
            expected_matching.reachable_right.as_slice()
        );
    }

    #[test]
    fn linear_scatter_matches_comparison_sort_with_ties_and_empty_vertices() {
        let mut storage = MatchingScratch::default();
        let scratch = &mut storage.conflict.finalize;
        let capacity = scratch.right_layout.capacity();
        let mut columns = [
            vec![],
            (0..63).map(u16_index).collect(),
            vec![0],
            (0..63).step_by(2).map(u16_index).collect(),
            (0..63).map(u16_index).collect(),
            vec![0, 62],
            vec![],
        ];
        for _ in 0..columns.len() {
            let mut expected = vec![Vec::<u16>::new(); 64];
            scratch.edge_count = 0;
            scratch.right_degrees.clear();
            scratch.right_degrees.resize(columns.len(), 0);
            // 扫线可以按 x 输出右组，而非按原始右编号输出。
            for right in [4, 1, 5, 0, 6, 3, 2] {
                let neighbors = &columns[right];
                let mut degree = 0u8;
                for &left in neighbors {
                    scratch.edge_buffer[scratch.edge_count] = SparseEdge {
                        left,
                        right: u16_index(right),
                    };
                    scratch.edge_count += 1;
                    expected[usize::from(left)].push(u16_index(right));
                    degree += 1;
                }
                scratch.right_degrees[right] = degree;
            }
            for row in &mut expected {
                row.sort_unstable_by_key(|&right| {
                    (scratch.right_degrees[usize::from(right)], right)
                });
            }
            let (actual, _) = build_sparse_adjacency_in_scratch(64, columns.len(), scratch);
            for (left, row) in expected.iter().enumerate() {
                assert_eq!(actual.neighbors(left), row);
            }
            columns.rotate_left(1);
        }
        assert_eq!(scratch.right_layout.capacity(), capacity);
        scratch.edge_count = 0;
        scratch.right_degrees.clear();
        let (empty, _) = build_sparse_adjacency_in_scratch(0, 0, scratch);
        assert_eq!(empty.offsets, &[0]);
        assert_eq!(empty.edges, []);
    }
}
