//! 稀疏冲突图构建：将 chord 列表转为 CSR 邻接。

use crate::matching::{
    ConflictFinalizeScratch, ConflictScratch, IMAGE64_AXIS_LEN, IMAGE64_AXIS_LIMIT,
    IMAGE64_MAX_CONFLICT_EDGES, SparseAdjacencyRef, SparseEdge, UNMATCHED_U16,
};
use crate::types::ChordAccess;
use crate::{copy, get_mut, slice, slice_mut, u16_index, u32_index};

const IMAGE64_INTERNAL_MIN: usize = 1;
const IMAGE64_INTERNAL_MAX: usize = IMAGE64_AXIS_LIMIT - 1;

const fn grid_index(x: usize, y: usize) -> usize {
    y * IMAGE64_AXIS_LEN + x
}

fn sort_right_slice_by_key(slice: &mut [u16], right_keys: &[u32]) {
    if slice.len() <= 1 {
        return;
    }

    if slice.len() <= 48 {
        for index in 1..slice.len() {
            let value = copy(slice, index);
            let value_key = copy(right_keys, usize::from(value));
            let mut cursor = index;
            while cursor > 0 && copy(right_keys, usize::from(copy(slice, cursor - 1))) > value_key {
                *get_mut(slice, cursor) = copy(slice, cursor - 1);
                cursor -= 1;
            }
            *get_mut(slice, cursor) = value;
        }
        return;
    }

    slice.sort_unstable_by_key(|&right| copy(right_keys, usize::from(right)));
}

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
    let offsets = &mut scratch.adjacency_offsets;
    for edge in edges {
        *get_mut(offsets, usize::from(edge.left) + 1) += 1;
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
    for edge in edges {
        let left = usize::from(edge.left);
        let slot = copy(&scratch.next_offsets, left);
        *get_mut(compact_edges, slot) = edge.right;
        *get_mut(&mut scratch.next_offsets, left) += 1;
    }

    scratch.right_keys.clear();
    scratch.right_keys.resize(right_size, 0);
    for (right, &degree) in right_degrees.iter().enumerate() {
        *get_mut(&mut scratch.right_keys, right) = (u32::from(degree) << 16) | u32_index(right);
    }

    for left in 0..left_size {
        let start = copy(offsets, left);
        let end = copy(offsets, left + 1);
        sort_right_slice_by_key(slice_mut(compact_edges, start..end), &scratch.right_keys);
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
    build_sparse_conflict_graph_grid_csr(horizontal_edges, vertical_edges, &mut scratch.finalize);
    build_sparse_adjacency_in_scratch(
        horizontal_edges.len(),
        vertical_edges.len(),
        &mut scratch.finalize,
    )
}

fn build_sparse_conflict_graph_grid_csr<H: ChordAccess, V: ChordAccess>(
    horizontal_edges: &[H],
    vertical_edges: &[V],
    scratch: &mut ConflictFinalizeScratch,
) {
    reset_grid_scratch(scratch);
    scratch.edge_count = 0;
    scratch.right_degrees.clear();
    scratch.right_degrees.resize(vertical_edges.len(), 0);

    for (index, horizontal) in horizontal_edges.iter().enumerate() {
        let horizontal = horizontal.chord();
        debug_assert_eq!(horizontal.y1, horizontal.y2);
        let y = usize::from(horizontal.y1);
        debug_assert!((IMAGE64_INTERNAL_MIN..=IMAGE64_INTERNAL_MAX).contains(&y));
        for x in internal_range(horizontal.x1, horizontal.x2) {
            let slot = grid_index(x, y);
            debug_assert_ne!(
                copy(&scratch.horizontal_grid_marks, slot),
                scratch.grid_mark
            );
            *get_mut(&mut scratch.horizontal_grid, slot) = u16_index(index);
            *get_mut(&mut scratch.horizontal_grid_marks, slot) = scratch.grid_mark;
            if copy(&scratch.horizontal_x_marks, x) != scratch.grid_mark {
                *get_mut(&mut scratch.horizontal_x_marks, x) = scratch.grid_mark;
                *get_mut(&mut scratch.horizontal_y_masks, x) = 0;
            }
            *get_mut(&mut scratch.horizontal_y_masks, x) |= 1u64 << y;
        }
    }

    for (index, vertical) in vertical_edges.iter().enumerate() {
        let vertical = vertical.chord();
        debug_assert_eq!(vertical.x1, vertical.x2);
        let x = usize::from(vertical.x1);
        debug_assert!((IMAGE64_INTERNAL_MIN..=IMAGE64_INTERNAL_MAX).contains(&x));
        let mut active = if copy(&scratch.horizontal_x_marks, x) == scratch.grid_mark {
            copy(&scratch.horizontal_y_masks, x) & internal_mask(vertical.y1, vertical.y2)
        } else {
            0
        };
        while active != 0 {
            let y =
                usize::try_from(active.trailing_zeros()).unwrap_or_else(|_| std::process::abort());
            let slot = grid_index(x, y);
            debug_assert_eq!(
                copy(&scratch.horizontal_grid_marks, slot),
                scratch.grid_mark
            );
            let left = copy(&scratch.horizontal_grid, slot);
            debug_assert_ne!(left, UNMATCHED_U16);
            push_conflict_edge(
                &mut scratch.edge_buffer,
                &mut scratch.edge_count,
                SparseEdge {
                    left,
                    right: u16_index(index),
                },
            );
            *get_mut(&mut scratch.right_degrees, index) += 1;
            active &= active - 1;
        }
    }
}

fn push_conflict_edge(
    edge_buffer: &mut [SparseEdge; IMAGE64_MAX_CONFLICT_EDGES],
    edge_count: &mut usize,
    edge: SparseEdge,
) {
    // 每条冲突边注入到唯一内部格点，64x64 输入最多 63*63 条。
    *get_mut(edge_buffer, *edge_count) = edge;
    *edge_count += 1;
}

fn reset_grid_scratch(scratch: &mut ConflictFinalizeScratch) {
    scratch.grid_mark = scratch.grid_mark.wrapping_add(1);
    if scratch.grid_mark != 0 {
        return;
    }

    scratch.grid_mark = 1;
    scratch.horizontal_grid_marks.fill(0);
    scratch.horizontal_x_marks.fill(0);
}

fn internal_range(start: u8, end: u8) -> std::ops::RangeInclusive<usize> {
    let start = usize::from(start).max(IMAGE64_INTERNAL_MIN);
    let end = usize::from(end).min(IMAGE64_INTERNAL_MAX);
    start..=end
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
