//! 贪心匹配初始化：度 1 匹配、按右部度数排序、3 跳增广。

use crate::fixed::FixedVec;
use crate::matching::{
    ChordBuffer, HkScratch, IMAGE64_AXIS_LIMIT, SparseAdjacencyRef, UNMATCHED_U16,
};
use crate::{copy, get_mut, slice, u16_index};

pub fn greedy_augment_len3_sparse_csr_u16(
    adjacency: &SparseAdjacencyRef<'_>,
    pair_left: &mut [u16],
    pair_right: &mut [u16],
    next_edges: &mut ChordBuffer<u16>,
) {
    next_edges.clear();
    next_edges.extend_from_slice(slice(adjacency.offsets, 0..pair_left.len()));
    'lefts: for left in 0..pair_left.len() {
        if copy(pair_left, left) != UNMATCHED_U16 {
            continue;
        }

        for &right in adjacency.neighbors(left) {
            let right = usize::from(right);
            let matched_left = copy(pair_right, right);
            if matched_left == UNMATCHED_U16 {
                *get_mut(pair_left, left) = u16_index(right);
                *get_mut(pair_right, right) = u16_index(left);
                break;
            }

            let matched_left = usize::from(matched_left);
            let end = copy(adjacency.offsets, matched_left + 1);
            let next_edge = get_mut(next_edges, matched_left);
            // 长度 1/3 的增广只会减少空闲右顶点，已检查的邻居不会再次可用。
            // 成功选中的右顶点也立即占用，所以每条内层边至多检查一次。
            while *next_edge < end {
                let alternate_right = copy(adjacency.edges, usize::from(*next_edge));
                *next_edge += 1;
                let alternate_right = usize::from(alternate_right);
                if copy(pair_right, alternate_right) != UNMATCHED_U16 {
                    continue;
                }

                *get_mut(pair_left, matched_left) = u16_index(alternate_right);
                *get_mut(pair_right, alternate_right) = u16_index(matched_left);
                *get_mut(pair_left, left) = u16_index(right);
                *get_mut(pair_right, right) = u16_index(left);
                continue 'lefts;
            }
        }
    }
}

pub fn greedy_initialize_sparse_matching_csr_u16(
    adjacency: &SparseAdjacencyRef<'_>,
    right_degrees: &[u8],
    scratch: &mut HkScratch,
) {
    let pair_left = &mut scratch.pair_left;
    let pair_right = &mut scratch.pair_right;
    for left in 0..pair_left.len() {
        let start = copy(adjacency.offsets, left);
        if copy(adjacency.offsets, left + 1) - start != 1 {
            continue;
        }

        let right = usize::from(copy(adjacency.edges, usize::from(start)));
        if copy(pair_right, right) == UNMATCHED_U16 {
            *get_mut(pair_left, left) = u16_index(right);
            *get_mut(pair_right, right) = u16_index(left);
        }
    }

    scratch.transpose_offsets.clear();
    scratch.transpose_offsets.resize(right_degrees.len() + 1, 0);
    for (right, &degree) in right_degrees.iter().enumerate() {
        let offset = copy(&scratch.transpose_offsets, right) + u16::from(degree);
        *get_mut(&mut scratch.transpose_offsets, right + 1) = offset;
    }
    scratch.transpose_edges.clear();
    scratch.transpose_edges.resize(adjacency.edges.len(), 0);

    scratch.write_offsets.clear();
    scratch
        .write_offsets
        .extend_from_slice(slice(&scratch.transpose_offsets, 0..right_degrees.len()));
    for left in 0..pair_left.len() {
        for &right in adjacency.neighbors(left) {
            let right = usize::from(right);
            let slot = copy(&scratch.write_offsets, right);
            *get_mut(&mut scratch.transpose_edges, usize::from(slot)) = u16_index(left);
            *get_mut(&mut scratch.write_offsets, right) += 1;
        }
    }

    counting_sort_right_order_u16(
        right_degrees,
        &mut scratch.right_order,
        &mut scratch.right_degree_counts,
    );

    for &right in &scratch.right_order {
        let right = usize::from(right);
        if copy(pair_right, right) != UNMATCHED_U16 {
            continue;
        }
        let start = copy(&scratch.transpose_offsets, right);
        let end = copy(&scratch.transpose_offsets, right + 1);
        for &left in slice(
            &scratch.transpose_edges,
            usize::from(start)..usize::from(end),
        ) {
            let left = usize::from(left);
            if copy(pair_left, left) == UNMATCHED_U16 {
                *get_mut(pair_left, left) = u16_index(right);
                *get_mut(pair_right, right) = u16_index(left);
                break;
            }
        }
    }
}

fn counting_sort_right_order_u16(
    right_degrees: &[u8],
    right_order: &mut ChordBuffer<u16>,
    degree_counts: &mut FixedVec<usize, IMAGE64_AXIS_LIMIT>,
) {
    let max_degree = usize::from(right_degrees.iter().copied().max().unwrap_or(0));
    degree_counts.clear();
    degree_counts.resize(max_degree + 1, 0);
    for &degree in right_degrees {
        *get_mut(degree_counts, usize::from(degree)) += 1;
    }

    let mut offset = 0usize;
    for count in degree_counts.iter_mut() {
        let current = *count;
        *count = offset;
        offset += current;
    }

    right_order.clear();
    right_order.resize(right_degrees.len(), 0);
    for (right, &degree) in right_degrees.iter().enumerate() {
        let slot = get_mut(degree_counts, usize::from(degree));
        *get_mut(right_order, *slot) = u16_index(right);
        *slot += 1;
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)] // 穷举图与配对的编号由显式尺寸界定。
mod tests {
    use std::{vec, vec::Vec};

    use super::*;

    // 独立保留逐次重扫的参考实现，验证游标不会改变任何配对选择。
    fn augment_len3_reference(
        adjacency: &SparseAdjacencyRef<'_>,
        pair_left: &mut [u16],
        pair_right: &mut [u16],
    ) {
        'lefts: for left in 0..pair_left.len() {
            if pair_left[left] != UNMATCHED_U16 {
                continue;
            }
            for &right in adjacency.neighbors(left) {
                let right = usize::from(right);
                let matched_left = pair_right[right];
                if matched_left == UNMATCHED_U16 {
                    pair_left[left] = u16_index(right);
                    pair_right[right] = u16_index(left);
                    break;
                }
                let matched_left = usize::from(matched_left);
                for &alternate_right in adjacency.neighbors(matched_left) {
                    let alternate_right = usize::from(alternate_right);
                    if alternate_right == right || pair_right[alternate_right] != UNMATCHED_U16 {
                        continue;
                    }
                    pair_left[matched_left] = u16_index(alternate_right);
                    pair_right[alternate_right] = u16_index(matched_left);
                    pair_left[left] = u16_index(right);
                    pair_right[right] = u16_index(left);
                    continue 'lefts;
                }
            }
        }
    }

    #[test]
    fn monotone_cursors_match_rescanning_for_every_three_by_three_seed() {
        let mut next_edges = ChordBuffer::new(0);
        for graph_bits in 0u16..(1 << 9) {
            let mut offsets = vec![0];
            let mut edges = Vec::new();
            for left in 0..3 {
                for right in 0..3 {
                    if graph_bits & (1 << (left * 3 + right)) != 0 {
                        edges.push(u16_index(right));
                    }
                }
                offsets.push(u16_index(edges.len()));
            }
            let adjacency = SparseAdjacencyRef {
                offsets: &offsets,
                edges: &edges,
            };

            // 每个左顶点可为空闲或选择三个右顶点，过滤非边与重复配对。
            for seed_bits in 0usize..64 {
                let mut seed = seed_bits;
                let mut pair_left = [UNMATCHED_U16; 3];
                let mut pair_right = [UNMATCHED_U16; 3];
                let mut valid = true;
                for (left, matched) in pair_left.iter_mut().enumerate() {
                    let choice = seed % 4;
                    seed /= 4;
                    if choice == 0 {
                        continue;
                    }
                    let right = choice - 1;
                    if pair_right[right] != UNMATCHED_U16
                        || !adjacency.neighbors(left).contains(&u16_index(right))
                    {
                        valid = false;
                        break;
                    }
                    *matched = u16_index(right);
                    pair_right[right] = u16_index(left);
                }
                if !valid {
                    continue;
                }
                let mut expected_left = pair_left;
                let mut expected_right = pair_right;
                augment_len3_reference(&adjacency, &mut expected_left, &mut expected_right);
                greedy_augment_len3_sparse_csr_u16(
                    &adjacency,
                    &mut pair_left,
                    &mut pair_right,
                    &mut next_edges,
                );
                assert_eq!(
                    pair_left, expected_left,
                    "graph={graph_bits}, seed={seed_bits}"
                );
                assert_eq!(
                    pair_right, expected_right,
                    "graph={graph_bits}, seed={seed_bits}"
                );
            }
        }
    }

    #[test]
    fn cursor_continues_after_multiple_successful_moves_of_one_left() {
        let offsets = [0, 7, 8, 9, 10, 11, 12, 13];
        let edges = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5];
        let adjacency = SparseAdjacencyRef {
            offsets: &offsets,
            edges: &edges,
        };
        let mut pair_left = [UNMATCHED_U16; 7];
        let mut pair_right = [UNMATCHED_U16; 7];
        pair_left[0] = 0;
        pair_right[0] = 0;
        let mut next_edges = ChordBuffer::new(0);
        greedy_augment_len3_sparse_csr_u16(
            &adjacency,
            &mut pair_left,
            &mut pair_right,
            &mut next_edges,
        );
        assert_eq!(pair_left, [6, 0, 1, 2, 3, 4, 5]);
        assert_eq!(pair_right, [1, 2, 3, 4, 5, 6, 0]);
        assert_eq!(next_edges[0], offsets[1]);
    }
}
