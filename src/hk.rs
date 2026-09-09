//! 稀疏 Hopcroft-Karp 算法。

#[cfg(feature = "profile")]
use std::time::Instant;

use crate::greedy::{
    greedy_augment_len3_sparse_csr_u16, greedy_initialize_sparse_matching_csr_u16,
};
use crate::matching::{HkScratch, SparseAdjacencyRef, UNMATCHED_U16};
#[cfg(feature = "profile")]
use crate::matching::{MatchingCounts, MatchingTimings};
use crate::{copy, get_mut, u16_index};

pub fn hopcroft_karp_sparse_csr_u16(
    adjacency: &SparseAdjacencyRef<'_>,
    right_size: usize,
    right_degrees: &[u8],
    scratch: &mut HkScratch,
) {
    let left_size = adjacency.offsets.len().saturating_sub(1);
    scratch.pair_left.clear();
    scratch.pair_left.resize(left_size, UNMATCHED_U16);
    scratch.pair_right.clear();
    scratch.pair_right.resize(right_size, UNMATCHED_U16);
    greedy_initialize_sparse_matching_csr_u16(adjacency, right_degrees, scratch);
    greedy_augment_len3_sparse_csr_u16(adjacency, &mut scratch.pair_left, &mut scratch.pair_right);
    scratch.distance.clear();
    scratch.distance.resize(left_size, u16::MAX);
    scratch.queue.clear();
    scratch.next_edge.clear();
    scratch.next_edge.resize(left_size, 0usize);
    scratch.unmatched_lefts.clear();
    scratch.reachable_left.clear();
    scratch.reachable_left.resize(left_size, false);
    scratch.reachable_right.clear();
    scratch.reachable_right.resize(right_size, false);
    scratch.unmatched_lefts.extend(
        scratch
            .pair_left
            .iter()
            .enumerate()
            .filter_map(|(left, &matched)| (matched == UNMATCHED_U16).then_some(u16_index(left))),
    );
    counting_sort_unmatched_lefts_by_degree(adjacency, scratch);
    scratch.touched_lefts.clear();

    loop {
        if scratch.unmatched_lefts.is_empty() {
            break;
        }

        for &left in &scratch.touched_lefts {
            *get_mut(&mut scratch.distance, usize::from(left)) = u16::MAX;
        }
        scratch.touched_lefts.clear();
        scratch.queue.clear();
        let mut head = 0usize;
        let mut found_augmenting_path = false;

        for &left in &scratch.unmatched_lefts {
            let left = usize::from(left);
            *get_mut(&mut scratch.distance, left) = 0;
            *get_mut(&mut scratch.next_edge, left) = copy(adjacency.offsets, left);
            scratch.touched_lefts.push(u16_index(left));
            scratch.queue.push(u16_index(left));
        }

        while head < scratch.queue.len() {
            let left = usize::from(copy(&scratch.queue, head));
            head += 1;
            for &right in adjacency.neighbors(left) {
                let right = usize::from(right);
                let matched = copy(&scratch.pair_right, right);
                if matched == UNMATCHED_U16 {
                    found_augmenting_path = true;
                } else {
                    let next_left = usize::from(matched);
                    if copy(&scratch.distance, next_left) == u16::MAX {
                        *get_mut(&mut scratch.distance, next_left) =
                            copy(&scratch.distance, left) + 1;
                        *get_mut(&mut scratch.next_edge, next_left) =
                            copy(adjacency.offsets, next_left);
                        scratch.touched_lefts.push(u16_index(next_left));
                        scratch.queue.push(u16_index(next_left));
                    }
                }
            }
        }

        if !found_augmenting_path {
            for &left in &scratch.touched_lefts {
                *get_mut(&mut scratch.reachable_left, usize::from(left)) = true;
            }
            for &left in &scratch.touched_lefts {
                let left = usize::from(left);
                for &right in adjacency.neighbors(left) {
                    let right = usize::from(right);
                    if copy(&scratch.pair_left, left) != u16_index(right) {
                        *get_mut(&mut scratch.reachable_right, right) = true;
                    }
                }
            }
            break;
        }

        let unmatched_len = scratch.unmatched_lefts.len();
        for i in 0..unmatched_len {
            let left = usize::from(copy(&scratch.unmatched_lefts, i));
            if copy(&scratch.pair_left, left) == UNMATCHED_U16 {
                dfs(left, adjacency, scratch);
            }
        }

        let pair_left = &scratch.pair_left;
        scratch
            .unmatched_lefts
            .retain(|&left| copy(pair_left, usize::from(left)) == UNMATCHED_U16);
    }
}

#[cfg(feature = "profile")]
pub fn hopcroft_karp_sparse_csr_u16_profile(
    adjacency: &SparseAdjacencyRef<'_>,
    right_size: usize,
    right_degrees: &[u8],
    scratch: &mut HkScratch,
) -> (MatchingTimings, MatchingCounts) {
    let total_start = Instant::now();
    let left_size = adjacency.offsets.len().saturating_sub(1);
    scratch.pair_left.clear();
    scratch.pair_left.resize(left_size, UNMATCHED_U16);
    scratch.pair_right.clear();
    scratch.pair_right.resize(right_size, UNMATCHED_U16);

    let greedy_start = Instant::now();
    greedy_initialize_sparse_matching_csr_u16(adjacency, right_degrees, scratch);
    greedy_augment_len3_sparse_csr_u16(adjacency, &mut scratch.pair_left, &mut scratch.pair_right);
    let greedy = greedy_start.elapsed();
    let greedy_matches = scratch
        .pair_left
        .iter()
        .filter(|&&right| right != UNMATCHED_U16)
        .count();

    let build_start = Instant::now();
    scratch.distance.clear();
    scratch.distance.resize(left_size, u16::MAX);
    scratch.queue.clear();
    scratch.next_edge.clear();
    scratch.next_edge.resize(left_size, 0usize);
    scratch.unmatched_lefts.clear();
    scratch.reachable_left.clear();
    scratch.reachable_left.resize(left_size, false);
    scratch.reachable_right.clear();
    scratch.reachable_right.resize(right_size, false);
    scratch.unmatched_lefts.extend(
        scratch
            .pair_left
            .iter()
            .enumerate()
            .filter_map(|(left, &matched)| (matched == UNMATCHED_U16).then_some(u16_index(left))),
    );
    counting_sort_unmatched_lefts_by_degree(adjacency, scratch);
    scratch.touched_lefts.clear();
    let dfs_build = build_start.elapsed();

    let mut bfs = std::time::Duration::ZERO;
    let mut dfs_search = std::time::Duration::ZERO;
    let mut cover = std::time::Duration::ZERO;
    let mut phases = 0usize;
    let mut augmentations = 0usize;

    loop {
        if scratch.unmatched_lefts.is_empty() {
            break;
        }

        phases += 1;
        let bfs_start = Instant::now();
        let found_augmenting_path = build_profile_levels(adjacency, scratch);
        bfs += bfs_start.elapsed();

        if !found_augmenting_path {
            let cover_start = Instant::now();
            for &left in &scratch.touched_lefts {
                *get_mut(&mut scratch.reachable_left, usize::from(left)) = true;
            }
            for &left in &scratch.touched_lefts {
                let left = usize::from(left);
                for &right in adjacency.neighbors(left) {
                    let right = usize::from(right);
                    if copy(&scratch.pair_left, left) != u16_index(right) {
                        *get_mut(&mut scratch.reachable_right, right) = true;
                    }
                }
            }
            cover += cover_start.elapsed();
            break;
        }

        let dfs_start = Instant::now();
        let unmatched_len = scratch.unmatched_lefts.len();
        for i in 0..unmatched_len {
            let left = usize::from(copy(&scratch.unmatched_lefts, i));
            if copy(&scratch.pair_left, left) == UNMATCHED_U16 && dfs(left, adjacency, scratch) {
                augmentations += 1;
            }
        }

        let pair_left = &scratch.pair_left;
        scratch
            .unmatched_lefts
            .retain(|&left| copy(pair_left, usize::from(left)) == UNMATCHED_U16);
        dfs_search += dfs_start.elapsed();
    }

    (
        MatchingTimings {
            total: total_start.elapsed(),
            greedy,
            bfs,
            dfs_build,
            dfs_search,
            cover,
            collect: std::time::Duration::ZERO,
        },
        MatchingCounts {
            greedy_matches,
            phases,
            augmentations,
        },
    )
}

#[cfg(feature = "profile")]
fn build_profile_levels(adjacency: &SparseAdjacencyRef<'_>, scratch: &mut HkScratch) -> bool {
    for &left in &scratch.touched_lefts {
        *get_mut(&mut scratch.distance, usize::from(left)) = u16::MAX;
    }
    scratch.touched_lefts.clear();
    scratch.queue.clear();
    let mut head = 0usize;
    let mut found_augmenting_path = false;

    for &left in &scratch.unmatched_lefts {
        let left = usize::from(left);
        *get_mut(&mut scratch.distance, left) = 0;
        *get_mut(&mut scratch.next_edge, left) = copy(adjacency.offsets, left);
        scratch.touched_lefts.push(u16_index(left));
        scratch.queue.push(u16_index(left));
    }

    while head < scratch.queue.len() {
        let left = usize::from(copy(&scratch.queue, head));
        head += 1;
        for &right in adjacency.neighbors(left) {
            let right = usize::from(right);
            let matched = copy(&scratch.pair_right, right);
            if matched == UNMATCHED_U16 {
                found_augmenting_path = true;
            } else {
                let next_left = usize::from(matched);
                if copy(&scratch.distance, next_left) == u16::MAX {
                    *get_mut(&mut scratch.distance, next_left) = copy(&scratch.distance, left) + 1;
                    *get_mut(&mut scratch.next_edge, next_left) =
                        copy(adjacency.offsets, next_left);
                    scratch.touched_lefts.push(u16_index(next_left));
                    scratch.queue.push(u16_index(next_left));
                }
            }
        }
    }
    found_augmenting_path
}

fn counting_sort_unmatched_lefts_by_degree(
    adjacency: &SparseAdjacencyRef<'_>,
    scratch: &mut HkScratch,
) {
    let max_degree = scratch
        .unmatched_lefts
        .iter()
        .map(|&left| left_degree(adjacency, usize::from(left)))
        .max()
        .unwrap_or(0);
    scratch.left_degree_counts.clear();
    scratch.left_degree_counts.resize(max_degree + 1, 0);
    for &left in &scratch.unmatched_lefts {
        *get_mut(
            &mut scratch.left_degree_counts,
            left_degree(adjacency, usize::from(left)),
        ) += 1;
    }

    let mut offset = 0usize;
    for count in &mut scratch.left_degree_counts {
        let current = *count;
        *count = offset;
        offset += current;
    }

    scratch.queue.clear();
    scratch.queue.resize(scratch.unmatched_lefts.len(), 0);
    for &left in &scratch.unmatched_lefts {
        let degree = left_degree(adjacency, usize::from(left));
        let slot = get_mut(&mut scratch.left_degree_counts, degree);
        *get_mut(&mut scratch.queue, *slot) = left;
        *slot += 1;
    }
    scratch.unmatched_lefts.copy_from_slice(&scratch.queue);
    scratch.queue.clear();
}

fn left_degree(adjacency: &SparseAdjacencyRef<'_>, left: usize) -> usize {
    copy(adjacency.offsets, left + 1) - copy(adjacency.offsets, left)
}

fn dfs(start_left: usize, adjacency: &SparseAdjacencyRef<'_>, scratch: &mut HkScratch) -> bool {
    let stack = &mut scratch.dfs_stack;
    stack.clear();
    stack.push((start_left, copy(&scratch.next_edge, start_left)));

    loop {
        let (left, mut edge) = copy(stack, stack.len() - 1);
        let end = copy(adjacency.offsets, left + 1);
        let mut found = false;

        while edge < end {
            let right = usize::from(copy(adjacency.edges, edge));
            edge += 1;
            let matched = copy(&scratch.pair_right, right);

            if matched == UNMATCHED_U16 {
                *get_mut(&mut scratch.pair_left, left) = u16_index(right);
                *get_mut(&mut scratch.pair_right, right) = u16_index(left);
                let n = stack.len() - 1;
                get_mut(stack, n).1 = edge;
                for i in (0..n).rev() {
                    let (prev_left, prev_edge) = copy(stack, i);
                    let prev_right = copy(adjacency.edges, prev_edge - 1);
                    *get_mut(&mut scratch.pair_left, prev_left) = prev_right;
                    *get_mut(&mut scratch.pair_right, usize::from(prev_right)) =
                        u16_index(prev_left);
                    *get_mut(&mut scratch.next_edge, prev_left) = prev_edge;
                }
                return true;
            }

            let next_left = usize::from(matched);
            if copy(&scratch.distance, next_left) == copy(&scratch.distance, left) + 1 {
                let n = stack.len() - 1;
                get_mut(stack, n).1 = edge;
                stack.push((next_left, copy(&scratch.next_edge, next_left)));
                found = true;
                break;
            }
        }

        if !found {
            *get_mut(&mut scratch.distance, left) = u16::MAX;
            *get_mut(&mut scratch.next_edge, left) = edge;
            stack.pop();
            if stack.is_empty() {
                return false;
            }
        }
    }
}
