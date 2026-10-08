//! 固定 64x64 sparse quad 输入的矩形分解。
//!
//! 输入为非零 dyadic square 列表或连续标签切面，内部派生区间、chord 与 cut。
//! 少量叶子使用区间事件；大量叶子使用 SIMD 栅格扫描。

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::num::NonZeroU16;
#[cfg(feature = "alloc")]
use core::simd::u16x32;
#[cfg(feature = "profile")]
use std::time::{Duration, Instant};

use crate::fixed::FixedVec;
use crate::matching::{ChordBuffer, MatchingScratch};
#[cfg(feature = "alloc")]
use crate::slice_mut;
use crate::types::{
    ActiveRect, ChordAccess, EffectiveChord, Orientation, PackedRectangles64, RangeU8, Rectangle,
    Run,
};
use crate::{get, slice};

mod dense;
pub use dense::DenseLabels64;

const EDGE: usize = 64;
const EDGE_U8: u8 = 64;
const MAX_LOD: u8 = 6;
const MAX_AXIS_INTERVALS: usize = EDGE * EDGE;
const MAX_RECTANGLES: usize = EDGE * EDGE;
const EMPTY_INTERVAL: Interval = Interval {
    start: 0,
    end: 0,
    value: 0,
};
const EMPTY_CHORD: ValuedChord = ValuedChord {
    value: 0,
    chord: EffectiveChord {
        orientation: Orientation::Horizontal,
        x1: 0,
        y1: 0,
        x2: 0,
        y2: 0,
    },
};

/// 64x64 sparse quad image 的非零叶子。
///
/// `lod` 表示正方形边长的 log2，`size = 1 << lod`。`value` 为非零 label，
/// 0 始终表示隐式 background。
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QuadLeaf64 {
    pub u: u8,
    pub v: u8,
    pub lod: u8,
    pub value: NonZeroU16,
}

/// sparse quad 输入或分解失败。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseQuadError {
    AllocationFailed,
    CapacityOverflow,
    LodOutOfRange,
    Misaligned,
    OutOfBounds,
    Overlap,
}

/// sparse 分解阶段耗时。
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default)]
pub struct SparseDecomposeTimings {
    pub total: Duration,
    pub build_axis: Duration,
    pub extract_chords: Duration,
    pub select_cuts: Duration,
    pub matching: Duration,
    pub matching_greedy: Duration,
    pub matching_bfs: Duration,
    pub matching_dfs_build: Duration,
    pub matching_dfs_search: Duration,
    pub matching_cover: Duration,
    pub matching_collect: Duration,
    pub emit_cuts: Duration,
    /// 兼容旧版 profiling 字段；切线直接合入位图，无需排序，始终为零。
    pub sort_cuts: Duration,
    pub partition: Duration,
}

/// sparse 分解阶段规模。
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default)]
pub struct SparseDecomposeCounts {
    pub leaves: usize,
    pub row_intervals: usize,
    pub column_intervals: usize,
    pub chord_groups: usize,
    pub horizontal_chords: usize,
    pub vertical_chords: usize,
    pub horizontal_cuts: usize,
    pub vertical_cuts: usize,
    pub rectangles: usize,
    pub greedy_matches: usize,
    pub matching_phases: usize,
    pub matching_augmentations: usize,
}

/// sparse 分解 profiling 结果。
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default)]
pub struct SparseDecomposeProfile {
    pub timings: SparseDecomposeTimings,
    pub counts: SparseDecomposeCounts,
}

#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default)]
struct SelectCutTimings {
    matching: Duration,
    matching_greedy: Duration,
    matching_bfs: Duration,
    matching_dfs_build: Duration,
    matching_dfs_search: Duration,
    matching_cover: Duration,
    matching_collect: Duration,
    emit_cuts: Duration,
    greedy_matches: usize,
    matching_phases: usize,
    matching_augmentations: usize,
}

/// 固定 64x64 sparse quad image。
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Default)]
pub struct SparseQuadImage64 {
    leaves: Vec<StoredLeaf>,
}

#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StoredLeaf {
    leaf: QuadLeaf64,
    morton_start: u16,
}

#[cfg(feature = "alloc")]
impl StoredLeaf {
    const fn morton_end(self) -> u16 {
        // 已验证的 dyadic square 占据一个连续 Morton 区间，终点最多为 4096。
        self.morton_start + (1u16 << (self.leaf.lod * 2))
    }
}

#[cfg(feature = "alloc")]
impl SparseQuadImage64 {
    /// 从非零 quad leaves 构造 sparse image。
    ///
    /// 输入不做预合并；只校验 dyadic 对齐、边界与互不重叠。
    ///
    /// # Errors
    ///
    /// leaf 越界、未按 lod 对齐、lod 超出范围、互相重叠或缓存无法分配时返回错误。
    pub fn from_leaves(leaves: &[QuadLeaf64]) -> Result<Self, SparseQuadError> {
        let mut stored = Vec::new();
        allocate_exact(&mut stored, leaves.len())?;

        let mut ordered = true;
        let mut previous_start = 0u16;
        if leaves.len() < EDGE {
            for &leaf in leaves {
                let validated = validate_leaf(leaf)?;
                ordered &= previous_start <= validated.morton_start;
                previous_start = validated.morton_start;
                stored.push(validated);
            }
        } else {
            let mut morton_starts = [0u16; EDGE];
            for chunk in leaves.chunks(EDGE) {
                batch_morton_starts(chunk, &mut morton_starts);
                for (&leaf, &morton_start) in chunk.iter().zip(&morton_starts) {
                    // 键计算没有 shape 错误；仍按输入顺序返回首个非法 leaf。
                    validate_leaf_shape(leaf)?;
                    ordered &= previous_start <= morton_start;
                    previous_start = morton_start;
                    stored.push(StoredLeaf { leaf, morton_start });
                }
            }
        }

        if !ordered {
            sort_stored_leaves_by_morton(&mut stored)?;
        }
        let mut previous_end = 0u16;
        for leaf in &stored {
            if previous_end > leaf.morton_start {
                return Err(SparseQuadError::Overlap);
            }
            previous_end = leaf.morton_end();
        }

        Ok(Self { leaves: stored })
    }

    /// 非零 leaf 数量。
    #[must_use]
    pub const fn len(&self) -> usize {
        self.leaves.len()
    }

    /// 是否为空图。
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }
}

/// sparse optimal 分解的固定容量内联 scratch。
///
/// 可以放在 worker 栈上复用；对象较大，调用时传递 `&mut`，避免按值传递。
pub struct SparseOptimalScratch64 {
    row_buckets: [IntervalBucket; EDGE],
    column_buckets: [IntervalBucket; EDGE],
    rows: AxisIntervals,
    columns: AxisIntervals,
    dense_labels: DenseLabels64,
    chord_groups: SparseChords,
    matching: MatchingScratch,
    horizontal_cuts: ChordBuffer<HorizontalCut>,
    vertical_cuts: ChordBuffer<VerticalCut>,
    horizontal_cut_masks: FixedVec<u64, EDGE>,
    vertical_cut_masks: FixedVec<u128, EDGE>,
    partition: PartitionScratch,
    rectangles: FixedVec<Rectangle, MAX_RECTANGLES>,
}

impl Default for SparseOptimalScratch64 {
    fn default() -> Self {
        const { Self::new() }
    }
}

impl SparseOptimalScratch64 {
    /// 创建可直接使用的内联缓存；初始化与借用分解均不分配堆内存。
    /// 在 worker 的任务循环外创建一次，并通过可变引用复用。
    #[must_use]
    #[allow(clippy::large_stack_arrays)] // 固定容量栈 scratch；整个初值在 const 块中求值。
    pub const fn new() -> Self {
        const {
            Self {
                row_buckets: [const { IntervalBucket::new() }; EDGE],
                column_buckets: [const { IntervalBucket::new() }; EDGE],
                rows: AxisIntervals::new(),
                columns: AxisIntervals::new(),
                dense_labels: DenseLabels64::new(),
                chord_groups: SparseChords::new(),
                matching: MatchingScratch::new(),
                horizontal_cuts: FixedVec::new(HorizontalCut {
                    y: 0,
                    start: 0,
                    end: 0,
                }),
                vertical_cuts: FixedVec::new(VerticalCut {
                    x: 0,
                    start: 0,
                    end: 0,
                }),
                horizontal_cut_masks: FixedVec::new(0),
                vertical_cut_masks: FixedVec::new(0),
                partition: PartitionScratch::new(),
                rectangles: FixedVec::new(Rectangle {
                    value: 0,
                    x: RangeU8::new(0, 0),
                    y: RangeU8::new(0, 0),
                }),
            }
        }
    }

    /// 兼容原预分配入口；现在等价于 `Ok(Self::new())`，不调用分配器。
    /// 新代码应直接使用 `new()`，避免大对象在 `Result` 中按值搬运。
    ///
    /// # Errors
    ///
    /// 当前固定容量实现始终返回 `Ok`。
    pub const fn try_new_preallocated() -> Result<Self, SparseQuadError> {
        const { Ok(Self::new()) }
    }

    /// 兼容原预分配入口；容量已内联，本方法不修改缓存或已有结果。
    ///
    /// # Errors
    ///
    /// 当前固定容量实现始终返回 `Ok`。
    pub const fn preallocate_64(&mut self) -> Result<(), SparseQuadError> {
        Ok(())
    }

    /// 对 sparse quad leaves 执行 64x64 最优分解。
    ///
    /// 输入不需要预排序；函数会校验 dyadic 对齐、边界与互不重叠。
    ///
    /// # Errors
    ///
    /// leaf 越界、未按 lod 对齐、lod 超出范围、互相重叠或内部缓存无法分配时返回错误。
    #[cfg(feature = "alloc")]
    pub fn decompose(&mut self, leaves: &[QuadLeaf64]) -> Result<Vec<Rectangle>, SparseQuadError> {
        Ok(self.decompose_borrowed(leaves)?.to_vec())
    }

    /// 对 sparse quad leaves 执行 64x64 最优分解，结果借用自 scratch。
    ///
    /// 资源契约：从 `new()` 创建后即可使用，本函数不调用堆分配器。
    ///
    /// # Errors
    ///
    /// leaf 越界、未按 lod 对齐、lod 超出范围、互相重叠或内部缓存无法分配时返回错误。
    pub fn decompose_borrowed(
        &mut self,
        leaves: &[QuadLeaf64],
    ) -> Result<&[Rectangle], SparseQuadError> {
        build_axis_intervals_from_leaves(
            leaves.iter().copied(),
            leaves.len(),
            &mut self.dense_labels,
            &mut self.rows,
            &mut self.columns,
            &mut self.row_buckets,
            &mut self.column_buckets,
        )?;
        self.decompose_intervals_borrowed()
    }

    /// 对 sparse leaves 分解，每个完成的矩形直接交给调用方。
    ///
    /// 输出顺序与 `decompose_borrowed` 相同；不写入 scratch 的矩形数组。
    /// 库不分配堆内存，sink 的资源行为由调用方决定。
    ///
    /// # Errors
    ///
    /// 输入错误在调用 sink 前返回。sink 错误会立即终止分区并原样返回；
    /// 已交给 sink 的前缀及失败调用产生的副作用不回滚。后续调用可继续复用 scratch。
    pub fn decompose_into<F>(
        &mut self,
        leaves: &[QuadLeaf64],
        sink: F,
    ) -> Result<usize, SparseQuadError>
    where
        F: FnMut(Rectangle) -> Result<(), SparseQuadError>,
    {
        build_axis_intervals_from_leaves(
            leaves.iter().copied(),
            leaves.len(),
            &mut self.dense_labels,
            &mut self.rows,
            &mut self.columns,
            &mut self.row_buckets,
            &mut self.column_buckets,
        )?;
        extract_sparse_chords(&self.rows, &self.columns, &mut self.chord_groups)?;
        self.select_cuts()?;
        self.sparse_partition_into(sink)
    }

    /// 将矩形直接写入调用方的 bounds / labels `SoA` 输出。
    ///
    /// 调用开始时清空输出；整个调用不分配堆内存，也不经过 `AoS` 中间结果。
    ///
    /// # Errors
    ///
    /// 输入非法或容量超限时返回错误。失败时保留本次已写入的输出前缀。
    pub fn decompose_packed(
        &mut self,
        leaves: &[QuadLeaf64],
        output: &mut PackedRectangles64,
    ) -> Result<usize, SparseQuadError> {
        output.clear();
        self.decompose_into(leaves, |rectangle| output.push(rectangle))
    }

    /// 对 `decompose` 同一路径计时。
    ///
    /// 仅用于 benchmark / profiling，默认 feature 不编译。
    ///
    /// # Errors
    ///
    /// leaf 越界、未按 lod 对齐、lod 超出范围、互相重叠或内部缓存无法分配时返回错误。
    #[cfg(feature = "profile")]
    pub fn decompose_profile(
        &mut self,
        leaves: &[QuadLeaf64],
    ) -> Result<SparseDecomposeProfile, SparseQuadError> {
        let total_start = Instant::now();
        let axis_start = Instant::now();
        build_axis_intervals_from_leaves(
            leaves.iter().copied(),
            leaves.len(),
            &mut self.dense_labels,
            &mut self.rows,
            &mut self.columns,
            &mut self.row_buckets,
            &mut self.column_buckets,
        )?;
        let build_axis = axis_start.elapsed();

        let chord_start = Instant::now();
        extract_sparse_chords(&self.rows, &self.columns, &mut self.chord_groups)?;
        let extract_chords = chord_start.elapsed();

        let counts_after_extract = SparseDecomposeCounts {
            leaves: leaves.len(),
            row_intervals: self.rows.logical_interval_count(),
            column_intervals: self.columns.logical_interval_count(),
            chord_groups: self.chord_groups.logical_group_count(),
            horizontal_chords: self.chord_groups.horizontal.len(),
            vertical_chords: self.chord_groups.vertical.len(),
            ..SparseDecomposeCounts::default()
        };

        let select_start = Instant::now();
        let select_timings = self.select_cuts_profile()?;
        let select_cuts = select_start.elapsed();

        let horizontal_cuts = self.horizontal_cuts.len();
        let vertical_cuts = self.vertical_cuts.len();

        let partition_start = Instant::now();
        let rectangles = self.sparse_partition()?.len();
        let partition = partition_start.elapsed();

        Ok(SparseDecomposeProfile {
            timings: SparseDecomposeTimings {
                total: total_start.elapsed(),
                build_axis,
                extract_chords,
                select_cuts,
                matching: select_timings.matching,
                matching_greedy: select_timings.matching_greedy,
                matching_bfs: select_timings.matching_bfs,
                matching_dfs_build: select_timings.matching_dfs_build,
                matching_dfs_search: select_timings.matching_dfs_search,
                matching_cover: select_timings.matching_cover,
                matching_collect: select_timings.matching_collect,
                emit_cuts: select_timings.emit_cuts,
                sort_cuts: Duration::ZERO,
                partition,
            },
            counts: SparseDecomposeCounts {
                horizontal_cuts,
                vertical_cuts,
                rectangles,
                greedy_matches: select_timings.greedy_matches,
                matching_phases: select_timings.matching_phases,
                matching_augmentations: select_timings.matching_augmentations,
                ..counts_after_extract
            },
        })
    }

    /// 对 sparse quad image 执行 64x64 最优分解。
    ///
    /// # Errors
    ///
    /// 当内部缓存无法分配时返回错误。
    #[cfg(feature = "alloc")]
    pub fn decompose_quads(
        &mut self,
        image: &SparseQuadImage64,
    ) -> Result<Vec<Rectangle>, SparseQuadError> {
        Ok(self.decompose_quads_borrowed(image)?.to_vec())
    }

    /// 对 sparse quad image 执行 64x64 最优分解，结果借用自 scratch。
    ///
    /// # Errors
    ///
    /// 当内部缓存无法分配时返回错误。
    #[cfg(feature = "alloc")]
    pub fn decompose_quads_borrowed(
        &mut self,
        image: &SparseQuadImage64,
    ) -> Result<&[Rectangle], SparseQuadError> {
        build_axis_intervals_from_leaves(
            image.leaves.iter().map(|stored| stored.leaf),
            image.leaves.len(),
            &mut self.dense_labels,
            &mut self.rows,
            &mut self.columns,
            &mut self.row_buckets,
            &mut self.column_buckets,
        )?;
        self.decompose_intervals_borrowed()
    }

    fn decompose_intervals_borrowed(&mut self) -> Result<&[Rectangle], SparseQuadError> {
        extract_sparse_chords(&self.rows, &self.columns, &mut self.chord_groups)?;
        self.select_cuts()?;
        self.sparse_partition()
    }

    fn select_cuts(&mut self) -> Result<(), SparseQuadError> {
        let groups = &self.chord_groups;
        let matching = &mut self.matching;
        let horizontal_cuts = &mut self.horizontal_cuts;
        let vertical_cuts = &mut self.vertical_cuts;

        horizontal_cuts.clear();
        vertical_cuts.clear();

        if groups.horizontal.is_empty() && groups.vertical.is_empty() {
            return Ok(());
        }

        let horizontal_cap = groups.horizontal.len();
        let vertical_cap = groups.vertical.len();
        reserve_exact(horizontal_cuts, horizontal_cap)?;
        reserve_exact(vertical_cuts, vertical_cap)?;

        // At every chord grid point at least three adjacent pixels have its
        // label. Different labels therefore cannot intersect, even at endpoints.
        // The full graph is the disjoint union of the former per-label graphs.
        // Matching and independent-set cardinalities add over that union.
        {
            let horizontal = groups.horizontal.as_slice();
            let vertical = groups.vertical.as_slice();
            matching.select_maximum_independent_set(horizontal, vertical);
            for &index in matching.selected_horizontal() {
                let index = usize::from(index);
                let Some(chord) = horizontal.get(index).map(|item| item.chord) else {
                    return Err(SparseQuadError::CapacityOverflow);
                };
                horizontal_cuts.push(HorizontalCut {
                    y: chord.y1,
                    start: chord.x1,
                    end: chord.x2,
                });
            }
            for &index in matching.selected_vertical() {
                let index = usize::from(index);
                let Some(chord) = vertical.get(index).map(|item| item.chord) else {
                    return Err(SparseQuadError::CapacityOverflow);
                };
                vertical_cuts.push(VerticalCut {
                    x: chord.x1,
                    start: chord.y1,
                    end: chord.y2,
                });
            }
        }

        Ok(())
    }

    #[cfg(feature = "profile")]
    fn select_cuts_profile(&mut self) -> Result<SelectCutTimings, SparseQuadError> {
        let groups = &self.chord_groups;
        let matching = &mut self.matching;
        let horizontal_cuts = &mut self.horizontal_cuts;
        let vertical_cuts = &mut self.vertical_cuts;

        horizontal_cuts.clear();
        vertical_cuts.clear();

        if groups.horizontal.is_empty() && groups.vertical.is_empty() {
            return Ok(SelectCutTimings::default());
        }

        let horizontal_cap = groups.horizontal.len();
        let vertical_cap = groups.vertical.len();
        reserve_exact(horizontal_cuts, horizontal_cap)?;
        reserve_exact(vertical_cuts, vertical_cap)?;

        let mut timings = SelectCutTimings::default();
        // At every chord grid point at least three adjacent pixels have its
        // label. Different labels therefore cannot intersect, even at endpoints.
        // The full graph is the disjoint union of the former per-label graphs.
        // Matching and independent-set cardinalities add over that union.
        {
            let horizontal = groups.horizontal.as_slice();
            let vertical = groups.vertical.as_slice();
            let profile = matching.maximum_independent_set_profile(horizontal, vertical);
            timings.matching += profile.timings.total;
            timings.matching_greedy += profile.timings.greedy;
            timings.matching_bfs += profile.timings.bfs;
            timings.matching_dfs_build += profile.timings.dfs_build;
            timings.matching_dfs_search += profile.timings.dfs_search;
            timings.matching_cover += profile.timings.cover;
            timings.matching_collect += profile.timings.collect;
            timings.greedy_matches += profile.counts.greedy_matches;
            timings.matching_phases += profile.counts.phases;
            timings.matching_augmentations += profile.counts.augmentations;

            let emit_start = Instant::now();
            for &index in matching.selected_horizontal() {
                let index = usize::from(index);
                let Some(chord) = horizontal.get(index).map(|item| item.chord) else {
                    return Err(SparseQuadError::CapacityOverflow);
                };
                horizontal_cuts.push(HorizontalCut {
                    y: chord.y1,
                    start: chord.x1,
                    end: chord.x2,
                });
            }
            for &index in matching.selected_vertical() {
                let index = usize::from(index);
                let Some(chord) = vertical.get(index).map(|item| item.chord) else {
                    return Err(SparseQuadError::CapacityOverflow);
                };
                vertical_cuts.push(VerticalCut {
                    x: chord.x1,
                    start: chord.y1,
                    end: chord.y2,
                });
            }
            timings.emit_cuts += emit_start.elapsed();
        }

        Ok(timings)
    }

    fn sparse_partition(&mut self) -> Result<&[Rectangle], SparseQuadError> {
        let cut_boundaries = build_cut_masks(
            &self.horizontal_cuts,
            &self.vertical_cuts,
            &mut self.horizontal_cut_masks,
            &mut self.vertical_cut_masks,
        )?;

        let result = &mut self.rectangles;
        result.clear();
        reserve_exact(result, self.rows.intervals.len())?;
        partition_rows(
            &self.rows,
            &self.horizontal_cut_masks,
            &self.vertical_cut_masks,
            cut_boundaries,
            &mut self.partition,
            |rectangle| {
                reserve_one(result)?;
                result.push(rectangle);
                Ok(())
            },
        )?;
        Ok(result.as_slice())
    }

    fn sparse_partition_into<F>(&mut self, sink: F) -> Result<usize, SparseQuadError>
    where
        F: FnMut(Rectangle) -> Result<(), SparseQuadError>,
    {
        let cut_boundaries = build_cut_masks(
            &self.horizontal_cuts,
            &self.vertical_cuts,
            &mut self.horizontal_cut_masks,
            &mut self.vertical_cut_masks,
        )?;
        partition_rows(
            &self.rows,
            &self.horizontal_cut_masks,
            &self.vertical_cut_masks,
            cut_boundaries,
            &mut self.partition,
            sink,
        )
    }
}

/// 固定 64x64 sparse layer 的增量构建器。
///
/// 只接受非零 dyadic square。builder 内部保留未展开的 leaves，
/// finish 使用与借用分解相同的稀疏／栅格路径，无需预合并相邻 square。
#[cfg(feature = "alloc")]
#[derive(Debug, Default)]
pub struct SparseLayerBuilder64 {
    leaves: Vec<QuadLeaf64>,
}

#[cfg(feature = "alloc")]
impl SparseLayerBuilder64 {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.leaves.clear();
    }

    /// 写入一个非零 dyadic square。
    ///
    /// # Errors
    ///
    /// square 越界、未按 lod 对齐、lod 超出范围、或内部缓存无法分配时返回错误。
    pub fn push_square(
        &mut self,
        u: u8,
        v: u8,
        lod: u8,
        value: NonZeroU16,
    ) -> Result<(), SparseQuadError> {
        let leaf = QuadLeaf64 { u, v, lod, value };
        validate_leaf_shape(leaf)?;
        allocate_more(&mut self.leaves, 1)?;
        self.leaves.push(leaf);
        Ok(())
    }

    /// 完成当前 layer 的最优矩形分解。
    ///
    /// # Errors
    ///
    /// square 互相重叠、或内部缓存无法分配时返回错误。
    pub fn finish(
        &mut self,
        scratch: &mut SparseOptimalScratch64,
    ) -> Result<Vec<Rectangle>, SparseQuadError> {
        if self.leaves.is_empty() {
            return Ok(Vec::new());
        }

        scratch.decompose(&self.leaves)
    }
}

// Leaf 校验

// 12 bit Morton 键直接映射源位置，再转成目的位置置换；每次交换固定一个位置。
// 小输入比较排序有固定上限，避免初始化和扫描 4096 键的常数代价。
#[cfg(feature = "alloc")]
fn sort_stored_leaves_by_morton(leaves: &mut [StoredLeaf]) -> Result<(), SparseQuadError> {
    if leaves.len() <= EDGE {
        leaves.sort_unstable_by_key(|leaf| leaf.morton_start);
        return Ok(());
    }
    if leaves.len() > MAX_AXIS_INTERVALS {
        return Err(SparseQuadError::Overlap);
    }
    let mut source_for_key = [0u16; MAX_AXIS_INTERVALS];
    for (source, leaf) in leaves.iter().enumerate() {
        let slot = crate::get_mut(&mut source_for_key, usize::from(leaf.morton_start));
        if *slot != 0 {
            return Err(SparseQuadError::Overlap);
        }
        // 0 表示键缺席，非零源位置编码为 index + 1，最大为 4096。
        *slot = crate::u16_index(source + 1);
    }
    let mut destination_for_source = [0u16; MAX_AXIS_INTERVALS];
    let mut rank = 0u16;
    for &source in &source_for_key {
        if source != 0 {
            *crate::get_mut(&mut destination_for_source, usize::from(source - 1)) = rank;
            rank += 1;
        }
    }
    for source in 0..leaves.len() {
        while usize::from(*get(&destination_for_source, source)) != source {
            let destination = usize::from(*get(&destination_for_source, source));
            leaves.swap(source, destination);
            destination_for_source.swap(source, destination);
        }
    }
    Ok(())
}

#[cfg(feature = "alloc")]
fn batch_morton_starts(leaves: &[QuadLeaf64], starts: &mut [u16; EDGE]) {
    for (chunk, output) in leaves.chunks(32).zip(starts.chunks_mut(32)) {
        let mut coordinates_u = [0u16; 32];
        let mut coordinates_v = [0u16; 32];
        for ((leaf, u), v) in chunk.iter().zip(&mut coordinates_u).zip(&mut coordinates_v) {
            *u = u16::from(leaf.u);
            *v = u16::from(leaf.v);
        }

        // 每个数据向量均为 32 个 u16（512 bit）；补零尾部不会写回有效范围外。
        let spread = |value: u16x32| {
            let value = (value | (value << 4)) & u16x32::splat(0x0f0f);
            let value = (value | (value << 2)) & u16x32::splat(0x3333);
            (value | (value << 1)) & u16x32::splat(0x5555)
        };
        let keys = (spread(u16x32::from_array(coordinates_u))
            | (spread(u16x32::from_array(coordinates_v)) << 1))
            .to_array();
        slice_mut(output, 0..chunk.len()).copy_from_slice(slice(&keys, 0..chunk.len()));
    }
}

#[cfg(feature = "alloc")]
fn validate_leaf(leaf: QuadLeaf64) -> Result<StoredLeaf, SparseQuadError> {
    validate_leaf_shape(leaf)?;

    let morton_start = u16::try_from(crate::morton::encode(leaf.u, leaf.v))
        .map_err(|_| SparseQuadError::CapacityOverflow)?;
    Ok(StoredLeaf { leaf, morton_start })
}

fn validate_leaf_shape(leaf: QuadLeaf64) -> Result<u8, SparseQuadError> {
    if leaf.lod > MAX_LOD {
        return Err(SparseQuadError::LodOutOfRange);
    }

    let size = quad_size(leaf.lod);
    let mask = size - 1;
    if (leaf.u & mask) != 0 || (leaf.v & mask) != 0 {
        return Err(SparseQuadError::Misaligned);
    }

    if u16::from(leaf.u) + u16::from(size) > u16::from(EDGE_U8)
        || u16::from(leaf.v) + u16::from(size) > u16::from(EDGE_U8)
    {
        return Err(SparseQuadError::OutOfBounds);
    }

    Ok(size)
}

const fn quad_size(lod: u8) -> u8 {
    1u8 << lod
}

// 几何类型

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Interval {
    start: u8,
    end: u8,
    value: u16,
}

/// 区间直接存入起点对应的槽；位图记录占用，清空只重置位图。
struct IntervalBucket {
    items: [Interval; EDGE],
    starts: u64,
}

impl IntervalBucket {
    const fn new() -> Self {
        Self {
            items: [Interval {
                start: 0,
                end: 0,
                value: 0,
            }; EDGE],
            starts: 0,
        }
    }

    fn try_push(&mut self, interval: Interval) -> Result<(), SparseQuadError> {
        let slot = self
            .items
            .get_mut(usize::from(interval.start))
            .ok_or(SparseQuadError::CapacityOverflow)?;
        let bit = 1u64 << u32::from(interval.start);
        if self.starts & bit != 0 {
            return Err(SparseQuadError::Overlap);
        }
        *slot = interval;
        self.starts |= bit;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LineRange {
    start: usize,
    end: usize,
}

struct AxisIntervals {
    lines: FixedVec<LineRange, EDGE>,
    intervals: FixedVec<Interval, MAX_AXIS_INTERVALS>,
    boundaries: u64,
}

impl AxisIntervals {
    const fn new() -> Self {
        Self {
            lines: FixedVec::new(LineRange { start: 0, end: 0 }),
            intervals: FixedVec::new(EMPTY_INTERVAL),
            boundaries: 0,
        }
    }

    fn clear_for_build(&mut self) -> Result<(), SparseQuadError> {
        self.lines.clear();
        self.intervals.clear();
        self.boundaries = 0;
        reserve_exact(&self.lines, EDGE)?;
        self.lines.resize(EDGE, LineRange::default());
        Ok(())
    }

    fn line(&self, index: usize) -> &[Interval] {
        let range = get(&self.lines, index);
        slice(&self.intervals, range.start..range.end)
    }

    // profiling 仍统计所有像素行的区间数，稳定行共享存储不改变该字段含义。
    #[cfg(feature = "profile")]
    fn logical_interval_count(&self) -> usize {
        self.lines.iter().map(|range| range.end - range.start).sum()
    }
}

// 从 leaves 构造轴区间

fn build_axis_intervals_from_leaves<I>(
    leaves: I,
    leaf_count: usize,
    dense_labels: &mut DenseLabels64,
    rows: &mut AxisIntervals,
    columns: &mut AxisIntervals,
    row_buckets: &mut [IntervalBucket; EDGE],
    column_buckets: &mut [IntervalBucket; EDGE],
) -> Result<(), SparseQuadError>
where
    I: IntoIterator<Item = QuadLeaf64>,
    I::IntoIter: Clone,
{
    if leaf_count >= dense::DENSE_LEAF_THRESHOLD {
        dense::rasterize_leaves(leaves, dense_labels, row_buckets)?;
        return dense::build_axes(dense_labels, rows, columns, column_buckets);
    }
    prepare_interval_buckets(row_buckets);
    prepare_interval_buckets(column_buckets);
    let mut all_lod_zero = true;
    for leaf in leaves {
        push_leaf_axis_intervals_one(leaf, row_buckets, column_buckets)?;
        all_lod_zero &= leaf.lod == 0;
    }

    if all_lod_zero {
        build_complete_axis_intervals_from_buckets(row_buckets, rows)?;
        build_complete_axis_intervals_from_buckets(column_buckets, columns)
    } else {
        build_axis_intervals_from_buckets(row_buckets, rows, true)?;
        build_axis_intervals_from_buckets(column_buckets, columns, false)
    }
}

fn prepare_interval_buckets(buckets: &mut [IntervalBucket; EDGE]) {
    for bucket in buckets {
        bucket.starts = 0;
    }
}

fn push_leaf_axis_intervals_one(
    leaf: QuadLeaf64,
    row_buckets: &mut [IntervalBucket],
    column_buckets: &mut [IntervalBucket],
) -> Result<(), SparseQuadError> {
    let size = validate_leaf_shape(leaf)?;
    let u1 = leaf.u + size;
    let v1 = leaf.v + size;
    let value = leaf.value.get();

    let row_interval = Interval {
        start: leaf.u,
        end: u1,
        value,
    };
    let Some(row_bucket) = row_buckets.get_mut(usize::from(leaf.v)) else {
        return Err(SparseQuadError::CapacityOverflow);
    };
    row_bucket.try_push(row_interval)?;

    let column_interval = Interval {
        start: leaf.v,
        end: v1,
        value,
    };
    let Some(column_bucket) = column_buckets.get_mut(usize::from(leaf.u)) else {
        return Err(SparseQuadError::CapacityOverflow);
    };
    column_bucket.try_push(column_interval)?;
    Ok(())
}

// 轴区间构建

fn build_complete_axis_intervals_from_buckets(
    buckets: &[IntervalBucket; EDGE],
    output: &mut AxisIntervals,
) -> Result<(), SparseQuadError> {
    output.clear_for_build()?;
    let mut previous = LineRange::default();
    for (line, bucket) in buckets.iter().enumerate() {
        let start = output.intervals.len();
        let mut starts = bucket.starts;
        reserve_exact(&output.intervals, starts.count_ones() as usize)?;
        while starts != 0 {
            let interval = *get(&bucket.items, starts.trailing_zeros() as usize);
            // 单位区间起点互异即不重叠；重复起点已在事件入桶时报告。
            push_merged_interval(&mut output.intervals, start, interval)?;
            starts &= starts - 1;
        }
        let end = output.intervals.len();
        let range = if end - start == previous.end - previous.start
            && slice(&output.intervals, start..end)
                == slice(&output.intervals, previous.start..previous.end)
        {
            output.intervals.resize(start, EMPTY_INTERVAL);
            previous
        } else {
            output.boundaries |= 1u64 << line;
            LineRange { start, end }
        };
        let Some(slot) = output.lines.get_mut(line) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        *slot = range;
        previous = range;
    }
    Ok(())
}

fn build_axis_intervals_from_buckets(
    buckets: &[IntervalBucket; EDGE],
    output: &mut AxisIntervals,
    validate_overlap: bool,
) -> Result<(), SparseQuadError> {
    output.clear_for_build()?;
    let mut active = IntervalBucket::new();
    let mut end_masks = [0u64; EDGE];
    let mut range = LineRange::default();
    for (line, bucket) in buckets.iter().enumerate() {
        let expired = *get(&end_masks, line);
        if bucket.starts | expired != 0 {
            output.boundaries |= 1u64 << line;
            active.starts &= !expired;
            let mut starts = bucket.starts;
            while starts != 0 {
                let interval = *get(&bucket.items, starts.trailing_zeros() as usize);
                active.try_push(interval)?;
                // square 的轴区间长度也等于沿扫描方向存活的行数。
                let end_line = line + usize::from(interval.end - interval.start);
                if end_line < EDGE {
                    *crate::get_mut(&mut end_masks, end_line) |= 1u64 << interval.start;
                }
                starts &= starts - 1;
            }
            let start = output.intervals.len();
            let mut active_starts = active.starts;
            reserve_exact(&output.intervals, active_starts.count_ones() as usize)?;
            let mut previous_end = 0u8;
            while active_starts != 0 {
                let interval = *get(&active.items, active_starts.trailing_zeros() as usize);
                if validate_overlap && previous_end > interval.start {
                    return Err(SparseQuadError::Overlap);
                }
                previous_end = interval.end;
                push_merged_interval(&mut output.intervals, start, interval)?;
                active_starts &= active_starts - 1;
            }
            range = LineRange {
                start,
                end: output.intervals.len(),
            };
        }
        let Some(slot) = output.lines.get_mut(line) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        *slot = range;
    }
    Ok(())
}

fn push_merged_interval(
    output: &mut FixedVec<Interval, MAX_AXIS_INTERVALS>,
    line_start: usize,
    interval: Interval,
) -> Result<(), SparseQuadError> {
    if let Some(last) = output
        .get_mut(line_start..)
        .and_then(|items| items.last_mut())
        && last.value == interval.value
        && last.end == interval.start
    {
        last.end = interval.end;
        return Ok(());
    }
    reserve_one(output)?;
    output.push(interval);
    Ok(())
}

// Chord 提取

struct SparseChords {
    horizontal: ChordBuffer<ValuedChord>,
    vertical: ChordBuffer<ValuedChord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ValuedChord {
    // Labels are retained only for profiling and the geometric invariant tests.
    #[cfg_attr(not(any(test, feature = "profile")), allow(dead_code))]
    value: u16,
    chord: EffectiveChord,
}

impl ChordAccess for ValuedChord {
    fn chord(&self) -> EffectiveChord {
        self.chord
    }
}

impl SparseChords {
    const fn new() -> Self {
        Self {
            horizontal: FixedVec::new(EMPTY_CHORD),
            vertical: FixedVec::new(EMPTY_CHORD),
        }
    }

    const fn clear(&mut self) {
        self.horizontal.clear();
        self.vertical.clear();
    }

    fn add_horizontal(&mut self, value: u16, chord: EffectiveChord) -> Result<(), SparseQuadError> {
        reserve_one(&self.horizontal)?;
        self.horizontal.push(ValuedChord { value, chord });
        Ok(())
    }

    fn add_vertical(&mut self, value: u16, chord: EffectiveChord) -> Result<(), SparseQuadError> {
        reserve_one(&self.vertical)?;
        self.vertical.push(ValuedChord { value, chord });
        Ok(())
    }

    // Preserve the public profiling field's meaning: number of distinct labels
    // with at least one chord. This read-only count is outside the ordinary path.
    #[cfg(any(test, feature = "profile"))]
    fn logical_group_count(&self) -> usize {
        let mut labels = [0u64; 1024];
        let mut count = 0;
        for item in self.horizontal.iter().chain(self.vertical.iter()) {
            let label = usize::from(item.value);
            let word = crate::get_mut(&mut labels, label / 64);
            let bit = 1u64 << (label % 64);
            if *word & bit == 0 {
                *word |= bit;
                count += 1;
            }
        }
        count
    }
}

fn extract_sparse_chords(
    rows: &AxisIntervals,
    columns: &AxisIntervals,
    groups: &mut SparseChords,
) -> Result<(), SparseQuadError> {
    groups.clear();
    for y in 1u8..EDGE_U8 {
        if rows.boundaries & (1u64 << y) == 0 {
            continue;
        }
        emit_horizontal_chords_for_boundary(
            rows.line(usize::from(y - 1)),
            rows.line(usize::from(y)),
            y,
            groups,
        )?;
    }
    for x in 1u8..EDGE_U8 {
        if columns.boundaries & (1u64 << x) == 0 {
            continue;
        }
        emit_vertical_chords_for_boundary(
            columns.line(usize::from(x - 1)),
            columns.line(usize::from(x)),
            x,
            groups,
        )?;
    }
    Ok(())
}

/// 输入是已合并的最大同色区间。
/// 两段同色且相交时，起点不等意味着交集左侧恰有一个同色像素，即凹角；
/// 终点不等给出另一端的凹角。无需重新查询相邻像素。
fn emit_chords_for_boundary(
    first: &[Interval],
    second: &[Interval],
    mut emit: impl FnMut(Interval) -> Result<(), SparseQuadError>,
) -> Result<(), SparseQuadError> {
    let (mut i, mut j) = (0, 0);
    while let (Some(&a), Some(&b)) = (first.get(i), second.get(j)) {
        let start = a.start.max(b.start);
        let end = a.end.min(b.end);
        if a.value == b.value && a.start != b.start && a.end != b.end && start < end {
            emit(Interval {
                start,
                end,
                value: a.value,
            })?;
        }
        if a.end <= b.end {
            i += 1;
        }
        // 终点相同时两段都耗尽，直接同时推进，避免再扫描一个空交集。
        if b.end <= a.end {
            j += 1;
        }
    }
    Ok(())
}

fn emit_horizontal_chords_for_boundary(
    first: &[Interval],
    second: &[Interval],
    y: u8,
    groups: &mut SparseChords,
) -> Result<(), SparseQuadError> {
    emit_chords_for_boundary(first, second, |interval| {
        groups.add_horizontal(
            interval.value,
            EffectiveChord {
                orientation: Orientation::Horizontal,
                x1: interval.start,
                y1: y,
                x2: interval.end,
                y2: y,
            },
        )
    })
}

fn emit_vertical_chords_for_boundary(
    first: &[Interval],
    second: &[Interval],
    x: u8,
    groups: &mut SparseChords,
) -> Result<(), SparseQuadError> {
    emit_chords_for_boundary(first, second, |interval| {
        groups.add_vertical(
            interval.value,
            EffectiveChord {
                orientation: Orientation::Vertical,
                x1: x,
                y1: interval.start,
                x2: x,
                y2: interval.end,
            },
        )
    })
}

// Cut / 分区类型

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HorizontalCut {
    y: u8,
    start: u8,
    end: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VerticalCut {
    x: u8,
    start: u8,
    end: u8,
}

// 扫描线分区

struct PartitionScratch {
    runs: FixedVec<Run, EDGE>,
    active: FixedVec<ActiveRect, EDGE>,
    next_active: FixedVec<ActiveRect, EDGE>,
}

impl PartitionScratch {
    const fn new() -> Self {
        Self {
            runs: FixedVec::new(Run {
                value: 0,
                x_start: 0,
                x_end: 0,
            }),
            active: FixedVec::new(ActiveRect::new(0, 0, 0, 0)),
            next_active: FixedVec::new(ActiveRect::new(0, 0, 0, 0)),
        }
    }
}

#[allow(clippy::mut_mut)] // 交换 buffer 引用，避免复制内联数组。
fn partition_rows<F>(
    rows: &AxisIntervals,
    horizontal_cut_masks: &FixedVec<u64, EDGE>,
    vertical_cut_masks: &FixedVec<u128, EDGE>,
    cut_boundaries: u64,
    scratch: &mut PartitionScratch,
    mut sink: F,
) -> Result<usize, SparseQuadError>
where
    F: FnMut(Rectangle) -> Result<(), SparseQuadError>,
{
    let runs = &mut scratch.runs;
    let mut active = &mut scratch.active;
    let mut next_active = &mut scratch.next_active;
    active.clear();
    next_active.clear();

    let mut count = 0usize;
    let mut counted_sink = |rectangle| {
        sink(rectangle)?;
        count += 1;
        Ok(())
    };
    let Some(&first_vertical_cut_mask) = vertical_cut_masks.first() else {
        return Err(SparseQuadError::CapacityOverflow);
    };
    build_runs_for_row(rows.line(0), first_vertical_cut_mask, runs)?;
    reserve_exact(active, runs.len())?;
    for &run in runs.as_slice() {
        active.push(ActiveRect::new(run.value, run.x_start, run.x_end, 0));
    }

    let mut boundaries = (rows.boundaries | cut_boundaries) & !1u64;
    while boundaries != 0 {
        let y = u8::try_from(boundaries.trailing_zeros())
            .map_err(|_| SparseQuadError::CapacityOverflow)?;
        let y_index = usize::from(y);
        let Some(&row_vertical_cut_mask) = vertical_cut_masks.get(y_index) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        let Some(&horizontal_cut_mask) = horizontal_cut_masks.get(y_index) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        build_runs_for_row(rows.line(y_index), row_vertical_cut_mask, runs)?;
        merge_sparse_runs(
            active.as_slice(),
            runs.as_slice(),
            y,
            horizontal_cut_mask,
            next_active,
            &mut counted_sink,
        )?;
        // 交换视图，避免逐行复制两个内联数组。
        core::mem::swap(&mut active, &mut next_active);
        boundaries &= boundaries - 1;
    }
    for &active_rect in active.as_slice() {
        emit(active_rect, EDGE_U8, &mut counted_sink)?;
    }
    Ok(count)
}

fn build_cut_masks(
    horizontal_cuts: &[HorizontalCut],
    vertical_cuts: &[VerticalCut],
    horizontal_masks: &mut FixedVec<u64, EDGE>,
    vertical_masks: &mut FixedVec<u128, EDGE>,
) -> Result<u64, SparseQuadError> {
    // 位图用 OR 合并，切线顺序无关；无需在收集后排序。
    horizontal_masks.clear();
    horizontal_masks.resize(EDGE, 0);
    vertical_masks.clear();
    vertical_masks.resize(EDGE, 0);

    let mut boundaries = 0u64;

    for cut in vertical_cuts {
        if cut.start >= cut.end {
            continue;
        }
        let bit = 1u128 << u32::from(cut.x);
        let Some(start_mask) = vertical_masks.get_mut(usize::from(cut.start)) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        *start_mask ^= bit;
        boundaries |= 1u64 << cut.start;
        if cut.end < EDGE_U8 {
            *crate::get_mut(vertical_masks, usize::from(cut.end)) ^= bit;
            boundaries |= 1u64 << cut.end;
        }
    }
    // 同向 chord 在相同 x 上连端点都互不重叠，因此每个 x 的覆盖数仅为 0/1。
    // 起止端点 XOR 后做前缀即可重建 OR 覆盖；end=64 在最终关闭时隐式处理。
    let mut active = 0u128;
    for mask in vertical_masks.iter_mut() {
        active ^= *mask;
        *mask = active;
    }

    for cut in horizontal_cuts {
        let Some(mask) = horizontal_masks.get_mut(usize::from(cut.y)) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        *mask |= cell_range_mask(cut.start, cut.end);
        boundaries |= 1u64 << cut.y;
    }

    Ok(boundaries)
}

fn build_runs_for_row(
    intervals: &[Interval],
    vertical_cut_mask: u128,
    runs: &mut FixedVec<Run, EDGE>,
) -> Result<(), SparseQuadError> {
    runs.clear();
    for &interval in intervals {
        let mut start = interval.start;
        let mut split_mask = vertical_cut_mask
            & coord_between_mask(
                interval.start.saturating_add(1),
                interval.end.saturating_sub(1),
            );
        reserve_exact(runs, split_mask.count_ones() as usize + 1)?;
        while split_mask != 0 {
            let split = u8::try_from(split_mask.trailing_zeros())
                .map_err(|_| SparseQuadError::CapacityOverflow)?;
            push_run(runs, interval.value, start, split)?;
            start = split;
            split_mask &= split_mask - 1;
        }
        push_run(runs, interval.value, start, interval.end)?;
    }
    Ok(())
}

fn push_run(
    runs: &mut FixedVec<Run, EDGE>,
    value: u16,
    start: u8,
    end: u8,
) -> Result<(), SparseQuadError> {
    if start >= end {
        return Ok(());
    }
    reserve_one(runs)?;
    runs.push(Run {
        value,
        x_start: start,
        x_end: end,
    });
    Ok(())
}

fn merge_sparse_runs<F>(
    active: &[ActiveRect],
    runs: &[Run],
    y: u8,
    horizontal_cut_mask: u64,
    next_active: &mut FixedVec<ActiveRect, EDGE>,
    result: &mut F,
) -> Result<(), SparseQuadError>
where
    F: FnMut(Rectangle) -> Result<(), SparseQuadError>,
{
    next_active.clear();
    let (mut active_index, mut run_index) = (0usize, 0usize);

    while active_index < active.len() && run_index < runs.len() {
        let (Some(&active_rect), Some(&run)) = (active.get(active_index), runs.get(run_index))
        else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        let run_key =
            u32::from(run.value) | (u32::from(run.x_start) << 16) | (u32::from(run.x_end) << 24);

        if active_rect.match_key() == run_key
            && !horizontal_cut_overlaps(horizontal_cut_mask, active_rect.x_start, active_rect.x_end)
        {
            reserve_one(next_active)?;
            next_active.push(active_rect);
            active_index += 1;
            run_index += 1;
        } else if (active_rect.x_start, active_rect.value) <= (run.x_start, run.value) {
            emit(active_rect, y, result)?;
            active_index += 1;
        } else {
            reserve_one(next_active)?;
            next_active.push(ActiveRect::new(run.value, run.x_start, run.x_end, y));
            run_index += 1;
        }
    }

    let Some(remaining_active) = active.get(active_index..) else {
        return Err(SparseQuadError::CapacityOverflow);
    };
    for &active_rect in remaining_active {
        emit(active_rect, y, result)?;
    }

    reserve_exact(next_active, runs.len().saturating_sub(run_index))?;
    let Some(remaining_runs) = runs.get(run_index..) else {
        return Err(SparseQuadError::CapacityOverflow);
    };
    for &run in remaining_runs {
        next_active.push(ActiveRect::new(run.value, run.x_start, run.x_end, y));
    }

    Ok(())
}

fn horizontal_cut_overlaps(mask: u64, start: u8, end: u8) -> bool {
    mask & cell_range_mask(start, end) != 0
}

fn coord_between_mask(start: u8, end: u8) -> u128 {
    if start > end {
        return 0;
    }

    let len = u32::from(end - start) + 1;
    ((1u128 << len) - 1) << u32::from(start)
}

fn cell_range_mask(start: u8, end: u8) -> u64 {
    if start >= end {
        return 0;
    }

    let start_mask = u64::MAX << u32::from(start);
    let end_mask = if end >= EDGE_U8 {
        u64::MAX
    } else {
        (1u64 << u32::from(end)) - 1
    };
    start_mask & end_mask
}

// 工具函数

#[cfg(feature = "alloc")]
fn allocate_exact<T>(items: &mut Vec<T>, additional: usize) -> Result<(), SparseQuadError> {
    if items.capacity().saturating_sub(items.len()) >= additional {
        return Ok(());
    }
    items
        .try_reserve_exact(additional)
        .map_err(|_| SparseQuadError::AllocationFailed)
}

#[cfg(feature = "alloc")]
fn allocate_more<T>(items: &mut Vec<T>, additional: usize) -> Result<(), SparseQuadError> {
    if items.capacity().saturating_sub(items.len()) >= additional {
        return Ok(());
    }
    items
        .try_reserve(additional)
        .map_err(|_| SparseQuadError::AllocationFailed)
}

const fn reserve_exact<T: Copy, const N: usize>(
    items: &FixedVec<T, N>,
    additional: usize,
) -> Result<(), SparseQuadError> {
    if items.capacity().saturating_sub(items.len()) >= additional {
        return Ok(());
    }
    Err(SparseQuadError::CapacityOverflow)
}

const fn reserve_one<T: Copy, const N: usize>(
    items: &FixedVec<T, N>,
) -> Result<(), SparseQuadError> {
    if items.len() < items.capacity() {
        return Ok(());
    }
    Err(SparseQuadError::CapacityOverflow)
}

fn emit<F>(active: ActiveRect, y_end: u8, result: &mut F) -> Result<(), SparseQuadError>
where
    F: FnMut(Rectangle) -> Result<(), SparseQuadError>,
{
    result(Rectangle {
        value: active.value,
        x: RangeU8::new(active.x_start, active.x_end),
        y: RangeU8::new(active.y_start, y_end),
    })
}

// 测试

#[cfg(test)]
mod tests {
    use std::num::NonZeroU16;
    use std::prelude::rust_2024::*;
    #[cfg(feature = "alloc")]
    use std::vec;

    use super::*;
    use crate::get_mut;

    const ROOT_LOD: u8 = 6;
    const GENERATED_TILE_LOD: u8 = 3;
    const GENERATED_TILE_EDGE: usize = 8;

    fn nz(value: u16) -> NonZeroU16 {
        NonZeroU16::new(value).unwrap_or_else(|| crate::invariant_failed())
    }

    #[test]
    #[allow(clippy::panic_in_result_fn)] // Result 传播被测错误，断言检查直接输出资源契约。
    fn direct_output_does_not_overwrite_borrowed_rectangle_storage() -> Result<(), SparseQuadError>
    {
        let mut scratch = SparseOptimalScratch64::new();
        let first = [leaf(0, 0, 6, NonZeroU16::MIN)];
        let expected = scratch.decompose_borrowed(&first)?.to_vec();
        let second = [leaf(8, 16, 3, NonZeroU16::MAX)];
        let mut seen = 0;
        assert_eq!(
            scratch.decompose_into(&second, |rectangle| {
                assert_eq!(rectangle.value, u16::MAX);
                seen += 1;
                Ok(())
            })?,
            1
        );
        assert_eq!(seen, 1);
        assert_eq!(scratch.rectangles.as_slice(), expected);
        let mut packed = PackedRectangles64::new();
        assert_eq!(scratch.decompose_packed(&second, &mut packed)?, 1);
        assert_eq!(scratch.rectangles.as_slice(), expected);
        assert_eq!(packed.bounds(), &[u32::from_le_bytes([8, 16, 16, 24])]);
        assert_eq!(packed.labels(), &[u16::MAX]);
        Ok(())
    }

    fn ok<T, E>(result: Result<T, E>) -> Option<T> {
        assert!(result.is_ok());
        result.ok()
    }

    const fn leaf(u: u8, v: u8, lod: u8, value: NonZeroU16) -> QuadLeaf64 {
        QuadLeaf64 { u, v, lod, value }
    }

    const fn tile_origin(tile: usize) -> u8 {
        match tile {
            0 => 0,
            1 => 8,
            2 => 16,
            3 => 24,
            4 => 32,
            5 => 40,
            6 => 48,
            7 => 56,
            _ => 64,
        }
    }

    const _: () = {
        assert!(1usize << (ROOT_LOD - GENERATED_TILE_LOD) == GENERATED_TILE_EDGE);
    };

    #[test]
    fn generated_tile_origins_cover_image() {
        for tile in 0..GENERATED_TILE_EDGE {
            let origin = tile_origin(tile);
            assert_eq!(origin % (1u8 << GENERATED_TILE_LOD), 0);
            if tile == GENERATED_TILE_EDGE - 1 {
                assert_eq!(origin + (1u8 << GENERATED_TILE_LOD), 64);
            }
        }
    }

    fn assert_from_leaves_error(leaves: &[QuadLeaf64], expected: SparseQuadError) {
        #[cfg(not(feature = "alloc"))]
        {
            let mut scratch = SparseOptimalScratch64::new();
            assert_eq!(scratch.decompose_borrowed(leaves), Err(expected));
        }
        #[cfg(feature = "alloc")]
        {
            let result = SparseQuadImage64::from_leaves(leaves);
            assert!(matches!(result, Err(error) if error == expected));
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn empty_sparse_image_outputs_no_rectangles() {
        let Some(sparse_image) = ok(SparseQuadImage64::from_leaves(&[])) else {
            return;
        };
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_quads(&sparse_image)) else {
            return;
        };
        assert_eq!(rectangles, []);
    }

    #[test]
    fn new_scratch_is_ready_for_borrowed_decomposition() {
        let mut scratch = SparseOptimalScratch64::new();
        assert_eq!(scratch.decompose_borrowed(&[]), Ok([].as_slice()));
    }

    #[test]
    fn inline_borrowed_decompose_outputs_slice() {
        let leaves = [leaf(0, 0, 6, nz(5))];
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_eq!(rectangles.len(), 1);
    }

    #[test]
    fn full_pixel_grid_fills_interval_capacity() {
        let leaves: Vec<_> = (0..EDGE_U8)
            .flat_map(|v| (0..EDGE_U8).map(move |u| leaf(u, v, 0, NonZeroU16::MIN)))
            .collect();
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_eq!(
            rectangles,
            &[Rectangle {
                value: 1,
                x: RangeU8::new(0, EDGE_U8),
                y: RangeU8::new(0, EDGE_U8),
            }],
        );
        assert_eq!(scratch.rows.intervals.len(), 1);
        assert_eq!(scratch.columns.intervals.len(), 1);
        assert_eq!(scratch.rows.boundaries, 1);
        assert_eq!(scratch.columns.boundaries, 1);
    }

    #[test]
    fn overfull_row_reports_overlap() {
        let leaves = [leaf(0, 0, 0, NonZeroU16::MIN); EDGE + 1];
        let mut scratch = SparseOptimalScratch64::new();
        assert_eq!(
            scratch.decompose_borrowed(&leaves),
            Err(SparseQuadError::Overlap)
        );
    }

    #[test]
    fn repeated_decomposition_clears_buckets() {
        let mut scratch = SparseOptimalScratch64::new();
        let _ = ok(scratch.decompose_borrowed(&[leaf(0, 0, ROOT_LOD, NonZeroU16::MIN)]));
        assert!(matches!(scratch.decompose_borrowed(&[]), Ok([])));
    }

    #[test]
    fn shuffled_mixed_lod_leaves_keep_optimum_and_exact_coverage() {
        // label 1 的 (63, 0) 与 (0, 63) 不能由同一矩形覆盖，故至少需要两个。
        // 上半区与左下区恰好给出两个矩形，右下两个色带各一个，独立最优值为 4。
        let leaves = [
            leaf(0, 0, 5, nz(1)),
            leaf(32, 0, 4, nz(1)),
            leaf(48, 0, 3, nz(1)),
            leaf(56, 0, 3, nz(1)),
            leaf(48, 8, 3, nz(1)),
            leaf(56, 8, 3, nz(1)),
            leaf(32, 16, 4, nz(1)),
            leaf(48, 16, 4, nz(1)),
            leaf(0, 32, 5, nz(1)),
            leaf(32, 32, 4, nz(2)),
            leaf(48, 32, 4, nz(2)),
            leaf(32, 48, 4, nz(3)),
            leaf(48, 48, 4, nz(3)),
        ];
        let mut expected = [[0u16; EDGE]; EDGE];
        for item in leaves {
            let size = quad_size(item.lod);
            for y in item.v..item.v + size {
                for x in item.u..item.u + size {
                    *get_mut(get_mut(&mut expected, usize::from(y)), usize::from(x)) =
                        item.value.get();
                }
            }
        }

        let mut scratch = SparseOptimalScratch64::new();
        let Some(canonical) = ok(scratch.decompose_borrowed(&leaves).map(<[_]>::to_vec)) else {
            return;
        };
        assert_eq!(canonical.len(), 4);
        for shift in 0..leaves.len() {
            for reverse in [false, true] {
                let mut shuffled = leaves;
                shuffled.rotate_left(shift);
                if reverse {
                    shuffled.reverse();
                }
                let Some(rectangles) = ok(scratch.decompose_borrowed(&shuffled)) else {
                    return;
                };
                assert_eq!(rectangles, canonical.as_slice());
                let mut actual = [[0u16; EDGE]; EDGE];
                for rectangle in rectangles {
                    for y in rectangle.y.start..rectangle.y.end {
                        for x in rectangle.x.start..rectangle.x.end {
                            let pixel =
                                get_mut(get_mut(&mut actual, usize::from(y)), usize::from(x));
                            assert_eq!(*pixel, 0, "矩形不得重叠");
                            *pixel = rectangle.value;
                        }
                    }
                }
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn last_coordinate_slot_reuses_scratch_after_failed_builds() {
        let mut scratch = SparseOptimalScratch64::new();
        let overlapping_pairs = [
            // 起点不同的包含关系必须由有序区间的 end 检查识别。
            [leaf(0, 0, 2, nz(1)), leaf(2, 2, 1, nz(2))],
            // 起点重复直接由占用位图识别。
            [leaf(0, 0, 1, nz(1)), leaf(0, 0, 0, nz(2))],
        ];
        for [first, second] in overlapping_pairs {
            for input in [[first, second], [second, first]] {
                assert_eq!(
                    scratch.decompose_borrowed(&input),
                    Err(SparseQuadError::Overlap)
                );
                let Some(rectangles) =
                    ok(scratch.decompose_borrowed(&[leaf(63, 63, 0, NonZeroU16::MAX)]))
                else {
                    return;
                };
                assert_eq!(
                    rectangles,
                    &[Rectangle {
                        value: u16::MAX,
                        x: RangeU8::new(63, 64),
                        y: RangeU8::new(63, 64),
                    }],
                );
                assert_eq!(scratch.decompose_borrowed(&[]), Ok([].as_slice()));
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn morton_key_sort_matches_comparison_order_for_every_pixel_key() {
        assert_eq!(std::mem::size_of::<StoredLeaf>(), 8);
        let leaves: Vec<_> = (0..MAX_AXIS_INTERVALS)
            .map(|index| {
                let shuffled = (index * 4051) % MAX_AXIS_INTERVALS;
                leaf(
                    u8::try_from(shuffled % EDGE).unwrap_or_default(),
                    u8::try_from(shuffled / EDGE).unwrap_or_default(),
                    0,
                    NonZeroU16::MIN,
                )
            })
            .collect();
        let Some(image) = ok(SparseQuadImage64::from_leaves(&leaves)) else {
            return;
        };
        let mut expected = Vec::new();
        for item in leaves {
            let Some(stored) = ok(validate_leaf(item)) else {
                return;
            };
            expected.push(stored);
        }
        expected.sort_unstable_by_key(|stored| stored.morton_start);
        assert_eq!(image.leaves, expected);
        let ordered_leaves: Vec<_> = expected.iter().map(|stored| stored.leaf).collect();
        let Some(ordered_image) = ok(SparseQuadImage64::from_leaves(&ordered_leaves)) else {
            return;
        };
        assert_eq!(ordered_image.leaves, expected);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn morton_key_sort_keeps_threshold_overlap_and_shape_error_semantics() {
        for count in [63usize, 64, 65, 512] {
            let mut leaves: Vec<_> = (0..count)
                .map(|index| {
                    leaf(
                        u8::try_from(index % EDGE).unwrap_or_default(),
                        u8::try_from(index / EDGE).unwrap_or_default(),
                        0,
                        NonZeroU16::MIN,
                    )
                })
                .collect();
            leaves.reverse();
            let Some(image) = ok(SparseQuadImage64::from_leaves(&leaves)) else {
                return;
            };
            assert!(
                image
                    .leaves
                    .windows(2)
                    .all(|pair| { get(pair, 0).morton_start < get(pair, 1).morton_start })
            );
            leaves.push(*get(&leaves, 0));
            assert_from_leaves_error(&leaves, SparseQuadError::Overlap);
        }
        let mut overfull = vec![leaf(0, 0, 0, NonZeroU16::MIN); MAX_AXIS_INTERVALS + 1];
        // 首键大于下一键，确保经过大输入置换入口的容量检查。
        *get_mut(&mut overfull, 0) = leaf(63, 63, 0, NonZeroU16::MIN);
        assert_from_leaves_error(&overfull, SparseQuadError::Overlap);
        *get_mut(&mut overfull, MAX_AXIS_INTERVALS) = leaf(0, 0, 7, NonZeroU16::MIN);
        assert_from_leaves_error(&overfull, SparseQuadError::LodOutOfRange);
    }

    #[test]
    fn axis_events_share_stable_rows_and_handle_end_64() {
        let mut scratch = SparseOptimalScratch64::new();
        let Some(full_rectangles) =
            ok(scratch.decompose_borrowed(&[leaf(0, 0, 6, NonZeroU16::MIN)]))
        else {
            return;
        };
        assert_eq!(full_rectangles.len(), 1);
        assert_eq!(scratch.rows.intervals.len(), 1);
        assert_eq!(scratch.columns.intervals.len(), 1);
        assert_eq!(scratch.rows.boundaries, 1);
        assert_eq!(scratch.columns.boundaries, 1);
        for line in 0..EDGE {
            assert_eq!(get(&scratch.rows.lines, line), get(&scratch.rows.lines, 0));
            assert_eq!(
                get(&scratch.columns.lines, line),
                get(&scratch.columns.lines, 0)
            );
        }

        let Some(band_rectangles) = ok(scratch.decompose_borrowed(&[leaf(16, 32, 4, nz(9))]))
        else {
            return;
        };
        assert_eq!(
            band_rectangles,
            &[Rectangle {
                value: 9,
                x: RangeU8::new(16, 32),
                y: RangeU8::new(32, 48),
            }]
        );
        assert_eq!(scratch.rows.intervals.len(), 1);
        assert_eq!(scratch.columns.intervals.len(), 1);
        assert_eq!(scratch.rows.boundaries, (1u64 << 32) | (1u64 << 48));
        assert_eq!(scratch.columns.boundaries, (1u64 << 16) | (1u64 << 32));
        #[cfg(feature = "profile")]
        {
            let Some(profile) = ok(scratch.decompose_profile(&[leaf(0, 0, 6, NonZeroU16::MIN)]))
            else {
                return;
            };
            assert_eq!(profile.counts.row_intervals, EDGE);
            assert_eq!(profile.counts.column_intervals, EDGE);
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn builder_reuses_leaf_events_after_clear_and_repeated_finish() {
        let mut builder = SparseLayerBuilder64::new();
        let mut scratch = SparseOptimalScratch64::new();
        for item in [leaf(0, 0, 4, nz(1)), leaf(0, 16, 4, nz(2))] {
            let _ = ok(builder.push_square(item.u, item.v, item.lod, item.value));
        }
        let Some(first) = ok(builder.finish(&mut scratch)) else {
            return;
        };
        assert_eq!(first.len(), 2);
        assert_eq!(builder.finish(&mut scratch), Ok(first));
        builder.clear();
        assert_eq!(builder.finish(&mut scratch), Ok(Vec::new()));
        let _ = ok(builder.push_square(32, 32, 5, nz(3)));
        assert_eq!(
            builder.finish(&mut scratch),
            Ok(vec![Rectangle {
                value: 3,
                x: RangeU8::new(32, 64),
                y: RangeU8::new(32, 64),
            }])
        );
    }

    #[test]
    fn endpoint_cut_masks_match_cell_coverage_with_adjacent_segments() {
        let vertical = [
            VerticalCut {
                x: 1,
                start: 0,
                end: 32,
            },
            VerticalCut {
                x: 1,
                start: 32,
                end: 64,
            },
            VerticalCut {
                x: 63,
                start: 1,
                end: 64,
            },
        ];
        let horizontal = [HorizontalCut {
            y: 63,
            start: 0,
            end: 64,
        }];
        let mut horizontal_masks = FixedVec::new(0u64);
        let mut vertical_masks = FixedVec::new(0u128);
        let Some(boundaries) = ok(build_cut_masks(
            &horizontal,
            &vertical,
            &mut horizontal_masks,
            &mut vertical_masks,
        )) else {
            return;
        };
        let mut expected = [0u128; EDGE];
        for cut in vertical {
            for y in cut.start..cut.end {
                *get_mut(&mut expected, usize::from(y)) |= 1u128 << cut.x;
            }
        }
        assert_eq!(vertical_masks.as_slice(), expected.as_slice());
        assert_eq!(*get(&horizontal_masks, 63), u64::MAX);
        assert_eq!(boundaries, 1 | (1u64 << 1) | (1u64 << 32) | (1u64 << 63));
    }

    #[test]
    fn serrated_border_pixels_produce_a_dense_valid_chord_graph() {
        let mut leaves = Vec::new();
        for y in 0..EDGE_U8 {
            for x in 0..EDGE_U8 {
                let vertical_border = x == 0 || x == EDGE_U8 - 1;
                let horizontal_border = y == 0 || y == EDGE_U8 - 1;
                let occupied = match (vertical_border, horizontal_border) {
                    (true, true) => false,
                    (true, false) => y % 2 == 1,
                    (false, true) => x % 2 == 1,
                    (false, false) => true,
                };
                if occupied {
                    leaves.push(leaf(x, y, 0, NonZeroU16::MIN));
                }
            }
        }
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_eq!(rectangles.len(), 124);
        let horizontal = &scratch.chord_groups.horizontal;
        let vertical = &scratch.chord_groups.vertical;
        assert_eq!(horizontal.len(), 122);
        assert_eq!(vertical.len(), 122);
        for chords in [horizontal, vertical] {
            let long_chords = chords
                .iter()
                .filter(|item| {
                    let chord = item.chord;
                    (chord.orientation == Orientation::Horizontal
                        && chord.x1 == 1
                        && chord.x2 == 63)
                        || (chord.orientation == Orientation::Vertical
                            && chord.y1 == 1
                            && chord.y2 == 63)
                })
                .count();
            assert_eq!(long_chords, 61);
        }
        let (graph, _) = crate::graph::build_sparse_conflict_graph_csr(
            horizontal,
            vertical,
            &mut scratch.matching.conflict,
        );
        assert_eq!(graph.edges.len(), 3964);
        for (left, horizontal_item) in horizontal.iter().enumerate() {
            let h = horizontal_item.chord;
            let expected: Vec<_> = vertical
                .iter()
                .enumerate()
                .filter_map(|(right, vertical_item)| {
                    let v = vertical_item.chord;
                    ((h.x1..=h.x2).contains(&v.x1) && (v.y1..=v.y2).contains(&h.y1))
                        .then(|| crate::u16_index(right))
                })
                .collect();
            let mut actual = graph.neighbors(left).to_vec();
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
        #[cfg(feature = "profile")]
        {
            let Some(profile) = ok(scratch.decompose_profile(&leaves)) else {
                return;
            };
            assert_eq!(profile.counts.greedy_matches, 122);
            assert_eq!(profile.counts.matching_phases, 0);
            assert_eq!(profile.counts.matching_augmentations, 0);
        }
    }

    #[test]
    fn single_full_leaf_outputs_one_rectangle() {
        let leaves = [leaf(0, 0, 6, nz(5))];
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_eq!(rectangles.len(), 1);
        let rectangle = rectangles.first();
        assert!(rectangle.is_some());
        let Some(rectangle) = rectangle else {
            return;
        };
        assert_eq!(rectangle.value, 5);
        assert_eq!(rectangle.x.start, 0);
        assert_eq!(rectangle.x.end, 64);
        assert_eq!(rectangle.y.start, 0);
        assert_eq!(rectangle.y.end, 64);
    }

    #[test]
    fn rejects_lod_out_of_range() {
        let leaves = [leaf(0, 0, 7, nz(1))];
        assert_from_leaves_error(&leaves, SparseQuadError::LodOutOfRange);
    }

    #[test]
    fn rejects_misaligned_leaf() {
        let leaves = [leaf(2, 0, 2, nz(1))];
        assert_from_leaves_error(&leaves, SparseQuadError::Misaligned);
    }

    #[test]
    fn rejects_out_of_bounds_leaf() {
        let leaves = [leaf(64, 0, 0, nz(1))];
        assert_from_leaves_error(&leaves, SparseQuadError::OutOfBounds);
    }

    #[test]
    fn rejects_overlapping_leaves() {
        let leaves = [leaf(0, 0, 6, nz(1)), leaf(0, 0, 5, nz(2))];
        assert_from_leaves_error(&leaves, SparseQuadError::Overlap);

        let mut scratch = SparseOptimalScratch64::new();
        assert_eq!(
            scratch.decompose_borrowed(&leaves),
            Err(SparseQuadError::Overlap)
        );

        #[cfg(feature = "alloc")]
        {
            let mut builder = SparseLayerBuilder64::new();
            for leaf in leaves {
                let _ = ok(builder.push_square(leaf.u, leaf.v, leaf.lod, leaf.value));
            }
            assert_eq!(builder.finish(&mut scratch), Err(SparseQuadError::Overlap));
        }
    }

    #[test]
    fn adjacent_same_value_leaves_decompose() {
        let leaves = [
            leaf(0, 0, 5, nz(1)),
            leaf(32, 0, 5, nz(1)),
            leaf(0, 32, 5, nz(1)),
            leaf(32, 32, 5, nz(1)),
        ];
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rects) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_ne!(rects, []);
    }

    #[test]
    fn l_shape_decomposes() {
        let leaves = [
            leaf(0, 0, 5, nz(1)),
            leaf(32, 0, 5, nz(1)),
            leaf(0, 32, 5, nz(1)),
        ];
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rects) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_ne!(rects, []);
    }

    #[test]
    fn multi_value_leaves_decompose() {
        let leaves = [
            leaf(0, 0, 5, nz(1)),
            leaf(32, 0, 5, nz(2)),
            leaf(0, 32, 5, nz(1)),
            leaf(32, 32, 4, nz(3)),
            leaf(48, 32, 4, nz(2)),
            leaf(32, 48, 4, nz(2)),
        ];
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rects) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_ne!(rects, []);
    }

    #[test]
    fn two_colored_regions_with_holes_keep_optimum_and_exact_coverage() {
        // 单个图案的冲突图为 C8；独立穷举像素矩形分区的最优值为 6。
        let pattern = [
            b"........",
            b"........",
            b"........",
            b"....#...",
            b"...####.",
            b".###.###",
            b".#######",
            b"...####.",
        ];
        let mut expected = [[0u16; EDGE]; EDGE];
        let mut leaves = Vec::new();
        for (origin, value) in [(0u8, 1u16), (16u8, u16::MAX)] {
            for (v, row) in (0u8..8).zip(pattern) {
                for (u, &pixel) in (0u8..8).zip(row) {
                    if pixel == b'#' {
                        *get_mut(
                            get_mut(&mut expected, usize::from(v)),
                            usize::from(origin + u),
                        ) = value;
                        leaves.push(leaf(origin + u, v, 0, nz(value)));
                    }
                }
            }
        }
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_eq!(rectangles.len(), 12);
        let mut actual = [[0u16; EDGE]; EDGE];
        for rectangle in rectangles {
            for y in rectangle.y.start..rectangle.y.end {
                for x in rectangle.x.start..rectangle.x.end {
                    let pixel = get_mut(get_mut(&mut actual, usize::from(y)), usize::from(x));
                    assert_eq!(*pixel, 0, "矩形不得重叠");
                    *pixel = rectangle.value;
                }
            }
        }
        assert_eq!(actual, expected);
        #[cfg(feature = "profile")]
        {
            let Some(profile) = ok(scratch.decompose_profile(&leaves)) else {
                return;
            };
            assert_eq!(profile.counts.rectangles, 12);
        }
    }

    #[test]
    fn alternating_holes_reach_both_chord_capacity_bounds() {
        let leaves: Vec<_> = (0..EDGE_U8)
            .flat_map(|v| {
                (0..EDGE_U8)
                    .filter(move |&u| u % 2 != 0 || v % 2 != 0)
                    .map(move |u| leaf(u, v, 0, NonZeroU16::MIN))
            })
            .collect();
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_eq!(rectangles.len(), 1025);
        for chords in [
            &scratch.chord_groups.horizontal,
            &scratch.chord_groups.vertical,
        ] {
            assert_eq!(chords.len(), crate::matching::IMAGE64_MAX_CHORDS);
            assert_eq!(chords.len(), chords.capacity());
        }
        #[cfg(feature = "alloc")]
        {
            let Some(image) = ok(SparseQuadImage64::from_leaves(&leaves)) else {
                return;
            };
            let Some(event_rectangles) = ok(scratch.decompose_quads_borrowed(&image)) else {
                return;
            };
            assert_eq!(event_rectangles.len(), 1025);
            for rectangle in event_rectangles {
                for y in rectangle.y.start..rectangle.y.end {
                    for x in rectangle.x.start..rectangle.x.end {
                        assert!(x % 2 != 0 || y % 2 != 0);
                    }
                }
            }
        }
    }

    #[test]
    fn many_four_pixel_colors_keep_independent_groups() {
        let mut leaves = Vec::new();
        for row in 0..32u8 {
            for column in 0..21u8 {
                let value = nz(u16::from(row) * 21 + u16::from(column) + 1);
                // 四像素 S 形，每种颜色恰好产生一条水平 chord。
                for (u, v) in [(1, 0), (2, 0), (0, 1), (1, 1)] {
                    leaves.push(leaf(column * 3 + u, row * 2 + v, 0, value));
                }
            }
        }
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_borrowed(&leaves)) else {
            return;
        };
        assert_eq!(rectangles.len(), 2 * 32 * 21);
        assert_eq!(scratch.chord_groups.logical_group_count(), 32 * 21);
        assert_eq!(scratch.chord_groups.horizontal.len(), 32 * 21);
        assert!(scratch.chord_groups.vertical.is_empty());
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // All indices are bounded by the fixed 3x3 oracle.
    fn global_chord_graph_preserves_colored_optima_on_all_three_by_three_images() {
        // Independent exact pixel-partition oracle. The first occupied pixel must
        // be the top-left corner of the rectangle that covers it; try every
        // legal bottom-right corner, then use the already solved smaller mask.
        let mut optimum = [0usize; 512];
        for mask in 1usize..512 {
            let first = mask.trailing_zeros() as usize;
            let (start_x, start_y) = (first % 3, first / 3);
            let mut best = 9;
            for end_y in start_y + 1..=3 {
                for end_x in start_x + 1..=3 {
                    let mut rectangle = 0usize;
                    for y in start_y..end_y {
                        for x in start_x..end_x {
                            rectangle |= 1 << (y * 3 + x);
                        }
                    }
                    if rectangle & mask == rectangle {
                        best = best.min(1 + optimum[mask ^ rectangle]);
                    }
                }
            }
            optimum[mask] = best;
        }

        let mut scratch = SparseOptimalScratch64::new();
        let mut leaves = Vec::with_capacity(9);
        for encoded in 0..3usize.pow(9) {
            leaves.clear();
            let mut remaining = encoded;
            let mut expected = [0u16; 9];
            let mut masks = [0usize; 3];
            for (pixel, expected_label) in expected.iter_mut().enumerate() {
                let label = remaining % 3;
                remaining /= 3;
                *expected_label = crate::u16_index(label);
                masks[label] |= 1 << pixel;
                if label != 0 {
                    leaves.push(leaf(
                        u8::try_from(pixel % 3).unwrap_or_default(),
                        u8::try_from(pixel / 3).unwrap_or_default(),
                        0,
                        nz(crate::u16_index(label)),
                    ));
                }
            }
            let Some(rectangles) = ok(scratch.decompose_borrowed(&leaves)) else {
                return;
            };
            assert_eq!(rectangles.len(), optimum[masks[1]] + optimum[masks[2]]);
            let mut actual = [0u16; 9];
            for rectangle in rectangles {
                assert!(rectangle.x.end <= 3 && rectangle.y.end <= 3);
                for y in rectangle.y.start..rectangle.y.end {
                    for x in rectangle.x.start..rectangle.x.end {
                        let pixel = usize::from(y) * 3 + usize::from(x);
                        assert_eq!(actual[pixel], 0);
                        actual[pixel] = rectangle.value;
                    }
                }
            }
            assert_eq!(actual, expected);

            // These geometric invariants justify both the single global graph
            // and the O(n^2) total chord-grid-writing bound across all labels.
            for chords in [
                &scratch.chord_groups.horizontal,
                &scratch.chord_groups.vertical,
            ] {
                for (index, first) in chords.iter().enumerate() {
                    for second in chords.iter().skip(index + 1) {
                        let (a, b) = (first.chord, second.chord);
                        if a.orientation == Orientation::Horizontal && a.y1 == b.y1 {
                            assert!(a.x2 < b.x1 || b.x2 < a.x1);
                        }
                        if a.orientation == Orientation::Vertical && a.x1 == b.x1 {
                            assert!(a.y2 < b.y1 || b.y2 < a.y1);
                        }
                    }
                }
            }
            for horizontal in &scratch.chord_groups.horizontal {
                for vertical in &scratch.chord_groups.vertical {
                    let (h, v) = (horizontal.chord, vertical.chord);
                    if (h.x1..=h.x2).contains(&v.x1) && (v.y1..=v.y2).contains(&h.y1) {
                        assert_eq!(horizontal.value, vertical.value);
                    }
                }
            }
        }
    }

    #[test]
    #[allow(clippy::indexing_slicing)] // 穷举固定的两个四像素行，所有访问由 0..4 界定。
    fn interval_chords_match_pixel_corner_oracle() {
        use crate::corners::{Corners, horizontal_support_from_corners};

        fn intervals(pixels: [u16; 4], origin: u8) -> Vec<Interval> {
            let mut output = Vec::<Interval>::new();
            for (x, value) in (origin..origin + 4).zip(pixels) {
                if value == 0 {
                    continue;
                }
                if let Some(last) = output.last_mut()
                    && last.value == value
                    && last.end == x
                {
                    last.end = x + 1;
                } else {
                    output.push(Interval {
                        start: x,
                        end: x + 1,
                        value,
                    });
                }
            }
            output
        }

        // 两行、每像素背景或两种颜色，覆盖全部 3^8 种组合；分别贴左右图边界。
        for mut encoded in 0..3usize.pow(8) {
            let mut first = [0u16; 4];
            let mut second = [0u16; 4];
            for pixel in first.iter_mut().chain(&mut second) {
                *pixel = crate::u16_index(encoded % 3);
                encoded /= 3;
            }
            for origin in [0u8, 60] {
                let mut expected = Vec::new();
                for start in 0u8..4 {
                    let s = usize::from(start);
                    let value = first[s];
                    if value == 0 {
                        continue;
                    }
                    for end in start + 1..=4 {
                        let e = usize::from(end);
                        if !first[s..e].iter().chain(&second[s..e]).all(|&v| v == value) {
                            continue;
                        }
                        let left = Corners::from_flags([
                            s > 0 && first[s - 1] == value,
                            true,
                            s > 0 && second[s - 1] == value,
                            true,
                        ]);
                        let right = Corners::from_flags([
                            true,
                            e < 4 && first[e] == value,
                            true,
                            e < 4 && second[e] == value,
                        ]);
                        if matches!(horizontal_support_from_corners(left), Some((true, _)))
                            && matches!(horizontal_support_from_corners(right), Some((_, true)))
                        {
                            expected.push(Interval {
                                start: origin + start,
                                end: origin + end,
                                value,
                            });
                        }
                    }
                }
                let mut actual = Vec::new();
                assert!(
                    emit_chords_for_boundary(
                        &intervals(first, origin),
                        &intervals(second, origin),
                        |chord| {
                            actual.push(chord);
                            Ok(())
                        },
                    )
                    .is_ok()
                );
                assert_eq!(
                    actual, expected,
                    "first={first:?}, second={second:?}, origin={origin}"
                );
            }
        }
    }
}
