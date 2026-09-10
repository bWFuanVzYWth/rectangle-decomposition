//! 固定 64x64 sparse quad 输入的矩形分解。
//!
//! 输入为非零 dyadic square 列表，内部派生事件、区间、chord 与 cut，
//! 不创建 dense 像素图。

use std::num::NonZeroU16;
#[cfg(feature = "profile")]
use std::time::{Duration, Instant};

use crate::corners::{Corners, horizontal_support_from_corners, vertical_support_from_corners};
use crate::matching::MatchingScratch;
use crate::types::{ActiveRect, ChordAccess, EffectiveChord, Orientation, RangeU8, Rectangle, Run};
use crate::{get, slice, slice_mut};

const EDGE: usize = 64;
const EDGE_U8: u8 = 64;
const MAX_LOD: u8 = 6;
const MAX_AXIS_INTERVALS: usize = EDGE * EDGE;
const MAX_AXIS_EVENTS: usize = MAX_AXIS_INTERVALS * 2;
const MAX_CHORDS_PER_ORIENTATION: usize = EDGE * (EDGE - 1);
const MAX_RECTANGLES: usize = EDGE * EDGE;

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
    sort_cuts: Duration,
    greedy_matches: usize,
    matching_phases: usize,
    matching_augmentations: usize,
}

/// 固定 64x64 sparse quad image。
#[derive(Clone, Debug, Default)]
pub struct SparseQuadImage64 {
    leaves: Vec<StoredLeaf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StoredLeaf {
    leaf: QuadLeaf64,
    morton_start: u64,
    morton_end: u64,
}

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

        for &leaf in leaves {
            stored.push(validate_leaf(leaf)?);
        }

        stored.sort_unstable_by_key(|leaf| leaf.morton_start);
        let mut previous_end = None::<u64>;
        for leaf in &stored {
            if previous_end.is_some_and(|end| end > leaf.morton_start) {
                return Err(SparseQuadError::Overlap);
            }
            previous_end = Some(leaf.morton_end);
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

/// sparse optimal 分解的可复用 scratch。
pub struct SparseOptimalScratch64 {
    row_events: Vec<AxisEvent>,
    column_events: Vec<AxisEvent>,
    row_buckets: [IntervalBucket; EDGE],
    column_buckets: [IntervalBucket; EDGE],
    rows: AxisIntervals,
    columns: AxisIntervals,
    active_intervals: Vec<Interval>,
    normalized_intervals: Vec<Interval>,
    chord_groups: SparseChordGroups,
    matching: MatchingScratch,
    horizontal_cuts: Vec<HorizontalCut>,
    vertical_cuts: Vec<VerticalCut>,
    horizontal_cut_masks: Vec<u64>,
    vertical_cut_masks: Vec<u128>,
    runs: Vec<Run>,
    active_rects: Vec<ActiveRect>,
    next_active_rects: Vec<ActiveRect>,
    rectangles: Vec<Rectangle>,
}

impl Default for SparseOptimalScratch64 {
    fn default() -> Self {
        Self {
            row_events: Vec::new(),
            column_events: Vec::new(),
            row_buckets: std::array::from_fn(|_| IntervalBucket::new()),
            column_buckets: std::array::from_fn(|_| IntervalBucket::new()),
            rows: AxisIntervals::default(),
            columns: AxisIntervals::default(),
            active_intervals: Vec::new(),
            normalized_intervals: Vec::new(),
            chord_groups: SparseChordGroups::default(),
            matching: MatchingScratch::default(),
            horizontal_cuts: Vec::new(),
            vertical_cuts: Vec::new(),
            horizontal_cut_masks: Vec::new(),
            vertical_cut_masks: Vec::new(),
            runs: Vec::new(),
            active_rects: Vec::new(),
            next_active_rects: Vec::new(),
            rectangles: Vec::new(),
        }
    }
}

impl SparseOptimalScratch64 {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建并预分配 64x64 hot path 所需 scratch。
    ///
    /// 资源契约：成功后使用 `decompose_borrowed` 不需要为固定上界缓存扩容。
    ///
    /// # Errors
    ///
    /// 内部缓存无法分配时返回错误。
    pub fn try_new_preallocated() -> Result<Self, SparseQuadError> {
        let mut scratch = Self::new();
        scratch.preallocate_64()?;
        Ok(scratch)
    }

    /// 预分配 64x64 hot path 所需 scratch。
    ///
    /// # Errors
    ///
    /// 内部缓存无法分配时返回错误。
    pub fn preallocate_64(&mut self) -> Result<(), SparseQuadError> {
        preallocate_exact(&mut self.row_events, MAX_AXIS_EVENTS)?;
        preallocate_exact(&mut self.column_events, MAX_AXIS_EVENTS)?;
        self.rows.preallocate_64()?;
        self.columns.preallocate_64()?;
        preallocate_exact(&mut self.active_intervals, EDGE)?;
        preallocate_exact(&mut self.normalized_intervals, EDGE)?;
        self.chord_groups.preallocate_64()?;
        self.matching.preallocate_64()?;
        preallocate_exact(&mut self.horizontal_cuts, MAX_CHORDS_PER_ORIENTATION)?;
        preallocate_exact(&mut self.vertical_cuts, MAX_CHORDS_PER_ORIENTATION)?;
        preallocate_exact(&mut self.horizontal_cut_masks, EDGE)?;
        preallocate_exact(&mut self.vertical_cut_masks, EDGE)?;
        preallocate_exact(&mut self.runs, EDGE)?;
        preallocate_exact(&mut self.active_rects, EDGE)?;
        preallocate_exact(&mut self.next_active_rects, EDGE)?;
        preallocate_exact(&mut self.rectangles, MAX_RECTANGLES)?;
        Ok(())
    }

    /// 对 sparse quad leaves 执行 64x64 最优分解。
    ///
    /// 输入不需要预排序；函数会校验 dyadic 对齐、边界与互不重叠。
    ///
    /// # Errors
    ///
    /// leaf 越界、未按 lod 对齐、lod 超出范围、互相重叠或内部缓存无法分配时返回错误。
    pub fn decompose(&mut self, leaves: &[QuadLeaf64]) -> Result<Vec<Rectangle>, SparseQuadError> {
        self.preallocate_64()?;
        Ok(self.decompose_borrowed(leaves)?.to_vec())
    }

    /// 对 sparse quad leaves 执行 64x64 最优分解，结果借用自 scratch。
    ///
    /// 资源契约：调用 `preallocate_64` 成功后，本函数不需要扩容固定上界缓存。
    ///
    /// # Errors
    ///
    /// leaf 越界、未按 lod 对齐、lod 超出范围、互相重叠或内部缓存无法分配时返回错误。
    pub fn decompose_borrowed(
        &mut self,
        leaves: &[QuadLeaf64],
    ) -> Result<&[Rectangle], SparseQuadError> {
        build_axis_intervals_from_leaves(
            leaves,
            &mut self.rows,
            &mut self.columns,
            &mut self.row_buckets,
            &mut self.column_buckets,
        )?;
        self.decompose_intervals_borrowed()
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
        self.preallocate_64()?;
        let total_start = Instant::now();
        let axis_start = Instant::now();
        build_axis_intervals_from_leaves(
            leaves,
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
            row_intervals: self.rows.intervals.len(),
            column_intervals: self.columns.intervals.len(),
            chord_groups: self.chord_groups.groups.len(),
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
                sort_cuts: select_timings.sort_cuts,
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
    pub fn decompose_quads(
        &mut self,
        image: &SparseQuadImage64,
    ) -> Result<Vec<Rectangle>, SparseQuadError> {
        self.preallocate_64()?;
        Ok(self.decompose_quads_borrowed(image)?.to_vec())
    }

    /// 对 sparse quad image 执行 64x64 最优分解，结果借用自 scratch。
    ///
    /// # Errors
    ///
    /// 当内部缓存无法分配时返回错误。
    pub fn decompose_quads_borrowed(
        &mut self,
        image: &SparseQuadImage64,
    ) -> Result<&[Rectangle], SparseQuadError> {
        build_axis_events(image, &mut self.row_events, &mut self.column_events)?;
        self.decompose_events_borrowed(false)
    }

    fn decompose_events_borrowed(
        &mut self,
        validate_row_overlap: bool,
    ) -> Result<&[Rectangle], SparseQuadError> {
        build_axis_intervals(
            &mut self.row_events,
            &mut self.rows,
            &mut self.active_intervals,
            &mut self.normalized_intervals,
            validate_row_overlap,
        )?;
        build_axis_intervals(
            &mut self.column_events,
            &mut self.columns,
            &mut self.active_intervals,
            &mut self.normalized_intervals,
            false,
        )?;

        self.decompose_intervals_borrowed()
    }

    fn decompose_intervals(&mut self) -> Result<Vec<Rectangle>, SparseQuadError> {
        Ok(self.decompose_intervals_borrowed()?.to_vec())
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

        let horizontal_cap = groups.horizontal.len();
        let vertical_cap = groups.vertical.len();
        reserve_exact(horizontal_cuts, horizontal_cap)?;
        reserve_exact(vertical_cuts, vertical_cap)?;

        for group in &groups.groups {
            let horizontal = groups.horizontal(group);
            let vertical = groups.vertical(group);
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

        horizontal_cuts.sort_unstable_by_key(|cut| (cut.y, cut.start, cut.end));
        vertical_cuts.sort_unstable_by_key(|cut| (cut.start, cut.end, cut.x));
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

        let horizontal_cap = groups.horizontal.len();
        let vertical_cap = groups.vertical.len();
        reserve_exact(horizontal_cuts, horizontal_cap)?;
        reserve_exact(vertical_cuts, vertical_cap)?;

        let mut timings = SelectCutTimings::default();
        for group in &groups.groups {
            let horizontal = groups.horizontal(group);
            let vertical = groups.vertical(group);
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

        let sort_start = Instant::now();
        horizontal_cuts.sort_unstable_by_key(|cut| (cut.y, cut.start, cut.end));
        vertical_cuts.sort_unstable_by_key(|cut| (cut.start, cut.end, cut.x));
        timings.sort_cuts = sort_start.elapsed();
        Ok(timings)
    }

    fn sparse_partition(&mut self) -> Result<&[Rectangle], SparseQuadError> {
        build_cut_masks(
            &self.horizontal_cuts,
            &self.vertical_cuts,
            &mut self.horizontal_cut_masks,
            &mut self.vertical_cut_masks,
        )?;

        let rows = &self.rows;
        let horizontal_cut_masks = &self.horizontal_cut_masks;
        let vertical_cut_masks = &self.vertical_cut_masks;
        let runs = &mut self.runs;
        let active = &mut self.active_rects;
        let next_active = &mut self.next_active_rects;

        let result = &mut self.rectangles;
        result.clear();
        reserve_exact(result, rows.intervals.len())?;
        active.clear();
        next_active.clear();

        let Some(&first_vertical_cut_mask) = vertical_cut_masks.first() else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        build_runs_for_row(rows.line(0), first_vertical_cut_mask, runs)?;
        reserve_exact(active, runs.len())?;
        for &run in runs.as_slice() {
            active.push(ActiveRect::new(run.value, run.x_start, run.x_end, 0));
        }

        for y in 1u8..EDGE_U8 {
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
                result,
            )?;
            std::mem::swap(active, next_active);
        }

        for &active_rect in active.as_slice() {
            emit(active_rect, EDGE_U8, result)?;
        }

        Ok(result.as_slice())
    }
}

/// 固定 64x64 sparse layer 的增量构建器。
///
/// 只接受非零 dyadic square。builder 内部直接维护 row / column interval list，
/// 不创建 dense image，也不要求调用方预合并相邻 square。
#[derive(Debug, Default)]
pub struct SparseLayerBuilder64 {
    row_intervals: Vec<LineInterval>,
    column_intervals: Vec<LineInterval>,
    square_count: usize,
}

impl SparseLayerBuilder64 {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.row_intervals.clear();
        self.column_intervals.clear();
        self.square_count = 0;
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
        push_leaf_line_intervals(
            QuadLeaf64 { u, v, lod, value },
            &mut self.row_intervals,
            &mut self.column_intervals,
        )?;
        self.square_count = self
            .square_count
            .checked_add(1)
            .ok_or(SparseQuadError::CapacityOverflow)?;
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
        if self.square_count == 0 {
            return Ok(Vec::new());
        }

        scratch.preallocate_64()?;
        build_axis_intervals_from_line_intervals(&mut self.row_intervals, &mut scratch.rows, true)?;
        build_axis_intervals_from_line_intervals(
            &mut self.column_intervals,
            &mut scratch.columns,
            false,
        )?;
        scratch.decompose_intervals()
    }
}

// Leaf 校验

fn validate_leaf(leaf: QuadLeaf64) -> Result<StoredLeaf, SparseQuadError> {
    validate_leaf_shape(leaf)?;

    let morton_start = crate::morton::encode(leaf.u, leaf.v);
    let morton_len = 1u64 << (u32::from(leaf.lod) * 2);
    let Some(morton_end) = morton_start.checked_add(morton_len) else {
        return Err(SparseQuadError::CapacityOverflow);
    };

    Ok(StoredLeaf {
        leaf,
        morton_start,
        morton_end,
    })
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

/// 行内有效区间为 `items[..len]`，最多 64 个；清空只重置长度。
struct IntervalBucket {
    items: [Interval; EDGE],
    len: u8,
}

impl IntervalBucket {
    const fn new() -> Self {
        Self {
            items: [Interval {
                start: 0,
                end: 0,
                value: 0,
            }; EDGE],
            len: 0,
        }
    }

    fn try_push(&mut self, interval: Interval) -> Result<(), SparseQuadError> {
        let slot = self
            .items
            .get_mut(usize::from(self.len))
            .ok_or(SparseQuadError::Overlap)?;
        *slot = interval;
        self.len += 1;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LineInterval {
    line: u8,
    interval: Interval,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AxisEvent {
    coord: u8,
    is_end: bool,
    interval: Interval,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LineRange {
    start: usize,
    end: usize,
}

#[derive(Default)]
struct AxisIntervals {
    lines: Vec<LineRange>,
    intervals: Vec<Interval>,
}

impl AxisIntervals {
    fn preallocate_64(&mut self) -> Result<(), SparseQuadError> {
        preallocate_exact(&mut self.lines, EDGE)?;
        preallocate_exact(&mut self.intervals, MAX_AXIS_INTERVALS)
    }

    fn clear_for_build(&mut self) -> Result<(), SparseQuadError> {
        self.lines.clear();
        self.intervals.clear();
        reserve_exact(&self.lines, EDGE)?;
        self.lines.resize(EDGE, LineRange::default());
        Ok(())
    }

    fn line(&self, index: usize) -> &[Interval] {
        let range = get(&self.lines, index);
        slice(&self.intervals, range.start..range.end)
    }
}

// 轴事件构造

fn build_axis_events(
    image: &SparseQuadImage64,
    row_events: &mut Vec<AxisEvent>,
    column_events: &mut Vec<AxisEvent>,
) -> Result<(), SparseQuadError> {
    row_events.clear();
    column_events.clear();

    let event_count = mesh_capacity(image.leaves.len(), 2)?;
    reserve_exact(row_events, event_count)?;
    reserve_exact(column_events, event_count)?;

    for stored in &image.leaves {
        let leaf = stored.leaf;
        let size = quad_size(leaf.lod);
        push_axis_events(leaf, size, row_events, column_events);
    }

    Ok(())
}

fn push_axis_events(
    leaf: QuadLeaf64,
    size: u8,
    row_events: &mut Vec<AxisEvent>,
    column_events: &mut Vec<AxisEvent>,
) {
    let u1 = leaf.u + size;
    let v1 = leaf.v + size;
    let value = leaf.value.get();

    let row_interval = Interval {
        start: leaf.u,
        end: u1,
        value,
    };
    row_events.push(AxisEvent {
        coord: leaf.v,
        is_end: false,
        interval: row_interval,
    });
    row_events.push(AxisEvent {
        coord: v1,
        is_end: true,
        interval: row_interval,
    });

    let column_interval = Interval {
        start: leaf.v,
        end: v1,
        value,
    };
    column_events.push(AxisEvent {
        coord: leaf.u,
        is_end: false,
        interval: column_interval,
    });
    column_events.push(AxisEvent {
        coord: u1,
        is_end: true,
        interval: column_interval,
    });
}

// 从 leaves 构造轴区间

fn build_axis_intervals_from_leaves(
    leaves: &[QuadLeaf64],
    rows: &mut AxisIntervals,
    columns: &mut AxisIntervals,
    row_buckets: &mut [IntervalBucket; EDGE],
    column_buckets: &mut [IntervalBucket; EDGE],
) -> Result<(), SparseQuadError> {
    prepare_interval_buckets(row_buckets);
    prepare_interval_buckets(column_buckets);
    push_leaf_axis_intervals(leaves, row_buckets, column_buckets)?;

    build_axis_intervals_from_buckets(row_buckets, rows, true)?;
    build_axis_intervals_from_buckets(column_buckets, columns, false)
}

fn prepare_interval_buckets(buckets: &mut [IntervalBucket; EDGE]) {
    for bucket in buckets {
        bucket.len = 0;
    }
}

fn push_leaf_axis_intervals(
    leaves: &[QuadLeaf64],
    row_buckets: &mut [IntervalBucket],
    column_buckets: &mut [IntervalBucket],
) -> Result<(), SparseQuadError> {
    for &leaf in leaves {
        push_leaf_axis_intervals_one(leaf, row_buckets, column_buckets)?;
    }
    Ok(())
}

fn push_leaf_line_intervals(
    leaf: QuadLeaf64,
    row_intervals: &mut Vec<LineInterval>,
    column_intervals: &mut Vec<LineInterval>,
) -> Result<(), SparseQuadError> {
    let size = validate_leaf_shape(leaf)?;
    let u1 = leaf.u + size;
    let v1 = leaf.v + size;
    let value = leaf.value.get();

    allocate_more(row_intervals, usize::from(size))?;
    let row_interval = Interval {
        start: leaf.u,
        end: u1,
        value,
    };
    for line in leaf.v..v1 {
        row_intervals.push(LineInterval {
            line,
            interval: row_interval,
        });
    }

    allocate_more(column_intervals, usize::from(size))?;
    let column_interval = Interval {
        start: leaf.v,
        end: v1,
        value,
    };
    for line in leaf.u..u1 {
        column_intervals.push(LineInterval {
            line,
            interval: column_interval,
        });
    }
    Ok(())
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
    for row in leaf.v..v1 {
        let Some(bucket) = row_buckets.get_mut(usize::from(row)) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        bucket.try_push(row_interval)?;
    }

    let column_interval = Interval {
        start: leaf.v,
        end: v1,
        value,
    };
    for column in leaf.u..u1 {
        let Some(bucket) = column_buckets.get_mut(usize::from(column)) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        bucket.try_push(column_interval)?;
    }
    Ok(())
}

// 轴区间构建

fn build_axis_intervals_from_line_intervals(
    line_intervals: &mut [LineInterval],
    output: &mut AxisIntervals,
    validate_overlap: bool,
) -> Result<(), SparseQuadError> {
    line_intervals.sort_unstable_by_key(|item| (item.line, item.interval.start));
    output.clear_for_build()?;

    let mut cursor = 0usize;
    for line in 0..EDGE {
        let line_coord = u8::try_from(line).map_err(|_| SparseQuadError::CapacityOverflow)?;
        let start = output.intervals.len();
        let mut previous_end = None::<u8>;
        while let Some(item) = line_intervals
            .get(cursor)
            .copied()
            .filter(|item| item.line == line_coord)
        {
            if validate_overlap && previous_end.is_some_and(|end| end > item.interval.start) {
                return Err(SparseQuadError::Overlap);
            }
            previous_end = Some(item.interval.end);
            push_merged_interval(&mut output.intervals, start, item.interval)?;
            cursor += 1;
        }
        let end = output.intervals.len();
        let Some(slot) = output.lines.get_mut(line) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        *slot = LineRange { start, end };
    }
    Ok(())
}

fn build_axis_intervals_from_buckets(
    buckets: &mut [IntervalBucket; EDGE],
    output: &mut AxisIntervals,
    validate_overlap: bool,
) -> Result<(), SparseQuadError> {
    output.clear_for_build()?;
    for (line, bucket) in buckets.iter_mut().enumerate() {
        let bucket = slice_mut(&mut bucket.items, 0..usize::from(bucket.len));
        bucket.sort_unstable_by_key(|interval| interval.start);
        if validate_overlap {
            validate_non_overlapping_line(bucket)?;
        }

        let start = output.intervals.len();
        reserve_exact(&output.intervals, bucket.len())?;
        for &interval in bucket.iter() {
            push_merged_interval(&mut output.intervals, start, interval)?;
        }
        let end = output.intervals.len();
        let Some(slot) = output.lines.get_mut(line) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        *slot = LineRange { start, end };
    }
    Ok(())
}

fn build_axis_intervals(
    events: &mut [AxisEvent],
    output: &mut AxisIntervals,
    active: &mut Vec<Interval>,
    normalized: &mut Vec<Interval>,
    validate_overlap: bool,
) -> Result<(), SparseQuadError> {
    events.sort_unstable_by_key(|event| (event.coord, !event.is_end));
    output.clear_for_build()?;
    active.clear();
    normalized.clear();

    let mut cursor = 0usize;
    for line in 0..EDGE {
        let line_coord = u8::try_from(line).map_err(|_| SparseQuadError::CapacityOverflow)?;
        while let Some(event) = events
            .get(cursor)
            .copied()
            .filter(|event| event.coord == line_coord)
        {
            if event.is_end {
                remove_active_interval(active, event.interval);
            } else {
                active.push(event.interval);
            }
            cursor += 1;
        }

        normalized.clear();
        reserve_exact(normalized, active.len())?;
        normalized.extend_from_slice(active);
        normalized.sort_unstable_by_key(|interval| interval.start);
        if validate_overlap {
            validate_non_overlapping_line(normalized)?;
        }

        let start = output.intervals.len();
        reserve_exact(&output.intervals, normalized.len())?;
        for &interval in normalized.as_slice() {
            push_merged_interval(&mut output.intervals, start, interval)?;
        }
        let end = output.intervals.len();
        let Some(slot) = output.lines.get_mut(line) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        *slot = LineRange { start, end };
    }

    Ok(())
}

fn push_merged_interval(
    output: &mut Vec<Interval>,
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

fn validate_non_overlapping_line(intervals: &[Interval]) -> Result<(), SparseQuadError> {
    let mut previous_end = None::<u8>;
    for interval in intervals {
        if previous_end.is_some_and(|end| end > interval.start) {
            return Err(SparseQuadError::Overlap);
        }
        previous_end = Some(interval.end);
    }
    Ok(())
}

fn remove_active_interval(active: &mut Vec<Interval>, interval: Interval) {
    let Some(position) = active.iter().position(|&item| item == interval) else {
        std::process::abort();
    };
    active.swap_remove(position);
}

// Chord 提取

#[derive(Default)]
struct SparseChordGroups {
    groups: Vec<SparseChordGroup>,
    horizontal: Vec<ValuedChord>,
    vertical: Vec<ValuedChord>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SparseChordGroup {
    horizontal_start: usize,
    horizontal_end: usize,
    vertical_start: usize,
    vertical_end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ValuedChord {
    value: u16,
    order: u16,
    chord: EffectiveChord,
}

impl ChordAccess for ValuedChord {
    fn chord(&self) -> EffectiveChord {
        self.chord
    }
}

impl SparseChordGroups {
    fn preallocate_64(&mut self) -> Result<(), SparseQuadError> {
        preallocate_exact(&mut self.groups, MAX_AXIS_INTERVALS)?;
        preallocate_exact(&mut self.horizontal, MAX_CHORDS_PER_ORIENTATION)?;
        preallocate_exact(&mut self.vertical, MAX_CHORDS_PER_ORIENTATION)
    }

    fn clear(&mut self) {
        self.groups.clear();
        self.horizontal.clear();
        self.vertical.clear();
    }

    fn add_horizontal(&mut self, value: u16, chord: EffectiveChord) -> Result<(), SparseQuadError> {
        reserve_one(&self.horizontal)?;
        let order = crate::u16_index(self.horizontal.len());
        self.horizontal.push(ValuedChord {
            value,
            order,
            chord,
        });
        Ok(())
    }

    fn add_vertical(&mut self, value: u16, chord: EffectiveChord) -> Result<(), SparseQuadError> {
        reserve_one(&self.vertical)?;
        let order = crate::u16_index(self.vertical.len());
        self.vertical.push(ValuedChord {
            value,
            order,
            chord,
        });
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SparseQuadError> {
        sort_valued_chords(&mut self.horizontal);
        sort_valued_chords(&mut self.vertical);
        self.groups.clear();

        let (mut horizontal_index, mut vertical_index) = (0usize, 0usize);
        while horizontal_index < self.horizontal.len() || vertical_index < self.vertical.len() {
            let value = match (
                self.horizontal.get(horizontal_index),
                self.vertical.get(vertical_index),
            ) {
                (Some(horizontal), Some(vertical)) => horizontal.value.min(vertical.value),
                (Some(horizontal), None) => horizontal.value,
                (None, Some(vertical)) => vertical.value,
                (None, None) => break,
            };

            let horizontal_start = horizontal_index;
            while self
                .horizontal
                .get(horizontal_index)
                .is_some_and(|item| item.value == value)
            {
                horizontal_index += 1;
            }
            let vertical_start = vertical_index;
            while self
                .vertical
                .get(vertical_index)
                .is_some_and(|item| item.value == value)
            {
                vertical_index += 1;
            }

            reserve_one(&self.groups)?;
            self.groups.push(SparseChordGroup {
                horizontal_start,
                horizontal_end: horizontal_index,
                vertical_start,
                vertical_end: vertical_index,
            });
        }
        Ok(())
    }

    fn horizontal(&self, group: &SparseChordGroup) -> &[ValuedChord] {
        slice(
            &self.horizontal,
            group.horizontal_start..group.horizontal_end,
        )
    }

    fn vertical(&self, group: &SparseChordGroup) -> &[ValuedChord] {
        slice(&self.vertical, group.vertical_start..group.vertical_end)
    }
}

fn sort_valued_chords(chords: &mut [ValuedChord]) {
    let mut previous = match chords.first() {
        Some(first) => (first.value, first.order),
        None => return,
    };

    for chord in chords.iter().skip(1) {
        let key = (chord.value, chord.order);
        if previous > key {
            chords.sort_unstable_by_key(|item| (item.value, item.order));
            return;
        }
        previous = key;
    }
}

fn extract_sparse_chords(
    rows: &AxisIntervals,
    columns: &AxisIntervals,
    groups: &mut SparseChordGroups,
) -> Result<(), SparseQuadError> {
    groups.clear();
    for y in 1u8..EDGE_U8 {
        emit_horizontal_chords_for_boundary(
            rows.line(usize::from(y - 1)),
            rows.line(usize::from(y)),
            y,
            groups,
        )?;
    }
    for x in 1u8..EDGE_U8 {
        emit_vertical_chords_for_boundary(
            columns.line(usize::from(x - 1)),
            columns.line(usize::from(x)),
            x,
            groups,
        )?;
    }
    groups.finish()?;
    Ok(())
}

fn emit_horizontal_chords_for_boundary(
    upper: &[Interval],
    lower: &[Interval],
    y: u8,
    groups: &mut SparseChordGroups,
) -> Result<(), SparseQuadError> {
    let (mut upper_index, mut lower_index) = (0usize, 0usize);
    while upper_index < upper.len() && lower_index < lower.len() {
        let (Some(&a), Some(&b)) = (upper.get(upper_index), lower.get(lower_index)) else {
            break;
        };
        let start = a.start.max(b.start);
        let end = a.end.min(b.end);
        if a.value == b.value && start < end {
            try_emit_horizontal_chord(
                LineCursor {
                    intervals: upper,
                    index: upper_index,
                },
                LineCursor {
                    intervals: lower,
                    index: lower_index,
                },
                y,
                start,
                end,
                a.value,
                groups,
            )?;
        }
        if a.end <= b.end {
            upper_index += 1;
        } else {
            lower_index += 1;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct LineCursor<'a> {
    intervals: &'a [Interval],
    index: usize,
}

fn try_emit_horizontal_chord(
    upper: LineCursor<'_>,
    lower: LineCursor<'_>,
    y: u8,
    start: u8,
    end: u8,
    value: u16,
    groups: &mut SparseChordGroups,
) -> Result<(), SparseQuadError> {
    let left_upper = start > 0 && value_at_cursor(upper, start - 1) == value;
    let left_lower = start > 0 && value_at_cursor(lower, start - 1) == value;
    let Some((supports_right, _)) =
        horizontal_support_from_corners(Corners::from_flags([left_upper, true, left_lower, true]))
    else {
        return Ok(());
    };
    if !supports_right {
        return Ok(());
    }

    let right_upper = end < EDGE_U8 && value_at_cursor(upper, end) == value;
    let right_lower = end < EDGE_U8 && value_at_cursor(lower, end) == value;
    let Some((_, supports_left)) = horizontal_support_from_corners(Corners::from_flags([
        true,
        right_upper,
        true,
        right_lower,
    ])) else {
        return Ok(());
    };
    if supports_left {
        groups.add_horizontal(
            value,
            EffectiveChord {
                orientation: Orientation::Horizontal,
                x1: start,
                y1: y,
                x2: end,
                y2: y,
            },
        )?;
    }
    Ok(())
}

fn emit_vertical_chords_for_boundary(
    left: &[Interval],
    right: &[Interval],
    x: u8,
    groups: &mut SparseChordGroups,
) -> Result<(), SparseQuadError> {
    let (mut left_index, mut right_index) = (0usize, 0usize);
    while left_index < left.len() && right_index < right.len() {
        let (Some(&a), Some(&b)) = (left.get(left_index), right.get(right_index)) else {
            break;
        };
        let start = a.start.max(b.start);
        let end = a.end.min(b.end);
        if a.value == b.value && start < end {
            try_emit_vertical_chord(
                LineCursor {
                    intervals: left,
                    index: left_index,
                },
                LineCursor {
                    intervals: right,
                    index: right_index,
                },
                x,
                start,
                end,
                a.value,
                groups,
            )?;
        }
        if a.end <= b.end {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    Ok(())
}

fn try_emit_vertical_chord(
    left: LineCursor<'_>,
    right: LineCursor<'_>,
    x: u8,
    start: u8,
    end: u8,
    value: u16,
    groups: &mut SparseChordGroups,
) -> Result<(), SparseQuadError> {
    let top_left = start > 0 && value_at_cursor(left, start - 1) == value;
    let top_right = start > 0 && value_at_cursor(right, start - 1) == value;
    let Some((supports_down, _)) =
        vertical_support_from_corners(Corners::from_flags([top_left, top_right, true, true]))
    else {
        return Ok(());
    };
    if !supports_down {
        return Ok(());
    }

    let bottom_left = end < EDGE_U8 && value_at_cursor(left, end) == value;
    let bottom_right = end < EDGE_U8 && value_at_cursor(right, end) == value;
    let Some((_, supports_up)) =
        vertical_support_from_corners(Corners::from_flags([true, true, bottom_left, bottom_right]))
    else {
        return Ok(());
    };
    if supports_up {
        groups.add_vertical(
            value,
            EffectiveChord {
                orientation: Orientation::Vertical,
                x1: x,
                y1: start,
                x2: x,
                y2: end,
            },
        )?;
    }
    Ok(())
}

fn value_at_cursor(cursor: LineCursor<'_>, coord: u8) -> u16 {
    let Some(interval) = cursor.intervals.get(cursor.index) else {
        return 0;
    };
    if interval.start <= coord && coord < interval.end {
        return interval.value;
    }
    if coord < interval.start {
        return cursor
            .index
            .checked_sub(1)
            .and_then(|index| cursor.intervals.get(index))
            .filter(|candidate| candidate.start <= coord && coord < candidate.end)
            .map_or(0, |candidate| candidate.value);
    }
    cursor
        .intervals
        .get(cursor.index + 1)
        .filter(|candidate| candidate.start <= coord && coord < candidate.end)
        .map_or(0, |candidate| candidate.value)
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

fn build_cut_masks(
    horizontal_cuts: &[HorizontalCut],
    vertical_cuts: &[VerticalCut],
    horizontal_masks: &mut Vec<u64>,
    vertical_masks: &mut Vec<u128>,
) -> Result<(), SparseQuadError> {
    horizontal_masks.clear();
    horizontal_masks.resize(EDGE, 0);
    vertical_masks.clear();
    vertical_masks.resize(EDGE, 0);

    for cut in vertical_cuts {
        let bit = 1u128 << u32::from(cut.x);
        for y in cut.start..cut.end {
            let Some(mask) = vertical_masks.get_mut(usize::from(y)) else {
                return Err(SparseQuadError::CapacityOverflow);
            };
            *mask |= bit;
        }
    }

    for cut in horizontal_cuts {
        let Some(mask) = horizontal_masks.get_mut(usize::from(cut.y)) else {
            return Err(SparseQuadError::CapacityOverflow);
        };
        *mask |= cell_range_mask(cut.start, cut.end);
    }

    Ok(())
}

fn build_runs_for_row(
    intervals: &[Interval],
    vertical_cut_mask: u128,
    runs: &mut Vec<Run>,
) -> Result<(), SparseQuadError> {
    runs.clear();
    for &interval in intervals {
        let mut start = interval.start;
        let mut split_mask = vertical_cut_mask
            & coord_between_mask(
                interval.start.saturating_add(1),
                interval.end.saturating_sub(1),
            );
        reserve_more(runs, split_mask.count_ones() as usize + 1)?;
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

fn push_run(runs: &mut Vec<Run>, value: u16, start: u8, end: u8) -> Result<(), SparseQuadError> {
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

fn merge_sparse_runs(
    active: &[ActiveRect],
    runs: &[Run],
    y: u8,
    horizontal_cut_mask: u64,
    next_active: &mut Vec<ActiveRect>,
    result: &mut Vec<Rectangle>,
) -> Result<(), SparseQuadError> {
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

    reserve_more(next_active, runs.len().saturating_sub(run_index))?;
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

fn preallocate_exact<T>(items: &mut Vec<T>, capacity: usize) -> Result<(), SparseQuadError> {
    if items.capacity() >= capacity {
        return Ok(());
    }
    items
        .try_reserve_exact(capacity - items.capacity())
        .map_err(|_| SparseQuadError::AllocationFailed)
}

fn allocate_exact<T>(items: &mut Vec<T>, additional: usize) -> Result<(), SparseQuadError> {
    if items.capacity().saturating_sub(items.len()) >= additional {
        return Ok(());
    }
    items
        .try_reserve_exact(additional)
        .map_err(|_| SparseQuadError::AllocationFailed)
}

fn allocate_more<T>(items: &mut Vec<T>, additional: usize) -> Result<(), SparseQuadError> {
    if items.capacity().saturating_sub(items.len()) >= additional {
        return Ok(());
    }
    items
        .try_reserve(additional)
        .map_err(|_| SparseQuadError::AllocationFailed)
}

const fn reserve_exact<T>(items: &Vec<T>, additional: usize) -> Result<(), SparseQuadError> {
    if items.capacity().saturating_sub(items.len()) >= additional {
        return Ok(());
    }
    Err(SparseQuadError::CapacityOverflow)
}

const fn reserve_one<T>(items: &Vec<T>) -> Result<(), SparseQuadError> {
    if items.len() < items.capacity() {
        return Ok(());
    }
    Err(SparseQuadError::CapacityOverflow)
}

const fn reserve_more<T>(items: &Vec<T>, additional: usize) -> Result<(), SparseQuadError> {
    if items.capacity().saturating_sub(items.len()) >= additional {
        return Ok(());
    }
    Err(SparseQuadError::CapacityOverflow)
}

const fn mesh_capacity(item_count: usize, per_item: usize) -> Result<usize, SparseQuadError> {
    match item_count.checked_mul(per_item) {
        Some(capacity) => Ok(capacity),
        None => Err(SparseQuadError::CapacityOverflow),
    }
}

fn emit(active: ActiveRect, y_end: u8, result: &mut Vec<Rectangle>) -> Result<(), SparseQuadError> {
    reserve_one(result)?;
    result.push(Rectangle {
        value: active.value,
        x: RangeU8::new(active.x_start, active.x_end),
        y: RangeU8::new(active.y_start, y_end),
    });
    Ok(())
}

// 测试

#[cfg(test)]
mod tests {
    use std::num::NonZeroU16;

    use super::*;
    use crate::get_mut;

    const ROOT_LOD: u8 = 6;
    const GENERATED_TILE_LOD: u8 = 3;
    const GENERATED_TILE_EDGE: usize = 8;

    fn nz(value: u16) -> NonZeroU16 {
        NonZeroU16::new(value).unwrap_or_else(|| std::process::abort())
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
        let result = SparseQuadImage64::from_leaves(leaves);
        assert!(matches!(result, Err(error) if error == expected));
    }

    #[test]
    fn empty_sparse_image_outputs_no_rectangles() {
        let Some(sparse_image) = ok(SparseQuadImage64::from_leaves(&[])) else {
            return;
        };
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_quads(&sparse_image)) else {
            return;
        };
        assert!(rectangles.is_empty());
    }

    #[test]
    fn borrowed_decompose_requires_preallocated_scratch() {
        let mut scratch = SparseOptimalScratch64::new();
        assert_eq!(
            scratch.decompose_borrowed(&[]),
            Err(SparseQuadError::CapacityOverflow)
        );
    }

    #[test]
    fn preallocated_borrowed_decompose_outputs_slice() {
        let leaves = [leaf(0, 0, 6, nz(5))];
        let Some(mut scratch) = ok(SparseOptimalScratch64::try_new_preallocated()) else {
            return;
        };
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
        let Some(mut scratch) = ok(SparseOptimalScratch64::try_new_preallocated()) else {
            return;
        };
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
    }

    #[test]
    fn overfull_row_reports_overlap() {
        let leaves = [leaf(0, 0, 0, NonZeroU16::MIN); EDGE + 1];
        let Some(mut scratch) = ok(SparseOptimalScratch64::try_new_preallocated()) else {
            return;
        };
        assert_eq!(
            scratch.decompose_borrowed(&leaves),
            Err(SparseQuadError::Overlap)
        );
    }

    #[test]
    fn repeated_decomposition_clears_buckets() {
        let Some(mut scratch) = ok(SparseOptimalScratch64::try_new_preallocated()) else {
            return;
        };
        let _ = ok(scratch.decompose_borrowed(&[leaf(0, 0, ROOT_LOD, NonZeroU16::MIN)]));
        assert!(matches!(scratch.decompose_borrowed(&[]), Ok([])));
    }

    #[test]
    fn single_full_leaf_outputs_one_rectangle() {
        let leaves = [leaf(0, 0, 6, nz(5))];
        let Some(sparse_image) = ok(SparseQuadImage64::from_leaves(&leaves)) else {
            return;
        };
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rectangles) = ok(scratch.decompose_quads(&sparse_image)) else {
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
        assert_eq!(scratch.decompose(&leaves), Err(SparseQuadError::Overlap));

        let mut builder = SparseLayerBuilder64::new();
        for leaf in leaves {
            let _ = ok(builder.push_square(leaf.u, leaf.v, leaf.lod, leaf.value));
        }
        assert_eq!(builder.finish(&mut scratch), Err(SparseQuadError::Overlap));
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
        let Some(rects) = ok(scratch.decompose(&leaves)) else {
            return;
        };
        assert!(!rects.is_empty());
    }

    #[test]
    fn l_shape_decomposes() {
        let leaves = [
            leaf(0, 0, 5, nz(1)),
            leaf(32, 0, 5, nz(1)),
            leaf(0, 32, 5, nz(1)),
        ];
        let mut scratch = SparseOptimalScratch64::new();
        let Some(rects) = ok(scratch.decompose(&leaves)) else {
            return;
        };
        assert!(!rects.is_empty());
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
        let Some(rects) = ok(scratch.decompose(&leaves)) else {
            return;
        };
        assert!(!rects.is_empty());
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
        let Some(mut scratch) = ok(SparseOptimalScratch64::try_new_preallocated()) else {
            return;
        };
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
}
