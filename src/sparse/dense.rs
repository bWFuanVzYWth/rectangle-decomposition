//! 连续标签切面：SIMD 比较相邻行／像素，直接构造最大同色区间。

use std::simd::{cmp::SimdPartialEq, u16x32};

use super::{
    AxisIntervals, EDGE, EDGE_U8, Interval, IntervalBucket, LineRange, QuadLeaf64,
    SparseOptimalScratch64, SparseQuadError, build_complete_axis_intervals_from_buckets,
    cell_range_mask, prepare_interval_buckets, reserve_exact, validate_leaf_shape,
};
use crate::types::{PackedRectangles64, Rectangle};
use crate::{copy, get, get_mut, slice, slice_mut};

// 少量大叶子保留事件路径；大量叶子用紧凑网格减少两轴散写。
pub(super) const DENSE_LEAF_THRESHOLD: usize = 512;

/// 行主序的 64×64 标签切面。0 为背景，其余值为调用方的颜色／材质键。
/// 固定网格隐含边界与无重叠条件，无需构造单位正方形列表。
#[repr(C, align(64))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseLabels64 {
    rows: [[u16; EDGE]; EDGE],
}

impl DenseLabels64 {
    /// 创建空切面，不调用堆分配器。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            rows: [[0; EDGE]; EDGE],
        }
    }

    /// 取得行主序标签。每行有两个 512-bit 数据向量。
    #[must_use]
    pub const fn rows(&self) -> &[[u16; EDGE]; EDGE] {
        &self.rows
    }

    /// 上游可直接写入目标布局；任意 u16 标签组合均是合法像素图。
    pub const fn rows_mut(&mut self) -> &mut [[u16; EDGE]; EDGE] {
        &mut self.rows
    }

    /// 清空标签，复用已有存储。
    pub fn clear(&mut self) {
        self.rows.as_flattened_mut().fill(0);
    }
}

impl Default for DenseLabels64 {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseOptimalScratch64 {
    /// 直接分解原生连续标签切面，返回借用结果，不分配堆内存。
    /// 输入无需叶子形状、排序或重叠校验；0 表示背景。
    ///
    /// # Errors
    /// 内部容量约束被违反时返回错误。
    pub fn decompose_labels_borrowed(
        &mut self,
        labels: &DenseLabels64,
    ) -> Result<&[Rectangle], SparseQuadError> {
        build_axes(
            labels,
            &mut self.rows,
            &mut self.columns,
            &mut self.column_buckets,
        )?;
        self.decompose_intervals_borrowed()
    }

    /// 连续标签切面的直接 sink 输出，库不分配堆内存或写入中间矩形数组。
    ///
    /// # Errors
    /// 内部容量错误或 sink 错误原样返回；已输出前缀不回滚。
    pub fn decompose_labels_into<F>(
        &mut self,
        labels: &DenseLabels64,
        sink: F,
    ) -> Result<usize, SparseQuadError>
    where
        F: FnMut(Rectangle) -> Result<(), SparseQuadError>,
    {
        build_axes(
            labels,
            &mut self.rows,
            &mut self.columns,
            &mut self.column_buckets,
        )?;
        super::extract_sparse_chords(&self.rows, &self.columns, &mut self.chord_groups)?;
        self.select_cuts()?;
        self.sparse_partition_into(sink)
    }

    /// 将连续标签切面的矩形直接写入调用方的 packed bounds / labels。
    ///
    /// # Errors
    /// 内部容量错误返回，输出保留本次已完成前缀。
    pub fn decompose_labels_packed(
        &mut self,
        labels: &DenseLabels64,
        output: &mut PackedRectangles64,
    ) -> Result<usize, SparseQuadError> {
        output.clear();
        self.decompose_labels_into(labels, |rectangle| output.push(rectangle))
    }
}

pub(super) fn rasterize_leaves<I>(
    leaves: I,
    labels: &mut DenseLabels64,
    occupied: &mut [IntervalBucket; EDGE],
) -> Result<(), SparseQuadError>
where
    I: IntoIterator<Item = QuadLeaf64>,
    I::IntoIter: Clone,
{
    labels.clear();
    let leaves = leaves.into_iter();
    if leaves.clone().all(|leaf| leaf.lod == 0) {
        // 单位像素直接查目标格；跳过 side、行跨度和占用区间的重复计算。
        for leaf in leaves {
            if (leaf.u | leaf.v) >= EDGE_U8 {
                return Err(SparseQuadError::OutOfBounds);
            }
            let index = usize::from(leaf.v) * EDGE + usize::from(leaf.u);
            let pixel = get_mut(labels.rows.as_flattened_mut(), index);
            if *pixel != 0 {
                return Err(SparseQuadError::Overlap);
            }
            *pixel = leaf.value.get();
        }
        return Ok(());
    }
    prepare_interval_buckets(occupied);
    for leaf in leaves {
        let side = validate_leaf_shape(leaf)?;
        let end_x = leaf.u + side;
        let mask = cell_range_mask(leaf.u, end_x);
        for y in leaf.v..leaf.v + side {
            let row_occupied = get_mut(occupied, usize::from(y));
            if row_occupied.starts & mask != 0 {
                return Err(SparseQuadError::Overlap);
            }
            row_occupied.starts |= mask;
            slice_mut(
                get_mut(&mut labels.rows, usize::from(y)),
                usize::from(leaf.u)..usize::from(end_x),
            )
            .fill(leaf.value.get());
        }
    }
    Ok(())
}

fn row_changes(current: &[u16; EDGE], previous: &[u16; EDGE]) -> u64 {
    let low = u16x32::from_slice(slice(current, 0..32))
        .simd_ne(u16x32::from_slice(slice(previous, 0..32)))
        .to_bitmask();
    let high = u16x32::from_slice(slice(current, 32..64))
        .simd_ne(u16x32::from_slice(slice(previous, 32..64)))
        .to_bitmask();
    low | (high << 32)
}

fn append_row(row: &[u16; EDGE], output: &mut AxisIntervals) -> Result<LineRange, SparseQuadError> {
    // 两次重叠比较覆盖边界 1..=63；边界 32 被重复比较但只写同一位。
    let low = u16x32::from_slice(slice(row, 1..33))
        .simd_ne(u16x32::from_slice(slice(row, 0..32)))
        .to_bitmask();
    let high = u16x32::from_slice(slice(row, 32..64))
        .simd_ne(u16x32::from_slice(slice(row, 31..63)))
        .to_bitmask();
    let mut boundaries = (low << 1) | (high << 32);
    let start = output.intervals.len();
    let mut left = 0u8;
    loop {
        let right = if boundaries == 0 {
            EDGE_U8
        } else {
            u8::try_from(boundaries.trailing_zeros()).unwrap_or_else(|_| std::process::abort())
        };
        let value = copy(row, usize::from(left));
        if value != 0 {
            reserve_exact(&output.intervals, 1)?;
            output.intervals.push(Interval {
                start: left,
                end: right,
                value,
            });
        }
        if boundaries == 0 {
            break;
        }
        boundaries &= boundaries - 1;
        left = right;
    }
    Ok(LineRange {
        start,
        end: output.intervals.len(),
    })
}

pub(super) fn build_axes(
    labels: &DenseLabels64,
    rows: &mut AxisIntervals,
    columns: &mut AxisIntervals,
    column_buckets: &mut [IntervalBucket; EDGE],
) -> Result<(), SparseQuadError> {
    rows.clear_for_build()?;
    prepare_interval_buckets(column_buckets);
    let first = get(&labels.rows, 0);
    let mut column_values = *first;
    let mut column_starts = [0u8; EDGE];
    let mut range = append_row(first, rows)?;
    *get_mut(&mut rows.lines, 0) = range;
    rows.boundaries = 1;
    for y in 1u8..EDGE_U8 {
        let current = get(&labels.rows, usize::from(y));
        let mut changes = row_changes(current, get(&labels.rows, usize::from(y - 1)));
        if changes != 0 {
            rows.boundaries |= 1u64 << y;
            range = append_row(current, rows)?;
        }
        *get_mut(&mut rows.lines, usize::from(y)) = range;
        while changes != 0 {
            let x =
                usize::try_from(changes.trailing_zeros()).unwrap_or_else(|_| std::process::abort());
            let value = copy(&column_values, x);
            if value != 0 {
                get_mut(column_buckets, x).try_push(Interval {
                    start: copy(&column_starts, x),
                    end: y,
                    value,
                })?;
            }
            *get_mut(&mut column_starts, x) = y;
            *get_mut(&mut column_values, x) = copy(current, x);
            changes &= changes - 1;
        }
    }
    for x in 0..EDGE {
        let value = copy(&column_values, x);
        if value != 0 {
            get_mut(column_buckets, x).try_push(Interval {
                start: copy(&column_starts, x),
                end: EDGE_U8,
                value,
            })?;
        }
    }
    build_complete_axis_intervals_from_buckets(column_buckets, columns)
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)] // 测试坐标由 0..64 和固定尺寸数组构造。
mod tests {
    use super::super::push_leaf_axis_intervals_one;
    use super::*;
    use std::num::NonZeroU16;

    fn buckets() -> Box<[IntervalBucket; EDGE]> {
        (0..EDGE)
            .map(|_| IntervalBucket::new())
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| std::process::abort())
    }

    #[test]
    fn dense_axes_match_independent_unit_buckets_for_all_lane_boundaries() {
        assert_eq!(std::mem::size_of::<DenseLabels64>(), 8192);
        assert_eq!(std::mem::align_of::<DenseLabels64>(), 64);
        let mut labels = DenseLabels64::new();
        let mut reference_rows = AxisIntervals::new();
        let mut reference_columns = AxisIntervals::new();
        let mut rows = AxisIntervals::new();
        let mut columns = AxisIntervals::new();
        let mut row_buckets = buckets();
        let mut column_buckets = buckets();
        let mut dense_buckets = buckets();
        let mut random = 0x9e37_79b9_7f4a_7c15u64;
        for pattern in 0..128 {
            prepare_interval_buckets(&mut row_buckets);
            prepare_interval_buckets(&mut column_buckets);
            for y in 0u8..64 {
                for x in 0u8..64 {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    let value = match pattern % 8 {
                        0 => 0,
                        1 => 65535,
                        2 => u16::from(x / 2) + 1,
                        3 => u16::from(y / 2) + 1,
                        4 => [0, 1, 32768, 65535][usize::from((x + y) % 4)],
                        5 => u16::from(
                            x == 31 || x == 32 || x == 63 || y == 31 || y == 32 || y == 63,
                        ),
                        _ => [0, 1, 2, 65535][(random & 3) as usize],
                    };
                    labels.rows[usize::from(y)][usize::from(x)] = value;
                    if let Some(value) = NonZeroU16::new(value) {
                        assert_eq!(
                            push_leaf_axis_intervals_one(
                                QuadLeaf64 {
                                    u: x,
                                    v: y,
                                    lod: 0,
                                    value
                                },
                                row_buckets.as_mut_slice(),
                                column_buckets.as_mut_slice()
                            ),
                            Ok(())
                        );
                    }
                }
            }
            assert_eq!(
                build_complete_axis_intervals_from_buckets(&row_buckets, &mut reference_rows),
                Ok(())
            );
            assert_eq!(
                build_complete_axis_intervals_from_buckets(&column_buckets, &mut reference_columns),
                Ok(())
            );
            assert_eq!(
                build_axes(&labels, &mut rows, &mut columns, &mut dense_buckets),
                Ok(())
            );
            for line in 0..64 {
                assert_eq!(rows.line(line), reference_rows.line(line));
                assert_eq!(columns.line(line), reference_columns.line(line));
            }
            assert_eq!(rows.boundaries & !1, reference_rows.boundaries & !1);
            assert_eq!(columns.boundaries & !1, reference_columns.boundaries & !1);
        }
    }
}
