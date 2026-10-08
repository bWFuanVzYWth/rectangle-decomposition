#![no_std]
#![cfg_attr(not(feature = "std"), feature(abort_immediate))]
#![feature(portable_simd)]

//! 将 64x64 带标签像素区域分解为最少同色矩形，输入为连续标签切面或 sparse quad 列表。
//!
//! 几何归约参考 Ferrari、Sankar、Sklansky (1984)，匹配使用 HKDW。
//! 算法原理、实现地图与完整出处见仓库 README 和 docs/。

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(test)]
mod corners;
mod fixed;
mod graph;
mod greedy;
mod hk;
mod matching;
#[cfg(any(feature = "alloc", test))]
mod morton;
mod sparse;
mod types;

pub use sparse::{DenseLabels64, QuadLeaf64, SparseOptimalScratch64, SparseQuadError};
#[cfg(feature = "profile")]
pub use sparse::{SparseDecomposeCounts, SparseDecomposeProfile, SparseDecomposeTimings};
#[cfg(feature = "alloc")]
pub use sparse::{SparseLayerBuilder64, SparseQuadImage64};
pub use types::{PackedRectangles64, Rectangle};

/// Only invariant violations reach this cold path; normal input errors use `Result`.
#[cold]
#[inline(never)]
pub(crate) fn invariant_failed() -> ! {
    #[cfg(feature = "std")]
    std::process::abort();
    #[cfg(not(feature = "std"))]
    core::process::abort_immediate();
}

pub(crate) fn get<T>(slice: &[T], index: usize) -> &T {
    debug_assert!(index < slice.len());
    slice.get(index).unwrap_or_else(|| invariant_failed())
}

pub(crate) fn get_mut<T>(slice: &mut [T], index: usize) -> &mut T {
    debug_assert!(index < slice.len());
    slice.get_mut(index).unwrap_or_else(|| invariant_failed())
}

pub(crate) fn copy<T: Copy>(slice: &[T], index: usize) -> T {
    *get(slice, index)
}

pub(crate) fn slice<T>(slice: &[T], range: core::ops::Range<usize>) -> &[T] {
    debug_assert!(range.start <= range.end);
    debug_assert!(range.end <= slice.len());
    slice.get(range).unwrap_or_else(|| invariant_failed())
}

pub(crate) fn slice_mut<T>(slice: &mut [T], range: core::ops::Range<usize>) -> &mut [T] {
    debug_assert!(range.start <= range.end);
    debug_assert!(range.end <= slice.len());
    slice.get_mut(range).unwrap_or_else(|| invariant_failed())
}

pub(crate) fn u16_index(index: usize) -> u16 {
    u16::try_from(index).unwrap_or_else(|_| invariant_failed())
}

pub(crate) fn u32_index(index: usize) -> u32 {
    u32::try_from(index).unwrap_or_else(|_| invariant_failed())
}
