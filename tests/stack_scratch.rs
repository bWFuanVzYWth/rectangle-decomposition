//! 资源契约回归：只统计当前测试线程，避免测试框架和其他线程的分配干扰。
#![allow(clippy::panic_in_result_fn)] // 测试通过 Result 传播被测错误，断言检查资源契约。

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::num::NonZeroU16;

use rectangle_decomposition::{
    QuadLeaf64, SparseOptimalScratch64, SparseQuadError, SparseQuadImage64,
};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn record_allocation() {
    if TRACK.try_with(Cell::get).unwrap_or(false) {
        let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
    }
}

// 仅测试计数器需要调用系统分配器；库代码继续禁止 unsafe。
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn start_tracking() {
    CALLS.set(0);
    TRACK.set(true);
}

fn finish_tracking() {
    TRACK.set(false);
    assert_eq!(CALLS.get(), 0, "初始化、分解和析构均不得调用堆分配器");
}

#[test]
#[allow(clippy::large_stack_arrays)] // 输入也放在栈上，覆盖端到端零分配。
fn construction_and_reused_borrowed_output_do_not_allocate() -> Result<(), SparseQuadError> {
    start_tracking();
    {
        let mut scratch = SparseOptimalScratch64::new();
        let mut leaves = [QuadLeaf64 {
            u: 0,
            v: 0,
            lod: 0,
            value: NonZeroU16::MIN,
        }; 4096];
        // 完整棋盘格达到 4096 个区间和矩形；稠密孔洞达到每方向 1953 条 chord。
        for holes in [false, true, false, true] {
            let mut count = 0;
            for v in 0..64u8 {
                for u in 0..64u8 {
                    if holes && u % 2 == 0 && v % 2 == 0 {
                        continue;
                    }
                    let value = if holes || (u + v) % 2 == 0 {
                        NonZeroU16::MIN
                    } else {
                        NonZeroU16::MAX
                    };
                    let Some(slot) = leaves.get_mut(count) else {
                        return Err(SparseQuadError::CapacityOverflow);
                    };
                    *slot = QuadLeaf64 {
                        u,
                        v,
                        lod: 0,
                        value,
                    };
                    count += 1;
                }
            }
            let (input, _) = leaves.split_at(count);
            let rectangles = scratch.decompose_borrowed(black_box(input))?;
            assert_eq!(rectangles.len(), if holes { 1025 } else { 4096 });
            // 返回结果必须位于 scratch 对象之内。
            let output_start = rectangles.as_ptr() as usize;
            let output_end = output_start + std::mem::size_of_val(rectangles);
            let scratch_start = std::ptr::from_ref(&scratch) as usize;
            assert!(output_start >= scratch_start);
            assert!(output_end <= scratch_start + std::mem::size_of_val(&scratch));
        }
        assert_eq!(scratch.decompose_borrowed(&[])?, []);
    }
    finish_tracking();
    Ok(())
}

#[test]
fn prepared_images_and_leaves_share_scratch_without_allocating() -> Result<(), SparseQuadError> {
    let fragmented =
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (8, 8, 1)].map(|(u, v, lod)| QuadLeaf64 {
            u,
            v,
            lod,
            value: NonZeroU16::MIN,
        });
    let full = [QuadLeaf64 {
        u: 0,
        v: 0,
        lod: 6,
        value: NonZeroU16::MAX,
    }];
    // 拥有型 image 的构造独立分配；只统计 scratch 和两种借用分解入口。
    let fragmented_image = SparseQuadImage64::from_leaves(&fragmented)?;
    let full_image = SparseQuadImage64::from_leaves(&full)?;
    let empty_image = SparseQuadImage64::from_leaves(&[])?;
    start_tracking();
    {
        let mut scratch = SparseOptimalScratch64::new();
        for (leaves, image, count) in [
            (fragmented.as_slice(), &fragmented_image, 3),
            (&[], &empty_image, 0),
            (full.as_slice(), &full_image, 1),
            (fragmented.as_slice(), &fragmented_image, 3),
        ] {
            assert_eq!(scratch.decompose_quads_borrowed(image)?.len(), count);
            assert_eq!(scratch.decompose_borrowed(leaves)?.len(), count);
        }
    }
    finish_tracking();
    Ok(())
}

#[test]
#[allow(clippy::large_stack_frames)] // 兼容 Result 入口包含结果搬运，另有直接 new 的 worker 测试。
fn compatibility_constructor_does_not_allocate() -> Result<(), SparseQuadError> {
    start_tracking();
    {
        let Ok(mut scratch) = SparseOptimalScratch64::try_new_preallocated() else {
            return Err(SparseQuadError::AllocationFailed);
        };
        scratch.preallocate_64()?;
        assert_eq!(scratch.decompose_borrowed(&[])?, []);
    }
    finish_tracking();
    Ok(())
}

#[test]
fn default_constructor_does_not_allocate() -> Result<(), SparseQuadError> {
    start_tracking();
    {
        let mut scratch = SparseOptimalScratch64::default();
        assert_eq!(scratch.decompose_borrowed(&[])?, []);
    }
    finish_tracking();
    Ok(())
}

#[test]
fn independent_workers_reuse_stack_storage() -> Result<(), Box<dyn std::error::Error>> {
    let mut workers = Vec::new();
    for _ in 0..4 {
        // 栈预算是回归测试约束，初始化及循环均在 worker 中执行。
        workers.push(
            std::thread::Builder::new()
                .stack_size(2 * 1024 * 1024)
                .spawn(|| construction_and_reused_borrowed_output_do_not_allocate().is_ok())?,
        );
    }
    for worker in workers {
        assert!(matches!(worker.join(), Ok(true)));
    }
    Ok(())
}
