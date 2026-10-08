//! Link-only bare-metal check: no std, alloc crate, global allocator, or OS runtime.
#![no_std]
#![no_main]

use core::hint::black_box;
use core::num::NonZeroU16;
use core::panic::PanicInfo;
use rectangle_decomposition::{
    DenseLabels64, PackedRectangles64, QuadLeaf64, SparseOptimalScratch64,
};

// The standalone firmware-style caller supplies its own entry point and panic handler.
// Exporting this symbol does not introduce an unsafe memory operation into the library.
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let mut scratch = SparseOptimalScratch64::new();
    let mut labels = DenseLabels64::new();
    for (y, row) in labels.rows_mut().iter_mut().enumerate() {
        for (x, value) in row.iter_mut().enumerate() {
            *value = 1 + u16::from((x + y) % 2 == 0);
        }
    }
    let mut packed = PackedRectangles64::new();
    let rectangles = scratch.decompose_labels_borrowed(black_box(&labels));
    assert_eq!(black_box(rectangles).map(<[_]>::len), Ok(4096));
    let mut checksum = 0u64;
    let count = scratch.decompose_labels_into(black_box(&labels), |rectangle| {
        checksum = checksum.wrapping_add(u64::from(rectangle.value));
        black_box(rectangle);
        Ok(())
    });
    assert_eq!(black_box((count, checksum)).0, Ok(4096));
    assert_eq!(
        scratch.decompose_labels_packed(black_box(&labels), &mut packed),
        Ok(4096)
    );
    black_box((packed.bounds(), packed.labels()));

    let leaves = [QuadLeaf64 {
        u: 0,
        v: 0,
        lod: 6,
        value: NonZeroU16::MIN,
    }];
    assert_eq!(
        scratch
            .decompose_borrowed(black_box(&leaves))
            .map(<[_]>::len),
        Ok(1)
    );
    assert_eq!(
        scratch.decompose_into(black_box(&leaves), |rectangle| {
            black_box(rectangle);
            Ok(())
        }),
        Ok(1)
    );
    assert_eq!(scratch.decompose_packed(&leaves, &mut packed), Ok(1));
    black_box((packed.bounds(), packed.labels()));
    idle()
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    idle()
}

fn idle() -> ! {
    loop {
        core::hint::spin_loop();
    }
}
