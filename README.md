# rectangle_decomposition

将 64x64 稀疏 quad 图像分解为同值矩形，基于 chord-and-matching 归约。
无第三方运行时依赖。

- `QuadLeaf64`：对齐的正方形叶子，边长为 `1 << lod`，非零值表示区域标签。
- `SparseQuadImage64`：校验叶子的范围、对齐与重叠。
- `SparseOptimalScratch64`：固定容量的内联缓存，支持栈上复用；`decompose_borrowed` 返回借用结果。
- `SparseLayerBuilder64`：逐个加入正方形并完成分解。

Chord 直接由相邻行／列的最大同色区间端点判定，无需重复查询邻近像素。
匹配后端使用 HKDW（Hopcroft–Karp 的 Duff–Wiberg 变体），冲突图按度数计数排序后
线性构建有序 CSR。支持含孔区域，保持最少矩形数量；当最优解不唯一时，具体切分可能变化。
将图像边长推广为 n 时，整体最坏复杂度仍为 O(n³)。详见 [匹配算法说明](docs/matching.md)。

在每个 worker 的任务循环外创建一次 scratch，通过 `&mut` 复用；初始化、借用分解和销毁
均不调用堆分配器。借用结果需在下一次分解前消费完毕：

```rust
use rectangle_decomposition::{QuadLeaf64, SparseOptimalScratch64, SparseQuadError};

fn process_layers(layers: &[&[QuadLeaf64]]) -> Result<(), SparseQuadError> {
    let mut scratch = SparseOptimalScratch64::new();
    for leaves in layers {
        let rectangles = scratch.decompose_borrowed(leaves)?;
        // 在这里读取 rectangles 并写入调用方的 mesh 缓冲。
        std::hint::black_box(rectangles);
    }
    Ok(())
}
```

当前 x86_64 布局约 366.0 KiB/份；已验证每个 worker 使用 2 MiB 栈的 debug/release 路径。
调用方仍需为自己的调用链留出栈空间。`new()` 直接得到可用缓存；原 `preallocate_64()`
保留为兼容空操作，`try_new_preallocated()` 返回 `Ok(new())`。新代码使用 `new()`，避免
大对象通过 `Result` 和辅助函数按值搬运。详见 [栈 scratch 与容量上界](docs/scratch.md)。

`decompose()` 等返回 `Vec<Rectangle>` 的便捷接口仍分配结果副本；
`SparseQuadImage64` 和 `SparseLayerBuilder64` 仍拥有其输入容器。

## 开发

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo bench --bench worst_case
```

`profile` feature 提供分阶段计时，criterion 仅用于基准测试。
