# rectangle_decomposition

将 64x64 稀疏 quad 图像分解为同值矩形，基于 chord-and-matching 归约。
无第三方运行时依赖。

- `QuadLeaf64`：对齐的正方形叶子，边长为 `1 << lod`，非零值表示区域标签。
- `SparseQuadImage64`：校验叶子的范围、对齐与重叠。
- `SparseOptimalScratch64`：复用计算缓存；预分配后，`decompose_borrowed` 返回借用结果。
- `SparseLayerBuilder64`：逐个加入正方形并完成分解。

匹配后端使用 HKDW（Hopcroft–Karp 的 Duff–Wiberg 变体），冲突图按度数计数排序后
线性构建有序 CSR。支持含孔区域，保持最少矩形数量；当最优解不唯一时，具体切分可能变化。
将图像边长推广为 n 时，整体最坏复杂度仍为 O(n³)。详见 [匹配算法说明](docs/matching.md)。

## 开发

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo bench --bench worst_case
```

`profile` feature 提供分阶段计时，criterion 仅用于基准测试。
