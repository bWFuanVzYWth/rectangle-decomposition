# rectangle_decomposition

将 64x64 稀疏 quad 图像分解为同值矩形，基于 chord-and-matching 归约。
无第三方运行时依赖。

- `QuadLeaf64`：对齐的正方形叶子，边长为 `1 << lod`，非零值表示区域标签。
- `SparseQuadImage64`：校验叶子的范围、对齐与重叠。
- `SparseOptimalScratch64`：复用计算缓存；预分配后，`decompose_borrowed` 返回借用结果。
- `SparseLayerBuilder64`：逐个加入正方形并完成分解。

## 开发

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo bench --bench worst_case
```

`profile` feature 提供分阶段计时，criterion 仅用于基准测试。
