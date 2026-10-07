# rectangle_decomposition

把 **64×64 彩色像素区域分解为数量最少的同色矩形**。适用于体素 mesh 中同一方向、同一平面上的面合并，也可用于一般像素区域分解。

每个非背景像素恰好被覆盖一次；矩形内部没有空洞、没有混色，矩形之间没有面积重叠。支持含孔区域和多个不相连的区域。不同最优解可以采用不同切法，输出允许 T 形接缝。

算法先找出能同时处理两个凹角的内部线段（**chord**），把全部颜色的候选一次构造成二分冲突图，通过最大匹配选出最多条互不冲突的 chord，再扫描生成矩形。不同颜色的 chord 不可能相交，所以全局求解仍等价于逐颜色求解；匹配结束时复用最后一次反向 BFS 提取最大独立集。具体解释见 [从像素到最少矩形](docs/algorithm.md)。

## 输入、输出和边界

- 原生输入为 `DenseLabels64`：64 字节对齐的行主序 `[[u16; 64]; 64]`，标签 0 是背景，`1..=65535` 是调用方定义的颜色／材质键。上游可直接写入 `rows_mut()`，省去单位叶子列表及布局转换。
- 也接受 `QuadLeaf64` 列表：`(u, v)` 是左上角，边长为 `2^lod`，`lod` 在 `0..=6`；坐标必须按边长对齐，叶子不得越界或重叠。`value: NonZeroU16` 是标签，未提供的像素是背景。
- 输出可选择借用的 `Rectangle` 切片、逐矩形 callback，或调用方持有的 `PackedRectangles64`。范围 `x.start..x.end`、`y.start..y.end` 左闭右开，终点可以是 64；矩形尺寸不受输入叶子的 LOD 限制。
- 体素调用方负责可见面筛选、方向分层和合并条件；需要区分的属性应反映在输入标签中。本库处理的是二维区域划分。

目标是最少矩形数量。输入不要求预排序或预合并；含孔区域使用同一套精确算法。

## 最小用法

将本仓库作为 Cargo 的本地 `path` 依赖加入项目，路径按实际位置调整：

当前 SIMD 实验使用 nightly 的 `std::simd`（`portable_simd`），需要 nightly Rust 工具链；显式 SIMD 数据向量统一为 512 bit。无需 AVX-512，编译器可将逻辑向量拆为目标平台支持的较窄指令。

```toml
[dependencies]
rectangle_decomposition = { path = "../rectangle_decomposition" }
```

然后使用可复用的 scratch：

```rust
use std::num::NonZeroU16;
use rectangle_decomposition::{QuadLeaf64, SparseOptimalScratch64, SparseQuadError};

fn main() -> Result<(), SparseQuadError> {
    // 三个同色像素构成 L 形：AA / A.，最少需要两个矩形。
    let value = NonZeroU16::MIN;
    let leaves = [
        QuadLeaf64 { u: 0, v: 0, lod: 0, value },
        QuadLeaf64 { u: 1, v: 0, lod: 0, value },
        QuadLeaf64 { u: 0, v: 1, lod: 0, value },
    ];

    // 每个 worker 在任务循环外构造一次，后续通过 &mut 复用。
    let mut scratch = SparseOptimalScratch64::new();
    let rectangles = scratch.decompose_borrowed(&leaves)?;
    assert_eq!(rectangles.len(), 2);
    for rectangle in rectangles {
        println!("{rectangle:?}");
    }
    Ok(())
}
```

引擎已有连续标签切面时，直接使用原生布局：

```rust
use rectangle_decomposition::{
    DenseLabels64, PackedRectangles64, SparseOptimalScratch64, SparseQuadError,
};

fn main() -> Result<(), SparseQuadError> {
    let mut labels = DenseLabels64::new();
    labels.rows_mut()[0][0] = 65535;
    let mut scratch = SparseOptimalScratch64::new();
    let mut output = PackedRectangles64::new();
    scratch.decompose_labels_packed(&labels, &mut output)?;
    // bounds 的四个字节依次是 x.start、x.end、y.start、y.end。
    assert_eq!(output.bounds(), &[u32::from_le_bytes([0, 1, 0, 1])]);
    assert_eq!(output.labels(), &[65535]);
    Ok(())
}
```

`decompose_labels_borrowed` 返回借用结果；`decompose_labels_into` 可将矩形直接送入 mesh 构建 callback。叶子输入也提供 `decompose_into` 和 `decompose_packed`。直接输出不经过 scratch 的矩形数组；callback 返回错误时立即终止，已输出前缀与失败 callback 的副作用不回滚，scratch 可继续复用。

初始化、借用分解、packed 输出和销毁不调用堆分配器；callback 自身的资源行为由调用方决定。返回的切片属于 scratch，需在下一次可变使用前消费完毕。scratch 为固定容量内联对象，在 x86_64 上为 **264,256 字节，约 258 KiB**，调用方需为 worker 配置足够的栈并避免按值搬运。资源契约与容量证明见 [scratch 与接口](docs/scratch.md)。

叶子数小于 512 时沿用稀疏事件路径；至少 512 个叶子时自动栅格化，再用 512-bit SIMD 构造两轴最大同色区间。原生 `DenseLabels64` 直接进入同一后端，所有入口保持整体 `O(n³)` 上界。

大量单位面适合由上游直接填充标签切面；只有少量大 LOD 块时保留叶子输入，避免扫描整个网格。直接 sink 便于写入下游已有布局，实际收益取决于消费者，不能保证每种输出布局都更快。

也提供拥有输入的 `SparseQuadImage64`、增量输入的 `SparseLayerBuilder64`，以及返回 `Vec<Rectangle>` 的便捷接口；这些拥有型容器可以分配堆内存。库本身没有第三方运行时依赖，库代码禁止 `unsafe`。

## 接手研究的阅读顺序

| 文档 | 回答的问题 |
| --- | --- |
| [算法原理](docs/algorithm.md) | 凹角、chord、冲突图是什么，为什么最大匹配能得到最少矩形？ |
| [实现与优化](docs/matching.md) | 数学步骤对应哪些代码，整数网格和 HKDW 如何减少工作？ |
| [scratch 与接口](docs/scratch.md) | 哪些操作分配内存，固定容量上界和线程栈怎样核算？ |
| [研究与验证](docs/research.md) | O(n³) 如何得到，已知下界是什么，修改算法怎样验证？ |
| [文献与出处](docs/references.md) | 先读哪些论文，文献算法与当前实现有哪些对应关系？ |

## 开发入口

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo bench --bench worst_case
cargo run --release --features profile --example worst_case_profile
```

`profile` feature 提供分阶段计时；Criterion 仅为开发依赖。基准的范围和正确使用方式见 [研究与验证](docs/research.md)。
