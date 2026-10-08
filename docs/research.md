# 研究与验证

[返回 README](../README.md) · 前置阅读：[实现与优化](matching.md)

## 复杂度中的 n 是什么

本仓库 API 的边长固定为 64。讨论渐近复杂度时，假设边长和相应缓冲一起推广，令 `n` 为图像边长；这不表示现有代码已经支持任意尺寸。

| 符号 | 含义 | 合法像素图的上界 |
| --- | --- | --- |
| `L` | 输入叶子数 | `n²` |
| `C` | 全部水平、竖直 chord 的总数 | `O(n²)` |
| `E` | 全局冲突图的边数 | `(n−1)²` |
| `Δ` | 冲突图最大度 | `n−1` |
| `G` | 有 chord 的不同标签数，仅用于规模统计 | `n²/4` |
| `R` | 输出矩形数 | `n²` |

几何依据见 [容量上界](scratch.md)。读取论文时应先换算符号：文献的 `n` 常表示多边形顶点数或图顶点数，与这里的图像边长不同。

## 当前 O(n³) 上界的组成

以下是对合法输入的保守上界，用于检查实现是否仍守住目标复杂度：

| 阶段 | 时间上界 | 说明 |
| --- | --- | --- |
| 构造拥有型 image | `O(L+n²)` | 有限 Morton 键排序；有序输入为 `O(L)`，固定小输入用比较排序 |
| 构造轴区间（所有入口） | `O(n²)` | 稀疏叶子走事件路径；至少 512 个叶子栅格化后 SIMD 提取行列 run；原生 grid 直接提取 |
| 提取 chord | `O(n²)` | 双指针扫描，保留提取顺序 |
| 事件建图与有序 CSR | `O(n² + C + E + n)` | 64×64 下扫线为 `O(n+C+E)`，推广后须计入多字位图范围查询 |
| 贪心和长度 3 预热 | `O(C + E + n)` | 单调游标使长度 3 内层每条邻接边至多扫描一次 |
| HKDW 搜索 | `O((C+E)√C)` | 每轮线性工作，沿用 HK 的轮数上界 |
| 从最终反向 BFS 提取独立集标记 | `O(C)` | 扫描最终距离和配对数组，无需再次遍历边 |
| 切线位图与矩形输出 | `O(C + n² + R)` | 端点事件生成切线位图，只在布局／切线变化边界处理 run |

将 `C,E=O(n²)` 代入，整体仍为 **O(n³)**，空间为 **O(n²)**。立方项来自 HKDW 搜索；预热、所有入口的轴区间构造以及拥有型 image 的有限键排序均守住二次上界。

原生标签输入固定保存 n² 个 u16 像素，0 为背景；行列 SIMD 比较和每次变化产生的 run 总数均为 `O(n²)`。自动栅格化先清空 n² 个标签，再处理叶子：单位像素逐个检查目标格，混合 LOD 的累计合法填充面积不超过 n²。对合法输入有 `L≤n²`，所以这两条构建路径均为二次上界。固定 512-bit 逻辑向量只改变批量工作量，推广边长时仍计入实际向量块数。

借用、直接 sink 和 packed 输出共用分区核心。直接输出仍至多生成 `R≤n²` 个矩形，不增加扫描次数；callback 自身的工作由调用方决定，不计入库内部复杂度。u16 CSR 偏移与 DFS 索引只改变存储宽度，固定 64×64 容量检查保证不截断；推广尺寸时需要同步扩大整数类型。

更紧的几何计数是 `Σ_horizontal(length+1) ≤ (n−1)²`，竖直方向同理：所有标签的同向 chord 闭格点集合两两不交。旧逐格构图的总工作已经是 `O(n²)`，逐条使用 `O(n)` 再乘 `C` 会高估；当前扫线进一步去掉了这项长度展开。全局构图也消除了原先逐颜色初始化计数桶的 `Gn` 项。

整数位图在 64×64 下由固定数量机器字实现；若推广边长，不能无条件把任意宽位图操作视为 O(1)。逐格写入的总长度界不依赖机器字宽度；位图范围操作则需要按被范围覆盖的机器字分块，每次成本为 `O(length/w+1)`，其中 `w` 为机器字位数，其总成本也受 `O(n²+C)` 约束。每次重新扫描整张宽位图的实现不能直接获得这个界。

稳定带优化降低的是输入敏感工作量，并未声明整体 `O(L+R)`：混合 LOD 的活动快照仍可能达到 `n²`，构造行列视图及位图也有 `O(n)` 初始化。Builder 追加摊还 `O(1)`，finish 与直接 leaves 共用后端并复制 `R` 个矩形。

## 上界不等于下界

显式读取 `L` 个叶子、输出 `R` 个矩形，至少需要 `Ω(L+R)` 的工作；原生连续标签输入的像素数为 n²，逐格读取与显式输出为 `Ω(n²+R)`。双色棋盘格必须输出 `n²` 个单像素矩形，因此一般输入的最坏情况至少为 **Ω(n²)**。

这是输入／输出规模下界，不能推出 Ω(n³)。当前仓库没有证明“任何算法都不能比立方时间更快”。能否继续利用像素 chord 图的结构，降低精确匹配或整个分解的上界，是独立的研究问题。

同样，不能把只适用于无孔简单多边形的更快算法直接套到含孔像素输入。比较研究结果时，应核对目标是否为最少矩形数、是否允许 T 形接缝、如何处理孔洞／点接触，以及复杂度按什么规模计量。相关论文见 [文献与出处](references.md)。

成果 120 的一般图近线性最大匹配预印本是继续研究的线索，其引言将二分图的近线性匹配归于已有精确整数流算法。当前实现保留 HKDW；全局建图和对称覆盖构造来自本像素模型及经典匹配性质。若将来匹配达到 `(C+E)^(1+o(1))`，还需实际实现上述分块位图操作，即可与现有二次非匹配步骤组合为完整分解的 `n^(2+o(1))` 条件上界。长度 3 预热的重复扫描已通过单调游标消除。64×64 的实际收益须另行量测，见 [流算法研究入口](references.md)。

## 修改算法时的验证层次

| 层次 | 要证明或检查什么 | 仓库中的入口 |
| --- | --- | --- |
| 几何提取 | 与四像素凹角判定相同，端点冲突不遗漏 | [sparse.rs](../src/sparse.rs)：`interval_chords_match_pixel_corner_oracle`，穷举 `3^8` 种局部图案 |
| 全局几何与图存储 | 同向格点不重叠，跨标签不相交；边集合、邻接顺序和事件复用正确 | [sparse.rs](../src/sparse.rs)：`global_chord_graph_preserves_colored_optima_on_all_three_by_three_images`；[graph.rs](../src/graph.rs)：扫线对旧逐格 oracle、乱序边段、有序 CSR 和匹配选择对照 |
| 精确匹配 | 匹配最大，独立集确实无冲突 | [hk.rs](../src/hk.rs)：全部 65,536 个 4×4 二分图对独立穷举，并检查等大小的覆盖证书 |
| 预热游标 | 单调跳过边仍保持原配对顺序 | [greedy.rs](../src/greedy.rs)：全部 3×3 图的 5,504 个合法部分匹配对独立重扫实现，以及连续迁移同一左顶点 |
| 几何输出 | 标签、覆盖、无重叠与最少矩形数 | [sparse.rs](../src/sparse.rs)：全部 `3^9` 个 3×3 背景／双色图对独立矩形分区 DP，以及含孔双色区域、L 形、极限 chord 测试 |
| 压缩表示与外围入口 | 混合 LOD、顺序和所有入口等价，排序与事件边界正确 | [axis_events.rs](../tests/axis_events.rs)：128 个随机四叉树层对独立单位像素展开；[sparse.rs](../src/sparse.rs)：Morton 全键／阈值／非法输入、cut 端点和真实稠密图 |
| 原生 grid 与直接输出 | SIMD 接缝、极端标签、最大容量、输入阈值、失败前缀及恢复 | [dense.rs](../src/sparse/dense.rs)：两轴区间对独立单位桶；[dense_labels.rs](../tests/dense_labels.rs)、[direct_output.rs](../tests/direct_output.rs) |
| 资源 | 不分配、容量不溢出、线程栈可承受 | [stack_scratch.rs](../tests/stack_scratch.rs) |

正确性论证与测试应互相补充。与旧版输出逐项相同只说明没有改变旧版行为，不是独立的最优性证明；改变匹配顺序时可能得到不同的最优划分，应比较覆盖、合法性、矩形数和独立证书。

修改论文算法前，先写清状态不变量，再构造会触发新逻辑的反例或穷举范围。至少覆盖含孔区域、多标签、边界接触、空图、重复调用及非法重叠输入。若改动数据容量或对象构造，还应在 debug 和 release 下检查资源契约。

```sh
cargo check --lib --no-default-features
cargo check --lib --features alloc
cargo check --lib --features std
cargo test
cargo test --features alloc
cargo test --all-features
cargo test --release --no-default-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

库始终为 `no_std`；默认 features 为空，仅依赖 `core`。`alloc` 开启拥有型 Image／Builder 和返回 Vec 的接口，`std` 包含 `alloc`，`profile` 包含 `std`。默认配置运行 63 项测试，`alloc`／全部 features 运行 71 项；默认测试仍覆盖随机 LOD、原生标签、sink／packed 输出和零分配资源契约。宿主测试程序自身使用 `std` 和测试分配器，不能单凭这些测试证明裸机链接。

默认库已另行检查 `wasm32-unknown-unknown` 目标：

```sh
cargo check --lib --no-default-features --target wasm32-unknown-unknown
```

裸机验证使用独立的 [no_std_link fixture](../checks/no_std_link/main.rs)，在 `x86_64-unknown-none` 上完成 release 链接，仅链接 `core`、`compiler_builtins` 和本库，没有 `alloc`、全局分配器或 OS 运行时。它把六个核心分解入口保留在最终可执行文件中；这是链接验证，未启动或执行裸机程序。调用方提供 `_start`、`panic_handler`，真实部署还需配置自己的启动和栈。复现需要 nightly 及已安装的 `rust-src`：

```sh
cargo -Z build-std=core -Z build-std-features=compiler-builtins-mem build --release --target x86_64-unknown-none --manifest-path checks/no_std_link/Cargo.toml --target-dir target/no-std-link
```

SIMD 使用 nightly 的 `core::simd`，逻辑数据向量保持 512 bit，无需 AVX-512。内部不变量失败使用 nightly 的 `core::process::abort_immediate`（`abort_immediate`），当前在 Rust 1.100.0-nightly 验证。未启用 `std` 时，通过平台 trap／abort 立即终止且不展开栈；启用 `std` 时保留进程 abort。库不定义 panic handler，调用方仍负责处理普通 panic 和 debug assertion。输入校验及 sink 错误仍通过 `Result` 传播。

## 性能测量入口与范围

| 命令 | 观察范围 |
| --- | --- |
| `cargo bench --features alloc --bench worst_case` | 含孔图及十一类图案：两种借用分解、image 构造、构造后立即分解、builder 整层构建；另测 scratch 初始化 |
| `cargo run --release --features profile --example worst_case_profile` | 轴区间、chord、匹配与分区的阶段耗时和规模 |
| `cargo run --release --example low_discrepancy_1000` | 一组含孔输入的平均／最大耗时，以及最慢样本的矩形数 |

`worst_case` 是基准名称，不是全局最坏输入的数学证明。`representative_64` 用确定性输入覆盖叶子／image／原生标签借用入口、直接 sink、拥有型 image 构造、构造后立即分解和 builder 整层构建，包括大 LOD、规则块、混合 LOD 行带、条纹、双色棋盘格、最大 chord 孔洞、稀疏／密集随机、稠密边框冲突、颜色小组和逆序单位像素。计时前检查各入口精确输出一致及简单图案的已知最优数；分解失败会使基准退出，不能当作零矩形计时。

Criterion 只用于开发；benchmark 的 `required-features = ["alloc"]`，未启用时 Cargo 会跳过它。`profile` 同时开启 `std`／`alloc`，阶段计时依赖宿主时钟；无 features 的 `low_discrepancy_1000` 示例自身是宿主程序，调用默认 no_std／no_alloc 库。

`voxel_checker_chunk_195_planes` 使用 195 份独立输入，模拟 64³ rock/air 棋盘格的三个轴各 63 个内层面与两个外边界面，总输出为 786432 个矩形。该基准测量分解与相同字段的 checksum 消费，排除面提取、顶点去重、最终 mesh 打包与 GPU 上传；独立的布局准备／材质映射实验见下方记录。全部分解基准消费矩形范围和标签，不再只读取数量，因此旧的 count-only Criterion 数字不能直接比较。

热路径测量应在计时外构造输入、复用 scratch，并消费返回结果；构造、输出复制和多线程场景另行测量。对比版本使用相同工具链、编译选项和样本，轮换运行顺序，保留分布而非只报最小值。阶段计时用于定位，总收益以不带阶段计时的完整调用为准。

profile 的 `chord_groups` 仍统计有 chord 的不同标签数，计数仅发生在 profile/test 路径。`matching_phases` 统计全局搜索批次，和旧版逐标签批次之和含义不同；比较时应以完整调用耗时、覆盖、最优数量及证书为准。

保留明显且可复现的收益；难以区别于噪声时，选择较简洁的实现。不要为了一个样本增加难以维护的分支，也不要以平均加速掩盖重要输入的稳定回退。正文记录稳定的算法与取舍依据；具体机器、提交之间的百分比和未落地尝试留在可复现的实验记录中。

区间位图与单调增广游标的实测结果、噪声排查和小输入取舍见 [2026-10-08 实验记录](experiments/algorithm-2026-10-08.md)。

后续的叶子事件、稳定行带、有限键排序、Builder 和冲突扫线测量见 [非匹配阶段实验记录](experiments/nonmatching-2026-10-08.md)。

固定 512-bit `std::simd` Morton 编码的局部收益、完整 pipeline 测量、nightly 要求和未保留尝试见 [SIMD 实验记录](experiments/simd512-2026-10-08.md)。SIMD 不增加原算法的工作量时保持现有复杂度；换成稠密扫描则必须重新证明上界。

连续标签切面、紧凑匹配索引、直接输出、独立输入的棋盘格 chunk 与随机种子尾部测量见 [布局与 SIMD 扫描实验](experiments/layout-simd-2026-10-08.md)。几何位集匹配在部分孔洞图上更快，但其它较坏输入稳定回退，拒绝默认合入的证据见 [几何匹配实验](experiments/geometric-matching-2026-10-08.md)。

默认 `no_std`／`no_alloc` 的 feature 分层、无分配器裸机链接检查、回归测试及局部性能回退见 [no_std 移植记录](experiments/no-std-2026-10-08.md)。
