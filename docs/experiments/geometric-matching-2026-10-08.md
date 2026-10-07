# 2026-10-08：几何位图 HKDW 实验

本次尝试利用 64×64 格点约束，把冲突邻接和访问状态直接表示为几何位平面，目标是改善含孔输入的完整分解耗时。三个候选均保持精确最小矩形分解、安全 Rust、借用入口零堆分配和总体 `O(n³)` 上界，均未合入主库。

**比较基线是隔离的 compact matcher，scratch 为 256,048 字节。** 它已将匹配 offset、游标及 DFS 栈改为 u16，仍使用原 CSR/HKDW；不是本轮输入、输出及其它主库优化的整体基线。这里的百分比不能与本轮其它实验相加。完整调用仍使用相同的 leaf 输入、轴区间构造和矩形输出，仅替换匹配后端。

## 三个候选

首版 `first` 的本地源码冻结于 `target/geometric-hk-first`，crate 为 `rectangle_decomposition_geometric_hk_first`，scratch 为 **290,112 字节**。

- 按 x 保存水平 chord 的 y 占用位，按 y 保存垂直 chord 的 x 占用位，各使用 64 个 u64；两份 u16[4096] owner 表把已置位格点映射回顶点编号。owner 只在相应占用位置读取，复用时不清空。
- 保留度 1 横 chord 优先和右部度数排序的 greedy 思路，之后用范围 AND 查找 free V 并执行长度 1/3 增广。首版按最低 x 选择候选，也优先尝试直接空闲邻居，因此不保证与旧 seed 的具体配对相同。
- 反向 BFS 从 free V 出发。每个 H 首次可达时，沿其完整 x 跨度清除 unseen-H 位；相应 mate V 入队，或把空闲 H 加入最短根列表。
- DFS 全轮共享 available-V 位平面。进入一个 matched H 时，沿其 mate V 的 y 跨度清位；仍可用的 matched V 因而必定指向未访问 H，省去逐边的 visited-bool 间接读取。free 终点也立即清位。DFS 保留最短层搜索和同轮追加的不相交路径搜索，最终失败 BFS 生成原 minimum-cover 编码。
- 连续跨度的完整八行块使用 `std::simd::u64x8`，逻辑宽度为 **512 bit**，不足八行的尾部标量处理。当前机器没有 AVX-512；更宽的逻辑向量不意味着一次硬件指令。

`rank` 源码保留于 `target/geometric-hk-candidate`，crate 为 `rectangle_decomposition_geometric_hk_candidate`，scratch 为 **294,032 字节**。在一次右部计数排序后构造 `(degree, right-id)` 的逆 rank；长度 3 seed 的外层邻居和内层 free alternate 都选最小 rank，右排序 seed 选择最小左输入编号。128 组 shuffled legal chord 测试确认其 seed 的两侧配对与原 CSR greedy 完全一致。HK 阶段仍按 x 位顺序搜索。

`ordered` 源码保留于 `target/geometric-hk-ordered-candidate`，crate 为 `rectangle_decomposition_geometric_hk_ordered_candidate`，scratch 同为 **294,032 字节**。它使用 rank seed，只在首次 BFS 成功后，把每个 H 的邻居 x 坐标按右 rank 预排一次，复用闲置的转置 offset/edge 缓冲。DFS 按单调游标读取这份坐标邻接，并用 available-V 位筛掉已访问目标。它引入了一份有序坐标邻接；不像首版那样完全省去邻接序列。

原通用 CSR/HK 保留在候选中供独立 oracle 使用，相关 scratch 也暂未删除。首版和 rank 的匹配热路不构造 CSR/transpose，只复用右度数存储和边数诊断；ordered 复用转置缓冲存放单方向的 x 序列。

## 复杂度与实际访问量

令 `C=H+V`，`E` 为冲突边数，`Δ≤n−1` 为最大度数，`S_H`、`S_V` 为两方向的闭端点内部跨度总和。同向 chord 的这些跨度互不重叠，因此分别至多 `(n−1)²`，且 `C,E=O(n²)`。

几何准备为 `O(S_H+S_V+C)`。每轮 BFS 对 H 至多访问一次，每轮 DFS 对 H 至多展开一次、对 V 跨度至多清除一次；候选边枚举至多为 E。每轮成本为 `O(S_H+S_V+C+E)`，保留 HKDW 的最短路径阻塞及不相交追加搜索，轮数为 `O(√C)`，总体仍为 `O(n³)`。

rank seed 重复扫描当前邻居以选择最小 rank，ordered 的一次预排也使用这种扫描，成本可达 `O(EΔ)=O(n³)`。这些扫描只在 seed/预排执行一次，没有放进每个 HK phase，因此不会产生 `O(n⁴)`。不过这个合法的非 phase 成本也可能成为新的最坏情况主导项。

首版 BFS 用 unseen 位省掉重复 H 的访问，DFS 用位图跳过多个不可用 V；但实际选中的候选仍要读取 owner 和 pair，清跨度也增加读写。ordered 为得到有序邻居必须推进序列游标，最坏每轮扫描 E 个坐标，削弱了位图一次跳过多个已访问候选的优势。`O(n³)` 保证本身不能证明常数更小。

## 一轮完整调用配对

每个候选与 compact 基线编入同一 release 进程，逻辑 CPU 2，scratch 在计时外构造并复用。每阶段连续调用 64 次，先交替预热 128 次，再进行 41 轮 ABBA/BAAB；每轮分别平均两段基线、两段候选时间。表中的变化为每轮 `(候选/基线−1)` 的中位数，中间 50% 为该成对变化的第 25/75 百分位。**每个版本仅有一次完整采样**，不能当作跨重复、机器或所有输入的稳定结论，也不能直接比较不同版本采样中的绝对微秒数。

下表准确摘录 `target/geometric-hk-comparison/run-1.csv`、`rank-run-1.csv`、`ordered-run-1.csv` 的四个主要含孔/稠密案例。负值表示候选更快；绝对耗时列为各自样本中位数，故百分比不一定等于两列中位数的直接比值。

| 版本 | 输入 | compact µs | 候选 µs | 成对变化中位数 | 中间 50% 变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| first | low_discrepancy_holes | 95.388 | 84.145 | -12.52% | -17.46% 至 -10.17% |
| first | alternating_holes_max_chords | 74.853 | 71.816 | -4.72% | -7.51% 至 -2.32% |
| first | dense_random_holes | 74.784 | 83.791 | **+11.37%** | **+7.75% 至 +14.87%** |
| first | dense_boundary_chords | 43.094 | 37.770 | -12.14% | -13.55% 至 -11.83% |
| rank | low_discrepancy_holes | 87.931 | 82.009 | -7.02% | -8.24% 至 -5.22% |
| rank | alternating_holes_max_chords | 75.850 | 74.472 | -1.94% | -4.66% 至 -1.31% |
| rank | dense_random_holes | 73.272 | 84.084 | **+14.51%** | **+11.83% 至 +16.91%** |
| rank | dense_boundary_chords | 43.273 | 38.277 | -11.74% | -12.72% 至 -9.91% |
| ordered | low_discrepancy_holes | 98.427 | 130.580 | **+33.45%** | **+26.85% 至 +38.26%** |
| ordered | alternating_holes_max_chords | 76.788 | 75.035 | -1.79% | -4.94% 至 +0.14% |
| ordered | dense_random_holes | 72.621 | 93.941 | **+27.23%** | **+23.09% 至 +32.48%** |
| ordered | dense_boundary_chords | 43.927 | 37.527 | -13.77% | -16.48% 至 -12.85% |

首版在低差异、最大 chord 和稠密边框上有局部收益，但 dense-random 的整段 IQR 都是回退。rank 没有解决这项回退。ordered 在 dense-random 和原低差异坏例上都出现整段 IQR 为正的明显代价；最大 chord 的 IQR 跨零，不能称为可靠收益。

其它八个案例也不能概括为全部无代价。以下是剩余案例中 IQR 全为正的回退；这属于完整调用结果，不据此把没有匹配工作的分布回退归因于某个匹配循环。

| 版本 | 输入 | 成对变化中位数 | 中间 50% 变化 |
| --- | --- | ---: | ---: |
| first | large_lod | +6.03% | +5.15% 至 +8.97% |
| rank | large_lod | +2.83% | +0.40% 至 +5.96% |
| ordered | large_lod | +3.39% | +1.67% 至 +5.53% |
| ordered | mixed_lod_bands | +2.27% | +0.45% 至 +5.15% |
| ordered | stripes | +1.85% | +0.08% 至 +6.14% |

first 的 mixed_lod_bands 为 +1.83%，IQR 为 0.00% 至 +3.70%；其它剩余案例中的正中位数均有 IQR 跨零，不能据此宣称稳定回退或收益。

## Phase 差异与未合入原因

另用 `decompose_profile` 检查确定性的 greedy/phase/augment 计数，不把短 profile 的绝对计时作为最终性能结论。这里的 phase 字段包含最后一次失败的 BFS attempt：例如最大 chord 输入的 1 表示一次失败 BFS、没有增广轮。

| 输入 | H / V | greedy matches | 后续增广数 | compact phases | first phases | rank phases | ordered phases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_discrepancy_holes | 770 / 750 | 716 | 25 | 4 | 3 | 3 | 4 |
| alternating_holes_max_chords | 1953 / 1953 | 1922 | 0 | 1 | 1 | 1 | 1 |
| dense_random_holes | 583 / 578 | 541 | 19 | 3 | 4 | 4 | 3 |
| dense_boundary_chords | 122 / 122 | 122 | 0 | 0 | 0 | 0 | 0 |

恢复旧 seed 并没有恢复 dense-random 的 phase 数，说明该差异还来自后续 DFS 次序。ordered 把 dense-random 的 phase 数恢复为 3，却把 low-discrepancy 的 3 恢复为 4；额外预排和序列扫描又抵消了减少 phase 的潜在收益。三个版本的最终匹配数、矩形数和精确覆盖都正确。

因此本次未将几何 HK 设为默认后端。当前证据不足以用预先可见的 C、E、Δ、度分布、跨度或 seed 空闲数稳定选择 x/rank：这些统计可估计每轮读写成本，不能可靠预测搜索次序引起的 phase 差异。按案例名称选路会掩盖反例，缺少独立分布和对抗输入的验证。用户优先要求坏情况速度，现有回退不能用其它案例的收益抵消后宣称已经普遍改善。

若继续研究，可以先把有序序列构造改为全局 rank 的线性 scatter，避免一次 `O(EΔ)` 预排，再用完整调用的尾部耗时与独立输入集验证选路规则；这仍需实测，未在本轮实现。

## 验证与复现边界

首版通过 62 项、rank/ordered 各通过 63 项 debug 全 feature 测试。包括全部 `3^9` 彩色像素输入的独立 minimum/coverage oracle、192 组随机 5+5 legal chord 的 brute-force MIS、64 组 shuffled 多段图的通用 CSR 最大匹配数及 matching/cover 证书、3969 边容量图、BFS 深度 62 的 63-H/63-V 种子长路径、端点 63、失败 BFS 与孤立顶点、重复 owner 复用、零分配及四个 2 MiB worker 栈。rank/ordered 另有 128 组 seed 配对与旧 CSR greedy 的精确对照。

候选、配对程序、CSV 和短 profile 均保存在被 Git 忽略的 `target/`，不会随仓库提交。本文件记录了算法、复杂度、输入案例、测量协议及结果；保留这些本地文件时可原样重跑，新的 checkout 需要按记录重建候选与配对程序，不能把忽略目录视为已归档的源码。

本地 `target/geometric-hk-comparison/Cargo.toml` 的 `before` 固定指向 `../compact-matching-candidate`；`after` 分别选择上述 first/rank/ordered 的 crate 与目录。然后执行：

```powershell
cargo build --offline --release --manifest-path target/geometric-hk-comparison/Cargo.toml --target-dir target/geometric-hk-comparison/build
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity=[IntPtr]4
./target/geometric-hk-comparison/build/release/geometric-hk-comparison.exe
```

配对程序使用现有 12 种 representative leaf 生成器，分别复用独立 scratch，以输出矩形数一致性检查后再计时。profile 程序与计数日志在 `target/geometric-seed-profile/`。重跑时保持各库相同 ISA 和 release 选项，隔离编译/其它采样的 CPU 竞争，并增加独立重复后再判断是否保留后端。
