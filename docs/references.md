# 文献与出处

[返回 README](../README.md) · 配合阅读：[算法原理](algorithm.md)、[实现与优化](matching.md)

本实现组合了几何归约、二分图匹配和像素网格工程化三层思路。下面按研究用途组织文献，不将整个实现归于单篇论文。

## 先理解矩形分解的几何归约

**Leonard A. Ferrari, P. V. Sankar, Jack Sklansky.** *Minimal rectangular partitions of digitized blobs*. Computer Vision, Graphics, and Image Processing, 28(1):58–71, 1984. [出版页面](https://www.sciencedirect.com/science/article/pii/0734189X84901397)，DOI: `10.1016/0734-189X(84)90139-7`。

这是理解数字区域最少矩形划分的直接入口：用连接凹角的线段建立二分冲突图，通过最大独立集决定最优切线。读本仓库前，先理解“为什么要最大化互不相交 chord”，再看具体匹配实现。

**Hiroshi Imai, Takao Asano.** *Efficient Algorithms for Geometric Graph Search Problems*. SIAM Journal on Computing, 15(2):478–494, 1986. [论文与摘要](https://epubs.siam.org/doi/10.1137/0215033)，DOI: `10.1137/0215033`。

该文研究水平／竖直线段相交图上的搜索、匹配及独立集，并应用于正交区域最少分解。适合继续研究如何利用几何结构。当前代码显式建立稀疏 CSR，没有照搬论文的通用几何搜索数据结构；其复杂度参数也不能直接当作图像边长。

## 再理解精确匹配

**John E. Hopcroft, Richard M. Karp.** *An n^(5/2) Algorithm for Maximum Matchings in Bipartite Graphs*. SIAM Journal on Computing, 2(4):225–231, 1973. [论文与摘要](https://epubs.siam.org/doi/10.1137/0202019)，DOI: `10.1137/0202019`。

提供分层搜索、成批增广以及 `O((顶点数 + 边数)√顶点数)` 的理论基础。论文标题中的指数不是本项目图像边长的指数。

**Iain S. Duff, Kamer Kaya, Bora Uçar.** *Design, Implementation, and Analysis of Maximum Transversal Algorithms*. ACM Transactions on Mathematical Software, 38(2), Article 13, 2011. [期刊 DOI](https://doi.org/10.1145/2049673.2049677)，[作者机构预印本](https://epubs.stfc.ac.uk/manifestation/5901)。

重点阅读 §3.3.1 的 HK、§3.3.2 的 HKDW 和 §4 的匹配预热。HKDW 在每轮最短增广之后追加本轮不相交 DFS，以增加一次 BFS 的产出。当前 [hk.rs](../src/hk.rs) 采用这一调度思路；具体缓冲布局、显式栈和预热组合以代码为准。预印本年份为 2010，正式发表年份为 2011。

## 流算法研究入口

**OpenAI.** *Almost-Linear-Time Maximum-Cardinality Matching in General Graphs*. 2026 年 9 月 24 日预印本，数学集成果 120。[论文](https://github.com/openai/math/blob/main/preprints/Almost-Linear-Time-Maximum-Cardinality-Matching-in-Sparse-General-Graphs-September-24-2026/main.pdf)。

该预印本给出一般图近线性时间精确最大基数匹配的随机化算法。其引言指出，二分匹配通过单位容量流归约已经可以使用已有近线性精确整数流算法；一般图还需处理奇块。当前像素 chord 图已经是二分图，研究后续渐近改进时应先核对这条流路线。下面两篇是其引用的原始流论文：

- **Li Chen、Rasmus Kyng、Yang P. Liu、Richard Peng、Maximilian Probst Gutenberg、Sushant Sachdeva.** *Maximum Flow and Minimum-Cost Flow in Almost-Linear Time*. FOCS 2022。[完整版本](https://arxiv.org/abs/2203.00671v2)，[DOI](https://doi.org/10.1109/FOCS54457.2022.00064)。
- **Jan van den Brand 等.** *A Deterministic Almost-Linear Time Algorithm for Minimum-Cost Flow*. 2023 年预印本。[论文](https://arxiv.org/abs/2309.16629)。

本实现仍使用 HKDW，全局建图与最终 BFS 的对称覆盖提取是对像素模型和经典匹配性质的专门化，没有移植这些近线性流算法。其渐近时间界也不直接保证 64×64 输入的加速；需要核算初始化、精度、存储以及完整调用成本。

## 研究孔洞与更强的几何条件

**W. T. Liou, J. J. M. Tan, R. C. T. Lee.** *Minimum Rectangular Partition Problem for Simple Rectilinear Polygons*. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 9(7):720–733, 1990. [机构保存的论文](https://ir.lib.nycu.edu.tw/bitstream/11536/4064/1/A1990DL35300004.pdf)。

第 II 节便于复习 chord 到最优划分的构造；后文利用无孔简单多边形的额外结构给出更快算法。其无孔前提不能从当前的像素输入契约中自动得到。

**Valeriu Soltan, Alexei Gorpinevich.** *Minimum dissection of a rectilinear polygon with arbitrary holes into rectangles*. Discrete & Computational Geometry, 9:57–79, 1993. [论文与摘要](https://link.springer.com/article/10.1007/BF02189307)，DOI: `10.1007/BF02189307`。

讨论任意孔洞、包括退化孔洞的更一般几何问题。适合在处理点接触、零宽障碍或推广输入模型时核对适用条件。不能把这类一般多边形问题与本库有限网格上的带标签像素集合不加区分地等同。

## 本项目的实现推导

最大同色区间的端点判定、跨标签 chord 不相交的全局建图依据、内部格点对总长度和冲突边的数量约束、固定容量 scratch、有序 CSR 的计数散布、最终反向 BFS 的对称覆盖提取和切线位图扫描，都在 [算法原理](algorithm.md)、[实现与优化](matching.md) 与 [容量上界](scratch.md) 中按当前代码说明。

这些是针对本项目输入模型的推导和工程选择，不应作为上述论文逐项实现的声明。继续优化时，需要分别说明：沿用了哪条已有定理、增加了什么输入前提、哪些部分由测试而非完整证明支持。
