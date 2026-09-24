---

title: "从 Roofline 到 Tensor Core：我终于开始看懂 GPU 为什么快了"
date: 2026-09-22
draft: false
tags: [CUDA, GPU, GEMM, Tensor Core, AI Infra]
categories: [AI Infra]
description: "从 Roofline 和 Arithmetic Intensity 出发，把 tiling、Occupancy、软件流水与 WMMA 连成一张 GPU 性能图，再用一版教学 GEMM 看见 Tensor Core 指令。"
summary: "从前两篇 CUDA GEMM 的手写优化继续往下：先用 Roofline 判断瓶颈，再理解数据复用、延迟隐藏与 Tensor Core 的 warp 级矩阵计算。"
math: true
ShowToc: true
TocOpen: true
---
前几天一直在写 GEMM。

最开始是 CPU 上的 `ijk`、`ikj`，后来是 CUDA naive kernel，再后来慢慢加 shared memory、tiling、register blocking，甚至做到了 double buffering。

每一次优化似乎都能解释：

> 因为 cache 好一点。
> 因为数据复用了。
> 因为 shared memory 快。
> 因为 latency 被隐藏了。

但这些解释总有一种零散的感觉。

我知道某个优化为什么“可能有效”，却还没有一个统一的模型告诉我：

> **GPU 到底为什么慢？**

今天终于把这件事慢慢串起来了。

从 Roofline 开始，一路走到了 Tensor Core。

回头看，这好像不只是在学 GEMM。

更像是在第一次理解，一块 GPU 到底是怎样被喂饱的。

这篇接着前两篇 [CUDA GEMM：把数据多用几次](../cuda-gemm/) 和 [CUDA GEMM：把时间也多用几次](../cuda-gemm-tuning/) 往下走：前面动手改了数据布局和搬运时机，这次试着找一个能解释这些优化的共同模型，最后再让 Tensor Core 跑进自己的 kernel。

这篇接着前两篇 [CUDA GEMM：把数据多用几次](../cuda-gemm/) 和 [CUDA GEMM：把时间也多用几次](../cuda-gemm-tuning/) 往下走：前面动手改了数据布局和搬运时机，这次试着找一个能解释这些优化的共同模型，最后再让 Tensor Core 跑进自己的 kernel。

---

## 先问一句：程序到底在等什么

GPU 性能优化最容易犯的错误，是看到代码之后立刻开始想：

> 这个循环能不能展开？
> 这个数据能不能放 shared memory？
> block 要不要调大一点？

但今天开始觉得，第一句话应该换成：

> **这个 kernel 现在到底在等什么？**

如果它在等计算，那么应该优化计算。

如果它在等显存，那么再怎么省两条乘法可能都没有什么意义。

这就是 Roofline Model 最简单，也最有用的地方。

它先定义一个量：

$$
AI = \frac{\text{FLOPs}}{\text{Bytes}}
$$

也就是 Arithmetic Intensity。

一份数据从内存搬过来之后，到底能做多少次计算。

Roofline 的核心关系非常简单：

$$
P =
\min(
P_{\text{peak}},
BW\times AI
)
$$

一边是 GPU 的峰值计算能力：

$$
P_{\text{peak}}
$$

另一边是内存带宽：

$$
BW
$$

这里的 `Bytes` 和 `BW` 必须对应同一层内存。Roofline 可以分别画 DRAM、L2 等不同的屋顶；眼下先把它当作定位瓶颈的简化模型，而不是不看缓存、指令和占用率就能精确预测耗时的公式。

于是性能大致受到两种上限约束。

如果：

$$
BW\times AI < P_{\text{peak}}
$$

那么程序是：

> **Memory Bound**

如果：

$$
BW\times AI > P_{\text{peak}}
$$

那么程序逐渐变成：

> **Compute Bound**

以前看 GPU benchmark，总觉得 TFLOPS 是最重要的数字。

现在才真正意识到：

> 一块 GPU 有再高的 TFLOPS，如果数据喂不过去，那些计算单元也只能闲着。

---

## GEMM 为什么那么特别

矩阵乘法：

$$
C_{M\times N}
=
A_{M\times K}B_{K\times N}
$$

计算量大约是：

$$
2MNK
$$

如果：

$$
M=N=K=N
$$

就是：

$$
2N^3
$$

但数据量只有：

$$
O(N^2)
$$

这意味着矩阵越大，同一份数据理论上可以被复用越来越多次。

这也是 GEMM 特别适合 GPU 的原因之一。

但前提是：

> **你真的把数据复用起来了。**

naive GEMM 里，每个 thread 不断从 global memory 读取 A 和 B：

```cpp
for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
}
```

如果粗略认为每次循环：

```text
读 A 4 Byte
读 B 4 Byte
做 2 FLOPs
```

那么：

$$
AI \approx \frac{2}{8}=0.25
$$

这是按每个线程在源码层面请求的操作数估出来的简化值，不等于实际 DRAM 流量：相邻线程可能复用缓存中的数据，硬件缓存也会合并请求。它帮助我看见 naive 写法里的重复工作，但不能直接拿来算真实带宽。

低得离谱。

这时候 GPU 最强的地方根本发挥不出来。

---

## Tiling 不是魔法

后来写 shared-memory tiled GEMM：

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
```

以前会说：

> shared memory 比 global memory 快。

当然没错。

但现在更喜欢另一个解释：

> **Tiling 真正重要的地方，是让一次 global memory load 服务于更多计算。**

假设：

$$
TILE=T
$$

需要加载：

$$
2T^2
$$

个 float。

也就是：

$$
8T^2\ Bytes
$$

然后计算：

$$
2T^3
$$

个 FLOPs。

于是：

$$
AI
=
\frac{2T^3}{8T^2}
=
\frac{T}{4}
$$

这个估算只计 A、B tile 的输入流量，并假设 tile 完整、能按预期复用；C 的写回、边界补零和缓存命中都先略过。它描述的是 tiling 带来的复用趋势，不是每个实际 kernel 的精确 DRAM Arithmetic Intensity。

这意味着：

$$
TILE\uparrow
\Rightarrow
AI\uparrow
$$

例如：

| Tile | Arithmetic Intensity |
| ---: | -------------------: |
|    8 |             2 FLOP/B |
|   16 |             4 FLOP/B |
|   32 |             8 FLOP/B |
|   64 |            16 FLOP/B |

所以 shared memory tiling 并不只是：

> “用了更快的内存。”

它更本质地改变了：

$$
\frac{\text{计算量}}{\text{数据搬运量}}
$$

这才是它为什么如此重要。

---

## 但 Tile 不能无限变大

看到：

$$
AI\propto TILE
$$

第一反应当然是：

> 那我把 TILE 开到 1024 不就天下无敌了？

显然不行。

比如：

```cpp
__shared__ float As[T][T];
__shared__ float Bs[T][T];
```

shared memory 占用：

$$
8T^2\ Bytes
$$

于是：

```text
T = 16 → 2 KB
T = 32 → 8 KB
T = 64 → 32 KB
T = 128 → 128 KB
```

而如果还是一个 thread 对应一个输出：

$$
threads/block=T^2
$$

那么：

```text
T = 16 → 256 threads
T = 32 → 1024 threads
T = 64 → 4096 threads
```

`T=64` 直接就没法这么干了。

所以 GPU 优化从这里开始出现一个非常熟悉的形状：

```text
收益 ↑
    │        ●
    │      ●   ●
    │    ●
    │  ●
    └────────────→ 参数
```

不是越大越好。

而是在多个资源之间找 sweet spot。

---

## Occupancy 并不是越高越好

以前很容易把 Occupancy 理解成：

> 越高越牛。

后来才发现这是个危险的误区。

Occupancy 大致表示：

$$
Occupancy=
\frac{\text{resident warps}}
{\text{maximum warps}}
$$

这是一个 SM 上活跃 warp 数的比例。高 Occupancy 能给调度器更多隐藏等待的机会，但真正需要看的是当前有没有足够多的 warp 可运行、它们是否有就绪指令；数字本身不是吞吐。可以对照 [NVIDIA CUDA Programming Guide 对 Occupancy 的定义](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。

它为什么有用？

因为 GPU 很擅长靠 warp switching 隐藏 latency。

比如：

```text
warp 0 → global load → 等数据

scheduler:
warp 1 → compute
warp 2 → compute
warp 3 → compute
```

等 scheduler 再回到 warp 0 时，数据也许已经到了。

所以更多 resident warp 意味着：

> GPU 有更多可以切换过去干活的东西。

但 Occupancy 并不是性能本身。

完全可能出现：

```text
Occupancy = 100%
SM utilization = 40%
```

也可能：

```text
Occupancy = 50%
SM utilization = 95%
```

于是今天终于把三个概念分开了：

$$
Occupancy \neq Utilization \neq Performance
$$

Occupancy 只是隐藏 latency 的一个手段。

---

## Register Blocking 的代价

这也解释了之前一个让我很困惑的问题。

从：

```text
1×1
```

做到：

```text
4×4
```

再做到：

```text
8×8
```

一个 thread 负责越来越多的输出。

好处非常明显：

```text
data reuse ↑
ILP ↑
```

但坏处也越来越明显：

```text
register usage ↑
```

比如 `8×8`：

$$
64
$$

个 accumulator。

光这些 accumulator 就可能占掉大量 registers。

于是：

```text
register pressure ↑
        ↓
resident blocks ↓
        ↓
active warps ↓
        ↓
occupancy ↓
```

更糟糕的时候甚至会 spill：

```text
register
    ↓
local memory
    ↓
device memory
```

性能可能直接掉下去。

所以以后看到：

```text
0 bytes spill stores
0 bytes spill loads
```

终于知道为什么这行值得开心了。

---

## 从“少搬”走到“早点搬”

前面的优化都在围绕一个问题：

> 数据怎样复用？

后来做 double buffering 时，开始出现一个新的问题：

> **数据什么时候搬？**

没有 pipeline：

```text
load tile 0
████

compute tile 0
    ████████

load tile 1
            ████

compute tile 1
                ████████
```

每个 tile：

$$
T=T_L+T_C
$$

如果可以重叠：

```text
compute tile 0
████████

load tile 1
████
```

那么 steady state 更接近：

$$
T=\max(T_L,T_C)
$$

这时候才真正理解：

> double buffering 已经不是单纯的内存层次优化，而是在做时间编排。

和 CPU pipeline 确实很像。

---

## `cp.async` 到底在解决什么

普通数据搬运逻辑大致是：

```text
Global Memory
      ↓
Register
      ↓
Shared Memory
```

thread 参与了整个过程。

而 `cp.async` 可以粗略理解成：

```text
Global Memory
      ↓
Shared Memory
```

异步发起。

于是：

```text
发起 copy(next tile)

继续 compute(current tile)
```

所以 `cp.async` 解决的是：

> **怎么让 copy 不把当前计算路径堵住。**

而不是：

> “让显存带宽突然翻倍。”

这一点后来尤其重要。

这里的“Global → Shared”说的是程序可见的数据路径：`cp.async` 不需要先把数据显式读进 thread 的普通寄存器，再由 thread 写入 shared。它并不意味着硬件的 DRAM 带宽变大，也不代表所有拷贝请求都会和计算完全重叠。

---

## Double Buffer 和 Multi-stage

刚开始觉得这两个好像是一个东西。

后来慢慢分清了。

`cp.async` 回答：

> **怎么异步搬？**

Double buffering / multi-stage 回答：

> **提前多少步搬？**

如果：

$$
T_{compute}>T_{load}
$$

那么：

```text
compute
██████████

load next
████
```

下一份数据早就准备好了。

这时候 two-stage 通常已经够用。

但如果：

$$
T_{compute}<T_{load}
$$

例如：

```text
compute
████

load next
██████████
```

compute 做完，数据还没回来。

那么只提前一个 tile 不够。

于是：

```text
prefetch k+1
prefetch k+2
prefetch k+3
```

目的不是“搬更多”。

而是：

> **让未来的数据更早出发。**

等真正需要 `k+2` 时，它已经在路上走了很久。

这就是 multi-stage pipeline 的意义：让足够多的独立拷贝在途，争取盖住请求延迟。stage 数不能只凭 `T_compute` 和 `T_load` 的比值推出；如果持续搬运量已经撞上带宽上限，继续加深 stage 也不会凭空提高带宽，反而会多占 shared memory。

---

## Latency 和 Bandwidth

这里也是今天很重要的一个分界。

Pipeline 擅长解决：

$$
\boxed{\text{Latency}}
$$

即：

> 请求发出以后，要多久回来？

但如果问题是：

$$
\boxed{\text{Bandwidth}}
$$

例如硬件最多：

$$
1TB/s
$$

而 kernel 持续需要：

$$
1.5TB/s
$$

那：

```text
2 stage
4 stage
8 stage
```

都没法救。

因为物理上每秒就只能搬这么多。

所以现在我喜欢用两句话总结：

> **Tiling：少搬一点。**

> **Pipeline：早点搬，别让我等。**

前者减少 traffic。

后者隐藏 latency。

---

## Tensor Core 出场

到这里才进入 Tensor Core。

以前普通 CUDA Core 最基本的是：

$$
c=a\times b+c
$$

也就是 scalar FMA。

Tensor Core 做的是：

$$
D=A\times B+C
$$

也就是 MMA：

> Matrix Multiply-Accumulate。

从：

```text
scalar
```

变成：

```text
matrix tile
```

计算基本单位发生了变化。

---

## 从 Thread 到 Warp

普通 CUDA 世界：

```text
thread 0 → 自己的 FMA
thread 1 → 自己的 FMA
...
thread 31 → 自己的 FMA
```

Tensor Core 世界则更像：

```text
lane 0 registers ┐
lane 1 registers │
lane 2 registers │
...              ├── MMA
lane 31 registers┘
```

32 个 lane 一起构成一条 matrix operation 的 operands。

所以：

$$
\boxed{\text{Warp-level computation}}
$$

真正进入了视野。

这也意味着：

> Tensor Core GEMM 的设计单位，逐渐不再是“一个 thread 算几个元素”，而是“一个 warp 负责哪一个 tile”。

---

## Fragment

然后遇到了 WMMA 的核心抽象：

```cpp
wmma::fragment
```

fragment 很容易被误认为是：

```cpp
float frag[16][16];
```

实际上不是。

它更像：

> **逻辑上的矩阵 tile，物理上分布在整个 warp 的 registers 中。**

例如逻辑上：

$$
A_{16\times16}
$$

物理上可能类似：

```text
lane 0  → 一部分元素
lane 1  → 一部分元素
lane 2  → 一部分元素
...
lane 31 → 一部分元素
```

所有 lane 合起来才构成一个 fragment。

所以：

$$
\boxed{
Fragment
=
logical matrix tile
+
distributed registers
}
$$

CUDA 文档把 `fragment` 定义为分布在 warp 各线程中的矩阵片段，同时明确说元素在各 lane 内的具体映射没有被 WMMA API 固定下来。因此，把它想成“全 warp 合起来的一块逻辑 tile”很有用；把它当成布局确定的 `float[16][16]` 则不行。详见 [CUDA Programming Guide 的 WMMA 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。

---

## WMMA 其实并没有想象中神秘

最小 WMMA 大概只有：

```cpp
fill_fragment()

load_matrix_sync()

mma_sync()

store_matrix_sync()
```

数据流：

```text
Memory
   ↓
load_matrix_sync
   ↓
Warp Registers / Fragment
   ↓
mma_sync
   ↓
Tensor Core
   ↓
Accumulator Fragment
   ↓
store_matrix_sync
   ↓
Memory
```

它和以前：

```cpp
acc += a * b;
```

其实完全对应。

只是：

```text
float a
float b
float acc
```

变成：

```text
A fragment
B fragment
C fragment
```

而：

```cpp
acc += a * b;
```

变成：

```cpp
mma_sync(c_frag, a_frag, b_frag, c_frag);
```

突然觉得 Tensor Core 也没有那么神秘了。

---

## 三层 Tile

今天最后终于把现代 GEMM 最重要的三层结构串了起来：

$$
CTA_M,CTA_N,CTA_K
$$

$$
Warp_M,Warp_N,Warp_K
$$

$$
MMA_M,MMA_N,MMA_K
$$

它们分别回答三个问题。

### CTA

一个 block 负责多大的输出：

```text
Entire GEMM
     ↓
CTA Tile
```

### Warp

CTA 内部，一个 warp 负责哪个区域：

```text
CTA Tile
    ↓
Warp Tile
```

### MMA

一个 warp 的 output 再拆成若干 Tensor Core 能吃下的最小 tile：

```text
Warp Tile
    ↓
MMA Tile
```

最终：

```text
Entire GEMM
    ↓
CTA Tile
    ↓
Warp Tile
    ↓
MMA Tile
    ↓
Tensor Core
```

以前看到 CUTLASS 里的：

```cpp
GemmShape<128, 128, 32>
GemmShape<64, 32, 32>
GemmShape<16, 8, 16>
```

总觉得像模板黑魔法。

现在大概已经能猜到它们在描述什么了。

---

## 第一次真的跑起来

最后写了一个教学版 WMMA GEMM：

$$
CTA=64\times64\times16
$$

$$
Warp=32\times32\times16
$$

$$
MMA=16\times16\times16
$$

矩阵：

$$
1024\times1024
$$

这份程序的两个 kernel 都读 FP16（`half`）输入、用 FP32 累加，并把 FP32 结果写回。这里的“普通 tiled”指逐元素循环的 CUDA Core 版本；它和 WMMA 版本使用同一组输入与输出类型，不能和上一篇不同尺寸、FP32 输入的 benchmark 混成一张表。

普通 tiled CUDA GEMM：

```text
time   ≈ 1.862 ms
TFLOPS ≈ 1.15
```

WMMA Tensor Core GEMM：

```text
time   ≈ 0.356 ms
TFLOPS ≈ 6.03
```

于是：

$$
\boxed{Speedup\approx5.23\times}
$$

而：

```text
Max abs error = 0
```

至少在当前测试输入下，结果完全一致。

校验遍历了整个 `1024×1024` 输出矩阵，但输入是手工构造的、以 `1/16` 为步长的数值；`max abs error = 0` 只说明这组可精确表示的测试数据完全一致，不代表一般随机输入都没有舍入误差。代码还用 `static_assert` 固定要求 M、N、K 能整除 tile；它不是可以直接处理任意矩阵尺寸的通用实现。

计时器先预热 5 次，再用 CUDA Events 测 20 次并取平均，每个 kernel 各测一批。这组结果适合看当前实现间的相对差异，不是多轮统计出的稳定范围。

第一次感觉：

> Tensor Core 不再只是文档里的“AI 加速单元”。

它真的在自己的代码里跑起来了。

---

## ptxas 也开始能看懂了

普通 tiled kernel：

```text
Used 35 registers
1024 bytes smem
0 bytes spill
```

WMMA：

```text
Used 56 registers
4096 bytes smem
0 bytes spill
```

这是当时这份源码和编译配置下 `ptxas` 的结果；寄存器数会随编译器、架构和代码变化，shared memory 数量则可以直接从下面的 tile 尺寸算出来。

普通 tiled：

$$
16\times16\times2B\times2
=
1024B
$$

WMMA：

A tile：

$$
64\times16
$$

B tile：

$$
16\times64
$$

FP16 每元素 2B：

$$
64\times16\times2
+
16\times64\times2
=
4096B
$$

正好对应：

```text
4096 bytes smem
```

而 register 从：

$$
35\rightarrow56
$$

也很好理解。

因为现在每个 warp 要保存：

```text
A fragments
B fragments
Accumulator fragments
```

register pressure 自然更高。

但：

```text
0 bytes spill
```

说明还处在健康范围内。

---

## SASS 给出的最后证据

最后：

```bash
cuobjdump --dump-sass ./wmma_gemm | grep HMMA
```

真的看到了：

```asm
HMMA.16816.F32
```

甚至还有：

```asm
R36.reuse
R48.reuse
```

`HMMA` 是这次编译结果里出现的矩阵乘加指令；指令名和 SASS 形式跟目标架构、工具链有关。这能确认编译器确实生成了 Tensor Core 路径，但不能单靠一条指令判断整个 kernel 已经高效。

这一下很有意思。

之前一直在讲：

```text
一个 A fragment
可以和多个 B fragment 组合

一个 B fragment
也可以参与多个输出 fragment
```

到了 SASS 里，居然真的开始看到 register reuse 的痕迹。

整个链条终于完整了：

```text
CUDA source
    ↓
WMMA
    ↓
fragment
    ↓
mma_sync
    ↓
HMMA
    ↓
Tensor Core
```

从代码一路走到了机器指令。

---

## 为什么还只有 6 TFLOPS

当然，6 TFLOPS 离这块 GPU 的 Tensor Core 峰值还差得很远。这个教学样例没有做 profiler 分析，所以我不能仅凭这份代码断言唯一瓶颈；但从执行顺序可以直接看见，它还没有把搬运和计算流水化：

现在的 kernel 还是：

```text
Global → Shared

Barrier

Shared → Fragment

MMA

Barrier

Global → Shared

...
```

也就是：

```text
LOAD
██████

WAIT

MMA
       ██

WAIT

LOAD
          ██████
```

Tensor Core 太快了。

它算两下就：

> 我的下一块数据呢？

所以真正的高性能实现还会继续做：

```text
cp.async
double buffering
multi-stage pipeline
shared-memory swizzle
ldmatrix
mma.sync
warp specialization
...
```

这些是下一层优化方向，不是这次 5.23 倍对比已经包含的东西。

但这里我决定先停下来。

不是因为这些不重要。

而是因为继续手搓下去，已经开始从“理解 GPU 性能”进入“专门做 GEMM kernel engineering”了。

对于现在想走的 AI Infra / AI Compiler 路线，下一步更有意义的是：

```text
手写 CUDA
    ↓
CUTLASS
    ↓
Triton
    ↓
TVM / MetaSchedule
```

也就是开始观察：

> **工业框架和编译器，是怎样把今天手动做的这些事情抽象起来的。**

---

## 源码与复现

这次的教学实现放在 [wmma-gemm.cu](../../downloads/wmma-gemm.cu)。它包含两个 kernel：一个使用 CUDA Core 做 FP16 输入、FP32 累加的 tiled GEMM；另一个用 WMMA 把 A/B tile 装入 warp fragment，再调用 `mma_sync`。

在支持这组 FP16 WMMA 形状的 NVIDIA GPU 上编译运行：

```bash
nvcc -O3 -arch=sm_89 wmma-gemm.cu -o wmma_gemm
./wmma_gemm
```

检查编译器报告的寄存器与 shared memory 用量，以及是否生成 HMMA 指令：

```bash
nvcc -O3 -arch=sm_89 -Xptxas -v wmma-gemm.cu -o wmma_gemm
cuobjdump --dump-sass ./wmma_gemm | grep HMMA
```

这份代码将 M、N、K 固定为 1024，并要求它们整除 CTA tile；要支持边界 tile，还需要补齐 WMMA store 的边界处理。文章里的耗时是在 RTX 4070 Laptop、`sm_89` 目标下记录的。换 GPU、CUDA 版本或矩阵尺寸后，时间、寄存器数和 SASS 都可能变化。

Occupancy、寄存器和资源限制可继续看 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)，WMMA 的 `fragment` 与 `mma_sync` 约定见 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。

---

## 回头看

今天最有价值的可能不是学会了 Tensor Core。

而是脑子里终于有了一张比较完整的 GPU 性能图：

```text
                     Performance
                          │
          ┌───────────────┴──────────────┐
          │                              │
       Compute                         Memory
          │                              │
      CUDA Core                       HBM
      Tensor Core                     L2
          │                           Shared
          │                              │
          └───────────────┬──────────────┘
                          │
                      Scheduling
                          │
              ┌───────────┼───────────┐
              │           │           │
             ILP      Occupancy    Pipeline
              │           │           │
              └───────────┴───────────┘
                          │
                     Resources
                  Registers / SMEM
```

以后再看到一个优化，应该不只是问：

> 它快了吗？

而是问：

> 它减少了 traffic 吗？
> 它提高了 reuse 吗？
> 它改变了 Arithmetic Intensity 吗？
> 它在隐藏 latency 吗？
> 它增加了 register pressure 吗？
> 它影响 occupancy 吗？
> 它是在优化空间布局，还是时间调度？

这样看，一个 CUDA kernel 就不再是一堆：

```cpp
__shared__
__syncthreads()
#pragma unroll
```

而开始变成一组相互制约的设计决策。

以前我是在学：

> **怎么写一个更快的 GEMM。**

今天之后，更像是在慢慢理解：

> **为什么一个 GEMM 能快。**

这两句话看起来差不多。

但好像已经不是一回事了。
