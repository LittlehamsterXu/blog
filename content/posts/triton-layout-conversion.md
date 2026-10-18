---
title: "Triton Layout：一次转换，到底搬了什么"
date: 2026-10-18
draft: true
tags: [Triton, Gluon, GPU, CUDA, AI Infra]
categories: [AI Infra]
description: "固定两端的 Blocked Layout，跟着一个元素从 lane 走到 warp，再沿着 PTX 里的地址，看一次 Layout Conversion 怎样借助 Shuffle 或 Shared Memory 完成。"
summary: "sizePerThread 从 2 改成 4，元素的下标没有变，拿着它的线程却可能变了。这次用一个只有 load、convert_layout、store 的小实验，追到 x[64] 真正经过的 shared-memory 地址。"
math: true
ShowToc: true
TocOpen: true
---

[上一篇](../triton-thread-mapping/)里，同样是 128 个数的求和，改一下 `num_warps`，编译出来的线程分配和通信方式就变了。

不过，求和会用到 Shuffle，并不能说明我们已经看懂了 Layout Conversion。求和需要把多个值合成一个值；转换布局时，值本身可以完全不变，只是换了线程来拿。

我想把这件事单独拿出来看看。

这次让输入原样成为输出，中间只留一次显式的 `convert_layout`。然后挑出一个数，看看它原来在哪个线程，后来去了哪里。

## 固定两端

普通 Triton 会替我们选择和传播布局。为了让这次实验的两端固定下来，这里换用 Gluon，直接给 `arange` 指定 Layout：

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def convert_demo(X, Y,
                 N: gl.constexpr,
                 SRC: gl.constexpr,
                 DST: gl.constexpr):
    src_idx = gl.arange(0, N, layout=SRC)
    x = gl.load(X + src_idx)

    y = gl.convert_layout(x, DST)

    dst_idx = gl.arange(0, N, layout=DST)
    gl.store(Y + dst_idx, y)
```

启动时传入两套布局：

```python
src = gl.BlockedLayout([2], [32], [nw], [0])
dst = gl.BlockedLayout([4], [32], [nw], [0])

kernel = convert_demo[(1,)](
    x, y, n, src, dst, num_warps=nw
)
```

`grid=(1,)`，整个数组由一个 program 处理。实验只用完整的固定形状，所以省掉了 mask。

这段代码里的两个 `arange` 都表示 `0, 1, ..., N-1`，区别在于这些位置分给了哪些线程。`gl.load` 按 SRC 的分工读入数据，`gl.store` 按 DST 的分工写回。夹在中间的转换，负责让数据接上后半段的分工。

因此转换前后，在逻辑坐标 `i` 上都有：

$$
y[i]=x[i]
$$

数组没有被转置，也没有把 `x[64]` 变成 `x[16]`。变化的是：哪个 warp、哪个 lane，以及线程里的哪个 value slot 持有 `x[64]`。

这里用 Gluon，是为了把这一项实验条件写死。平时写普通 Triton kernel，并不需要因此把每个 `tl.arange` 都改成手工指定布局。

## 每线程几个值

先看一个 warp、`N=128` 的情况：

```text
SRC = BlockedLayout([2], [32], [1], [0])
DST = BlockedLayout([4], [32], [1], [0])
```

这四个数组在 Gluon 的接口中依次叫 `size_per_thread`、`threads_per_warp`、`warps_per_cta` 和 `order`。到了 TTGIR，会看到 `sizePerThread` 这样的拼写。它们描述的是同一套分工：

| 参数 | SRC 中的值 | 在这个一维例子里怎样理解 |
| --- | --- | --- |
| `sizePerThread` | `[2]` | 先给每个线程连续的两个元素 |
| `threadsPerWarp` | `[32]` | 把这组分配沿着一维方向铺给 32 个 lane |
| `warpsPerCTA` | `[1]` | 再沿这个方向铺给一个 warp |
| `order` | `[0]` | 只有第 0 维，没有多维展开顺序的选择 |

在二维布局里，数组各项分别对应两个维度，`order=[1, 0]` 就表示先沿第 1 维展开，再沿第 0 维展开。这里先停留在一维，把元素归属看清楚就够了。[官方 Layout 教程](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)有更完整的二维展开示例。

SRC 这一轮能覆盖：

$$
2\times32\times1=64
$$

但 tensor 有 128 个元素，一轮还不够。于是同样的分工再覆盖一次，增加线程内部持有的值。

lane 0 第一轮拿 `x[0], x[1]`，第二轮接着拿 `x[64], x[65]`。它最终仍然有四个值。`sizePerThread=[2]` 描述的是基本分组，不能直接读成“整个 tensor 里，每个线程总共拿两个数”。

DST 的一轮则能覆盖 `4 × 32 × 1 = 128` 个元素，不需要重复。把两边列出来，差别就很明显了：

| Lane | SRC 持有的下标 | DST 应该持有的下标 |
| ---: | --- | --- |
| 0 | 0, 1, 64, 65 | 0, 1, 2, 3 |
| 1 | 2, 3, 66, 67 | 4, 5, 6, 7 |
| 16 | 32, 33, 96, 97 | 64, 65, 66, 67 |
| 31 | 62, 63, 126, 127 | 124, 125, 126, 127 |

两边都是每线程四个值。只数元素数量，根本看不出这次转换做了什么。

把线程内的四个 slot 编号为 `0、1、2、3`，还能写出两条很短的下标式：

```text
SRC：i = 2*lane + (slot % 2) + 64*(slot // 2)
DST：i = 4*lane + slot
```

SRC 的 slot 0、1 属于第一轮，slot 2、3 属于第二轮。这是同一个线程里同时需要持有的四个值，不是说执行时必须先跑完第一轮，再跑第二轮。“重复覆盖”在这里是在描述布局怎样把整个逻辑范围铺满。

![一个 warp 内的布局转换：SRC 按两个连续值分组并重复覆盖两轮，DST 按四个连续值分组；x64 从 lane 0 的 slot 2 经 Shuffle 到 lane 16 的 slot 0。](/images/triton/layout-one-warp-shuffle.svg "两端都拿四个值，但拿的不是同样四个。颜色跟着逻辑元素走，线程和 slot 可以改变。")

## 同一个 warp

沿着上面的表找 `x[64]`：

```text
SRC：warp 0，lane 0，第 2 个 value slot
                         │
                         ▼
DST：warp 0，lane 16，第 0 个 value slot
```

slot 从 0 开始计数，只用来表示当前线程持有的第几个逻辑值，不是 PTX 的寄存器编号。

lane 16 原先没有 `x[64]`，所以仅仅调整自己的寄存器顺序还不够。但它与 lane 0 同属一个 warp，可以通过 Shuffle 从别的 lane 取得值。

这次编译出来的 PTX 里出现了四条 `shfl.sync`，没有 shared-memory 读写，metadata 报告的 shared memory 也是 `0 B`。完整文件放在 [一个 warp 的 PTX](../../downloads/triton-layout-conversion/convert_w1_n128.ptx) 中。

四条是汇编文本中的静态指令条数。它并不表示整个 kernel 只交换了四个元素；一条 warp 指令会由参与的 lane 一起执行，分别处理各自的寄存器。

TTGIR 中的转换则仍然很简洁。去掉与阅读无关的变量名后，关键部分是：

```mlir
#src = #ttg.blocked<{sizePerThread = [2],
                    threadsPerWarp = [32],
                    warpsPerCTA = [1], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [4],
                    threadsPerWarp = [32],
                    warpsPerCTA = [1], order = [0]}>

%y = ttg.convert_layout %x
    : tensor<128xf32, #src> -> tensor<128xf32, #dst>
```

这段是便于对照的节选，完整的 [TTGIR](../../downloads/triton-layout-conversion/convert_w1_n128.ttgir) 保留了原始名字和上下文。`tensor<128xf32>` 没变，附在类型上的布局变了。往下到 PTX，才看见这项变化落成了具体的交换指令。

## 跨过 warp

再把配置换成四个 warp、`N=512`：

```text
SRC = BlockedLayout([2], [32], [4], [0])
DST = BlockedLayout([4], [32], [4], [0])
```

SRC 的基本覆盖范围变成 `2 × 32 × 4 = 256`，同样重复两轮；DST 则一轮覆盖全部 512 个值。两边还是每线程四个元素。

这次用 `tid = warp_id × 32 + lane_id` 表示整个线程块里的线程编号。SRC 中，前两个值的下标是 `2*tid`、`2*tid+1`，后两个再各加 256；DST 中，则是 `4*tid` 到 `4*tid+3`。

例如：

| 线程 | SRC 持有的下标 | DST 应该持有的下标 |
| --- | --- | --- |
| tid 0：warp 0 / lane 0 | 0, 1, 256, 257 | 0, 1, 2, 3 |
| tid 16：warp 0 / lane 16 | 32, 33, 288, 289 | 64, 65, 66, 67 |
| tid 32：warp 1 / lane 0 | 64, 65, 320, 321 | 128, 129, 130, 131 |

`x[64]` 原来在 tid 32，转换后要交给 tid 16：

```text
warp 1 / lane 0                 warp 0 / lane 16
       x[64] ─── Shared Memory ───→ x[64]
```

Shuffle 无法从另一个 warp 读取寄存器。这次编译结果用了 shared memory：源线程先写入一块共同可见的存储，线程块同步，再由目标线程按新的分工读出。

在 [四个 warp 的 PTX](../../downloads/triton-layout-conversion/convert_w4_n512.ptx) 里，可以找到两条向量化的 `st.shared`、一条 `bar.sync`，以及两条向量化的 `ld.shared`。没有出现 `shfl.sync`，shared memory 用量为 `2048 B`。

“先写，再同步，再读”已经解释了为什么可以跨 warp 换手。但到这里，还有一个容易凭直觉猜错的地方：`x[64]` 究竟写到了 shared memory 的哪里？

## 它写到了哪里

输入是 `float32`，每个元素四字节。tid 32 按 SRC 布局读入 `x[64]` 时，global-memory 地址自然是：

```text
X + 64 * 4 = X + 256 bytes
```

shared memory 是转换时临时安排的一块空间，没有义务继续保持输入数组的顺序。沿着这份 PTX 的地址计算往下代，tid 32 写入 `x[64]` 的位置，其实是：

```text
shared_base + 128 bytes
```

先摘出真正执行读写的几条指令，省略中间的地址计算，以及其他值的搬运：

```ptx
ld.global.v2.b32 { %r2, %r3 }, [ %rd1 + 0 ];
st.shared::cta.v2.b32 [ %r1 + 0 ], { %r2, %r3 };

bar.sync 0;

ld.shared.v2.b32 { %r7, %r8 }, [ %r22 ];
st.global.v4.b32 [ %rd3 + 0 ], { %r7, %r8, %r9, %r10 };
```

每个线程执行同一段程序，但寄存器里的地址各不相同。这里需要分开看两个线程：tid 32 的 `%rd1` 是 `X+256`，`%r1` 是 `shared_base+128`；tid 16 的 `%r22` 是 `shared_base+128`，`%rd3` 是 `Y+256`。

`v2.b32` 一次搬两个 32-bit 值，所以 tid 32 的 `%r2, %r3` 中是 `x[64], x[65]`。同步之后，tid 16 通过同一个 shared-memory 地址，把它们读进自己的 `%r7, %r8`。最后一条 `v4.b32` 连同另外两个值一起写回。

只看 `x[64]`，走过的地址可以连成：

```text
tid 32                                 tid 16
X + 256 ──→ shared_base + 128 ──barrier──→ Y + 256
```

这条路径来自 PTX 的静态地址推导。没有在运行时采集访存轨迹，也不需要猜测“值应该还在 shared 数组的第 64 项”。

稍微展开写地址的计算，就能看见原因。把原始寄存器名换成有含义的变量，首对元素的 shared-memory 字节偏移等价于：

```python
write_offset = ((tid << 2) & 504) ^ (-(tid & 1) & 1088)
```

这里的 `^` 是按位异或。tid 32 是偶数，后一项为零，前一项为 128，于是得到刚才的地址。

但是，同样要交给 tid 16 的 `x[66]`，来自 tid 33。代入之后，写偏移变成：

```text
((33 << 2) & 504) XOR 1088
= 128 XOR 1088
= 1216 bytes
```

也就是说，最终由同一线程持有的 `x[64]` 和 `x[66]`，在这块临时空间里隔得很远。

读的一端同样配套安排好了。tid 16 读第一对值时，偏移为 `tid << 3 = 128`；读第二对时，偏移再异或 1088，得到 1216：

| Shared Memory 偏移 | tid 16 读回的值 |
| ---: | --- |
| 128 bytes | `x[64], x[65]` |
| 1216 bytes | `x[66], x[67]` |

![四个 warp 的布局转换中，tid32 的 x64、x65 经过 shared 偏移128字节，tid33 的 x66、x67 经过偏移1216字节；同步后都交给 tid16，写回连续的输出位置。](/images/triton/layout-shared-memory-trace.svg "把源线程、shared 地址、目标线程分开追。中间地址不连续，写回的逻辑下标仍然连续。")

让我觉得有意思的是这层区别：逻辑上始终是下标 64 的值，global memory 里也确实从下标 64 的位置读、写。但中间怎样经过 shared memory，完全可以是另一套排列。看布局转换时，不能把这几层地址混在一起。

## 会撞 bank 吗

之前写 CUDA 的 shared-memory kernel 时，我问过：“什么是 bank conflict，为什么会有 bank conflict？”现在再看到这些异或，问题就更具体了：既然数据都放进 shared memory，为什么不顺着排？

先把 bank 编号算清楚。本文这张 NVIDIA GPU 的 shared memory 有 32 个 bank，连续的 32-bit word 轮流落在不同 bank。用字节地址 `b` 来算，是：

```text
bank = (b // 4) % 32
```

如果从 bank 0 对齐的起点看一个 `float` 数组，下标 `i` 才可以直接对应 `i % 32`。前面算出的 `128` 和 `1216` 是字节偏移，不能直接拿来 `% 32`，少掉一次除以 4，算的就不是同一件事了。

先只看同一个 warp 的一条普通 32-bit 读取。lane 0 读第 0 个 word，lane 1 读第 1 个，依次下去，32 个 lane 的请求就分散到了 32 个 bank。换成每个 lane 都去读第 `32*lane` 个 word，请求会落在同一个 bank 的不同位置，需要分开服务，这才是这里说的 bank conflict。

但如果大家都读同一个 word，情况又不同：硬件可以把它广播给这些 lane，不能因为“都落在 bank 0”，就把它叫作 32 路冲突。[CUDA 文档的 shared-memory 访问示例](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns)把这几种情况画在了一起。

![同一 warp 的标量读取示意：两个 lane 读取同一个32位字可广播，读取偏移0和128字节的不同字则落在同一bank，需要分开服务。](/images/triton/bank-conflict-basics.svg "同一个 bank，还要看是不是同一个 word。这张图只画普通 32-bit 读取，不代替本次向量访问的分析。")

回到这次转换，假设 shared 起点落在 bank 0，那么偏移 `128 B` 对应 word 32、bank 0；`1216 B` 对应 word 304、bank 16。

这能帮我们检查地址，却还不能证明整个转换没有冲突。尤其在读的一端，这两个地址是同一个 tid 16 通过两条不同的 `ld.shared` 读取的，光比较它们落在哪个 bank，本来就不足以判断同一轮访问有没有冲突。

要继续判断，需要看同一条机器指令中，各个 lane 的完整访问。向量读写还可能需要分几次服务，不能把本来就需要的拆分都算成冲突；[Nsight Compute 的说明](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-bank-conflicts-are-significant)也区分了理想情况下的分次访问和额外的冲突开销。

所以，这里先把三件事分开：值归哪个线程，值暂存在 shared 的哪个地址，那次访问又落在哪些 bank。前两件已经沿着 PTX 对上了；至于这套临时排列到底省下多少访存开销，还得接着看最终机器指令和性能计数。

## 补一个对照

最开始的两组实验，一组一个 warp、128 个元素，另一组四个 warp、512 个元素。运行和编译结果如下：

| 配置 | 输出与输入一致 | Shared Memory | `shfl.sync` 条数 |
| --- | --- | ---: | ---: |
| 1 warp，N=128 | 是 | 0 B | 4 |
| 4 warps，N=512 | 是 | 2048 B | 0 |

这两组同时改了线程数和数据量。为了排除“只是因为 N 变大才换了通信路径”这个解释，又固定 `N=512`，补跑了一个 warp 的版本：

| 配置 | 输出与输入一致 | Shared Memory | `shfl.sync` 条数 |
| --- | --- | ---: | ---: |
| 1 warp，N=512 | 是 | 0 B | 16 |
| 4 warps，N=512 | 是 | 2048 B | 0 |

一个 warp 的版本，每线程现在持有 16 个值，交换的工作更多，但仍然没有跨出这个 warp，所以这次编译依旧走 Shuffle。对应的 [TTGIR](../../downloads/triton-layout-conversion/convert_w1_n512.ttgir) 和 [PTX](../../downloads/triton-layout-conversion/convert_w1_n512.ptx) 也一并保留。

这里核对的是通信路径，没有计时。静态指令条数、shared memory 用量，以及一个 warp 或四个 warp 的配置，不能直接代替性能测试。至少从这些输出里，还得不出“Shuffle 比 Shared Memory 快多少”的结论。

同样，看到 `convert_layout`，也不能立即认定一定发生了跨线程搬运。如果转换前后，值仍由同一个线程持有，就可能只需要线程内重排，甚至无需生成实际的搬运指令。[官方教程对 trivial conversion 的说明](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)也专门提到了这种情况。

## 为何要转换

回头看这个小 kernel，它做的事情确实有点多余。

输入和输出都是连续的一维数组，完全可以用同一套布局完成拷贝。中间放一次转换，只是为了让平时藏在编译结果里的搬运过程单独露出来。

真实 kernel 中，不同阶段却未必能一直沿用同一套分工。读取时，希望各线程合起来访问连续地址；进入计算时，指令可能要求特定的 lane/value 分布；写回时，又要照顾输出的地址排列。上一篇 CuTe 里看到的 MMA fragment，就是这种具体约定。

一套布局适合前半段，另一套适合后半段，中间就有了需要协调的地方。转换会带来成本，但让某一段长期使用不合适的分工，也有它的成本。[官方教程的二维拷贝例子](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)里，输入和输出的连续方向不同，便为 load 和 store 分别选择布局，在中间做转换。

至于这笔交换是否划算，要把它放回整个算子里测。这个只复制一个 tile 的实验还回答不了。

不过，再看到 TTGIR 里的 `ttg.convert_layout`，至少可以先去找一个具体的元素：它之前归谁，之后归谁，两个线程之间用什么办法递过去。如果需要 shared memory，再沿着写地址和读地址把两端接起来。

这次接起来的是 `x[64]`。从 warp 1 的 lane 0，到 warp 0 的 lane 16，中间停在 shared memory 基址后偏移 128 字节的位置。

## 源码与复现

完整实验见 [triton_layout_conversion.py](../../downloads/triton_layout_conversion.py)。学习环境为 WSL Ubuntu 26.04、Triton 3.8.0、RTX 4070 Laptop，编译目标为 `sm_89`。[metadata.json](../../downloads/triton-layout-conversion/metadata.json) 记录了实际运行环境、检查结果和各组配置。

准备好兼容的 PyTorch、Triton 和 NVIDIA GPU 环境后，进入脚本所在目录运行；`python` 应指向这个环境中的解释器。

```bash
python triton_layout_conversion.py --dump-dir ./layout-conversion-ir
```

脚本运行上面的三组配置，逐元素比较输出与输入，并保存 TTGIR、PTX。随文产物已移除本地源码路径和调试元数据，布局、执行指令及地址计算保持不变。结果可以直接下载：

| 配置 | TTGIR | PTX |
| --- | --- | --- |
| 1 warp，N=128 | [convert_w1_n128.ttgir](../../downloads/triton-layout-conversion/convert_w1_n128.ttgir) | [convert_w1_n128.ptx](../../downloads/triton-layout-conversion/convert_w1_n128.ptx) |
| 4 warps，N=512 | [convert_w4_n512.ttgir](../../downloads/triton-layout-conversion/convert_w4_n512.ttgir) | [convert_w4_n512.ptx](../../downloads/triton-layout-conversion/convert_w4_n512.ptx) |
| 1 warp，N=512 | [convert_w1_n512.ttgir](../../downloads/triton-layout-conversion/convert_w1_n512.ttgir) | [convert_w1_n512.ptx](../../downloads/triton-layout-conversion/convert_w1_n512.ptx) |

这些指令和地址是这次环境中的编译结果。换编译器版本、目标 GPU 或布局，具体生成方式也可能变化；复现时可以先核对元素归属，再查看本机生成的指令。
