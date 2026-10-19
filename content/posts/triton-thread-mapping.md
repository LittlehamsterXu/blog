---
title: "看懂 Triton：当线程映射交给编译器"
date: 2026-10-18
draft: false
tags: [CUDA, GPU, Triton, GPU Compiler, AI Infra]
categories: [AI Infra]
description: "从向量加法到 128 个数的求和，改一改 num_warps，再沿着 TTGIR 和 PTX 看看，那些没写出来的线程究竟去了哪里。"
summary: "CuTe 里，一个线程拿哪些值，需要把映射写出来。到了 Triton，threadIdx 不见了，分工却还在。这次只求 128 个数的和，看看同一段代码交给一个 warp 和四个 warp，会有什么不同。"
ShowToc: true
TocOpen: true
---

上一篇 [看 CuTe](../cutlass-cute-mapping/)，花了不少篇幅去弄清楚：一个 thread，到底拿着哪几个值。

先构造 Layout，再把数据 partition 给线程，最后打印出坐标。一张表看下来，才知道那些模板参数安排了什么工作。

到了 Triton，代码里却很少再出现 thread。写一个 `tl.arange(0, 128)`，就能对着 128 个位置做加载、计算和求和。

那这 128 个数，是交给了 128 个线程，还是让更少的线程各自多拿几个？

先不急着写 GEMM。这次只求 128 个数的和，再改一改 `num_warps`，看看编译结果里发生了什么。

## threadIdx 不见了

不过，在求和之前，还是先看一小段向量加法。它能把 Triton 里几个容易混在一起的东西分开。

手写 CUDA 时，很熟悉这样的下标：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = x[i] + y[i];
}
```

这里一个线程处理一个元素，代码写的是当前线程的工作。Triton 的写法换了一个范围：

```python
@triton.jit
def add_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = tl.load(Y + offsets, mask=mask, other=0.0)
    tl.store(Out + offsets, x + y, mask=mask)
```

这里的 `offsets` 有 `BLOCK_SIZE` 个逻辑位置。`X + offsets` 是一组指针，`tl.load` 返回的 `x` 也有相同的逻辑形状。接下来写 `x + y`，表达的是这整块数据上的逐元素运算。

这就是 Tile-Based 编程在这个例子里的样子：先写一块数据的工作，再让编译器把它分给线程。Tile 可以是一维的，也可以是二维、多维的，并不一定是一块矩阵。

但一个 program 应该取哪一块，仍然由代码决定。比如总长度是 `1003`，每块处理 `128` 个位置：

```python
block_size = 128
grid = (triton.cdiv(n, block_size),)

kernel = add_kernel[grid](
    x, y, out, n,
    BLOCK_SIZE=block_size,
    num_warps=4,
)
```

`grid` 是 `(8,)`，也就是启动八个 program。`tl.program_id(0)` 取到当前 program 在这一维的编号，`pid * 128` 决定它的起点：

```text
program 0    offsets =    0 ... 127
program 1    offsets =  128 ... 255
...
program 7    offsets =  896 ... 1023
```

最后一块只有下标 `896` 到 `1002` 有效，后面的访问由 mask 挡住。这里加载时显式写了 `other=0.0`，让屏蔽位置返回零；如果省略 `other`，这些位置的值是未定义的，并不会自动补零。这个区别在归约时尤其要留意，可以对照 [tl.load 的参数说明](https://triton-lang.org/main/python-api/generated/triton.language.load.html)。

在本文这种 NVIDIA、单 CTA 的普通 kernel 中，可以把一个 program 对应到一个 CUDA block。`program_id` 接近这里的 `blockIdx`，却不能顺手把 `tl.arange` 当成 `threadIdx`。

`tl.arange(0, 128)` 说的是这块数据里有哪 128 个位置。至于它们交给多少线程，每个线程拿哪些位置，这一行没有给出答案。

## 线程块和数据块

这个区别，之前写 CUDA 时其实已经遇到过。

那时有一个 `16×16` 的线程块，一个 warp 的 32 个线程会跨过线程块的两行。把 block 改成 `32×8`，一个 warp 又能刚好落在一行里。看起来像是换了个矩阵形状，但这里画出来的“矩阵”，首先是线程的坐标。

CUDA 按 x 方向优先递增来给二维线程块编号：

```cpp
int tid = threadIdx.y * blockDim.x + threadIdx.x;
```

因此，`16×16` 的 warp 0 包含 `(x=0..15, y=0)` 和 `(x=0..15, y=1)`；`32×8` 的 warp 0 则是 `(x=0..31, y=0)`。这一步只确定了哪些线程在同一个 warp，还没有说明它们各自访问矩阵的哪里。

后来，我把另一个实验改成了“`32×8` 个线程共同覆盖 `32×32` 的 tile”。256 个线程，要算 1024 个输出，每个线程便负责四个结果。当时为了把这件事看清楚，还把累加数组展开成了 `sum0`、`sum1`、`sum2`、`sum3`。

那份 [Register 1×4 kernel](../cuda-gemm/) 中，决定输出位置的是这几行：

```cpp
// blockDim = (32, 8)，这里只摘出输出坐标的分配。
int tid = threadIdx.y * blockDim.x + threadIdx.x;
int row_in_tile = tid / 8;
int col_in_tile = (tid % 8) * 4;

// 当前线程的四份累加结果，对应同一行的连续四列。
float sum0 = 0.0f;
float sum1 = 0.0f;
float sum2 = 0.0f;
float sum3 = 0.0f;
```

每八个线程负责输出的一行，每个线程再负责四列。例如 tid 9，也就是线程块坐标 `(x=9, y=0)`，对应的输出位置却是行下标 1、列下标 4 到 7。整个 warp 0 的线程都在 block 的第一行，算出来的结果覆盖了输出 tile 的前四行。

![线程块的二维坐标与输出 tile 的坐标：16×16 的 warp 0 跨两行，32×8 的 warp 0 在线程块第一行，但按每线程四列的映射覆盖输出前四行。](/images/triton/cuda-tile-vs-threads.svg "上面画线程，下面画输出。两张图的行列不能直接当成一回事。")

所以，“一个 warp 跨了两行”还需要接着问：是哪张坐标表里的两行？实际地址是否连续，要把线程编号代入访问下标之后再看，不能只凭 block 的外形判断。

回到 Triton，也是同样的区分。`tl.arange` 描述逻辑位置，`num_warps` 约束参与的线程数。以前在 CUDA 里手写的那几行分配，现在需要到编译结果中去找。

## 只改一个 num_warps

向量加法里还有 grid、边界和两份输入，先把它们都拿掉。剩下的 kernel 很短：

```python
@triton.jit
def reduce128(X, Y):
    offsets = tl.arange(0, 128)
    x = tl.load(X + offsets)
    s = tl.sum(x, axis=0)
    tl.store(Y, s)
```

输入正好是 128 个 FP32 数，所以这里不需要 mask。只启动一个 program，也就暂时用不到 `program_id`。

```python
x = torch.arange(128, device="cuda", dtype=torch.float32)

for nw in (1, 4):
    y = torch.empty(1, device="cuda", dtype=torch.float32)
    kernel = reduce128[(1,)](x, y, num_warps=nw)
    print(nw, y.item(), kernel.metadata.shared)
```

在这张 NVIDIA GPU 上，一个 warp 有 32 个线程。`num_warps=1` 时，这个 program 用 32 个线程协作；改成 `4`，就用了 128 个线程。这是启动配置给编译器的约束，[num_warps 的文档](https://triton-lang.org/main/python-api/generated/triton.Config.html)也用这个方式解释它。

两个版本都得到了 `8128.0`，也就是 `0 + 1 + ... + 127`。但后面那个数不一样：

```text
num_warps    result    shared bytes
1            8128.0   0
4            8128.0   16
```

源码里没有 shared memory，也没有同步。只是把 warp 数改了一下，它们就出现了。

先记着这个变化，往下看这 128 个元素分别落到了哪里。

## 去编译结果里找线程

启动时返回的 `kernel` 可以用来查看编译产物：

```python
print(kernel.asm["ttir"])
print(kernel.asm["ttgir"])
print(kernel.asm["ptx"])
```

TTIR 中还能看到 `tt.make_range`、`tt.load`、`tt.reduce` 这些操作。比如加载结果的类型是 `tensor<128xf32>`，保留了“128 个 FP32 值”的意思。

到了 TTGIR，类型上多了一项：`tensor<128xf32, #blocked>`。文件开头则定义了 `#blocked`：

```text
// num_warps = 1
#blocked = #ttg.blocked<{
  sizePerThread = [4],
  threadsPerWarp = [32],
  warpsPerCTA = [1],
  order = [0]
}>
```

这里终于能看见线程了。对这份一维、正好覆盖 128 个元素的布局，每个线程先拿连续四个值，32 个线程组成一个 warp，一个 CTA 只有这一个 warp。

于是，输入的分配是：

| Warp | Lane | 持有的输入元素 |
| ---: | ---: | --- |
| 0 | 0 | `x[0]`、`x[1]`、`x[2]`、`x[3]` |
| 0 | 1 | `x[4]`、`x[5]`、`x[6]`、`x[7]` |
| 0 | 2 | `x[8]`、`x[9]`、`x[10]`、`x[11]` |
| 0 | 31 | `x[124]`、`x[125]`、`x[126]`、`x[127]` |

用 lane 编号 `l` 和线程内的 value 编号 `v` 表示，就是 `i = 4*l + v`，其中 `v` 从 0 到 3。

这和上一篇打印 Thread–Value mapping 时做的事情很像。区别是，这次没有自己构造那张映射表，而是从编译结果里把它读出来。

再看四个 warp 的版本：

```text
// num_warps = 4
#blocked = #ttg.blocked<{
  sizePerThread = [1],
  threadsPerWarp = [32],
  warpsPerCTA = [4],
  order = [0]
}>
```

每个线程拿一个值，128 个线程刚好覆盖输入。设 warp 编号为 `w`，lane 编号为 `l`，这次是 `i = 32*w + l`：

| Warp | Lane | 持有的输入元素 |
| ---: | ---: | --- |
| 0 | 0 | `x[0]` |
| 0 | 31 | `x[31]` |
| 1 | 0 | `x[32]` |
| 2 | 0 | `x[64]` |
| 3 | 31 | `x[127]` |

![同一个 128 元素逻辑 tile，在一个 warp 下分成每 lane 四个连续值，在四个 warp 下分成每 lane 一个值。](/images/triton/reduce128-thread-ownership.svg "数据量一直是 128。变化的是参与的线程数，以及每个线程持有哪几个值。")

先算 `128 / 32 = 4`，确实可以知道一个 warp 的版本平均每线程拿四个值。但总量还不能告诉我们，它们是 `0、1、2、3`，还是 `0、32、64、96`。只有读到具体布局，才知道这次采用了前一种分工。

源码中的 `x` 也不表示“每个线程各有一个 128 元素数组”。它是 program 视角下的 tensor；落实到这份布局，每个线程只持有自己那四个输入。之前 CUDA 中显式展开的四份值，在这里仍然存在，只是没有逐个写在源码里。

`order=[0]` 在这个一维例子里没有多少选择余地。多维时，它才会涉及优先沿哪个维度分配。这里先不展开，够读懂这两张表就好。

还有一个范围需要记住：刚才“每线程四个值”的解释，依赖这次的 shape 和布局恰好匹配。以后看到 `sizePerThread=[4]`，不能脱离 tensor 的形状，就认定一个线程总共只有四个值。更大的 tile 还会重复使用这套分配规则。下一篇会碰到这样的例子。

这里的 Layout 描述的是值怎样分布在寄存器、lane 和 warp 之间。它和输入数组的地址布局要分开看：同一个 `x[64]`，内存位置没变，持有它的线程却可以变。

## 多出的 16 字节

求和需要把分散在各个线程里的数汇到一起。前面的两张分配表不同，后面的汇合方式也跟着变了。

一个 warp 的版本中，每个线程先把自己拿到的四个值做局部加法。这样，128 个输入先变成 32 份局部结果，再在 warp 内继续归约。

PTX 里能看到五条 `shfl.sync.bfly.b32`，配合加法把这些结果合起来。Shuffle 可以让同一 warp 内的 lane 交换寄存器中的值；它的语义可以对照 [PTX 的 shfl.sync 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。这个版本没有出现 shared memory 的读写，也没有 CTA barrier。

五条也不是一个碰巧数出来的数字。这里用的是蝶形交换，五轮的 lane 异或掩码依次为 `16、8、4、2、1`。第一轮，lane `l` 取 lane `l ^ 16` 的局部结果来相加；下一轮再取 `l ^ 8` 的结果。每轮合并的范围翻倍，五轮后就覆盖了 32 个 lane。

四个 warp 的版本，每个线程只有一个输入。先在各自的 warp 内求和，得到四份局部结果。问题是，普通的 warp shuffle 不能直接从另一个 warp 拿数据。

于是生成代码多了一段 shared-memory 路径：先把各 warp 的局部结果写到 shared memory 的四个位置；CTA 同步之后，各组 lane 读回这四份值，再做两轮 Shuffle。最后由 thread 0 把标量结果写到 `Y`。

四份 FP32 局部结果，每份四字节，正好对应前面看到的 `16 B`。

它的 PTX 中一共出现七条 Shuffle：前五条让四个 warp 各自求和，后两条再合并从 shared memory 读回的四份结果。四个 warp 执行的是同一组前五条指令，所以这里不会因为有四个 warp，就在文本里出现 `4×5` 条。

把相关指令在生成 PTX 中出现的次数列出来，会更直观：

| PTX 指令 | 1 个 warp | 4 个 warp |
| --- | ---: | ---: |
| `shfl.sync` | 5 | 7 |
| `st.shared` | 0 | 1 |
| `ld.shared` | 0 | 1 |
| `bar.sync` | 0 | 1 |

这里数的是 PTX 文本中的指令条目。有的指令会由很多线程执行，有的带有谓词，只让部分线程执行，不能把表里的数字直接当成运行时的执行次数。PTX 后面还要经过 `ptxas` 生成机器码，表里也没有包含耗时测量。

所以，这个实验还回答不了哪个版本更快。它已经能回答的是：为什么同样一行 `tl.sum`，在这两种分工下，需要的通信不一样。

也别把这段 shared memory 直接称作 Layout Conversion。当前 TTGIR 中主要是 `tt.reduce`，这里的通信是在完成归约。下一篇会专门构造一次显式的布局转换，观察值本身怎样换到别的线程手里。

## 谁决定分工

回头再看 `reduce128`，它仍然写得很具体：处理 `X` 开头的 128 个元素，沿第 0 维求和，把标量结果存到 `Y`。向量加法里的 grid、tile 大小、地址和边界处理，也都需要自己写。

交出去的是更细的分工：在给定的配置下，让哪些线程持有哪些值，怎样执行每线程的操作，以及归约需要怎样的线程间通信。

`num_warps` 又把两边连在了一起。程序员没有指定每个 lane 的坐标，却仍然能通过它改变参与的线程数。编译器接到不同的约束，给出的布局和通信代码也会不同。这里的“交给编译器”，并不意味着它会替我选择算法、tile 大小，或者自动找到最快的配置。

这次也没有继续追问：为什么编译器为一个 warp 选了连续四个值，而不是让 lane 0 拿 `0、32、64、96`？要回答这个问题，还得继续看布局分配、访存合并和后续优化的实现。仅凭这两份输出，能确定的是它这次选成了什么样，不能反推出一条适用于所有 kernel 的选择规则。

以前写 CUDA，分工直接留在下标和循环里。看 CuTe 时，可以把它构造成对象，再打印出来。到了 Triton，源码里少了这部分细节，想知道它们去了哪里，就需要再往编译结果里走一层。

至少这次再看到 `tl.arange(0, 128)`，不会急着给它配上 128 个线程了。

接下来还有一个很近的问题：如果同一批值前后需要采用不同的分布，编译器怎样把它们搬过去？[下一篇](../triton-layout-conversion/)就从这里接着看。

## 源码与复现

本文的实验从 `mini-tensor/triton_example` 中的向量加法和 `reduce128` 整理而来，写作时重新运行过。环境是 WSL Ubuntu、Triton 3.8.0 和 RTX 4070 Laptop，GPU 编译目标为 `sm_89`。

[下载完整实验脚本](../../downloads/triton_thread_mapping.py)。它会校验长度为 `1003` 的向量加法，运行两个求和版本，并输出布局、shared-memory 用量和相关 PTX 指令的静态计数。

在已安装兼容的 PyTorch 和 Triton 的 Python 环境中，进入脚本所在目录运行：

```bash
python triton_thread_mapping.py --dump-dir ./thread-mapping-ir
```

这里的 `python` 应指向准备好的环境，`--dump-dir` 用于保存编译产物。具体布局及指令可能随版本和目标架构变化。

随文产物已移除本地源码路径和调试元数据，布局、执行指令及地址计算保持不变。

这次核对过的文件也放在下面，正文中的分配和指令计数对应这些产物：

| 配置 | TTIR | TTGIR | PTX |
| --- | --- | --- | --- |
| 1 个 warp | [下载](../../downloads/triton-thread-mapping/reduce128_w1.ttir) | [下载](../../downloads/triton-thread-mapping/reduce128_w1.ttgir) | [下载](../../downloads/triton-thread-mapping/reduce128_w1.ptx) |
| 4 个 warp | [下载](../../downloads/triton-thread-mapping/reduce128_w4.ttir) | [下载](../../downloads/triton-thread-mapping/reduce128_w4.ttgir) | [下载](../../downloads/triton-thread-mapping/reduce128_w4.ptx) |
