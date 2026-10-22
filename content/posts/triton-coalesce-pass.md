---
title: "Triton Coalesce：布局是怎样选出来的"
date: 2026-10-19
draft: true
tags: [Triton, MLIR, GPU Compiler, CUDA, AI Infra]
categories: [AI Infra]
description: "用一个 32×32 的矩阵转置，沿着 AxisInfo、Coalesce 和 RemoveLayoutConversions，看看布局怎样被选出来，又怎样变成最后的 Shared Memory 交换。"
summary: "上一篇把两端的布局写死，再去找数据怎样换手。这次把布局参数拿掉，交给编译器。一个很短的转置 kernel，转换数量却先从六次涨到十次，最后又只剩一次。"
ShowToc: true
TocOpen: true
---

[上一篇](../triton-layout-conversion/)里，为了看清 Layout Conversion，我用 Gluon 把两端的布局固定下来，再沿着 PTX 追一个元素去了哪里。

这次想把那些手工指定的参数拿掉。

毕竟，平时写普通 Triton kernel，并不会先替每个中间结果选好布局。读取时该让哪些线程拿连续元素，写回时又该换成什么分工，这些事情还是要交给编译器。

于是写了一个矩阵转置。代码很短，编译出来的过程却比想象中热闹：`ttg.convert_layout` 先有六个，经过 Coalesce 变成十个，后面又只剩一个。

一个优化 Pass，怎么还把转换加多了？

这篇就从这里接着看。

## 两边都想连续

还是先把例子缩小。输入、输出都是连续的 `32×32` FP32 矩阵，只启动一个 program，使用四个 warp：

```python
@triton.jit
def transpose_kernel(X, Y, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)

    x_ptrs = X + rows[:, None] * BLOCK + cols[None, :]
    values = tl.load(x_ptrs)

    y_ptrs = Y + cols[None, :] * BLOCK + rows[:, None]
    tl.store(y_ptrs, values)
```

这里的 `BLOCK=32`，矩阵也恰好是 `32×32`，所以省掉了边界处理。它只演示一个 tile，没有处理任意大小矩阵的 grid 和 stride。

启动和检查也很直接：

```python
x = torch.arange(32 * 32, device="cuda", dtype=torch.float32).reshape(32, 32)
y = torch.empty_like(x)

kernel = transpose_kernel[(1,)](x, y, 32, num_warps=4)
assert torch.equal(y, x.T.contiguous())
```

注意 `values` 的逻辑坐标。`values[row, col]` 读的是 `X[row, col]`，写回时仍然使用这份值，但目的地变成了 `Y[col, row]`。

把两个元素偏移并排放着看：

```text
读取：row * 32 + col
写回：col * 32 + row
```

读取时，固定 row、增加 col，地址连续。写回时，固定 col、增加 row，地址才连续。比如 `values[1, 2]` 从输入的元素偏移 34 读入，再写到输出的元素偏移 65；这些偏移还要乘四，才是 FP32 的字节偏移。

![转置的输入和输出地址对照：两张网格都按 values 的 row、col 排列，读取沿 col 连续，写回沿 row 连续；values 的位置1、2从X偏移34写到Y偏移65。](/images/triton/coalesce-contiguous-directions.svg "同一份 values，读写两端的连续方向不同。图里的数字都是元素偏移，Y 的实际存储坐标是 col、row。")

如果一直沿用同一套线程分工，读写两端就未必都舒服。这个问题在上一个实验里，是由我们手工选两套布局来处理的。这次源码里没有布局参数，正好可以看看编译器会怎么做。

## 六次变成十次

前两篇主要看 `kernel.asm["ttgir"]` 和 `kernel.asm["ptx"]`。它们适合看一个编译阶段结束时留下了什么，却看不出中间某个 Pass 单独改了哪里。

这次打开每个 Pass 的 IR 输出：

```bash
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 \
python coalesce_lab.py > run.log 2>&1
```

`coalesce_lab.py` 就是上面的小实验，加上运行和保存编译结果的部分。`TRITON_ALWAYS_COMPILE` 用来强制重新编译，免得命中缓存后，想看的过程根本没有发生。文末另放了一份会自动收集快照的[完整脚本](../../downloads/triton_coalesce.py)。

日志很长，这里先只找相邻的两个标记：

```text
IR Dump Before TritonGPUCoalesce
...
IR Dump Before TritonGPUF32DotTC
```

这份日志写的是 **Before**。第一段是进入 Coalesce 之前的状态，第二段才是它执行完、下一个 Pass 还没开始时的状态。不能看到标题里有 Coalesce，就把下面的内容当成它的产出。

两段里数一数 `ttg.convert_layout`，分别是六个和十个。

再看新增的布局，读写方向就对上了。下面把原始别名换成 `#load`、`#store`，方便后面一直用这两个名字：

| 参数 | Load：`#load` | Store：`#store` |
| --- | --- | --- |
| `sizePerThread` | `[1, 4]` | `[4, 1]` |
| `threadsPerWarp` | `[4, 8]` | `[8, 4]` |
| `warpsPerCTA` | `[4, 1]` | `[1, 4]` |
| `order` | `[1, 0]` | `[0, 1]` |

Load 一端每个 lane 先沿 col 持有连续四个值；每行八个 lane，四行合成一个 warp。Store 则把 row、col 的角色换过来。各项乘起来都能覆盖 32 个 lane、四个 warp，正好还是启动时给定的线程数。

但 Coalesce 没有顺手把整段 IR 全部改成这两套分工。它先在内存操作周围接上转换。这次新增的四个是：

```text
Load 前：原布局的指针 → #load
Load 后：#load 的值   → 原布局

Store 前：原布局的指针 → #store
         原布局的值   → #store
```

指针 tensor 也有布局。哪条线程拿着哪个地址，同样需要和内存操作对得上。

这六个、十个里面，既有指针和索引的转换，也有 FP32 值的转换。不能把它们全算成“输入数据已经跨线程搬了十遍”；转换是否需要实际交换，还要接着看两端布局和后面的 Lowering。

读到这里，MLIR 里的 Operation、SSA Value 和类型终于有了具体的落点：`tt.load` 产生一份结果，后面的操作使用它；结果类型里还带着布局。改写 Load 的分工以后，后面的使用者如果仍然要求旧布局，就得先把接口接起来。

所以这四个转换有它们各自的位置。Coalesce 先让 Load、Store 用上选好的布局，至于中间能不能少绕几步，还要接着处理。[v3.8.0 的 Coalesce 源码](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)里，可以对照着找布局选择和这部分改写。

不过，读写方向相反是我们看地址公式得出的判断。编译器又是凭什么知道的？

## 连续性要有依据

沿着 Coalesce 往前找，会碰到 `AxisInfoAnalysis`。

它让我想起以前在 Tai-e 里做的常量传播：分析器给程序里的值记一些信息，再沿着操作把这些信息传下去。这里记的东西换了，关注的是 tensor 各个维度上的连续、整除和重复性质。

先只看一个一维整数例子：

```python
r = tl.arange(0, 8)
a = r + 8                       # 8, 9, 10, 11, 12, 13, 14, 15
b = 2 * a                       # 16, 18, 20, 22, 24, 26, 28, 30
c = tl.full((8,), 16, tl.int32)  # 16, 16, 16, 16, 16, 16, 16, 16
```

三个属性分别在问：

| 属性 | 在这个例子里怎样理解 |
| --- | --- |
| Contiguity | 按多长的一组切分，每组里能保证相邻值加一 |
| Divisibility | 这些连续组的首元素，共同保证能被多大的二次幂整除 |
| Constancy | 按多长的一组切分，每组里能保证值相同 |

分析保留的是能证明的保证，复杂表达式得出的信息可能比实际数据的性质更弱。按这次版本的传播规则，上面三个值的一维信息是：

| 值 | Contiguity | Divisibility | Constancy |
| --- | ---: | ---: | ---: |
| `a` | 8 | 8 | 1 |
| `b` | 1 | 2 | 1 |
| `c` | 1 | 16 | 8 |

`a` 的八个数连续加一，可以作为一组。这组从 8 开始，所以 Divisibility 是 8。它没有说组里的 9、10、11 也都能被 8 整除。

`b` 仍然很有规律，但步长变成了 2，已经不满足这里“相邻加一”的连续性。组长降到 1，每个元素都成了组首，能共同保证的整除因子便是 2。不能只盯着第一个数 16，把 Divisibility 继续记成 16。

`c` 则是一整组相同的值。这里 Constancy 描述的是重复，不等于“编译器知道这个值是多少”；即使同一组重复的是运行时才知道的数，也可能具有 Constancy 信息。[AxisInfo 的定义](https://github.com/triton-lang/triton/blob/v3.8.0/include/triton/Analysis/AxisInfo.h)和[各操作的传播规则](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Analysis/AxisInfo.cpp)把这些区别写得很具体。

这张表里的数值是用来解释整数序列的。到了指针，要留意单位：连续性按相邻的目标元素理解，FP32 指针相差四字节也可以是连续的；Divisibility 对应的则是字节对齐。

因此，“地址连续”和“适合一次读多少个元素”也还差一步。选一组四个 FP32 值，不光要能连着读，还要考虑起点的对齐、单个元素的位宽，以及线程和 tile 的分配。这次的 `sizePerThread=[1, 4]`，不能只从 `tl.arange` 的长度里直接抄出来。

回到转置，Coalesce 查询的是 `x_ptrs`、`y_ptrs` 的信息。它关心哪些地址连续，不关心从这些地址里读出的浮点数是不是恰好递增。

`rows[:, None] * 32 + cols[None, :]` 中，乘法、扩维、广播和加法会一步步传播分析结果。Load 的指针沿 col 连续，Store 的指针沿 row 连续；Coalesce 据此确定优先方向，再结合对齐、形状和线程配置选择布局。这部分可以对照 [CoalesceUtils.cpp](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp) 看。Constancy 也参与上游分析的传播，但不能把三个属性都说成 Coalesce 直接拿来打分的三个指标。

## 组变短以后

Divisibility 和连续组的关系，在看 Join 时尤其容易漏掉。这个转置本身没有分支，下面单独拿两条路径举个例子。

假设一份长度为八的值，从两条路径过来。分析得到的保证分别是：

```text
路径 A：[8, 9, 10, 11, 12, 13, 14, 15]   C=8，D=8
路径 B：[8, 9, 10, 11, 16, 17, 18, 19]   C=4，D=8
```

A 的八个数是一整组。B 每四个数一组，两组起点分别是 8、16，也都能被 8 整除。

汇合以后，连续组最多只能共同保证四个。假如 Divisibility 仍然保留为 8，问题就来了：A 现在也得按四个数切开，第二组从 12 开始，而 12 不能被 8 整除。

所以这里合并出的保证应该是 `C=4，D=4`。原来两边都写着 `D=8`，并不足以让结果继续保持 8，因为它依赖的分组已经变了。[`AxisInfo::join`](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Analysis/AxisInfo.cpp)里专门处理了连续性变化时的这个约束。

之前补的数据流分析概念，到这里才和具体问题连上。操作怎样把输入信息变成输出信息，是 Transfer Function；不同路径的保证怎样保守合并，是 Join。分析会继续传播，直到这些信息不再变化，也就是到达 Fixed Point。

即使没有枚举运行时 tensor 的每个值，也能依照这些规则保留一部分可靠的保证。后面的优化能用多少，就取决于前面能证明多少。

这段规则值得单独看一下：布局选择依赖的是有适用条件的分析结果，不能把“连续”“对齐”当成两个互不相关的标签。

## 剩下的一次

知道 Coalesce 为什么增加转换以后，再看后面的 `RemoveLayoutConversions` 就顺了。

这次只截取**第一次**执行它前后的状态。前面仍然有十个 `ttg.convert_layout`，执行以后只剩一个：

| IR 状态 | 静态 `convert_layout` 数量 | 快照 |
| --- | ---: | --- |
| Coalesce 前 | 6 | [before-coalesce.ttgir](../../downloads/triton-coalesce/before-coalesce.ttgir) |
| Coalesce 后 | 10 | [after-coalesce.ttgir](../../downloads/triton-coalesce/after-coalesce.ttgir) |
| 第一次 RemoveLayoutConversions 后 | 1 | [after-remove-layout-conversions.ttgir](../../downloads/triton-coalesce/after-remove-layout-conversions.ttgir) |

后两行的“后”，仍然来自下一份 Before 快照。具体标记分别是 `TritonGPUF32DotTC`、`TritonGPUOptimizeThreadLocality`；后面如果再次出现同名清理 Pass，不能混进这一次的统计。

![布局转换数量随编译阶段变化：Coalesce前6个，后10个，第一次RemoveLayoutConversions后1个；最后的一次values布局转换通过Shared Memory写入、同步和读取完成。](/images/triton/coalesce-conversion-counts.svg "先让读写用上各自的布局，再沿数据依赖调整中间操作，最后留下连接两端的值转换。")

从十个变成一个，光看数字还不够。

翻一下清理后的 IR，会看到有些索引被直接放到需要的布局里重新生成。同样是 `0..31`，可以在 Load 的分工下生成一份，也可以在 Store 的分工下生成另一份。`tt.make_range`、扩维、广播和地址计算接着沿各自的布局往下走，就不必总是先算出一种分布，再把结果转过去。

在这份快照里，连行、列索引都能找到按两套布局生成的版本。代码不一定因此少几行，但需要这些索引的线程可以直接生成自己的那一份。[RemoveLayoutConversions 的实现](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp)会传播布局，并尝试重写、重建相应的操作；它并不是看见转换就直接删除。

最后留下的是读入的 FP32 值：

```mlir
%converted = ttg.convert_layout %values
    : tensor<32x32xf32, #load>
      -> tensor<32x32xf32, #store>
```

这里省掉了无关上下文，`#load`、`#store` 仍然对应前面那张表。完整的[最终 TTGIR](../../downloads/triton-coalesce/transpose.ttgir)里，也只剩下这一次转换。

这批值已经从输入矩阵读进来了，两端又采用不同的持有方式。按布局算一个具体位置 `values[4, 0]`：Load 时在 warp 1、lane 0，Store 时要交给 warp 0、lane 1。

这就和上一篇接上了。索引可以在当地重新算出来，输入里的那份数据却仍要交到需要它的线程手上。这次结果选择保留两端的布局，并在中间完成交换。

至少在这个例子里，Coalesce 和 RemoveLayoutConversions 做的事情并不矛盾。前者为内存操作选分工，后者沿着依赖关系消化由此增加的转换；仍然需要协调的数据，就继续留给后面的 Lowering。

## 再看 PTX

走到这里，最后一个转换已经有了来处。再顺着 LLVM / NVVM Lowering 往下看，便能在 [PTX](../../downloads/triton-coalesce/transpose.ptx) 中找到它的实现。

这次生成的相关指令如下，统计的是汇编文本里的静态条数：

| 指令 | 条数 |
| --- | ---: |
| `ld.global.v4.b32` | 2 |
| `st.shared::cta.b32` | 8 |
| `bar.sync` | 1 |
| `ld.shared.b32` | 8 |
| `st.global.v4.b32` | 2 |
| `shfl.sync` | 0 |

global-memory 两端都是 `v4` 形式，每条指令处理四个 32-bit 值。shared-memory 中间则是标量读写，不能因为两端都有 `v4`，就把中间也想成向量化的搬运。

metadata 记录的 shared memory 是 `4096 B`，正好等于 `32 × 32 × 4`。Load 按第一套布局读到寄存器里，写入 shared memory，同步，再按第二套布局读回寄存器，最后写出。

这里没有 Shuffle。前面挑出的 `values[4, 0]` 已经需要跨 warp 交接，仅靠同一 warp 内的寄存器交换，接不上两端。

IR 里的一次 `convert_layout`，到了 PTX 就展开成多条指令。所以六次、十次、一次也好，八条 shared 写、八条 shared 读也好，都要先看清楚是在数哪一层的东西。这些静态数量还不能直接回答运行时要花多久。

这次没有计时，也没有做 bank-conflict 测量。能确认的是，两端生成了向量化的 global-memory 访问，中间通过 shared memory 换了分工。要比较它和其他实现的性能，还得另外做实验。

## 暂时先到这里

从 CUDA 到 CuTe，再到 Triton，来来回回看的，好像一直是那块数据该交给谁。

最开始，它写在 `threadIdx`、下标和循环里。后来把映射拿出来，变成可以构造和打印的 Layout。到了 Triton，程序里先写一块 tensor 的工作，再去编译结果里找那些没有直接写出来的线程。

这次又往前追了一点，开始看分析结果怎么影响布局，某个 Pass 怎么改掉一段 IR。

这当然还算不上弄懂了 Triton 编译器。现在让我拿一份复杂 kernel，解释每个 Pass 为什么放在那里，多半还是会卡住。有些源码能跟着这个小例子读过去，换个场景，大概又要重新翻。

不过，现在再看到一个 `ttg.convert_layout`，至少可以先找它两边的布局，再去找是哪一步把它放进来的。那条从程序到 PTX 的路，终于不只剩下几个阶段的名字了。

继续往下，还有 Tensor Core、异步拷贝、流水线，也有更多没看明白的布局和 Pass。每点开一个，似乎都能再花上不少时间。

只是最开始想学的，也不只有 Triton。

接下来想先往上看看：一个 PyTorch 模型交给 `torch.compile` 之后，计算图怎样被捕获、拆开，Inductor 又怎样生成这些 kernel。GPU 上的一小块数据已经追过几次了，模型走到这里之前的那段过程，还很模糊。

所以 Triton 暂时先学到这里。以后真碰到绕不过去的问题，再回来翻源码、补实验。

能把这个转置从头接到尾，这一段时间就已经没有白花。

## 源码与复现

完整脚本是 [triton_coalesce.py](../../downloads/triton_coalesce.py)，从这次的转置实验整理而来。环境为 WSL Ubuntu 26.04、Triton 3.8.0、RTX 4070 Laptop，目标 `sm_89`；[metadata.json](../../downloads/triton-coalesce/metadata.json)保留了运行检查、编译配置、快照来源和指令统计。

准备好兼容的 PyTorch、Triton 和 NVIDIA GPU 环境，进入脚本所在目录运行：

```bash
python triton_coalesce.py --dump-dir ./coalesce-ir
```

脚本会强制重新编译，自动开启 Pass Dump，在内存中截取 Coalesce 和第一次 RemoveLayoutConversions 前后的快照，同时检查输出与 PyTorch 转置结果一致。[实验说明和统计](../../downloads/triton-coalesce/README.md)、[TTIR](../../downloads/triton-coalesce/transpose.ttir)、[最终 TTGIR](../../downloads/triton-coalesce/transpose.ttgir) 和 [PTX](../../downloads/triton-coalesce/transpose.ptx) 都可以直接下载。

随文文件只保留源码文件名，已移除本地目录和 PTX 调试元数据。布局和执行指令保持不变。换版本、GPU 或配置后，布局、Pass 顺序和指令条数都可能变化，可以先核对各个 Pass 实际输出的标记，再做对比。

文中的源码链接固定到 Triton `v3.8.0`。想接着看，可以从 [AxisInfo.h](https://github.com/triton-lang/triton/blob/v3.8.0/include/triton/Analysis/AxisInfo.h)、[AxisInfo.cpp](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Analysis/AxisInfo.cpp)、[Coalesce.cpp](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp) 和 [RemoveLayoutConversions.cpp](https://github.com/triton-lang/triton/blob/v3.8.0/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp) 这几处慢慢找。
