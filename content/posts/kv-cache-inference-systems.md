---
title: "KV Cache：从显存管理到高效推理系统"
date: 2026-10-26T14:31:00+08:00
draft: false
tags: [KV Cache, PagedAttention, vLLM, PyTorch, LLM 推理]
categories: [AI Infra]
description: "用五个小实验理解 PagedAttention、Prefix Caching、缓存量化、Offloading 和 Continuous Batching，看看推理系统怎样管理已经产生的 KV Cache。"
summary: "缓存按块分配，相同前缀共享，数值换成更小的表示，暂时用不到的数据搬到 CPU，请求在每轮结束后重新组批。从 Python 模拟器到真实 CUDA Tensor，把这五件事分别拆开看。"
math: true
ShowToc: true
TocOpen: true
---

{{< study-summary id="知识点速记" >}}

**这篇看缓存怎么管：分块、复用、量化、搬运、调度。**

| 方法 | 记住这个动作 |
| --- | --- |
| 分页 | KV 按固定大小分块。逻辑位置查 Block Table，找到物理块。 |
| 前缀复用 | 相同的完整前缀块直接复用；哈希包含前面的上下文。 |
| KV 量化 | 用整数和 scale 保存近似值，读取时还原，减少字节数。 |
| 卸载 | GPU 缓存复制到 CPU 后释放 GPU 引用；使用前再搬回来。 |
| 连续组批 | 每轮结束移除完成请求，下一轮把等待请求补进来。 |

- **分页寻址**：位置 `t` 对应逻辑块 `t // P`、块内偏移 `t % P`。物理块编号从 `block_table` 读取。
- **前缀复用**：块内 Token 相同，前面的上下文也要相同。当前模拟器只登记完整块。
- **INT8**：每个向量取 `scale = max_abs / 127`，`q = round(x / scale)`，还原值约为 `q * scale`。存储时还要加上 scale。
- **卸载**：显存占用下降，CPU 内存和传输成本增加。复制结束后释放旧 GPU 引用，才能让对应 Tensor 的显存腾出来。
- **连续组批**：谁先完成谁先退出，空位下一轮就能用。缓存管理器随之领块、保留共享块或回收块。

{{< /study-summary >}}

[上一篇](../kv-cache-attention-optimization/)主要看 Attention 里面的缓存：每个位置保存多少组 K/V，表示有多宽，历史能压成多少条，这一轮又会读哪些。

模型把这些数据算出来以后，还有一串很实际的问题。

一个请求刚开始只有几百个 Token，后面会生成多长，暂时还不知道。要先替它留一整段显存吗？两个请求开头完全一样，能不能用同一份缓存？显存快满了，有些数据能不能换个格式，或者先搬出去？

还有，请求有长有短。短的已经结束了，长的还在生成，新来的请求要等到什么时候？

这次把这些问题拆成了五个实验：三个 Python 模拟器观察块和请求的状态，两个 CUDA Tensor 实验观察字节数、误差、传输和显存。先把每一件事单独看清楚，再想它们怎样接进同一个推理系统。

![GPU的KV物理块池位于中心，分页通过块表定位、前缀通过相同块复用、量化改变表示字节、卸载连接CPU内存、连续组批决定本轮使用哪些缓存。](/images/kv-cache-systems/cache-system-overview.svg "五项机制围着同一份 KV Cache 工作，下面分别拆开看。")

## 分页分配

### 从一整段缓存到 Block

先看最容易实现的分配方式：每个请求都按最大长度留一段连续空间。

假设最多支持 32 个 Token。请求 A 现在只有 8 个 Token，也先占住 32 个槽位；另一个请求能否使用剩下的空间，还得看管理方式。请求越来越多，预留给未来长度的那些槽位，就会挤占同时处理其他请求的空间。

PagedAttention 把缓存拆成固定大小的 **Block**。一个请求需要多少空间，就从池里领多少块；长到下一块时再领新的。逻辑上连续的历史，可以落在物理上不连续的块里。[PagedAttention 论文](https://arxiv.org/html/2309.06180)把这套设计与操作系统的分页联系起来。

在这个实验里，每个 Block 放 4 个 Token 的缓存位置，池里一共有 32 块。图里一格表示一个 Token 的 KV 槽位，实际数据还带着层、Head 和特征维度。

比如 A 先有 8 个 Token，领到物理块 `0、1`。其他请求陆续领走一批块以后，A 又增加 3 个 Token，新领到的块是 `17`：

```text
A 的长度：     8 → 11
A 的 Block Table：[0, 1] → [0, 1, 17]
```

A 的历史顺序依然是 `0..10`，Block Table 负责告诉读取方这些位置分别放在哪里。

![请求A的逻辑缓存从8个Token增长到11个Token，块表从物理块0、1扩展到0、1、17；下方展开32个物理块，显示已分配与空闲状态。](/images/kv-cache-systems/paged-blocks.svg "顺序属于逻辑位置；物理块可以从池中的空闲位置分配。")

### 查表怎样找到位置

块大小记为 `P`，Token 位置记为 `t`，从零开始编号：

```python
logical_block = t // P
offset = t % P
physical_block = block_table[logical_block]
```

现在读取 A 的位置 9：

```text
P = 4，t = 9
逻辑块 = 2
块内偏移 = 1
block_table[2] = 17
最终读取：物理块 17 的偏移 1
```

模拟器维护块编号、长度和空闲列表。对应到真正的 Attention，Kernel 也要理解这张表，按映射读取各个块并完成同一个历史上的 Attention。分页管理与读取 Kernel 在这里接起来。

这次脚本里，追加位置的核心逻辑很短：

```python
if current_len % block_size == 0:
    block_table[request_id].append(allocate_block())

lengths[request_id] += 1
```

旧块满了就领下一块，块还没满就继续往里面写。

### 槽位到底省在哪里

`paged_kv_sim.py` 使用四个请求：

| 请求 | 提示词长度 | 后续增加 | 最终长度 |
| --- | ---: | ---: | ---: |
| A | 8 | 3 | 11 |
| B | 19 | 5 | 24 |
| C | 5 | 2 | 7 |
| D | 30 | 0 | 30 |

固定预留方案给每个请求 32 个槽位，总共是 128。分页方案把每个请求的长度 `S` 向上取整到块：

$$
N_{\mathrm{blocks}}=\left\lceil\frac{S}{P}\right\rceil
$$

运行后得到：

| 状态 | 有效 Token 槽位 | 固定预留槽位 | 分页占用块数 | 分页占用槽位 | 块内空余 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill 后 | 62 | 128 | 17 | 68 | 6 |
| Decode 后 | 72 | 128 | 19 | 76 | 4 |

最终真正使用 72 个位置，分页给这些请求分配了 76 个槽位。剩下的 4 个空位散在各请求尚未填满的尾块里。

每个请求最后一块的空余都小于 `P`。块越小，尾部空余越少，块表和访问管理也会更细；块越大，一次管理的内容更多，尾块可能留下的空位也更多。

这里数的是**分给请求的槽位**。真实引擎可以先在 GPU 上建好一个大块池，再让请求按需领用。尚未领用的块留在池中，供其他请求使用；这时观察请求占用了多少块，比只看进程显存更能解释池内空间去了哪里。

## 前缀复用

### 两个请求能共用哪些块

有了 Block，接着看重复的内容。

两个请求可能带着相同的系统提示词、文档或对话前缀。相同模型、相同输入前缀下，这部分已经计算过的 KV 可以继续使用，后一个请求只处理后续内容。

`prefix_cache_sim.py` 把每个字符串当作一个抽象 Token。两个请求的前 8 个位置完全相同：

```text
共同前缀：请 解释 KV Cache | 工作 原理 以及 例子

A 的后缀：先 讲 原理
B 的后缀：再 比较 GQA MLA
```

块大小为 4，因此共同前缀正好对应两个完整块。A 总长 11，尾部还剩 3 个 Token；B 总长 12，后缀又组成一个完整块。

模拟器用物理块 ID 代表复用的结果。先处理 A，再处理 B：

| 请求 | 完整块 | 命中块 | 复用 Token | 本次待处理 Token | 完整块 ID |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 2 | 0 | 0 | 11 | `[0,1]` |
| B | 3 | 2 | 8 | 4 | `[0,1,2]` |

B 的前两项直接指向 A 已经登记的块 `0、1`，只为后面那一块登记新的 ID。A 的尾部 3 个 Token 本次照常处理，等完整成块后才适合按这套规则登记复用。

![两个请求使用相同的前8个抽象Token并指向同样的物理块0和1。请求A有3个尾部Token未登记为完整块，请求B新增一个完整块2。](/images/kv-cache-systems/prefix-reuse.svg "共同前缀的两个块只保留一份，两个请求都能指向它们。")

### 为什么哈希要带上前面的块

KV 取决于当前位置之前的上下文。假设两个块的 Token 都是 `工作 原理 以及 例子`，但前面的内容不同，它们算出的 KV 也可能不同。

所以模拟器的键同时包含父哈希和当前块的 Token：

```python
content = repr((previous_hash, tuple(tokens))).encode("utf-8")
key = sha256(content).hexdigest()
```

第一块从 `ROOT` 开始；第二块的键包含第一块的哈希；继续往后，前面的上下文就沿着哈希链传下来。请求 B 逐块查询时，前两块得到相同的键，于是命中已有的 ID。

[vLLM 的前缀缓存文档](https://docs.vllm.ai/en/stable/design/prefix_caching/)也按父哈希、块内 Token 和额外标识组织键。真实系统还会考虑 LoRA、多模态输入等影响 KV 的信息，让缓存对应到正确的计算结果。

这套实验只登记完整块，因此命中统计也以完整块为单位。一个请求有 11 个 Token、块大小为 4，能登记的是前两个完整块。

复用以后，还要管理共享关系：哪些请求正在引用同一块，何时能回收，缓存池紧张时先淘汰谁。这些状态由实际引擎的缓存管理器维护。这里先用哈希到块 ID 的字典，把“相同前缀怎样找到同一份结果”看清楚。

## 缓存量化

### 一个数怎样变成 INT8

分页改变数据的位置，前缀复用减少重复的副本。接下来直接改变每个数占用的字节。

`kv_cache_quant.py` 在 CUDA 上生成 K、V，各自形状为 `[1,4,1024,64]`，原始格式是 FP32。然后分别做两种转换：

- FP32 → FP16 → FP32，观察半精度重建误差。
- FP32 → INT8 加 scale → FP32，观察整数重建误差。

这一版沿最后一个维度量化：每 64 个数共用一个 scale。先取向量里的最大绝对值，让它对应整数范围中的 127，再把其他值按同一比例缩放、四舍五入：

$$
\begin{aligned}
a&=\max_i|x_i|\\
s&=\max(a/127,10^{-8})\\
r_i&=\operatorname{round}(x_i/s)\\
q_i&=\operatorname{clip}(r_i,-127,127)\\
\hat{x}_i&=q_i s
\end{aligned}
$$

这叫逐向量的**对称 INT8 量化**，零点是 0。scale 的下限让全零向量也能正常处理。

对应代码是：

```python
max_abs = x.abs().amax(dim=-1, keepdim=True)
scale = (max_abs / 127.0).clamp(min=1e-8)
q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
restored = q.float() * scale
```

这次第一个 K 向量有 64 个数，最大绝对值为 `2.35513091`，所以 scale 约为 `0.01854434`。截取其中四个坐标：

| 原值 | INT8 | 还原值 |
| ---: | ---: | ---: |
| `0.194019` | 10 | `0.185443` |
| `2.161374` | 117 | `2.169688` |
| `-0.172050` | -9 | `-0.166899` |
| `0.849060` | 46 | `0.853040` |

整数只记录它落在哪一个刻度上，scale 记录这个刻度有多宽。还原出来的数落在附近，舍入误差也就出现在这里。

![同一个64维向量取出前8个坐标，按真实scale转换成INT8再恢复浮点值。下方比较FP32、FP16、INT8数据和scale的存储字节。](/images/kv-cache-systems/quantization.svg "scale 来自完整 64 维向量，图中截取前 8 维观察舍入。")

### 存储与误差

K、V 合计有 `524288` 个元素。每个 scale 对应 64 个元素，K、V 合计有 `8192` 个 FP32 scale：

| 表示 | K/V 存储 | K 的平均绝对误差 | K 的最大绝对误差 |
| --- | ---: | ---: | ---: |
| FP32 原值 | 2 MiB | 0 | 0 |
| FP16 | 1 MiB | `0.00014044` | `0.00184584` |
| INT8 + FP32 scale | 0.53125 MiB | `0.00502959` | `0.01719800` |

INT8 数据本身是 `0.5 MiB`，scale 占 `0.03125 MiB`，加起来是 `0.53125 MiB`。相对 FP32 减少 `73.4375%`，相对 FP16 减少 `46.875%`。

表里的容量统计对应某一种选定的缓存表示。实验为了比较误差，同时保留原值、不同格式和还原张量，它们会各自占空间。部署时则要看最终保留哪一种表示，以及计算时怎样还原。

scale 的粒度也会影响取舍。按整张 Tensor 共用一个 scale，元数据更少，数值范围差异较大的向量会挤在同一套刻度里；按更小的组取 scale，可以更贴近局部范围，元数据也会增加。

这份脚本把 INT8 先恢复成 FP32，再计算重建误差。推理 Kernel 怎样读取低精度缓存、何时反量化，可以沿着 [vLLM 的量化缓存实现说明](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)继续对照。存下来的字节和计算时展开的临时数据，都需要算进执行过程。

## 缓存卸载

### 从 GPU 搬到 CPU

显存里暂时放不下的缓存，还可以先搬到 CPU 内存。等它再次参与计算时，再搬回 GPU。

这里数据依然保留，只是换了存放位置。完整 Attention 需要读取历史 K/V，执行到这部分数据时，就要把它取回，或者安排分块搬运。

`kv_cache_offload.py` 使用四个独立 Tensor 作为搬运块。每个 Tensor 同时包含 K 和 V，形状为 `[2,1,8,2048,128]`，使用 FP16：

```text
2 × 1 × 8 × 2048 × 128 × 2 bytes = 8 MiB
4 个搬运块合计：32 MiB
```

这一节按含有 2048 个 Token 的 Tensor 搬运。前面分页模拟器的教学页大小是 4，两组实验各自用自己的单位观察问题。

先把块 2、3 复制到 pinned CPU 内存，等复制完成后，移除 GPU 列表中对应的引用：

```python
cpu_tensor, d2h_ms = measure_d2h(chunks[idx])
cpu_cache[idx] = cpu_tensor
chunks[idx] = None
```

旧 GPU Tensor 的引用释放以后，才能回收对应的存储。切片视图会共享原 Tensor 的底层存储；这里把每个搬运块单独分配，释放块 2 时，其他块还可以各自保留。

![四块各8MiB的GPU缓存中，块2和3搬到pinned CPU后释放GPU引用，GPU有效分配32降到16MiB；随后通过H2D恢复，GPU回到32MiB且数据一致。](/images/kv-cache-systems/offloading.svg "缓存从 GPU 搬走，再带着传输成本搬回来。")

### 显存读数怎样变化

脚本以创建这些 Tensor 之前的 `memory_allocated()` 为基线。运行后，缓存占用按下面的顺序变化：

| 状态 | GPU Tensor 已分配空间 | CPU 缓存数据 |
| --- | ---: | ---: |
| 四块都在 GPU | 32 MiB | 0 |
| 卸载块 2 | 24 MiB | 8 MiB |
| 再卸载块 3 | 16 MiB | 16 MiB |
| 两块全部恢复 | 32 MiB | 0 |

最后 `memory_reserved()` 的读数是 `40 MiB`。这个值统计 PyTorch 分配器管理的空间，其中可以包含已经空闲、等待复用的部分；`memory_allocated()` 则跟着仍被 Tensor 占用的空间变化。[PyTorch 的内存管理说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management)区分了这两个指标。

因此，观察卸载时，可以分别看数据还占着多少 GPU 存储，以及分配器手里还保留着多少空间。

### 搬运需要多久

这次使用 pinned CPU 缓冲区和 `non_blocking=True` 复制。目标 Tensor 先分配好，每个方向预热两次，再用 CUDA Event 测七次，取中位数：

| 搬运块 | GPU → CPU | CPU → GPU |
| --- | ---: | ---: |
| 2，8 MiB | `0.638816 ms` | `0.632000 ms` |
| 3，8 MiB | `0.640160 ms` | `0.633888 ms` |

Event 结束后先同步，再使用复制结果。恢复时，两个块的 `torch.equal()` 都为 `True`，逐元素一致。

这里计的是复制区间，目标分配、垃圾回收和一致性检查放在区间外。每次取回都要占用链路带宽，也可能让后续计算等待。

`non_blocking=True` 影响 Host 是否在调用处等待，数据使用顺序仍由 Stream 和同步关系保证。若要让搬运与其他计算重叠，还要安排独立 Stream、预取时间和依赖；[PyTorch 的 pinned memory 教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)给出了这些关系的具体例子。

## 连续组批

### 请求结束以后，空位给谁

前面几节都在看缓存，最后把请求本身放进来。

同一批请求里，有的生成两三个 Token 就结束，有的还要继续很多轮。如果整批都结束后才重新组批，已经空出来的执行位置就会等着。

Continuous Batching 把调度粒度缩到迭代：这一轮先推进活动请求，结束的退出；下一轮开始时，再把等待请求补进空位。[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)用 iteration-level scheduling 描述了这种调度粒度。

`continuous_batching_sim.py` 最多同时放三个请求。每轮，每个活动请求生成一个 Token：

| 请求 | 到达时刻 | 需要生成 |
| --- | ---: | ---: |
| A | 0 | 8 |
| B | 1 | 5 |
| C | 2 | 3 |
| D | 4 | 6 |
| E | 5 | 2 |

静态策略先接到 A，从第 0 轮运行到第 8 轮边界。B、C、D、E 陆续到达，只能在等待队列里排着。A 完成以后，B、C、D 进入下一批；C 先结束，对应位置继续保留到这批全部完成。

连续策略则在第 1 轮加入 B，第 2 轮加入 C。C 在第 5 轮边界结束，D 立即进入下一轮；B 在第 6 轮边界结束，E 接着补入。

![上方静态组批的3个批次位置覆盖16轮，已完成槽位用斜纹保留；下方连续组批在每轮边界补入D、E，11轮完成相同24个Token。](/images/kv-cache-systems/continuous-batching.svg "颜色跟着请求走。图中的行表示本轮批次位置，活动列表会随请求退出而压紧。")

### 同一批任务的两种结果

五个请求总共需要生成 24 个 Token。两种策略跑完后的统计：

| 指标 | 静态组批 | 连续组批 |
| --- | ---: | ---: |
| 总轮数 | 16 | 11 |
| 平均等待 | 5.2 | 0.4 |
| 平均周转时间 | 10.0 | 5.2 |
| 模拟吞吐，Token / 时间单位 | 1.500 | 2.182 |
| 有用 Token 槽位利用率 | 50.0% | 72.7% |

等待时间是“开始处理减到达时刻”，周转时间是“完成减到达时刻”。槽位利用率按下面的方式计算：

$$
U=\frac{\text{生成的 Token 总数}}{\text{最大批次大小}\times\text{总轮数}}
$$

静态策略是 `24 / (3 × 16)`，连续策略是 `24 / (3 × 11)`。这张时间线直接数出了多少执行位置真正推进了 Token。

模拟器把每轮定义为同样长的时间单位，只调度 Decode，每个活动请求一轮推进一个 Token。真实系统中，批次大小、上下文长度、Prefill 工作量和缓存容量，还会影响一轮能安排多少工作。

连续组批也会把问题带回缓存管理器：新请求进入，要找到可复用的前缀、领取需要的块；请求结束，要减少引用并回收可释放的块。请求来来去去，缓存池也跟着变化。

## 把五件事接起来

现在再看这几个名字，各自负责的事情就具体了：

| 机制 | 主要管理什么 | 需要付出的成本 |
| --- | --- | --- |
| 分页 | Token 位置到物理块的映射、按需领用 | 块表访问、尾块空余和对应的 Kernel |
| 前缀复用 | 已计算前缀对应的共享结果 | 哈希、引用状态和淘汰管理 |
| 量化 | 缓存里每个数的表示方式 | scale、重建误差和反量化计算 |
| 卸载 | 缓存此刻放在 GPU 还是 CPU | CPU 内存、传输带宽和等待 |
| 连续组批 | 下一轮执行哪些请求 | 调度、缓存容量和不同请求的资源竞争 |

它们也能围着同一个块协作。一个新请求先命中已有前缀，剩余位置从池里领块；块内数据使用某种精度保存；暂时不执行的请求可以把缓存移到 CPU；调度器接纳它时，再让需要的数据准备好。

具体安排要沿着数据依赖走：哪些位置已经算过，哪些块被共享，这一轮要读哪些块，数据现在又在哪里。每一项优化省下来的东西不同，额外增加的工作也不同。

上一篇在看模型怎样表示历史，这篇在看系统怎样存放和使用这些历史。把这五个实验摆在一起后，“推理效率”也开始有了可以逐项追踪的内容：槽位给了谁，结果复用了多少，每个数占几字节，搬了多少数据，这一轮哪些请求真正向前走了。

## 源码与复现

五个原始学习脚本与本次运行记录放在同一个包里：

- [完整实验包](../../downloads/kv-cache-systems/experiments.zip)
- [运行说明](../../downloads/kv-cache-systems/README.md)
- [分页、前缀与调度结果](../../downloads/kv-cache-systems/simulation_results.json)
- [量化与卸载结果](../../downloads/kv-cache-systems/gpu_results.json)

| 文件 | 观察内容 |
| --- | --- |
| `paged_kv_sim.py` | 领块、追加位置、块表和槽位统计 |
| `prefix_cache_sim.py` | 父哈希、完整块命中和待处理 Token |
| `continuous_batching_sim.py` | 到达队列、每轮批次、请求完成和指标 |
| `kv_cache_quant.py` | CUDA Tensor 的精度转换、误差和存储字节 |
| `kv_cache_offload.py` | pinned CPU 传输、GPU 引用释放和数据还原 |
| `record_simulations.py` | 运行前三个原脚本，并收集 JSON 与控制台输出 |

前三个只依赖 Python 标准库，量化脚本需要 PyTorch，卸载脚本需要支持 CUDA 的 PyTorch 环境。解压后分别运行：

```bash
python -B paged_kv_sim.py
python -B prefix_cache_sim.py
python -B continuous_batching_sim.py
python -B kv_cache_quant.py
python -B kv_cache_offload.py
```

要保存三个模拟器的结构化结果，可以运行附加的记录脚本：

```bash
python -B record_simulations.py --output-dir ./my-results
```

本次环境是 Python `3.12.15`、PyTorch `2.14.1+cu130`、CUDA `13.0`、RTX 4070 Laptop GPU，显存 8 GiB。两个 Tensor 实验固定种子 42。模拟器的原始输出保存在 [simulation_stdout.txt](../../downloads/kv-cache-systems/simulation_stdout.txt)，CUDA 实验输出保存在 [gpu_stdout.txt](../../downloads/kv-cache-systems/gpu_stdout.txt)。

这组实验分开观察元数据管理、随机 Tensor 重建、缓存复制和调度。分页与前缀部分用块 ID 表示缓存状态；调度使用等长的抽象轮次。量化记录的是随机 K/V Tensor 的重建误差，卸载记录的是复制与显存变化。五个脚本各自运行，正文中的数字按这些具体设置解释，源码和完整记录可直接对照。
