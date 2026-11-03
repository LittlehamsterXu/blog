---
title: "nano-vLLM 学习笔记"
date: 2026-10-30T15:00:00+08:00
draft: false
tags: [nano-vLLM, LLM 推理, KV Cache, Continuous Batching, CUDA Graph]
categories: [AI Infra]
description: "结合 Qwen3-0.6B 的运行日志，整理 nano-vLLM 的推理流程、连续组批、分页 KV Cache 和 CUDA Graph。"
summary: "从一次 step 开始，看请求怎样被调度、Token 与 KV 怎样逐轮增长，再把块表接到 Attention，理解连续组批、前缀复用和 CUDA Graph 的实现。"
ShowToc: true
TocOpen: true
---

{{< study-summary id="知识点速记" >}}

**1. 一轮推理：选请求 → 算输入 → 更新序列**

`LLMEngine.step()` 驱动三步：`Scheduler` 选本轮请求，`ModelRunner` 准备输入、执行模型并采样，`postprocess` 更新缓存进度、追加 Token、移除完成请求。

每条请求的 Token、缓存进度和块表保存在 `Sequence`；物理块由 `BlockManager` 分配、共享和回收。

**2. Prefill / Decode：新 Token 的 KV 晚一轮**

| 本轮做什么 | 本轮结束时 |
| --- | --- |
| Prefill 输入 `ABC`，采样 `D` | 序列 `ABCD`；KV 只有 `ABC`。 |
| Decode 输入 `D`，采样 `E` | 序列 `ABCDE`；KV 推进到 `ABCD`。 |

**刚采样出的 Token 先加入序列，下一轮输入模型时才算自己的 KV。** 上面两轮结束时，序列长度都比已计算 KV 的位置数多 1。

**3. 连续组批：每轮重新选请求**

A 正在生成，B 中途到达。这版代码采用 Prefill 优先：

- **这一轮**：B 做 Prefill，A 暂停。
- **下一轮**：A、B 共同 Decode。

`waiting` 等待 Prefill，`running` 等待继续生成。队列里的请求不一定本轮都执行；谁结束谁退出，下一轮重新组批。

**4. 分页缓存：逻辑位置 → 物理槽位**

块表记录“第几个逻辑块，放在哪个物理块”。位置从 0 计数，每块 4 个位置、块表为 `[2, 7, 4]` 时：

- 位置 9：`9 // 4 = 2`、`9 % 4 = 1`，属于逻辑块 2、块内偏移 1。
- 查 `block_table[2] = 4`，得到物理块 4，最终槽位是 `4×4+1 = 17`。

`slot_mapping` 指定**本轮 KV 写哪里**；`block_tables` 配合 `context_lens` 指定**历史 KV 从哪里读、读多长**。

**5. 复用与抢占：共享块，保留 Token**

- **前缀复用**：完整块里的 Token 和它之前的上下文都相同，才能共享这块 KV。
- **释放**：`ref_count` 归零，块回到空闲队列；旧内容没被覆盖时，仍可能再次命中。
- **抢占**：块不够时，释放某条请求的块引用，保留 Token 序列，放回 `waiting`，之后重新 Prefill。

**6. CUDA Graph：地址不变，数据每轮更新**

- **初始化**：分配固定缓冲区 → 把模型前向的 GPU 操作捕获成图。
- **每一轮**：填入新输入 → `replay` → logits / 采样。

新输入包括 Token、位置和块表。GPU 仍然计算本轮结果，省下的是 CPU 重复提交模型计算的开销；这版的 logits 和采样在图外执行。

{{< /study-summary >}}

[上一篇](../kv-cache-inference-systems/)用几个小实验，分别看了缓存分块、前缀复用和连续组批。每个机制单独看已经比较清楚，接下来想看它们在推理引擎里怎样配合。

这次看的是 nano-vLLM。先在 WSL 里用 Qwen3-0.6B 跑通生成，再打印每轮的请求与缓存状态，顺着 `LLMEngine`、`Scheduler` 和 `ModelRunner` 看执行过程。

日志里最先注意到的是，Prefill 结束后，序列长度会比缓存进度多一。后来又加入了中途到达的请求，观察它什么时候做 Prefill、什么时候与旧请求一起 Decode。下面从一次 `step()` 开始。

## 整体架构

提交请求以后，引擎反复执行 `step()`，直到所有请求结束：

```python
llm.add_request(prompt, sampling_params)

while not llm.is_finished():
    outputs, num_tokens = llm.step()
```

`add_request()` 把文本编码成 Token，建立一个 `Sequence`，交给调度器。`step()` 的核心是三次调用：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

`schedule()` 选择本轮请求和每条请求处理的 Token 数；`run()` 准备模型输入，执行前向并采样；`postprocess()` 更新缓存进度、追加输出 Token，并移除已经完成的请求。

下面展开一次双请求 Decode：请求 1 的末尾是 `D`，请求 2 的末尾是 `Y`。执行器把这两个 Token 组成一批输入 `[D, Y]`，两条请求各自读取自己的历史。模型输出 `[E, Z]`，再按同样的顺序追加回对应序列。图中的字母是示意 Token。

![从左到右展示选请求、同批执行和更新序列。请求1输入D、采样E，序列从ABCD变为ABCDE；请求2输入Y、采样Z，序列从XY变为XYZ。两条请求的缓存进度分别从3到4、从1到2。](/images/nano-vllm/engine-step.svg "沿着两行分别看：请求 1 从 D 到 E，请求 2 从 Y 到 Z。同批计算，结果各自回填。")

几个类的分工可以列成一张表：

| 类 | 负责的内容 |
| --- | --- |
| `LLMEngine` | 接收请求，驱动迭代，收集完成结果。 |
| `Sequence` | 保存单条请求的 Token、长度、采样参数和缓存状态。 |
| `Scheduler` | 管理等待与运行队列，选择本轮请求。 |
| `BlockManager` | 分配、共享和回收 KV 缓存块。 |
| `ModelRunner` | 准备输入和缓存元数据，执行模型与采样。 |

每轮选出的请求可以不同，模型输入也要随之重新整理。先看单条请求的输入和输出怎样变化。

## Prefill 与 Decode

### 序列与缓存进度

假设 Prompt 是 `A B C`，接下来生成 `D E F`。

Prefill 把 `A B C` 输入模型，计算各层的 K/V，再用最后一个位置的 logits 采样 `D`。这时序列已经是 `A B C D`，缓存里处理完的仍然是 `A B C`。

下一轮 Decode 输入 `D`，计算它自己的 Q/K/V，把 K/V 写入缓存，再读取包含当前位置在内的历史并采样 `E`。序列变成 `A B C D E`，缓存推进到 `A B C D`。

![Token沿位置排列，序列和KV分别增长；采样出的D跨到下一轮成为输入，写入自己的KV后参与Attention，随后采样E。](/images/nano-vllm/token-kv-timing.svg "沿着 D 看两轮：先被采样，再作为输入计算 KV。序列末尾的新 Token 尚未进入缓存。")

**采样得到 Token ID，和计算这个 Token 的 KV，发生在相邻两轮。**

### 双请求日志

本地保存的示例一次提交了两个请求，Prompt 分别有 11 和 17 个 Token。第一轮同时做 Prefill，下面只保留进度相关字段：

```text
[AFTER SCHEDULE] mode=PREFILL
seq=1 tokens=11 cached=0 scheduled=11 blocks=[0]
seq=2 tokens=17 cached=0 scheduled=17 blocks=[1]

[MODEL OUTPUT] token_ids=[151667, 151667]

[AFTER POSTPROCESS] mode=PREFILL
seq=1 tokens=12 cached=11 scheduled=0 blocks=[0]
seq=2 tokens=18 cached=17 scheduled=0 blocks=[1]
```

`tokens` 是序列里已有的 Token 数，`cached` 是已经计算出 KV 的位置数，`scheduled` 是本轮送进模型的 Token 数。

请求 1 本轮处理 11 个位置，缓存进度从 0 增加到 11；同时追加一个采样结果，序列长度变成 12。请求 2 同样从 `17 / 0` 变成 `18 / 17`。

下一轮，两条请求共同 Decode，各自输入末尾的一个 Token：

| 状态 | 请求 1：tokens / cached | 请求 2：tokens / cached |
| --- | --- | --- |
| Prefill 后 | `12 / 11` | `18 / 17` |
| 下一轮输入数量 | `1` | `1` |
| Decode 后 | `13 / 12` | `19 / 18` |

正常生成时，后处理后通常保持 `tokens = cached + 1`。末尾那个新采样的 Token，等待下一轮计算自己的 KV。

这版代码也支持分块 Prefill。长 Prompt 超过一轮的 Token 预算时，可以分几次处理。中间的 chunk 只推进缓存进度，处理到序列末尾以后，才把采样结果追加进序列。

## 连续组批

### 等待与运行队列

上一篇的模拟器按轮次补入请求。到了 nano-vLLM，还要把请求的 Prefill 算进去。

调度器维护两个队列：

- `waiting`：等待执行 Prefill，包括新请求、未完成的分块 Prefill，以及被抢占后等待恢复的请求。
- `running`：已经进入生成阶段，等待继续 Decode 的请求。

`schedule()` 先尝试安排 `waiting` 中的请求。**本轮只要选出了 Prefill 请求，就直接返回，Decode 放到后续轮次。**

因此，A 正在生成时，B 新到达且能被接纳，这一轮先处理 B 的 Prefill，A 暂停。之后再把 A、B 放进同一个 Decode 批次。

### 新请求到达

`test_continuous_batching.py` 先提交机器人故事请求 A，执行两次 `step()`；然后依次加入三个解释类请求，每加入一个就执行一轮，最后继续运行到所有请求结束。

前六次调度结果如下，请求 1 到 4 分别记为 A 到 D：

```text
mode=PREFILL running=1 scheduled=[1]
mode=DECODE  running=1 scheduled=[1]
mode=PREFILL running=2 scheduled=[2]
mode=PREFILL running=3 scheduled=[3]
mode=PREFILL running=4 scheduled=[4]
mode=DECODE  running=4 scheduled=[1, 2, 3, 4]
```

![前六轮的请求到达与选择过程：新请求进入等待队列并执行Prefill，A的生成进度在第三至第五轮暂停，第六轮四条请求组成Decode批次。](/images/nano-vllm/continuous-batching.svg "看 A 的进度：新请求做 Prefill 时保持不动，四条请求共同 Decode 时再继续增长。")

第三到第五轮，`running` 从 2 增加到 4，`scheduled` 却每次只有新请求。队列里已有多少请求，和这一轮执行多少请求，是两个不同的数。

第六轮，四条请求一起 Decode。A 先完成以后退出队列，后续批次变成 `[2, 3, 4]`，其余请求继续生成。这次运行共执行 68 轮，完整记录放在文末附件中。

连续组批允许每轮重新组织请求。这里采用 Prefill 优先，同一轮选择一种阶段；新请求进入以后，先处理输入，再参与后续 Decode。图中的轮次只表示执行顺序，每轮的计算量和耗时可以不同。

## 分页 KV Cache

### 块表与缓存池

前面日志中的 `blocks=[0]`、`blocks=[1]`，是两条请求领到的物理块编号。本地每块放 **256 个 Token**，两个 Prompt 各用一块，后续生成继续填入尾块，跨入下一块时再申请新块。

`ModelRunner` 初始化时分配 KV 池，整体形状是：

```text
[2, 层数, 物理块数, 每块Token数, KV Heads数, Head维度]
```

第一维区分 K、V，每层 Attention 使用池中属于自己的部分。`BlockManager` 分配池里的块，`Sequence.block_table` 保存这条请求的逻辑块到物理块的映射。

池中的 Tensor 提前分配，请求按需领取块编号。请求增长或结束时，主要更新块的归属、引用和内容。

### 写入与读取

把块大小缩到 `P=4`，展开一个具体地址。假设请求的块表是：

```text
逻辑块：       0  1  2
block_table = [2, 7, 4]
```

逻辑位置 `t=9` 对应：

```python
logical_block = 9 // 4           # 2
offset = 9 % 4                  # 1
physical_block = block_table[2]  # 4
slot = physical_block * 4 + 1    # 17
```

这个位置的 K/V 写入物理块 4 的偏移 1，展平以后就是槽位 17。

![请求的逻辑位置0到9分布在物理块2、7、4中，当前位置9写入槽位17，Attention按块表顺序读取这10个有效位置，不读取尾块中的空槽。](/images/nano-vllm/paged-addressing.svg "写入只定位当前槽位；读取沿块表访问完整有效历史。逻辑顺序连续，物理地址可以分散。")

`ModelRunner` 准备输入时，把这些地址整理成 Kernel 使用的元数据：

| 数据 | 用途 |
| --- | --- |
| `slot_mapping` | 本轮每个输入 Token 的 K/V 写入哪个物理槽位。 |
| `block_tables` | 每条请求的逻辑历史对应哪些物理块。 |
| `context_lens` | 每条请求当前有效的上下文长度。 |

Decode 取 `seq.last_token` 作为输入，位置编号是 `len(seq)-1`，上下文长度是 `len(seq)`。模型计算当前 Q/K/V 后，`store_kvcache_kernel` 按 `slot_mapping` 写入 K/V，`flash_attn_with_kvcache` 再根据块表和上下文长度读取缓存、计算 Attention。

调度器管理块的归属，输入准备负责展开地址，Attention Kernel 按这些地址写入和读取数据。

### 前缀复用

两个请求有相同的完整前缀块，后来的请求可以让块表指向已有物理块，跳过这部分 KV 计算。

缓存键包含父块哈希和当前块的 Token，查询时还会比较 Token 内容。父哈希把之前的上下文带进来，避免把不同前缀下相同的一段 Token 当作同一份缓存。

共享关系由 `ref_count` 管理。多个请求引用同一块，计数随之增加；请求结束时减少引用，归零的块回到空闲队列。内容和哈希暂时保留，后续仍可能命中，真正重新分配这块时才清理旧登记。

这里还有一个实现细节：前缀查询会保留最后一个逻辑块继续计算，即使它恰好完整。执行这部分以后，才能得到末尾位置用于预测下一个 Token 的 logits。

### 抢占

Decode 需要新块却没有空闲块时，调度器会从 `running` 尾部选择请求抢占。

被抢占的请求释放块引用，清空块表和缓存进度，再放回 `waiting`。Token 序列仍然保留，之后重新做 Prefill；仍留在池中的完整前缀，也可能再次命中。

上一篇看过把缓存搬到 CPU 的卸载方式。这份实现选择释放块、随后恢复计算，用额外的 Prefill 换出缓存空间。请求能否继续执行，也就和可用的物理块数量联系起来了。

## CUDA Graph

Decode 每轮都会重复相似的前向计算：更新 Token、位置和块表，再执行同一套模型层。普通 eager 模式下，CPU 每轮都要组织和提交这些算子调用。

CUDA Graph 把一段 GPU 工作提前捕获，后续通过 `graph.replay()` 重放，减少重复提交的开销。

### 固定缓冲区

图重放时使用相同的内存地址，新一轮的数据需要写入原来的缓冲区。nano-vLLM 因此先分配 `input_ids`、`positions`、`slot_mapping`、`context_lens`、`block_tables` 和输出缓冲区，再捕获模型前向。

输入的值每轮变化，缓冲区地址继续复用。请求的物理块编号也作为块表数据更新。

![eager逐个提交模型Kernel，CUDA Graph通过一次replay提交捕获的GPU工作；两轮使用同样地址的缓冲区，但Token和位置内容发生变化。](/images/nano-vllm/cuda-graph.svg "对比 CPU 到 GPU 的提交箭头：计算仍需执行，重复组织和提交这些计算的工作减少。")

### 不同批次大小

活动请求数量会变化，因此需要准备多张图。这版先捕获批次大小 `1、2、4、8`，之后按 16 递增到配置上限。

实际有 3 条请求时，选择批次大小为 4 的图，前三项填入真实输入。额外位置的 `slot_mapping=-1`、上下文长度为 0，重放以后只取前三条输出，再计算 logits 和采样。

计算图复用的是模型执行过程，调度器仍然逐轮选择请求，输入和块表也仍然逐轮更新。它减少的是 CPU 重复提交 GPU 操作的工作。

本次运行使用 `enforce_eager=True`，这一节按源码整理。后续可以固定模型和请求，关掉调度打印，对照两条执行路径的 Decode 延迟。

## 源码与复现

源码按 [nano-vLLM 的本次阅读版本](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8) 整理，本地增加了请求和调度状态打印。主要文件如下：

| 文件 | 内容 |
| --- | --- |
| [llm_engine.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/llm_engine.py) | 请求入口与推理循环。 |
| [scheduler.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py) | 队列、调度和后处理。 |
| [block_manager.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py) | 分配、前缀缓存与引用计数。 |
| [model_runner.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py) | 输入准备、模型执行与图捕获。 |
| [attention.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py) | KV 写入与 Attention 调用。 |

CUDA Graph 的固定地址与重放约定，可对照 [PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)。

运行环境是 WSL Ubuntu 26.04、Python `3.12.15`、PyTorch `2.8.0+cu128`、Triton `3.4.0`、FlashAttention `2.8.3.post2`、Transformers `4.57.6`，使用 RTX 4070 Laptop GPU（8 GiB）单卡加载 Qwen3-0.6B。

生成设置保存在脚本中：

```python
llm = LLM(
    path,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    gpu_memory_utilization=0.7,
)
```

实际 KV 块大小为 256。双请求日志记录序列和缓存进度，连续组批日志记录新请求到达后的执行顺序：

- [双请求日志摘录](../../downloads/nano-vllm/prefill-decode-excerpt.txt)
- [连续组批日志](../../downloads/nano-vllm/continuous-batching.log)与[逐轮调度记录](../../downloads/nano-vllm/schedule.json)
- [实验脚本](../../downloads/nano-vllm/test_continuous_batching.py)与[调度记录工具](../../downloads/nano-vllm/record_schedule.py)
- [运行说明](../../downloads/nano-vllm/README.md)与[源码校验](../../downloads/nano-vllm/sources.json)

在已经配置好环境和模型的 nano-vLLM 项目中，运行记录工具可以重新观察每轮调度：

```bash
.venv/bin/python record_schedule.py --output-dir ./my-schedule
```

它会执行同目录下的 `test_continuous_batching.py`，输出调度日志和 JSON。前六轮可以直接与连续组批图对照，后续继续记录请求的退出与批次变化。
