---
title: "KV Cache：从缓存机制到 Attention 架构优化"
date: 2026-10-26
draft: false
tags: [Transformer, KV Cache, GQA, MLA, DeepSeek]
categories: [AI Infra]
description: "从自回归生成的重复计算出发，理解 KV Cache、MHA/GQA/MQA、MLA，以及 DeepSeek-V4 的 CSA 与 HCA。"
summary: "旧 Token 为什么可以复用，缓存为什么越长越贵？从自回归生成出发，沿着 KV Heads、表示宽度和历史条目三个方向，整理 MHA/MQA/GQA、MLA 与 CSA/HCA 怎样改变缓存需求和历史访问。"
math: true
ShowToc: true
TocOpen: true
---

{{< study-summary id="知识点速记" >}}

**KV Cache：把旧 Token 的 K/V 存起来，下次接着用。**

- 每层存自己的 K/V。新 Q 和历史 K 算分数，再按权重对 V 求和。旧 Q 用完就丢。
- 因果 Mask 下，追加 Token 不影响旧位置的表示，所以旧结果可以复用。新 Token 仍要算自己的 Q/K/V、Attention 和 FFN。
- **Prefill**：输入 `ABC` → 存 `ABC` 的 K/V → 预测 `D`。
- **Decode**：输入 `D` → 缓存追加到 `ABCD` → 预测 `E`。

**显存怎么算**

MHA/MQA/GQA：每层 K、V 各一份，形状都是 `[B,H_kv,S,d_h]`：

$$
M_{\mathrm{KV}}=2LBSH_{\mathrm{kv}}d_hb
$$

`L` 层数，`B` 序列数，`S` 历史长度，`H_kv` KV Heads 数，`d_h` Head 宽度，`b` 每元素字节数。BF16 的 `b=2`。其他不变，`B` 或 `S` 翻倍，缓存也翻倍。

**几种架构的区别**

| 方法 | 改了什么 |
| --- | --- |
| MHA | 每个 Q Head 有自己的 K/V。 |
| MQA | 所有 Q Heads 共用一组 K/V。 |
| GQA | Q Heads 分组，组内共用 K/V。 |
| MLA | 每个 Token 改存联合潜在向量 `cKV` 和位置 Key `kR`。 |
| CSA | 每 4 个新 Token 出一条压缩记录，再用 Indexer 选 Top-k。 |
| HCA | 每 128 个 Token 压成一条，压缩后全部读取。 |

MHA/MQA/GQA 改的是 KV 组数，各个 Q 仍有自己的权重，仍然读取全部可见历史。

**MLA 的矩阵吸收**

- 算分数：先用 Key 上投影的转置变换当前 Q，再和历史 `cKV` 做点积。
- 取结果：先对 `cKV` 加权求和，再做 Value 上投影。
- 内容与 RoPE 分开，分别缓存 `cKV` 和 `kR`。两部分精度相同时：

$$
M_{\mathrm{MLA}}=LBS(d_c+d_R)b
$$

`d_c` 是 `cKV` 的宽度，`d_R` 是 `kR` 的宽度。

**CSA/HCA 存什么、读什么**

- V4 的 CSA/HCA：同一条缓存向量同时作为核心 Attention 的 K 和 V 输入，见[论文式 (19)、(26)](https://arxiv.org/html/2606.19348v1#S2.SS3)。两种机制都另保留最近 128 个 Token 的未压缩窗口。
- **CSA**：相邻块重叠压缩。完整压缩池一直保留，Indexer 扫描可见候选；主 Attention 读 Top-k 条和近期窗口。Flash 的 `k=512`，Pro 的 `k=1024`。
- **HCA**：128 个 Token 非重叠压成一条；主 Attention 读全部可见压缩条目和近期窗口。

**取舍**：共享减少独立表示，压缩把信息放进更少的维度或条目。模型要在这些约束下保留有用信息，换取更小的缓存和更少的历史读取。

{{< /study-summary >}}

[上一篇](../transformer-position-encoding/)写到增量生成时，输入虽然只剩一个 Token，位置编号还要接着历史往下数。那时已经有了一个缓存，只是主要盯着 `past_len`，还没仔细看里面存的东西。

这次顺着缓存往下看。

最开始的问题很直白：生成下一个 Token，为什么还要把前面的内容再算一遍？等到知道可以保存 K、V，问题又变成了：历史越来越长，这些东西到底要占多少显存，每一轮又要读多少？

MQA、GQA、MLA，再到 CSA、HCA，名字一下多了起来。整理时，我觉得还是把缓存摊开比较好理解：**每个位置存几组向量、每组有多宽、总共保留多少位置，这一轮又会读取哪些。**

这篇按这个顺序记。前半段接回之前的 Decoder，后半段结合论文整理压缩 Attention 的结构。

## 生成中的重复计算

### Prefill 与 Decode

假设提示词只有三个 Token：`[A, B, C]`，接下来依次生成 `D、E、F`。

最直接的写法，是每一轮把已经得到的完整序列送进模型，然后取最后一个位置的 logits：

```text
A B C     → D
A B C D   → E
A B C D E → F
```

这样很容易写，也能生成结果。只是第二轮里，`A、B、C` 又经过了一遍投影、Attention 和 FFN；第三轮，又多算了一遍。

在因果 Attention 中，旧位置只能读取自己和更早的位置。给序列末尾追加 `D`，`A、B、C` 原先能看到的内容保持不变。固定模型和推理状态，它们在各层算出的表示也就可以沿用。

于是可以先处理完整提示词，把各层历史 K、V 保存下来。这个阶段叫 **Prefill**。提示词已经给定，各个位置可以在因果 Mask 下并行计算；最后一个位置的 logits 用来采样 `D`。

接下来把新得到的 `D` 送进模型，为它计算各层表示，并把它的 K、V 追加到缓存，再预测 `E`。这就是 **Decode**。下一轮输入 `E`，继续往后生成。

这里有个小细节：预测出 `D` 时，缓存里刚处理完的是 `A、B、C`。等到下一轮真的把 `D` 输入模型，缓存里才会多出 `D`。

![上方完整前缀每一轮重新进入模型，下方先Prefill三个提示Token，再逐轮只输入D和E，读取并扩展各层KV缓存。](/images/kv-cache/generation-reuse.svg "Prefill 处理提示词；Decode 接着输入刚生成的 Token。")

### 每层缓存什么

单看某一层、某一个 Head，当前 Token 的 Attention 可以写成：

$$
o_t=\operatorname{softmax}\!\left(
\frac{q_tK_{\leq t}^{\mathsf T}}{\sqrt{d_h}}
\right)V_{\leq t}
$$

当前 Query \(q_t\) 要和所有可见历史 Key 匹配，再用得到的权重聚合 Value。

所以缓存要留住的是**每一层自己的 K、V**。下一轮有新的 Query，它还会读取这些历史 K、V；旧 Query 已经完成了当时那一轮查询，可以随那一轮结束。

对于常见的缓存布局，某一层保存两份张量：

```text
K cache: [B, H_kv, S, d_h]
V cache: [B, H_kv, S, d_h]
```

`S` 是这一层已经写入的历史长度。模型有多少层，就要为多少层分别保存缓存。每层输入不同，投影参数也不同。

新 Token 仍然要经过完整的模型主干：算自己的 Q/K/V，读取历史做 Attention，经过输出投影和 FFN，再送到下一层。省下来的，是对旧 Token 重复执行的那部分计算。

因此，KV Cache 解决了“旧内容反复算”的问题，历史读取本身还在。上下文越长，每个新 Query 面前的候选 Key、Value 就越多。

## 缓存容量

先看标准 MHA、GQA、MQA：K 和 V 各存一份，两者的 Head 宽度相同，数据类型也相同。

总缓存字节数可以写成：

$$
M_{\mathrm{KV}}=2LBSH_{\mathrm{kv}}d_hb
$$

| 符号 | 含义 |
| --- | --- |
| `2` | K、V 两份 |
| \(L\) | Attention 层数 |
| \(B\) | 同时保留的序列数量 |
| \(S\) | 每条序列的缓存长度 |
| \(H_{\mathrm{kv}}\) | 独立 KV Heads 数量 |
| \(d_h\) | 每个 KV Head 的宽度 |
| \(b\) | 每个元素占用的字节数 |

先取一组方便心算的配置：32 层、单条序列、上下文 8192、32 个 Query Heads、Head 宽度 128，使用 BF16，每个元素 2 字节。

只改变 KV Heads：

| 结构 | KV Heads | 理论缓存容量 |
| --- | ---: | ---: |
| MHA | 32 | 4 GiB |
| GQA | 8 | 1 GiB |
| MQA | 1 | 128 MiB |

这里用二进制单位，`1 GiB = 2³⁰ bytes`。这些数由公式计算，统计的是缓存张量本身。

![在相同层数、上下文长度、Head宽度和数据类型下，MHA、GQA、MQA的缓存分别为4GiB、1GiB、128MiB。下方展示上下文翻倍时缓存线性增长。](/images/kv-cache/cache-growth.svg "容量随历史长度线性增长，也随同时保留的序列数量增长。")

如果 batch 里每条序列长度不同，可以把 \(BS\) 换成各序列有效长度之和；连续预分配实现则可能按容量上限占用空间。部署时还会有模型权重、临时张量、缓存管理信息和分配器预留空间，所以这张表适合估算 KV 数据量，显存监控要结合具体实现看。

这也解释了为什么长上下文会让推理变得吃力。存储端要容纳不断增长的缓存；执行端则要反复读取历史，计算匹配分数和加权和。缓存容量与读取流量在这里开始联系起来。

## 共享 KV Heads

### MHA、MQA 与 GQA

最直接的方向，是减少独立的 KV Heads。

MHA 中，每个 Query Head 都有对应的一组 K、V。假设有 4 个 Query Heads，就有 4 组独立的历史表示。

MQA 保留多个 Query Heads，让它们共同读取一组 K 和一组 V。各个 Query 仍然不同，算出的 Attention 权重也可以不同；共享的是用来被查询的历史表示。[MQA 论文](https://arxiv.org/abs/1911.02150)把减少增量解码中的 KV 读取带宽作为主要动机。

GQA 则把 Query Heads 分组。同组 Query 使用同一组 K、V，不同组各自保留独立的 K、V。[GQA 论文](https://aclanthology.org/2023.emnlp-main.298/)用这种方式连接 MHA 与 MQA 两个端点。

![三个面板分别画出4个Query的投影和历史K、V矩阵。MHA各用一组，GQA每两个Query共享一组，MQA的全部Query共享同一组。](/images/kv-cache/kv-head-sharing.svg "沿着同色连线看：Q 仍然各自查询，历史 K/V 的独立组数逐渐减少。")

图里固定 4 个 Query Heads，GQA 取 2 组，缓存容量比例就是 `4 : 2 : 1`。前面的容量表取 32 个 Query Heads、8 组 GQA，是另一组更接近常见模型规模的示例。

用一个具体 shape 看会更清楚。`B=1、S=100、d_h=64`，8 个 Query Heads 分成 2 组：

```text
当前 Q: [1, 8,   1, 64]
历史 K: [1, 2, 100, 64]
历史 V: [1, 2, 100, 64]
```

Query 0 到 3 读取 KV 组 0，Query 4 到 7 读取 KV 组 1。每个 Query 都会生成自己的 100 个分数，再聚合对应组的 Value。

### 共享发生在哪

实现上，K/V 的投影宽度从 `H_q * d_h` 变成 `H_kv * d_h`，缓存也按较小的 Head 数保存。

有些说明代码会先用 `repeat_interleave` 把 K/V 展开到 Query Heads 的数量，让后面的矩阵乘法更直观。[PyTorch SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)也用这种方式说明 GQA 的语义。

实际执行时，支持 GQA 的 Kernel 可以让同组 Query 直接复用较少的 KV Heads。缓存保存的形状、计算时的逻辑对应关系，以及 Kernel 是否创建展开副本，是三个可以分别检查的地方。

在固定其他维度时，KV Heads 越少，缓存数据就越少。表示能力则要连同模型规模、训练方法和任务表现一起比较。GQA 论文给出了由 MHA checkpoint 转换后继续训练的方案，并在其评估设置下获得了接近 MHA 的质量。

到这里，优化作用在 \(H_{\mathrm{kv}}\) 这个维度上。历史位置仍然逐个保存，每个位置的表示宽度也保持原样。

## 潜在表示与 MLA

### 联合压缩 K、V

继续盯着单个 Token。MHA 要为它保存多个 Head 的 K、V，能不能让这些表示共享一份更紧凑的底层信息？

[DeepSeek-V2](https://arxiv.org/html/2405.04434v5#S2.SS1)提出的 **Multi-head Latent Attention，MLA**，使用一个低维潜在向量联合生成内容 Key 与 Value。

下面统一使用列向量。第 \(j\) 个 Token 在这一层的输入记为 \(h_j\)，先得到：

$$
c_j^{KV}=W^{DKV}h_j
$$

再由各个 Head 的上投影，得到内容相关的 K、V：

$$
\begin{aligned}
k_{j,i}^{C}&=W_i^{UK}c_j^{KV}\\
v_{j,i}&=W_i^{UV}c_j^{KV}
\end{aligned}
$$

\(i\) 表示 Head。不同 Head 使用不同的上投影，所以能从同一份潜在向量中得到各自的内容表示。

缓存端保存这份 \(c_j^{KV}\)。它同时支撑 K 和 V，两者通过参数学习出各自的读取方式。这里的压缩维度 \(d_c\)，是相对于多头 K/V 的整体表示而言；它完全可能大于单个 Head 的宽度。

![左侧逐Token保留各个Head的完整K与V，右侧从隐藏状态分别投影出联合潜在向量和位置Key，再以两列字段逐Token存入历史缓存。小格显示示意特征维度。](/images/kv-cache/mla-latent.svg "先看每个 Token 最后存了什么：多头 K/V，或联合潜在向量与位置 Key。")

### 矩阵吸收

读到这里，接着会遇到一个问题：下一轮读取缓存时，难道还要把所有历史 K、V 恢复出来？

先只看内容分数，暂时省略缩放，把 Key 的定义代入：

$$
\begin{aligned}
q_{t,i}^{\mathsf T}k_{j,i}^{C}
&=q_{t,i}^{\mathsf T}W_i^{UK}c_j^{KV}\\
&=\left((W_i^{UK})^{\mathsf T}q_{t,i}\right)^{\mathsf T}c_j^{KV}
\end{aligned}
$$

第二行把变换放到当前 Query 上。先为当前 Query 算一次 \((W_i^{UK})^{\mathsf T}q_{t,i}\)，然后直接与历史潜在向量做点积，历史端就可以一直保留压缩表示。

Value 一侧也有类似的结合顺序。得到该 Head 对历史位置的权重 \(\alpha_{t,j,i}\) 以后：

$$
\begin{gathered}
\sum_j\alpha_{t,j,i}v_{j,i}\\
=\sum_j\alpha_{t,j,i}W_i^{UV}c_j^{KV}\\
=W_i^{UV}\left(\sum_j\alpha_{t,j,i}c_j^{KV}\right)
\end{gathered}
$$

先对潜在向量加权求和，再做这个 Head 的 Value 上投影。进一步还可以把它与对应的输出投影组合。

这就是这里的**矩阵吸收**：利用线性变换的结合顺序，把历史端的大规模恢复，改成当前 Query 的变换和聚合后的变换。Softmax 仍然放在算分数之后，权重算完以后才做 Value 聚合。

![上半部分并排展示逐条恢复历史Key与先变换当前Query的路径，下半部分并排展示历史Value逐条上投影后聚合与先聚合潜在向量后上投影的路径。](/images/kv-cache/mla-absorption.svg "Key 侧先变换当前 Query；Value 侧先把历史潜在向量聚合成一行。")

### 解耦位置编码

再把[上一文的 RoPE](../transformer-position-encoding/#旋转位置编码)接回来。

如果内容 Key 先做上投影，再按历史位置 \(j\) 旋转，那么匹配分数中会出现：

$$
q_{t,i}^{\mathsf T}R_t^{\mathsf T}R_jW_i^{UK}c_j^{KV}
$$

中间的旋转矩阵随着当前位置和历史位置变化。前面那个可以统一搬到 Query 一侧的固定变换，现在夹进了位置依赖。

MLA 因而把内容和位置拆成两条路径：内容部分保持方便吸收的形式，额外的位置 Query、Key 单独应用 RoPE，再把两个点积加到同一个分数里。这里用 \(d_h\) 表示内容 Q/K 的 Head 宽度，\(d_R\) 表示额外的位置维度。

$$
\begin{gathered}
\operatorname{score}_{t,j,i}\\
=\frac{
(q_{t,i}^{C})^{\mathsf T}k_{j,i}^{C}
+(q_{t,i}^{R})^{\mathsf T}k_j^{R}
}{\sqrt{d_h+d_R}}
\end{gathered}
$$

其中位置 Key \(k_j^R\) 在各个 Query Heads 之间共享。因此每个位置缓存的是：

```text
内容潜在向量 c^KV: [d_c]
位置 Key      k^R: [d_R]
```

在各部分使用相同字节宽度时，总容量变成：

$$
M_{\mathrm{MLA}}=LBS(d_c+d_R)b
$$

与前面的标准 KV 公式相比，这里计算的是一份联合潜在向量加一份位置 Key。它已经包含两条实际保存的内容。

缓存缩小以后，计算形状也跟着改变了：内容点积可能在 \(d_c\) 维上完成，聚合也发生在潜在空间。执行时间还要看这些矩阵乘法、数据布局和 Kernel 怎样落到硬件上。[DeepSeek-V3 的官方推理实现](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)保留了显式 K/V 路径和吸收后的路径，很适合对着上面的两步变换读。

## 压缩历史与 CSA/HCA

### 换一个压缩方向

前面的 MLA 仍然为每个历史位置留下一份表示。把历史排成一张表，行数依然是 \(S\)，只是每行变窄了。

再换个方向：能不能把若干个相邻 Token 聚合成一条历史记录，让表的行数也减少？

[DeepSeek-V4 技术报告 §2.3](https://arxiv.org/html/2606.19348v1#S2.SS3)里的两种机制就在处理这件事：

- **CSA，Compressed Sparse Attention**：较低比例的历史压缩，再选择部分压缩条目。
- **HCA，Heavily Compressed Attention**：较高比例的历史压缩，再读取全部可见的压缩条目。

V4 的这两种机制采用 **Shared Key-Value MQA**。先把整体路径放在这里，接下来分别看 K/V 共享、历史压缩和条目选择。

![CSA使用步长4的重叠压缩，再经过索引器和Top-k选择；HCA使用128的非重叠压缩，直接访问压缩池。两路都结合近期未压缩窗口。](/images/kv-cache/compressed-history.svg "两种机制都保留近期细节，远处历史使用不同的粒度和访问方式。")

### 共享 K 与 V

普通 MQA 中，K、V 各自生成，所有 Query Heads 共用一组 K，也共用一组 V。V4 再往前走了一步：**同一条缓存向量，既作为 Key，也作为 Value。** [论文式 (19)、(26)](https://arxiv.org/html/2606.19348v1#S2.SS3)将核心 Attention 的 `key` 和 `value` 设为同一组条目。

设当前 Head 的 Query 是 \(q\)，本轮可读的第 \(j\) 条缓存向量是 \(c_j\)。它先参与匹配，再参与加权求和：

$$
\begin{aligned}
s_j&=\frac{q^{\mathsf T}c_j}{\sqrt{d}}\\
o&=\sum_j\alpha_jc_j
\end{aligned}
$$

\(d\) 是向量宽度，\(\alpha_j\) 是由分数归一化得到的 Attention 权重。第一行把 \(c_j\) 当作 Key，决定这一条有多相关；第二行把同一个 \(c_j\) 当作 Value，取出它的内容。各个 Head 的 Q 仍然不同，因此可以算出不同的权重，得到各自的加权结果。

和前面的共享方式放在一起看：

| 结构 | 共享了什么 |
| --- | --- |
| 普通 MQA | 多个 Q Heads 共用 K、共用 V；K 和 V 分开生成。 |
| MLA | 内容 K/V 来自同一个 `cKV`，经过不同的上投影得到。 |
| V4 的 Shared KV | 多个 Q Heads 共用历史条目，同一个条目同时作为 K 和 V 输入。 |

[官方 Kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py#L325)也正好对应这两步：同一份 `kv_shared`，先与 `q_shared` 相乘计算分数，再与权重 `acc_s_cast` 相乘得到输出。

标准容量公式里的 `2` 对应分别保存 K、V 两份数据；V4 的核心缓存按一份共享向量计数，再分别统计压缩池、近期窗口和额外索引池。

位置处理也跟着变了：共享 KV 的部分维度应用了 RoPE，这部分既参与匹配，也参与 Value 聚合。V4 因而在聚合输出的对应维度上，再按当前 Query 的位置做逆旋转，使输出携带相对位置关系。

### 学习式压缩

这里的“聚合”可以先理解成带权压缩。

模型从各个 Token 的隐藏状态投影出候选表示，同时预测压缩权重；权重再结合块内可学习的位置偏置，沿时间位置做 Softmax，最后逐通道加权求和。

也就是说，每个通道可以从块内不同位置提取信息。这套参数与模型一起训练，让压缩条目承担后续 Attention 所需要的表示。

CSA 的压缩步长为 4，每处理 4 个 Token 产生一个条目。细看构造时，一个条目会结合相邻两块共 8 个候选表示，两块使用不同投影，形成**重叠压缩**。所以“4 倍压缩”说的是输出条目的频率，压缩时实际读取的范围更宽。

HCA 使用更大的块，128 个 Token 聚合成一条，采用非重叠压缩。相同长度的历史，在这里会得到少得多的条目。

### CSA：压缩后再选

即便已经缩短到原来的四分之一，长历史里仍然可能有很多压缩条目。CSA 接着使用 **Lightning Indexer**，先做一轮较轻量的相关性评分。

可以把这两套表示分开看：核心 Attention 保存较宽的压缩 KV 条目；Indexer 另外保存较窄的压缩索引 Key。当前 Token 生成索引 Query，对所有因果可见的索引条目评分，再选出 Top-k 对应的核心 KV 条目。

后面的核心 Attention，把这些选中的压缩条目与近期局部窗口放在一起，完成分数计算和 Value 聚合。各个核心 Query Heads 使用同一组选中位置。

这样分开以后，有两件事就容易看清了：

1. 完整压缩池继续保留，下一轮 Query 可以选择别的历史条目。
2. 本轮核心 Attention 只读取选中的那部分；Indexer 仍然要处理可见候选池，并付出评分和 Top-k 的开销。

“存储的历史”和“本轮核心计算访问的历史”，在这里有了不同的大小。

### HCA：压得更短，全部读取

HCA 每 128 个 Token 产生一条压缩记录。序列已经大幅缩短，核心 Attention 可以直接读取全部因果可见的压缩条目。

因此，它省去了索引器和 Top-k 选择。这里的 Dense Attention，覆盖的是**压缩后的历史池**。

近期未压缩窗口在两种机制里都保留。远处内容通过压缩记录传递，最近的细粒度依赖则直接读取原始 KV 表示；尚未形成完整压缩块的近期内容，也由局部窗口接住。压缩块和窗口各自遵循因果可见范围。

### 存多少，读多少

用一组条目数把区别摆出来。历史长度取 `S=8192`，局部窗口取 128，CSA 的 Top-k 先用 V4-Flash 的 512。

只看某一层、完整压缩块对应的主数据池：

| 项目 | CSA | HCA |
| --- | ---: | ---: |
| 核心压缩历史条目 | `8192 / 4 = 2048` | `8192 / 128 = 64` |
| 额外索引条目 | 2048 | — |
| 本轮核心 Attention 读取的压缩条目 | 最多 512 | 64 |
| 本轮近期窗口条目 | 最多 128 | 最多 128 |

索引条目的宽度与核心条目不同。在官方配置里，前者宽 128，后者宽 512。所以计数时分别列出来，估算字节数时再乘各自的宽度与精度。

![以8192历史长度为例，CSA保留2048个压缩核心条目和较窄的索引池，当前Query选择512个；HCA保留64个压缩条目并全部读取。两者另加近期128窗口。](/images/kv-cache/history-access.svg "压缩改变池的大小，稀疏选择改变本轮读取的集合。")

如果只给核心池写一个粗略的元素量，CSA 是 \((S/4+128)\times512\)，HCA 是 \((S/128+128)\times512\)。CSA 还要加上索引池，以及压缩尾部、局部窗口管理等状态。这里的共享条目同时承担 K 和 V，直接按实际保存的张量计数即可。

实际每轮能选多少，还取决于已经因果可见的完整压缩块数量。候选不足 `k` 时，就读取已有候选。

继续把历史拉到 `1,048,576` 个 Token，CSA 的压缩池会增长到 `262,144` 条，HCA 的压缩池则是 `8192` 条。Flash 的 CSA 核心仍然最多选 512 条，再加近期窗口；HCA 会读取全部 8192 条压缩历史，再加窗口。前者通过索引控制核心访问量，后者通过更高压缩率控制池的增长，两边付出的工作也就不同。

### 两种机制怎样搭配

CSA 保留较细的远程历史粒度，通过选择控制核心读取量；HCA 把远程表示压得更粗，用更短的池直接完成全量读取。

它们在压缩粒度、候选数量和选择开销上有不同的取舍。V4 把两种层交替放进主干，让表示在多层计算中经过不同尺度的历史读取。

按官方发布配置，几个关键参数可以放在一起：

| 配置 | V4-Flash | V4-Pro |
| --- | ---: | ---: |
| CSA 压缩步长 | 4 | 4 |
| CSA Top-k | 512 | 1024 |
| HCA 压缩步长 | 128 | 128 |
| 近期窗口大小 | 128 | 128 |
| 最初两层 | 纯局部窗口 Attention | HCA |
| 后续主干 | CSA / HCA 交替 | CSA / HCA 交替 |

参数分别来自 [Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)和 [Pro 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json)。压缩、索引和缓存更新可以接着读[官方推理源码中的 `Compressor`、`Indexer`、`Attention`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)。

压缩把多个位置的信息放进较少的条目，能保留多少细节，取决于学到的表示和模型后续怎样使用它。CSA 还要判断哪些条目值得读取，Indexer、Top-k 和收集数据都会产生工作量。比较最终推理性能时，这些部分要与核心 Attention 一起算进去。

## 缓存复用的检查

### 完整前向与增量前向

回到之前的 Decoder，可以把同一段输入按两种方式送进模型：一次完整前向，或者先处理前缀、再带着各层缓存输入后面的 Token。把相同位置的 logits 对齐，就能检查复用路径是否接回了原来的计算。

上一篇已经做过这组比较：先处理四个 Token，再输入两个 Token，后两个位置的 logits 与完整六 Token 前向相差约 `3.33e-16`。当时使用 CPU、float64、固定随机初始化，位置编号正确接到 `[4,5]`。[实验与结果](../transformer-position-encoding/#源码与复现)保留在前文。

这里要一起核对几件事：缓存是否属于正确的层，历史 K/V 的形状是否一致，当前位置是否接着前缀增长，因果 Mask 是否开放了相应的历史。它们共同决定新 Query 读到的内容。

先把结果接回去，再测它用了多少时间，会比较踏实。

### Prefill 与 Decode 的耗时

Prefill 一次处理较长序列，矩阵乘法可以沿 Token 维度展开；单请求 Decode 每轮只有一个新位置，长历史的读取、较小的矩阵乘法以及 Kernel 启动开销会更显眼。Batch、上下文和具体实现改变以后，瓶颈也会跟着变化。

所以性能记录至少要分清两段：提示词处理用了多久，后续每生成一个 Token 又用了多久。

CUDA 还会带来计时上的差别。[PyTorch 的异步执行说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution)提到，Host 发出操作以后，GPU 可以稍后才完成。CPU 两次读时钟之间的间隔，需要结合同步点才能解释。

如果测包含 Python 循环在内的整段延迟，可以在计时开始前与结束时同步 GPU，再读 Host 时钟；如果关心某条 Stream 上的执行间隔，可以用 CUDA Event 记录，再等待结束 Event 完成。多 Stream 的任务要把依赖和计时范围一起定义好。

预热后，按同样的 batch、提示词长度和生成长度重复测量。缓存写入、索引、采样、Python 调度和 Kernel 启动，都可能进入实际耗时；具体占比随模型和负载变化。

架构给出了需要保存和读取的数据，执行时间还要看这些数据怎样进入 Kernel。这里先沿着已有 Decoder 理解缓存复用，MLA、CSA/HCA 则按上面的论文与官方结构整理。

## 回头看缓存

把几种方法放回同一张表，会发现它们分别改动了不同的部分：

| 方向 | 代表方法 | 主要改变 |
| --- | --- | --- |
| 复用旧计算 | KV Cache | 保存旧位置的结果，只计算新增位置 |
| 减少独立组数 | MQA、GQA | 每个位置保存更少组 K/V |
| 压缩单个位置的表示 | MLA | 联合潜在向量与位置 Key 替代完整多头 K/V |
| 减少历史条目 | CSA、HCA | 多个位置聚合成较少的压缩记录 |
| 选择本轮访问的历史 | CSA | 从压缩池中选择核心 Attention 要读的条目 |

MHA 允许每个 Head 独立学习 K/V；GQA 让多个 Query Heads 共享 K/V；MLA 用联合潜在向量压缩 K/V 相关表示；DeepSeek-V4 则让 K 和 V 本身共享表示，并结合 CSA/HCA 压缩历史序列、调整历史访问方式。

**每增加一种共享或压缩，都可能减少一部分表示自由度或历史细节，换取更小的缓存和更少的历史读取。** GQA 减少了独立的 KV 组数，MLA 让内容 K/V 受同一潜在表示约束，共享 K=V 将匹配与取值放在同一向量上，CSA/HCA 则让多个位置的信息通过压缩条目传递。

这些设计把取舍放在不同的位置。模型需要在相应的约束下学会保留有用信息，最后的表现由架构与训练共同决定。

接下来的[《KV Cache：从显存管理到高效推理系统》](../kv-cache-inference-systems/)继续看模型产生这些缓存以后，推理引擎如何分配、复用和调度它们：PagedAttention、Prefix Caching、KV Cache 量化与 Offloading，以及 Continuous Batching。

两篇分别沿着模型和系统往下看。这篇关注 Attention 怎样表示历史、减少缓存需求与访问量；下一篇关注这些缓存怎样存放，以及 GPU 怎样同时服务更多推理请求。

这次先把 Attention 内部的缓存结构理了一遍。以后再遇到一种新的压缩方法，我想先问清楚三件事：**保存下来的是什么，这一轮实际读取什么，省掉的工作又搬到了哪里。**

能把这三个问题放回公式和图里，那些缩写也就没那么容易混在一起了。
