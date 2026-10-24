---
title: "Attention 与 Transformer 基础"
date: 2026-10-24
draft: false
tags: [PyTorch, Attention, Transformer, 深度学习]
categories: [AI Infra]
description: "用 PyTorch 拆开 Attention、多头、位置与 Mask，再把 Encoder 和 Decoder 接成一个能反转变长序列的 Seq2Seq Transformer。"
summary: "从 Q、K、V 和张量形状开始，逐步理解 Attention、Encoder 与 Decoder。最后用一个完整的序列反转案例，验证 Cross-Attention、PAD 处理、训练和自回归生成怎样接在一起。"
math: true
ShowToc: true
TocOpen: true
---

[上一篇](../pytorch-mlp-to-cuda-kernels/)用一个 XOR MLP，把模型、梯度和参数更新接了起来。最后写到，接下来准备看 Transformer。

这几天就从 Attention 开始，写了几个很小的 PyTorch 实验。

刚看公式时，感觉它好像也不复杂。两次矩阵乘法，中间放一个 Softmax，再加上几个 Linear。可真正把每一步打印出来，又会冒出一些很具体的问题。

Q、K、V 都从同一份输入算出来，为什么还需要三套投影？多头里的“拆分”，到底是在拆 Token，还是在拆特征？Mask 已经把 PAD 屏蔽了，怎么 PAD 的输出还不是零？

这些地方如果只记住一张结构图，过几天大概还是会混。所以这次继续把例子缩小，先看数据怎样走，再看模块怎样拼起来。后面又补了一个完整的 Seq2Seq 模型，用变长序列反转，把前面分开看的部分接到一起。

## Attention

### 不同位置怎样交换信息

先用一串数字，把位置之间的关系看清楚。

随机生成一条长度为 5 的 0/1 序列，让每个位置都预测这条序列的第一个 Token：

```text
输入：0 1 1 0 1       目标：0 0 0 0 0
输入：1 0 1 1 0       目标：1 1 1 1 1
```

如果模型只有 Token Embedding、位置向量和逐位置的 Linear，那么第 4 个位置拿到的，只是自己的 Token 和自己的位置。即使知道这里是第 4 个位置，也不知道这一条随机序列开头是 0 还是 1。

第一个位置当然能答对，因为答案就在它自己的输入里。后四个位置需要从别处拿信息。

Attention 在这里提供的，就是这条跨位置的读取通路。某个位置可以根据当前表示，给其他位置分配权重，再把那些位置的信息取回来。

`attention_ablation.py` 做的就是这个任务：每一步重新生成随机序列，让模型学会把开头的信息传到后面。后面再来看它的训练结果。

Transformer 把这种跨位置交互放进了网络的基本模块里。一层里各位置的 Attention 可以一起组织成矩阵运算。后面会看到，这个实现把所有位置两两配对，形成一张随序列长度平方增长的分数矩阵。

### 匹配分数与加权输出

先看单头 Self-Attention。输入已经是 Token 的向量表示：

```python
# B: batch size，T: 序列长度，D: 输入特征维度
x = torch.randn(B, T, D)

q = q_proj(x)  # [B, T, d_k]
k = k_proj(x)  # [B, T, d_k]
v = v_proj(x)  # [B, T, d_v]
```

这里的三个投影都是可学习的 Linear。虽然输入相同，参数并不共享，所以算出来的 Q、K、V 也不必相同。

可以先把 Q 理解成当前位置拿来发起匹配的表示，把 K 理解成各位置拿来参与匹配的表示。V 则是匹配完成后，真正参与加权汇总的内容。

三个名字对应三种计算用途。训练时，投影参数会随着下游任务的 loss 一起更新，让匹配与汇总逐渐适合当前任务。

```python
scores = q @ k.transpose(-2, -1)  # [B, T, T]
scores = scores / math.sqrt(q.size(-1))
weights = torch.softmax(scores, dim=-1)
output = weights @ v             # [B, T, d_v]
```

`scores[b, i, j]` 是第 `i` 个 Query 和第 `j` 个 Key 的匹配分数。**行对应 Query，列对应 Key**，后面看 Mask 时，这个方向很重要。

Softmax 沿最后一维做，也就是固定一个 Query，在所有可见的 Key 之间分配权重。没有 Attention Dropout 时，每一行的和为 1。

![单头Attention从Q和K的匹配开始，经过缩放、Mask和沿Key维的Softmax得到权重A，再乘V得到输出；权重与输出的形状分别为B乘T乘T和B乘T乘d_v。](/images/transformer/attention-flow.svg "Softmax 后得到的是读取各位置的权重。再乘 V，才得到这个位置的新表示。")

`weights` 和 `output` 很容易在刚开始时被叫成同一个“Attention 结果”，但它们承载的东西不同。

例如某个 Query 的权重是 `[0.2, 0.3, 0.5]`，那它得到的表示就是：

$$
o_i=0.2v_0+0.3v_1+0.5v_2
$$

权重有几个数，取决于有多少个 Key；输出有几个特征，取决于 V 的宽度。前者描述怎样读，后者是读回来后得到的表示。

### 缩放与 Softmax

把上面的过程写成公式，就是 Scaled Dot-Product Attention：

$$
\begin{gathered}
A=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)\\
\operatorname{Attention}(Q,K,V)=AV
\end{gathered}
$$

为什么要除以 \(\sqrt{d_k}\)？

先用一个简化假设看量级：如果 Q、K 各分量相互独立，均值为 0、方差为 1，点积累加 \(d_k\) 项后的方差就是 \(d_k\)。维度变大，分数的波动也会变大；Softmax 更容易把分布压得很尖，进入梯度很小的区域。除以根号维度，是把这种量级变化压回来。[原论文 §3.2.1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1)给出了这段动机。

缩放要发生在 Softmax **之前**，因为它要改变的是归一化后的相对分布：

```python
torch.softmax(scores / math.sqrt(d_k), dim=-1)
```

Softmax 是非线性的。先把分数缩小，会让归一化后的分布更平缓；先算概率再缩小，只会整体改变概率的总和。这也是两种写法得到不同结果的原因。

还要把两种 Softmax 分开记。这里的 Softmax 是在 Key 位置之间分配读取权重；语言模型最后如果要得到下一个 Token 的概率，才是在词表维度上做 Softmax。后者训练时仍然可以直接把 logits 交给 Cross-Entropy，和上一篇一样。

## 多头注意力

单头里，每个 Query 只有一套匹配分数。多头则让不同的投影子空间，各自算一套权重和加权表示，最后再合起来。

这次 `multi_head_attention.py` 的例子很小：`B=1, T=3, D=4, H=2`，每个 Head 的宽度 `d=2`。先保留这个尺寸，看清楚它究竟拆了什么。

### 特征拆分

三个投影仍然可以写成 `Linear(D, D)`：

```python
q = self.q_proj(x)  # [B, T, D]
q = q.reshape(B, T, H, d).transpose(1, 2)
# [B, T, D] -> [B, T, H, d] -> [B, H, T, d]
```

第一步 `reshape` 把**投影后的特征维**拆成 `H × d`，第二步 `transpose` 把 Head 维挪到序列维前面，方便后面按 batch、按 Head 做矩阵乘法。

每个 Head 都拥有完整的 `T` 个位置，只是每个位置的特征宽度变成了 `d`。

把大 Linear 的输出特征按 Head 分组后，每组对应权重矩阵中不同的输出参数块。它们都能读取完整的输入特征，各自得到一份投影。

![多头注意力的张量形状变化：以batch为2、序列长3、模型宽4、两个head为例，投影后拆分特征并交换维度，计算各head的context，再交换回来合并并做输出投影。](/images/transformer/multi-head-shapes.svg "图里用 B=2 把 batch 维也保留下来。Head 拆的是特征；每个 Head 都保留完整序列。")

形状按这个顺序往下走：

| 中间结果 | 形状 |
| --- | --- |
| 拆分后的 Q、K、V | `[B, H, T, d]` |
| `q @ k.transpose(-2, -1)` | `[B, H, T, T]` |
| Softmax 权重 | `[B, H, T, T]` |
| `weights @ v` | `[B, H, T, d]` |
| 合并后的 context | `[B, T, D]` |
| 输出投影后 | `[B, T, D]` |

这里缩放用的是 `sqrt(d)`，也就是**每个 Head 的 Key 宽度**。

### 合并与输出投影

算完各个 Head 后，要先把维度交换回来：

```python
context = weights @ v
context = context.transpose(1, 2).contiguous()
context = context.reshape(B, T, D)
output = self.out_proj(context)
```

这次 `transpose` 把 Token 维放回 Head 维前面，后面的 `reshape` 才能把同一个 Token 在各个 Head 上的结果拼到一起。直接 reshape 会沿当前元素次序重新分组，把 Head 和 Token 混在一起。

这里显式写 `contiguous()`，让交换维度后的数据按新次序连续存放，再合并最后两个维度。原来的多头小脚本用的是 `transpose(...).reshape(...)`，其中 `reshape` 会在需要时完成这份复制。

合并 Head 只是把结果拼起来。后面的 `out_proj` 才进一步混合各 Head 的特征。本文实现的 Q/K/V 和输出宽度都取 `D`，因此 Attention 的输入和输出可以保持同样的 shape，方便后面做残差相加。

打印 `weights[0, 0]`、`weights[0, 1]`，就能分别看到两个 Head 的权重。先看随机初始化后的状态：

固定种子为 7，关闭 Causal Mask，得到：

```text
Head 0                  Head 1
0.331  0.333  0.336      0.309  0.352  0.340
0.310  0.354  0.335      0.295  0.363  0.342
0.263  0.398  0.339      0.281  0.369  0.351
```

每一张都是“三个 Query，各自在三个 Key 之间分配权重”。两套投影给出了不同的分布，而序列长度始终是 3。

### 最小实现

把投影、拆分、计算和合并放在一起，就是下面这个版本。它沿用学习代码里“`True` 表示屏蔽”的约定，额外把两种 Mask 放在同一个接口里，便于对照：

```python
import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, *, causal=False, padding_mask=None):
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head

        q = self.q_proj(x).reshape(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, d).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

        blocked = torch.zeros_like(scores, dtype=torch.bool)
        if causal:
            future = torch.ones(T, T, device=x.device, dtype=torch.bool)
            blocked |= torch.triu(future, diagonal=1)
        if padding_mask is not None:
            if padding_mask.shape != (B, T):
                raise ValueError("padding_mask must have shape [B, T]")
            padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
            blocked |= padding_mask[:, None, None, :]

        # 这个朴素版本不接受某行所有 Key 都被屏蔽。
        if blocked.all(dim=-1).any():
            raise ValueError("A query must have at least one visible key")

        weights = torch.softmax(
            scores.masked_fill(blocked, float("-inf")), dim=-1
        )
        context = weights @ v
        context = context.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(context), weights
```

为了逐步观察权重，这里直接写出矩阵乘法和 Softmax，并保持 Dropout 关闭。上面的接口把学习文件里分开的 Causal Mask 与 Padding Mask 收在了一起，接下来分别看它们的作用。

## 位置与 Mask

位置表示和 Mask 都会影响 Attention，但它们负责的事情不同。

位置表示告诉模型：这个 Token 在哪里。Mask 则规定：这个 Query 允许读取哪些 Key。

### 位置表示

先去掉位置表示和 Mask，只看普通的全序列 Self-Attention。如果把输入 Token 的顺序一起置换，Q、K、V 也跟着置换，输出会按相同方式置换。要把“我在第 3 个位置”这条信息带进计算，还需要给输入加上位置表示。

更直观的验证在 `positional_encoding_experiment.py` 里：三个位置都使用相同的内容向量 `[1,0,1,0]`，并把 Q/K/V 投影设成恒等映射，避免投影参数干扰。

不加位置编码，三个 Key 完全一样。每个 Query 的匹配分数也一样，权重就都是：

```text
0.333  0.333  0.333
0.333  0.333  0.333
0.333  0.333  0.333
```

原始 Transformer 使用的正弦位置编码，可以写成：

$$
\begin{aligned}
PE_{(p,2i)}&=\sin\left(p/10000^{2i/D}\right)\\
PE_{(p,2i+1)}&=\cos\left(p/10000^{2i/D}\right)
\end{aligned}
$$

同一个位置在不同特征维度上使用不同频率，再把这个向量加到 Token Embedding 上。长度为 `T`、宽度为偶数 `D` 时，学习代码的写法是：

```python
position = torch.arange(T, dtype=torch.float32).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, D, 2, dtype=torch.float32)
    * (-math.log(10000.0) / D)
)
pe = torch.zeros(T, D)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
x = embeddings + pe.unsqueeze(0)
```

加完以后，即使三个位置的 Token 内容完全相同，进入 Attention 的向量也已经不同，匹配分数和权重会随之改变。把输入、分数和权重并排打印出来，位置编码怎样影响结果就很直观了。

![相同Token在不加位置编码时得到均匀的三乘三Attention权重；加入正弦位置编码后，三行权重分别发生变化。两张热力图使用相同的零到一色标，数据来自恒等投影、无Mask的实际实验。](/images/transformer/position-weights.svg "两边使用相同内容和恒等投影，对比加入正弦位置编码前后的权重。")

后面的完整 Encoder 选用**可学习的位置参数**：

```python
self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
x = self.token_emb(token_ids) + self.pos_emb[:, :T, :]
```

这两种做法都给每个位置提供一个 `D` 维向量，但一个由公式固定生成，一个由训练更新。独立实验可以分别比较，读代码时要记得当前模型选的是哪种。

### 因果约束

如果要用前面的 Token 预测下一个 Token，当前位置就不能从输入里偷看未来答案。

以长度为 4 的序列为例，保留对角线，屏蔽右上三角：

```python
future_mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool, device=x.device),
    diagonal=1,
)
scores = scores.masked_fill(future_mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
```

为什么填 `-inf`，不是 0？因为 0 仍然是合法分数，`exp(0)=1`，可能分走不少权重。把某项设为负无穷，Softmax 后它的权重才是零，前提是这一行还有可见的 Key。

保留对角线也有原因：输入位置 `i` 上放的是已知 Token，这一行负责预测它后面的 Token，所以可以读取当前位置本身。是否“未来”，要结合输入和标签的对齐方式看。

`multi_head_attention.py` 里用**同一组参数、同一份输入**，只切换 `causal`。第一个 Query 在因果模式下只有一个可见 Key，权重必然是 `[1,0,0]`；最后一个 Query 已经能看见整段序列，所以它的那一行，和关闭 Mask 时一样。

这比重新初始化两个模型再比较更清楚。否则权重变了，到底是因为 Mask，还是因为参数根本就不同，还得再分辨一次。

### 填充位置

一个 batch 中序列长度不同，可以先补齐到同样长度。Encoder 例子里，`0` 是 PAD：

```python
token_ids = torch.tensor([
    [11, 25, 70, 0, 0],
    [9,   3,  2, 4, 1],
])
padding_mask = token_ids.eq(0)  # [B, T]，True 表示 PAD
```

为了不让任何 Query 读取 PAD Key，把 `[B,T]` 的 Mask 扩展成 `[B,1,1,T]`：

```python
scores = scores.masked_fill(
    padding_mask[:, None, None, :],
    float("-inf"),
)
```

最后一维是 Key，因此屏蔽的是列。两个中间的 `1` 会广播到所有 Head、所有 Query。

![两种Mask并排对比：因果Mask只允许每行读取当前位置及之前的Key；Padding Mask屏蔽PAD对应的Key列，包含PAD Query的行仍可以读取有效Key。](/images/transformer/mask-comparison.svg "行是 Query，列是 Key。Padding Mask 屏蔽后两列，不会顺手把后两行的输出也清零。")

原来的六层 Encoder 脚本已经逐层检查了：所有 PAD Key 的权重为零，每行权重和为 1，输出全部有限。这里设置 `dropout=0.0`，所以“每行和为 1”也适用于返回的权重；若在 Softmax 后启用 Attention Dropout，单次结果就不一定满足这一点。

例如第一个样本、第一个 Head 的第一个 Query，在相同参数下对比：

```text
未屏蔽：0.143  0.190  0.247  0.209  0.211
屏蔽后：0.247  0.327  0.426  0.000  0.000
```

最后两列清零，前三列重新归一化，原先分给 PAD 的权重就回到了有效位置上。

再看 PAD 自己作为 Query 时会发生什么。

PAD 位置作为 Query，仍然可以读取前面三个有效 Key。即使 `nn.Embedding(..., padding_idx=0)` 的 PAD 行初始为零，也会加上位置向量，后面还会经过 Attention、残差和 FFN。这个 PAD 位置的隐藏状态完全可能非零。

下游做池化时，把有效位置筛出来；逐 Token 算 loss 时，让损失忽略 PAD 目标。这样就能把 Attention 中的信息读取和下游的数据使用接起来。

这个手写 Softmax 要求每行至少有一个可见 Key。整行都是 `-inf` 时会得到 NaN，所以最小实现里加了一次检查，在进入 Softmax 前直接报错。

换接口时还有一个容易弄反的约定：本文手写代码、`nn.MultiheadAttention` 的布尔 Mask 以 `True` 表示屏蔽；`F.scaled_dot_product_attention` 的布尔 `attn_mask` 以 `True` 表示允许参与。[SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)里专门说明了这一点。

## Encoder

把 Attention 放进网络，还需要残差、归一化和 FFN。`transformer_encoder.py` 这次使用的是 Post-LN：先算子层、加残差，再 LayerNorm。

### 残差与归一化

省掉 Dropout 后，一个 Block 的前向可以写成：

```python
attn_out, weights = self.attn(x, padding_mask=padding_mask)
x = self.norm1(x + attn_out)
x = self.norm2(x + self.ffn(x))
```

残差相加要求两个分支形状一致，所以 Attention 和 FFN 最后都要回到 `[B,T,D]`。它给原表示保留一条直接通路，反向时也有直接相加的梯度通路。

这里的 `nn.LayerNorm(D)` 沿最后的特征维计算均值、方差。对 `[B,T,D]` 而言，就是每个样本、每个位置，各自归一化自己的 `D` 个特征。

LayerNorm 默认还带可学习的缩放和偏移，训练与推理时都使用当前输入的统计量。[接口说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)可以对照 `normalized_shape` 来看。

![Post-LN Encoder Block横向流程：输入先经过多头Attention，与残差相加后LayerNorm，再经过逐位置的FFN，与第二条残差相加后LayerNorm，输入输出均为B乘T乘16。](/images/transformer/encoder-block.svg "这份 Encoder 是 Post-LN。后面训练用的 TinyDecoder 是 Pre-LN，读两份代码时不要把顺序记混。")

### 前馈网络

FFN 还是上一篇熟悉的 MLP：

```python
self.ffn = nn.Sequential(
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
)
```

它把每个位置的特征从 16 扩展到 64，加入非线性，再映射回 16。同一层的各位置使用同一套 FFN 参数，但每个位置分别计算，不在这里直接交换 Token 信息。

Attention 负责让位置之间读取信息，FFN 则进一步变换各位置已经聚合到的特征。两者交替执行，一个位置先从别处读取，再把读回来的信息融进自己的表示。

### 多层隐藏状态

完整 Encoder 把这个 Block 放进 `nn.ModuleList`，依次执行六次。六层结构相同，各自持有一套参数。

实验配置是 `D=16, H=4, d_ff=64, n_layers=6`，输入 `[2,5]` 的 Token ID，输出 `[2,5,16]`。每一层的 Attention 权重都是 `[2,4,5,5]`。

代码还存了进入第一个 Block 前的表示，以及每层结束后的隐藏状态，一共七份：

```python
all_hidden_states = [x.detach().clone()]
for layer in self.layers:
    x, weights = layer(x, padding_mask=padding_mask)
    all_hidden_states.append(x.detach().clone())
```

这里 `detach().clone()` 保存了一份观察快照，后续层接着使用原来的 `x`，层间的计算图也继续连着。

固定一个 Token 的位置，逐层打印它的向量、范数或与初始表示的 cosine，就能追踪这一个位置怎样经过六层网络。输入、每层输出以及对应的 Attention 权重，都可以在这里对上。

## Decoder

Encoder 拼好以后，接着看原始 Transformer 的另一半：Decoder。

原始 Encoder–Decoder 结构里，Encoder 处理源序列；Decoder 一边读取已有目标前缀，一边读取 Encoder 给出的表示，再预测下一个目标 Token。

![原始Transformer的Encoder与Decoder横向结构：源Token经过Encoder形成memory，右移目标进入Decoder的因果Self-Attention，再以Decoder状态作为Q、memory作为K和V做Cross-Attention，经过FFN输出logits。](/images/transformer/encoder-decoder.svg "先看原始 Encoder–Decoder 的关系。前面的独立实验分开验证各个模块，文末的序列反转案例再把两侧接起来。")

### 自注意力与交叉注意力

先看输入来自哪里：

| 模块 | Q 的来源 | K、V 的来源 | 读取范围 |
| --- | --- | --- | --- |
| Encoder Self-Attention | 当前层的源序列表示 | 同一份源序列表示 | 所有有效源位置 |
| Decoder Causal Self-Attention | 当前层的目标序列表示 | 同一份目标序列表示 | 当前位置及之前 |
| Cross-Attention | Decoder 当前表示 | Encoder 最终输出 | 所有有效源位置 |

Cross-Attention 继续使用同一套 Attention 计算，Q 和 K/V 分别从两份表示生成：

```python
q = self.q_proj(decoder_hidden)
k = self.k_proj(encoder_output)
v = self.v_proj(encoder_output)
```

目标长度和源长度可以不同。`cross_attention.py` 里，Decoder 输入 `[2,3,16]`，Encoder 输出 `[2,5,16]`，使用四个 Head：

```text
Q       [2, 4, 3, 4]
Kᵀ      [2, 4, 4, 5]
weights [2, 4, 3, 5]
V       [2, 4, 5, 4]
context [2, 4, 3, 4]
output  [2, 3, 16]
```

所以一般写成 `[B,H,T_dec,T_enc]`。每个目标 Query，在源序列的各个 Key 之间分配权重。输出长度跟着 Query，也就是 `T_dec`，不会因为读了五个源位置就变成五个输出位置。

源序列已经给定，每个目标 Query 都可以读取整段有效源序列。这个小脚本使用五个有效源位置；文末的变长序列案例会再加上源端 Padding Mask。

### 两份实现的区别

前面的六层 Encoder 和 Cross-Attention 分别展示了两侧表示的计算。接下来用一个 **decoder-only** 的 `TinyDecoder` 看训练：输入目标前缀，经因果 Self-Attention 和 FFN，预测下一个 Token。

它的 Block 还采用了 Pre-LN，先归一化，再进入子层：

```python
attn_out, _ = self.attn(self.norm1(x))
x = x + attn_out
x = x + self.ffn(self.norm2(x))
```

这份 FFN 用 GELU，整个 Decoder 最后还有一次 LayerNorm。前面的 Encoder 使用 Post-LN、ReLU；这里的 TinyDecoder 使用 Pre-LN、GELU。沿着各自的 `forward` 看清子层顺序，再往下接训练对齐。

## 训练

### 目标序列右移

训练时先给模型一个具体目标：根据已有前缀预测下一个 Token。

`teacher_forcing_loss.py` 用六个符号：`PAD=0, BOS=1, A=2, B=3, C=4, EOS=5`。其中 BOS 表示序列开始，EOS 表示结束。这次输入没有 PAD。

```python
sequences = torch.tensor([
    [1, 2, 3, 4, 5],  # BOS A B C EOS
    [1, 3, 2, 4, 5],  # BOS B A C EOS
])
decoder_input = sequences[:, :-1]  # [2, 4]
targets = sequences[:, 1:]        # [2, 4]
```

第一条样本对齐后是：

```text
输入：BOS  A  B  C
目标： A   B  C  EOS
```

“目标右移作为输入”说的就是这个关系。训练输入使用真实的历史 Token，这就是这里的 Teacher Forcing。

![Teacher Forcing将BOS、A、B、C输入分别对齐A、B、C、EOS标签；两条样本产生2乘4乘6的logits，展平后与8个标签计算交叉熵，再反向传播并更新参数。](/images/transformer/teacher-forcing.svg "每个输入位置预测它后面的 Token，Causal Mask 限定这一行可读取的前缀。")

为什么训练时可以一次把这四个位置全算出来？因为真实前缀已经在训练数据里，可以同时构造出所有位置的输入。Causal Mask 限定每行的读取范围，各行就可以在保留因果关系的同时一起计算。

如果只有输入、标签错开，却没有因果约束，第一个位置可能直接读到后一个输入 Token，而那个 Token 恰好就是自己的标签。loss 可能降得很好看，但预测时没有这份答案可读。

生成时，前缀会随着模型的预测逐步增长。训练一次计算整段，生成一步接一步，差别就在输入是怎样得到的。

### 损失与参数更新

TinyDecoder 使用两层 Block，`D=16, H=4, d_ff=32`，词表大小为 6。每个位置输出六个类别分数：

```python
import torch.nn.functional as F
from decoder_kv_cache import TinyDecoder

torch.manual_seed(7)
model = TinyDecoder(
    vocab_size=6, max_len=8, d_model=16,
    n_heads=4, d_ff=32, n_layers=2,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

logits = model(decoder_input)  # [2, 4, 6]
B, T, V = logits.shape
loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T),
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

把 `[2,4,6]` 展成 `[8,6]`，就是把 batch 和位置合起来，当成八个分类样本；目标则是八个整数类别。

`CrossEntropyLoss` 的这个整数标签接口把第二维作为类别维，所以这里显式展平成 `[B*T,V]`，也和上一篇的 `[N,C]` 写法接上。

Q、K、V 的投影怎样学到东西，也可以沿这条链往回看：

```text
loss ← logits ← 隐藏状态 ← Attention 输出
                             ↑
                         weights @ V
                             ↑
                     softmax(QKᵀ / √d)
```

loss 对输出的梯度，会经由加权求和传给 V 和权重，再经 Softmax、点积传给 Q、K，最后传到三组投影参数。优化器更新这些 Parameter 后，下一次前向就会用新参数重新算出 `weights`。

### 训练实验

先用 `teacher_forcing_loss.py` 跑一次完整更新，比较 step 前后的 loss、梯度和参数：

| 检查项 | 本次 CPU 实测 |
| --- | ---: |
| 更新前的 loss | `2.071973` |
| LM head 权重的梯度范数 | `1.578478` |
| LM head 权重最大绝对变化 | `0.001000002` |
| 更新后，同一 batch 的 loss | `2.010830` |

38 份参数 tensor 都有有限梯度，并且在这一步后发生了变化。前向、loss、反向和参数更新，到这里就连成了一轮。

接着用 `train_attention.py` 训练两条交替的 0/1 序列，连续跑 300 步，观察 loss、Q/K/V 梯度和 Attention 权重怎样变化。

首步 Q、K、V 投影权重的梯度范数，分别约为 `0.05931`、`0.07418`、`0.13933`。训练后 Q 权重的最大绝对变化约为 `0.17986`。这就把前面那条抽象的反向链路，落到了具体参数上。

交替序列可以用“看到 0 预测 1，看到 1 预测 0”来解决。如果想进一步观察信息怎样跨位置传递，就换回开头的首 Token 任务。

在首 Token 任务里，后四个输入 Token 都是独立随机生成的，各位置要通过 Attention 从开头读取答案。

两种模型各训练 1200 步，再各取 4096 条新随机序列评估，按位置统计准确率：

| 模型 | 位置 0 | 位置 1 | 位置 2 | 位置 3 | 位置 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 没有 Attention | 100% | 49.85% | 49.73% | 49.61% | 50.42% |
| 加入 Attention | 100% | 100% | 100% | 100% | 100% |

第一个位置两者都能答对，差距出现在后四个位置。加入 Attention 后，这几个位置也能读到开头的信息，并把答案传到自己的输出里。

还可以固定模型，修改输入较后的 Token，再比较较早位置的 logits，直接观察 Causal Mask 是否限定了信息传播的范围。

这次修改最后一个输入位置，较早三个位置的 logits 最大绝对变化是 `0`，最后位置的变化约为 `1.5271`，符合预期的依赖范围。

## 完整案例：序列反转

前面的 TinyDecoder 根据目标前缀预测下一个 Token。现在再加一份源序列：Decoder 从 BOS 开始，通过 Cross-Attention 读取 Encoder 为当前源输入生成的 memory。相同的目标前缀，就能结合不同的源内容，得到不同预测。

任务选得很简单：给一串 Token，输出它的反转序列，最后补上 EOS。输入和输出都直接用整数编号，方便逐个位置检查。

### 输入与目标

`seq2seq_transformer.py` 重新定义了一套编号：`PAD=0, BOS=1, EOS=2`，内容 Token 使用 `3..14`，词表大小为 15。这里的 EOS 是 2，和前面 Teacher Forcing 小实验里的 5 不同。

源序列长度从 1 到 5 随机生成，右侧补 PAD。例如下面这条**示意样本**：

```text
src            [4, 7, 9, 0, 0]
target_full    [1, 9, 7, 4, 2, 0, 0]
decoder_input  [1, 9, 7, 4, 2, 0]
targets        [9, 7, 4, 2, 0, 0]
```

`target_full` 最长要容纳 BOS、五个内容 Token 和 EOS，因此宽度为 7。错开一位后，Decoder 输入和标签的宽度都是 6。

注意 EOS 后面的位置。在这条长度为 3 的样本中，输入里的 EOS 和后续 PAD 对应的目标都是 PAD，后面算 loss 时会忽略它们；EOS 自己仍然是需要预测的有效标签。

![完整Seq2Seq Transformer双塔结构：左侧源序列经过Token与位置嵌入、两层Encoder，右侧右移目标经过两层Decoder。每个子层都画出残差和Add与Norm，Encoder最终memory提供Cross-Attention的K和V；Decoder栈外经过最终LayerNorm与Linear输出logits，用交叉熵训练。](/images/transformer/seq2seq-example.svg "从下往上读。Add & Norm 表示先加残差，再做 LayerNorm；两侧各重复两层。")

图的布局参考 [《Attention Is All You Need》图 1](https://arxiv.org/html/1706.03762v7#S3)，按这份实现重新绘制。位置嵌入、GELU 和 Decoder 栈外的最终 LayerNorm 都与代码对应；训练时 logits 直接交给交叉熵。

### 模型与 Mask

新模型的宽度是 `D=32`，四个 Head，FFN 中间宽度为 64，Encoder 和 Decoder **各两层**。两侧的 Block 都采用 Post-LN，FFN 使用 GELU；源、目标还分别有自己的 Token Embedding 和可学习位置 Embedding。

整个前向的入口很短：

```python
def forward(self, src, tgt_input):
    memory, src_pad_mask = self.encode(src)
    return self.decode(tgt_input, memory, src_pad_mask)
```

形状可以顺着这条链往下看：

| 数据 | 形状 |
| --- | --- |
| 源 Token ID | `[B, 5]` |
| Encoder 最终 memory | `[B, 5, 32]` |
| 右移目标 Token ID | `[B, 6]` |
| Decoder 隐藏状态 | `[B, 6, 32]` |
| 每层 Cross-Attention 权重 | `[B, 4, 6, 5]` |
| 最终 logits | `[B, 6, 15]` |

Decoder 每层都有三个子层。这次 Cross-Attention 真正进入了前向计算，接在 Causal Self-Attention 后面：

```python
attn_out, _ = self.self_attn(
    x, x, key_padding_mask=tgt_pad_mask, causal=True,
)
x = self.norm1(x + attn_out)

cross_out, _ = self.cross_attn(
    x, memory, key_padding_mask=src_pad_mask,
)
x = self.norm2(x + cross_out)
x = self.norm3(x + self.ffn(x))
```

每层的 Q 来自当前 Decoder 表示，K、V 来自同一份 Encoder 最终 memory，再由该层自己的参数分别投影。

变长数据把前面的 Mask 问题都带进来了：

| 计算位置 | 需要屏蔽什么 |
| --- | --- |
| Encoder Self-Attention | 源序列中的 PAD Key |
| Decoder Self-Attention | 未来位置，以及目标输入中的 PAD Key |
| Decoder Cross-Attention | 源序列中的 PAD Key |
| Cross-Entropy | 标签中的 PAD 位置 |

最后一项写在损失函数里：

```python
loss = F.cross_entropy(
    logits.reshape(-1, VOCAB_SIZE),
    targets.reshape(-1),
    ignore_index=PAD,
)
```

Attention Mask 决定信息可以从哪里读，`ignore_index` 决定哪些目标参与监督。两处各自处理好，变长序列就能一起组成 batch。默认取均值时，loss 在有效标签上平均。[CrossEntropyLoss 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)说明了这个约定。

这里继续通过 Key Mask 处理 PAD；各位置经过 Embedding、位置相加和子层计算后，再按各自的 Mask 参与读取。源序列至少有一个内容 Token，目标输入以 BOS 开始，每个 Query 都有可见的 Key。

### 训练与生成

训练使用 Adam，学习率 `3e-3`，batch size 为 64，跑 1000 步。每一步都会重新抽取随机长度、随机内容的源序列，反转后构造目标。

这次按原脚本固定种子 7，在 RTX 4070 Laptop 上重新运行，记录到的训练 loss 是：

| step | 当前训练 batch 的 loss，更新前 |
| ---: | ---: |
| 1 | `2.826636` |
| 100 | `0.022996` |
| 1000 | `0.000295` |

第一步反向时，92 份参数 tensor 都有有限梯度，并且在更新后发生变化。接下来用训练后的模型生成完整序列。

训练时，真实的右移目标一次性送进 Decoder。生成时，从源序列和 BOS 开始，后面的输入由模型自己接着写。

`greedy_decode()` 先编码源序列，再从 BOS 开始：

```python
memory, src_pad_mask = model.encode(src)
generated = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)

for _ in range(max_new_tokens):
    logits = model.decode(generated, memory, src_pad_mask)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    # 原脚本在这里处理已生成 EOS 的样本：之后补 PAD。
    generated = torch.cat([generated, next_token], dim=1)
    # 整个 batch 都生成 EOS 后，提前结束。
```

上面只保留主要计算；完整脚本还维护每条样本的 `finished` 状态。取的是最后一个位置的 logits，选最大分数的 Token，再把自己的预测追加到前缀。遇到 EOS 表示这条输出已经结束，不要求不同长度的样本同一轮结束。

这里 Encoder 执行一次，memory 反复使用。Decoder 则在每一步重新计算**完整目标前缀**，各层 Cross-Attention 也重新投影 memory。这些重复计算，正好是后面看 KV Cache 时可以继续追下去的地方。

### 结果与源序列扰动

训练结束后，再看完整生成序列：反转后的内容和 EOS 全部匹配，就记作一条正确输出。

训练后再抽取 64 条长度为 1 到 5 的随机序列，这次全部生成正确，包括 EOS。例如其中一条是：

```text
源序列：[8, 7, 13, 4]
期望值：[4, 13, 7, 8, 2]
生成值：[4, 13, 7, 8, 2]
```

生成值从 BOS 开始，由模型逐步预测并追加得到。

再做一组源序列扰动：固定训练好的模型和目标生成方式，只修改源输入。把每个有效 Token 在 `3..14` 之间循环加一，长度和 PAD 位置不动：

```python
changed_src[valid_mask] = (
    3 + (changed_src[valid_mask] - 3 + 1) % (VOCAB_SIZE - 3)
)
```

重新生成后，同时统计输出的变化和新目标的匹配情况：

```text
原始源序列：完整输出正确 64 / 64
修改源序列：完整输出正确 64 / 64
生成结果随源输入改变   64 / 64
```

上面那条样本循环加一以后，源序列变成 `[9,8,14,5]`，模型生成 `[5,14,8,9,2]`，也与新的反转目标一致。

再把每层权重拿出来，源 PAD Key、目标 PAD Key、未来 Key 和 Cross-Attention 的源 PAD Key 权重都为零。固定源序列，修改最后一个目标输入位置，前五个位置的 logits 保持不变。前面分开看过的信息依赖，在完整模型里也能逐层对上。

到这里，源和目标两条路径就接在了一起。Decoder 先读取已有目标前缀，再通过 Cross-Attention 读取源 memory，最后预测下一个 Token；loss 沿着这条路径传回两侧参数。

## 源码与复现

源码来自这几天的 Attention 学习实验，公开副本和新增的核验放在同一个目录。需要相互导入的文件一起打包，解压后可以直接在装有 PyTorch 的环境中运行：

- [完整实验包](../../downloads/attention-transformer/experiments.zip)
- [运行说明](../../downloads/attention-transformer/README.md)
- [本次核验结果](../../downloads/attention-transformer/results.json)

| 文件 | 对应内容 |
| --- | --- |
| `single_attention.py` | 单头、Causal Mask、权重行和 |
| `multi_head_attention.py` | 多头拆分，同参数开关 Causal Mask |
| `positional_encoding_experiment.py` | 相同内容向量，加位置编码前后对比 |
| `transformer_encoder.py` | 六层 Encoder、PAD Key 检查、隐藏状态 |
| `cross_attention.py` | 两份输入、不同序列长度的 Cross-Attention |
| `teacher_forcing_loss.py` | 右移标签、交叉熵、一次反向与更新 |
| `decoder_kv_cache.py` | 上一项依赖的 TinyDecoder，也保留缓存实验 |
| `train_attention.py`、`attention_ablation.py` | 交替序列训练、读取首 Token 的对照任务 |
| `verify_experiments.py` | 固定种子，补充 shape、Mask、参数更新等核验 |

解压后，在该目录执行：

```bash
python verify_experiments.py --output-dir ./my-results
```

复现记录使用 PyTorch `2.14.1+cu130`。补充核验使用 CPU、float32、种子 7、单线程；原 `single_attention.py` 在本次运行时自动选择了 CUDA，其余原脚本在 CPU 运行。这些实验也支持 CPU 环境。

正文的最小多头实现是合并接口后的说明版本，原实验包保留各脚本原本的结构。具体配置、矩阵和数值以结果文件及运行说明为准。

完整 Seq2Seq 案例的源码和运行记录放在这里：

- [Seq2Seq 完整实验包](../../downloads/seq2seq-transformer/experiments.zip)
- [原模型与训练脚本](../../downloads/seq2seq-transformer/seq2seq_transformer.py)
- [Seq2Seq 运行说明](../../downloads/seq2seq-transformer/README.md)
- [Seq2Seq 本次结果](../../downloads/seq2seq-transformer/results.json)

解压后可以直接运行原脚本，也可以运行附加核验，保存训练日志和结构化结果：

```bash
python -B seq2seq_transformer.py
python -B verify_seq2seq.py --output-dir ./my-seq2seq-results
```

这组结果使用相同的 PyTorch `2.14.1+cu130`，在 CUDA 上训练；脚本也会按环境选择 CPU。逐步训练 loss 保存在 [loss.csv](../../downloads/seq2seq-transformer/loss.csv)，原控制台输出保存在 [original_stdout.txt](../../downloads/seq2seq-transformer/original_stdout.txt)。

几项实验设置一并记在这里：单头、多头和六层 Encoder 的观察从随机初始化开始；TinyDecoder 的 Teacher Forcing 片段执行一次更新，使用等长样本。首 Token 对照的两个模型各自生成评估数据；Seq2Seq 的 64 条评估序列是训练后重新抽样，未与训练序列去重。具体配置和完整矩阵都保存在两份结果文件中。

## 后续学习

这次先把一条计算过程接起来：Token 变成向量，投影成 Q、K、V，经过 Attention 和 FFN 更新表示，再输出 logits，与右移的目标计算 loss。反向传播和参数更新，还是上一篇那套流程，只是中间的函数复杂了一些。

最后的序列反转案例，把之前分开看的部分接成了一个完整模型。源序列经过 Encoder，目标前缀经过 Decoder，两边在 Cross-Attention 相遇，再一起走到 loss 和参数更新。

再回头看最开始那张结构图，每条箭头终于都能在代码里找到对应的位置。生成时怎样重新读取前缀，哪些结果已经算过，也开始变成可以继续追的问题。

代码里已经开始做完整序列前向与 KV Cache 增量前向的等价检查。下一篇再接着看缓存和推理时的重复计算。这篇先把结构、训练和最小生成过程整理到这里，免得下次又从那几个 shape 开始重新想一遍。
