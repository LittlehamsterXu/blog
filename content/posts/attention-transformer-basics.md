---
title: "Attention 与 Transformer 基础"
date: 2026-10-24
draft: false
tags: [PyTorch, Attention, Transformer, 深度学习]
categories: [AI Infra]
description: "从一个读取序列首 Token 的小实验开始，用 PyTorch 拆开 Attention、多头、位置与 Mask，再拼出 Encoder，走到 Decoder 的一次训练。"
summary: "公式不长，真正写起来却有不少地方需要停一下：Q、K、V 从哪来，Head 拆的是哪一维，PAD Key 被屏蔽后为什么还有输出，以及训练时为什么能一次预测整段序列。"
math: true
ShowToc: true
TocOpen: true
---

[上一篇](../pytorch-mlp-to-cuda-kernels/)用一个 XOR MLP，把模型、梯度和参数更新接了起来。最后写到，接下来准备看 Transformer。

这几天就从 Attention 开始，写了几个很小的 PyTorch 实验。

刚看公式时，感觉它好像也不复杂。两次矩阵乘法，中间放一个 Softmax，再加上几个 Linear。可真正把每一步打印出来，又会冒出一些很具体的问题。

Q、K、V 都从同一份输入算出来，为什么还需要三套投影？多头里的“拆分”，到底是在拆 Token，还是在拆特征？Mask 已经把 PAD 屏蔽了，怎么 PAD 的输出还不是零？

这些地方如果只记住一张结构图，过几天大概还是会混。所以这次继续把例子缩小，先看数据怎样走，再看模块怎样拼起来，最后跑一次训练。

## Attention

### 不同位置怎样交换信息

先不用句子，也先不讨论一个词应该关注另一个词的什么含义。

随机生成一条长度为 5 的 0/1 序列，让每个位置都预测这条序列的第一个 Token：

```text
输入：0 1 1 0 1       目标：0 0 0 0 0
输入：1 0 1 1 0       目标：1 1 1 1 1
```

如果模型只有 Token Embedding、位置向量和逐位置的 Linear，那么第 4 个位置拿到的，只是自己的 Token 和自己的位置。即使知道这里是第 4 个位置，也不知道这一条随机序列开头是 0 还是 1。

第一个位置当然能答对，因为答案就在它自己的输入里。后四个位置需要从别处拿信息。

Attention 在这里提供的，就是这条跨位置的读取通路。某个位置可以根据当前表示，给其他位置分配权重，再把那些位置的信息取回来。

这也是 `attention_ablation.py` 比较有意思的地方。它没有只训练两条固定序列，而是每一步都重新生成随机样本，评估时也换一批。后面会回到这个实验，看训练结果是否真的支持这个解释。

Transformer 把这种跨位置交互放进了网络的基本模块里。训练时，一层里各位置的 Attention 可以组织成矩阵运算；不必为了得到第 5 个位置的这一层输出，先依次算完第 1、2、3、4 个位置的同层输出。代价也很直观：下面这个朴素实现会显式构造一张随序列长度平方增长的分数矩阵。

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

这些名字是在描述它们在计算里的用途。初始的随机参数并不知道哪个词是主语，也没有人为给某个维度写上“语义”的标签。

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

权重有几个数，取决于有多少个 Key；输出有几个特征，取决于 V 的宽度。即使碰巧 `T == d_v`、两者 shape 相同，也不能因此把它们当成同一种东西。

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

这里并不是保证实际训练中的 Q、K 永远满足这个分布，只是解释这个缩放因子从哪里来。

缩放要发生在 Softmax **之前**，因为它要改变的是归一化后的相对分布：

```python
torch.softmax(scores / math.sqrt(d_k), dim=-1)
```

如果先 Softmax，再把结果除以 `sqrt(d_k)`，得到的是另一种计算，连每行和为 1 都不再满足。Softmax 是非线性的，不能把这个除法随手挪出去。

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

每个 Head 仍然拥有全部 `T` 个位置。不是第一个 Head 看前半句话，第二个 Head 看后半句话。

一个大 Linear 也不表示各 Head 共用同一组投影结果。把输出特征按 Head 分组后，每组对应的是权重矩阵中不同的输出参数块；它们都能读取完整的输入特征。

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

这里缩放用的是 `sqrt(d)`，也就是**每个 Head 的 Key 宽度**，不是整个 `D`。

### 合并与输出投影

算完各个 Head 后，要先把维度交换回来：

```python
context = weights @ v
context = context.transpose(1, 2).contiguous()
context = context.reshape(B, T, D)
output = self.out_proj(context)
```

不能从 `[B,H,T,d]` 直接 `reshape(B,T,D)`，那样只是按当前元素次序重新分组，Head 和 Token 会混在错误的位置上。

这里显式写 `contiguous()`，是让交换维度后的数据按新次序连续存放，再合并最后两个维度。原来的多头小脚本用了 `transpose(...).reshape(...)`，同样可以得到正确结果，因为 `reshape` 必要时会复制；它不保证永远只是改一下元数据。

合并 Head 只是把结果拼起来。后面的 `out_proj` 才进一步混合各 Head 的特征。本文实现的 Q/K/V 和输出宽度都取 `D`，因此 Attention 的输入和输出可以保持同样的 shape，方便后面做残差相加。

观察不同 Head 时，打印 `weights[0, 0]`、`weights[0, 1]` 就能分别看到两张矩阵。随机初始化时它们通常已经不同，但这只能说明计算不同，还不能说某个 Head 已经学会了语法或指代。

补充核验固定种子为 7，关闭 Causal Mask 后，这两个未训练 Head 的权重是：

```text
Head 0                  Head 1
0.331  0.333  0.336      0.309  0.352  0.340
0.310  0.354  0.335      0.295  0.363  0.342
0.263  0.398  0.339      0.281  0.369  0.351
```

每一张仍然是“三个 Query，各自在三个 Key 之间分配权重”。两张矩阵不同，输入长度却没有被切成两份。

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

这里为了观察权重没有加入 Dropout，也没有使用融合 Attention 接口。原学习文件中，带 Causal Mask 的多头版本和带 Padding Mask 的 Encoder 版本是分开的；上面把公共部分收在一起，不影响接下来分别看它们的作用。

## 位置与 Mask

位置表示和 Mask 都会影响 Attention，但它们负责的事情不同。

位置表示告诉模型：这个 Token 在哪里。Mask 则规定：这个 Query 允许读取哪些 Key。

### 位置表示

先去掉位置表示和 Mask，只看普通的全序列 Self-Attention。如果把输入 Token 的顺序一起置换，Q、K、V 也跟着置换，输出会按相同方式置换。这种计算本身没有额外给一个 Token 标出“我是第 3 个位置”。

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

加完以后，即使三个位置的 Token 内容完全相同，它们进入 Attention 的向量也已经不同，匹配分数和权重会随之改变。这里验证的是“位置进入了计算”，还没有训练一个模型去理解这些差异。

![相同Token在不加位置编码时得到均匀的三乘三Attention权重；加入正弦位置编码后，三行权重分别发生变化。两张热力图使用相同的零到一色标，数据来自恒等投影、无Mask的实际实验。](/images/transformer/position-weights.svg "这张图只改变了位置输入。两边均没有训练，也没有换投影参数。")

后面的完整 Encoder 使用的是**可学习的位置参数**，不是这份正弦编码：

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

最后两列清零，前三列重新归一化，而不是把后两项删掉后保留旧权重。

但 **PAD Key 权重为零，不等于 PAD Query 的输出为零**。

PAD 位置作为 Query，仍然可以读取前面三个有效 Key。即使 `nn.Embedding(..., padding_idx=0)` 的 PAD 行初始为零，也会加上位置向量，后面还会经过 Attention、残差和 FFN。这个 PAD 位置的隐藏状态完全可能非零。

如果下游要做池化，就要排除这些位置；如果要逐 Token 算 loss，就需要让损失忽略相应目标。这里没有必要为了“看起来像 PAD”，强求每一层的那几行都自动归零。

还要留意全被屏蔽的行。这个手写版本直接对 `[-inf,-inf,...]` 做 Softmax 会得到 NaN。当前两个样本都有有效 Key，不触发这个问题；前面合并后的最小实现则显式拒绝这种输入。不能为了屏蔽 PAD Query，把整行填成 `-inf` 后就不再管它。

最后记一个接口差异：本文手写代码、`nn.MultiheadAttention` 的布尔 Mask 都以 `True` 表示屏蔽；`F.scaled_dot_product_attention` 的布尔 `attn_mask` 却以 `True` 表示允许参与。换接口时要重新核对，不能直接照搬。[SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)明确说明了这个相反的约定。

## Encoder

把 Attention 放进网络，还需要残差、归一化和 FFN。`transformer_encoder.py` 这次使用的是 Post-LN：先算子层、加残差，再 LayerNorm。

### 残差与归一化

省掉 Dropout 后，一个 Block 的前向可以写成：

```python
attn_out, weights = self.attn(x, padding_mask=padding_mask)
x = self.norm1(x + attn_out)
x = self.norm2(x + self.ffn(x))
```

残差相加要求两个分支形状一致，所以 Attention 和 FFN 最后都要回到 `[B,T,D]`。它给原表示保留一条直接通路，反向时也有直接相加的梯度通路；这不表示堆多深都不用关心训练稳定性。

这里的 `nn.LayerNorm(D)` 沿最后的特征维计算均值、方差。对 `[B,T,D]` 而言，是每个样本、每个位置，各自归一化自己的 `D` 个特征，不是把整个 batch 或整句 Token 混在一起求统计量。

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

Attention 负责让位置之间读取信息，FFN 则进一步变换各位置已经聚合到的特征。经过 Attention 后，一个位置的表示已经包含其他位置的信息，因此“FFN 逐位置”也不意味着它处理的仍然只是最初那个孤立 Token。

### 多层隐藏状态

完整 Encoder 把这个 Block 放进 `nn.ModuleList`，依次执行六次。每一层都有自己的参数，结构相同不代表共享权重。

实验配置是 `D=16, H=4, d_ff=64, n_layers=6`，输入 `[2,5]` 的 Token ID，输出 `[2,5,16]`。每一层的 Attention 权重都是 `[2,4,5,5]`。

代码还存了进入第一个 Block 前的表示，以及每层结束后的隐藏状态，一共七份：

```python
all_hidden_states = [x.detach().clone()]
for layer in self.layers:
    x, weights = layer(x, padding_mask=padding_mask)
    all_hidden_states.append(x.detach().clone())
```

这里 `detach().clone()` 只用于保存观察快照，后续层接着使用的仍然是原来的 `x`，没有把层间的梯度链断开。如果把 `x = x.detach()` 放进循环，那才是另一回事。

固定一个 Token 的位置，逐层打印它的向量、范数或与初始表示的 cosine，可以看到表示确实在变化。不过当前 Encoder 还没有经过任务训练，这些变化只能说明计算发生了，不能直接解释成“层越深，语义越高级”。

## Decoder

到这里已经拼出了 Encoder，但原始 Transformer 的另一半还没有接上。

原始 Encoder–Decoder 结构里，Encoder 处理源序列；Decoder 一边读取已有目标前缀，一边读取 Encoder 给出的表示，再预测下一个目标 Token。

![原始Transformer的Encoder与Decoder横向结构：源Token经过Encoder形成memory，右移目标进入Decoder的因果Self-Attention，再以Decoder状态作为Q、memory作为K和V做Cross-Attention，经过FFN输出logits。](/images/transformer/encoder-decoder.svg "这是原始 Encoder–Decoder 的关系图。这次分别实现了 Encoder 和 Cross-Attention；训练实验使用的 TinyDecoder 没有中间的 Cross-Attention。")

### 自注意力与交叉注意力

先看输入来自哪里：

| 模块 | Q 的来源 | K、V 的来源 | 读取范围 |
| --- | --- | --- | --- |
| Encoder Self-Attention | 当前层的源序列表示 | 同一份源序列表示 | 所有有效源位置 |
| Decoder Causal Self-Attention | 当前层的目标序列表示 | 同一份目标序列表示 | 当前位置及之前 |
| Cross-Attention | Decoder 当前表示 | Encoder 最终输出 | 所有有效源位置 |

Cross-Attention 没有换掉 Attention 公式，只是 Q 和 K/V 来自两份表示：

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

这里不需要按目标位置去屏蔽“后面的源 Token”，因为源序列已经给定；如果源序列含 PAD，仍然需要屏蔽那些源 Key。当前独立脚本的五个源位置都有效，因此还没有加入源端 Padding Mask。

### 当前实现的范围

原始结构可以顺着[论文 §3](https://arxiv.org/html/1706.03762v7#S3)看，但这次代码没有把它完整训练成一个翻译模型。

已经连起来的是六层 Encoder；Cross-Attention 单独跑通了输入来源和 shape。接下来训练使用的 `TinyDecoder` 是另一份 **decoder-only** 模型，没有 Encoder，也没有 Cross-Attention。

它的 Block 还采用了 Pre-LN，先归一化，再进入子层：

```python
attn_out, _ = self.attn(self.norm1(x))
x = x + attn_out
x = x + self.ffn(self.norm2(x))
```

这份 FFN 用 GELU，整个 Decoder 最后还有一次 LayerNorm。前面的 Encoder 则是 Post-LN、ReLU。两份代码都可以用来理解结构，但不能一边展示前者的图，一边把后者的输出当成原始 Encoder–Decoder 已经训练完成的证据。

## 训练

### 目标序列右移

训练语言模型时，不需要单独给每个 Token 写一句“应该关注第几位”。先给它一个下游目标：根据已有前缀预测下一个 Token。

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

“目标右移作为输入”说的就是这个关系。输入使用真实的历史 Token，而不是模型上一位置自己猜出来的 Token，这就是这里的 Teacher Forcing。

![Teacher Forcing将BOS、A、B、C输入分别对齐A、B、C、EOS标签；两条样本产生2乘4乘6的logits，展平后与8个标签计算交叉熵，再反向传播并更新参数。](/images/transformer/teacher-forcing.svg "每个输入位置预测它后面的 Token。输入右移与 Causal Mask 要一起看，才能保证没有偷看答案。")

为什么训练时可以一次把这四个位置全算出来？因为真实前缀已经在训练数据里，可以同时构造出所有位置的输入。Causal Mask 负责限制每行只能读取到哪里，所以矩阵一起算，不代表当前位置能够读取未来内容。

如果只有输入、标签错开，却没有因果约束，第一个位置可能直接读到后一个输入 Token，而那个 Token 恰好就是自己的标签。loss 可能降得很好看，但预测时没有这份答案可读。

推理时则不同：下一个 Token 尚未产生，不能预先把真实未来放进输入。这也是并行训练和逐步生成看起来不同的原因。

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

把 `[2,4,6]` 展成 `[8,6]`，只是把 batch 和位置合起来，当成八个分类样本；目标则是八个整数类别。展平没有让原来不同位置之间多出一种新的依赖。

不要直接把 `[B,T,V]` 原样交给这个整数标签接口，误以为它默认把最后一维当成类别维。这里显式展平，是为了对齐 `CrossEntropyLoss` 的类别维约定，也和上一篇的 `[N,C]` 写法接上。

Q、K、V 的投影怎样学到东西，也可以沿这条链往回看：

```text
loss ← logits ← 隐藏状态 ← Attention 输出
                             ↑
                         weights @ V
                             ↑
                     softmax(QKᵀ / √d)
```

loss 对输出的梯度，会经由加权求和传给 V 和权重，再经 Softmax、点积传给 Q、K，最后传到三组投影参数。`weights` 是当前输入、当前参数计算出的中间结果；优化器更新的是投影矩阵等 Parameter，不是把这张权重表存起来逐项更新。

某个具体样本、某个被 Mask 的位置，梯度当然可能为零。这不妨碍用实际训练中的投影梯度，检查整条计算链有没有连通。

### 用实验检查结论

这次把 `teacher_forcing_loss.py` 按原配置重新跑了一遍，另外保存 step 前后的参数做比较。它只执行一次更新，所以首先检查的是 shape、有限的 loss、非零梯度，以及参数确实发生变化。

| 检查项 | 本次 CPU 实测 |
| --- | ---: |
| 更新前的 loss | `2.071973` |
| LM head 权重的梯度范数 | `1.578478` |
| LM head 权重最大绝对变化 | `0.001000002` |
| 更新后，同一 batch 的 loss | `2.010830` |

38 份参数 tensor 都有有限梯度，并且在这一步后发生了变化。当前数据没有 PAD，TinyDecoder 也还没有实现 Padding Mask；前面 Encoder 的 PAD 验证不能直接当成这份训练脚本也支持任意补齐 batch 的证据。

这里还有一个小限制：两条样本都以 BOS 开始，但第一个目标分别是 A 和 B。在没有额外条件输入的 decoder-only 模型里，完全一样的 BOS 前缀对应两个不同答案，本来就不能指望两个位置都预测成确定的正确类别。

因此，一次 `optimizer.step()` 成功，只说明训练链路跑通了，不表示模型已经学会生成这两条序列。

另一份 `train_attention.py` 确实训练了 300 步，任务是两条交替的 0/1 序列。它可以用来观察 loss 下降、Q/K/V 的非零梯度，以及 Q 投影参数和 Attention 权重随训练改变。

首步 Q、K、V 投影权重的梯度范数，分别约为 `0.05931`、`0.07418`、`0.13933`。训练后 Q 权重的最大绝对变化约为 `0.17986`。这就把前面那条抽象的反向链路，落到了具体参数上。

但交替序列还有一个取巧办法：看到 0 预测 1，看到 1 预测 0。即使不读取很远的位置，也可能做好这个任务。所以只凭它训练成功，不能证明 Attention 已经学到了长距离依赖。

再回到开头的“各位置预测首 Token”实验，情况就清楚一些。后四个输入 Token 是独立随机生成的，模型如果只看当前位置，没法知道首 Token；加入 Attention 后才有读取它的通路。

两种模型各训练 1200 步，再各取 4096 条新随机序列评估，按位置统计准确率：

| 模型 | 位置 0 | 位置 1 | 位置 2 | 位置 3 | 位置 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 没有 Attention | 100% | 49.85% | 49.73% | 49.61% | 50.42% |
| 加入 Attention | 100% | 100% | 100% | 100% | 100% |

两个模型各自生成评估样本，不是完全相同的一批；这里展示这一次运行的结果。第一个位置两者都能答对，真正拉开差距的是需要读取别处的后四个位置。

这里的对比验证的是这个具体任务需要跨位置信息，以及这份带 Attention 的模型能够学会使用它。它没有验证真实语言理解，也不意味着每个 Head 都会学出人能命名的分工。

为了检查因果约束，复现脚本还增加了一组对照：固定模型，只修改输入较后的 Token，再比较较早位置的 logits。带 Causal Mask 时，较早输出应保持不变。这样检查的是模型实际算出来的依赖，而不只是 Mask 的图案看起来对不对。

这次修改最后一个输入位置，较早三个位置的 logits 最大绝对变化是 `0`，最后位置的变化约为 `1.5271`，符合预期的依赖范围。

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

复现记录使用 PyTorch `2.14.1+cu130`。补充核验使用 CPU、float32、种子 7、单线程；原 `single_attention.py` 在本次运行时自动选择了 CUDA，其余原脚本在 CPU 运行。所有实验也都能在没有 CUDA 的环境里执行。

正文的最小多头实现是合并接口后的说明版本，原实验包保留各脚本原本的结构。具体配置、矩阵和数值以结果文件及运行说明为准。

## 后续学习

这次先把一条计算过程接起来：Token 变成向量，投影成 Q、K、V，经过 Attention 和 FFN 更新表示，再输出 logits，与右移的目标计算 loss。反向传播和参数更新，还是上一篇那套流程，只是中间的函数复杂了一些。

有些结论现在可以直接打开代码验证。Head 没有切断序列，PAD Key 的权重确实归零，Cross-Attention 的输出长度跟着 Query，Teacher Forcing 也没有取消因果约束。

还有些地方，只看这些小实验远远不够。随机模型的权重矩阵不能直接解释成语义；一次更新不能当成训练收敛；单独跑通 Encoder 和 Cross-Attention，也不等于完整的 Encoder–Decoder 已经训练好了。

代码里已经开始做完整序列前向与 KV Cache 增量前向的等价检查。下一篇再接着看自回归生成、缓存和推理时的重复计算。这篇先把结构与训练整理到这里，免得下次又从那几个 shape 开始重新想一遍。
