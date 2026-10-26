---
title: "Transformer 位置编码"
date: 2026-10-26
draft: false
tags: [PyTorch, Transformer, 位置编码, RoPE]
categories: [AI Infra]
description: "从位置向量相加开始，画出正弦编码的频率，再用二维旋转理解 RoPE，最后检查增量生成中的位置编号。"
summary: "位置怎样进入 Attention？沿着可学习位置表、正弦编码和 RoPE，看看绝对坐标怎样影响匹配，再用完整前向与缓存前向对照，理解生成时的位置为什么要接着往下数。"
math: true
ShowToc: true
TocOpen: true
---

[上一篇](../attention-transformer-basics/)把 Encoder、Decoder 和训练流程接了起来。位置编码只在中间停了一小会儿：给 Token 向量加一个位置向量，再送进 Attention。

写的时候很顺手：

```python
x = token_emb(token_ids) + pos_emb(positions)
```

可这句加法背后，其实还有不少东西可以展开。为什么位置要变成一个向量？正弦公式里那些不同的频率在做什么？到了 RoPE，位置又为什么变成了对 Q、K 的旋转？

这篇就沿着位置往下走。先看它怎样进入输入，再看它怎样影响两个 Token 的匹配，最后回到生成代码，检查位置编号怎样随前缀一起增长。

## 位置向量

### 从编号到表示

Token ID 和位置 ID 是两件事。

同一个 Token 可以出现在第 0、1、2 个位置，内容 Embedding 查到的是同一行；位置 Embedding 则根据它所在的位置，查出另一行向量。

现在的 Decoder 里，两张表都是 `nn.Embedding`：

```python
self.token_emb = nn.Embedding(vocab_size, d_model)
self.pos_emb = nn.Embedding(max_seq_len, d_model)
```

把输入形状写出来，就更清楚了：

```python
# token_ids: [B, T]
positions = torch.arange(T, device=token_ids.device)

content = self.token_emb(token_ids)        # [B, T, D]
position = self.pos_emb(positions)         # [T, D]
x = content + position[None, :, :]         # [B, T, D]
```

位置表有 `max_seq_len` 行，每行 `D` 个数。位置 3 取第 3 行，位置 4 取第 4 行；同一 batch 里，各条序列的相同位置使用相同的位置向量。

两份向量逐元素相加，主干宽度继续保持 `D`。后面的 Linear 会读取这个组合表示，训练时内容表、位置表和投影参数一起更新。

![Token ID与Position ID分别查两张Embedding表，得到同宽度的内容向量和位置向量，相加后经过Q与K投影，进入Attention分数计算。](/images/position-encoding/position-addition.svg "位置先从一个整数变成 D 维向量，再和内容一起进入投影。")

这里的位置表属于**可学习的绝对位置表示**。每个编号对应一行参数，模型在训练中调整这些参数。

### 换一个顺序

先把位置输入拿开，单独看全序列 Self-Attention。把 Token 按 `[2,0,3,1]` 换序，Q、K、V 也随之换序，输出会跟着做相同的置换。

用同一组投影参数做一次对照：

```python
permutation = torch.tensor([2, 0, 3, 1])

y = attention(x)
y_permuted = attention(x[:, permutation])
error = (y_permuted - y[:, permutation]).abs().max()
```

这次实验使用全可见、关闭 Dropout 的 Self-Attention。float64 下，两个结果的最大误差约为 `2.22e-16`，落在浮点舍入的量级。

再把每个槽位的位置向量固定住，只换 Token 内容。内容重新落入不同的位置，输出也随之改变；按同样方式比较，这次最大差异约为 `0.726824`。

这个对照把位置的作用放到了眼前：内容的顺序变了，它与位置的组合也就变了。

## 正弦位置编码

可学习位置表把每一行交给训练。原始 Transformer 还给出了一种直接由公式生成位置向量的方式：不同频率的正弦和余弦。[论文 §3.5](https://arxiv.org/html/1706.03762v7#S3.SS5)描述了这套构造。

### 每两个维度用一个频率

先把公式拆小一点。位置记为 \(p\)，位置向量宽度为偶数 \(D\)，第 \(j\) 对维度使用频率：

$$
\theta_j=10000^{-2j/D}
$$

于是这一对坐标就是：

$$
\begin{aligned}
PE_{p,2j}&=\sin(p\theta_j)\\
PE_{p,2j+1}&=\cos(p\theta_j)
\end{aligned}
$$

`2j` 和 `2j+1` 共用一个角度，一个取 sin，一个取 cos。位置增加时，这对数沿着单位圆变化。频率大的维度转得快，频率小的维度转得慢。

例如 `D=8`，一共有四对坐标：

| 特征维度 | 频率 \(\theta_j\) | 位置 \(p\) 对应的角度 |
| --- | ---: | --- |
| 0、1 | `1` | \(p\) |
| 2、3 | `0.1` | \(0.1p\) |
| 4、5 | `0.01` | \(0.01p\) |
| 6、7 | `0.001` | \(0.001p\) |

同样向后走一个位置，第一对变化比较明显，最后一对只移动一点点。把这些快慢不同的变化拼起来，就得到了这个位置的整条向量。

![D等于8时的正弦位置编码。左侧画出四种频率的sin曲线，右侧画出位置0到63对应的完整八维位置向量，偶数维为sin、奇数维为cos。](/images/position-encoding/sinusoidal-patterns.svg "左边看变化速度，右边看它们怎样交错组成 PE。两边使用同一组频率。")

公式对应的 PyTorch 写法也很短：

```python
positions = torch.arange(T, dtype=torch.float64)
theta = 10000.0 ** (
    -torch.arange(0, D, 2, dtype=torch.float64) / D
)
angles = positions[:, None] * theta[None, :]  # [T, D/2]

pe = torch.stack(
    (angles.sin(), angles.cos()), dim=-1
).flatten(-2)                               # [T, D]
```

`stack` 把每对 sin、cos 放在一起，`flatten` 再把最后两个维度展开成 `D`。拿到 `pe` 后，仍然加到内容向量上：

```python
x = token_embeddings + pe[None, :, :]
```

这套数值由位置编号和频率决定。放进 `nn.Module` 时，可以把预先生成的表注册成 buffer，让它随模型一起迁移设备；可学习的位置表则作为 Parameter 参与优化。

### 相同内容，不同位置

上一篇的小实验用了三个完全相同的内容向量 `[1,0,1,0]`，Q/K/V 都取恒等投影。

加位置编码前，每个 Query 看见三个相同的 Key，权重都是三分之一。加完以后，进入 Attention 的表示已经随位置变化，权重变成：

```text
0.333  0.405  0.261
0.221  0.426  0.352
0.149  0.369  0.482
```

沿着代码看，变化发生在 Softmax 之前：先加 PE，接着 Q/K 的数值改变，点积改变，最后分配给各个位置的权重也改变。

## 绝对位置与相对距离

绝对位置在问“你是第几个”，相对距离在问“你离我几步”。

位置 3 和位置 5 相隔两步，位置 13 和位置 15 也相隔两步。正弦编码里，恰好有一条很有用的关系：一对同频率的 sin、cos 做点积，会合成位置差的余弦。

把所有维度对加起来，就是：

$$
\begin{gathered}
\langle PE_m,PE_n\rangle\\
=\sum_j\cos((m-n)\theta_j)
\end{gathered}
$$

它来自三角恒等式 `sin(a)sin(b) + cos(a)cos(b) = cos(a-b)`。位置分别写在两个向量里，做完点积，相对距离就出现了。

用前面的 `D=8` 计算：

```text
PE(3)  · PE(5)  = 2.563718
PE(13) · PE(15) = 2.563718
```

### 位置怎样进入匹配分数

实际 Attention 还会把位置和内容组合，再经过 Q/K 投影。用列向量写一个 Token 的计算：

$$
\begin{aligned}
q_m&=W_Q(x_m+p_m)\\
k_n&=W_K(x_n+p_n)
\end{aligned}
$$

这里 \(p_m\) 就是位置向量。暂时省掉 bias，记 \(A=W_Q^{\mathsf T}W_K\)，展开点积：

$$
\begin{aligned}
q_m^{\mathsf T}k_n
&=x_m^{\mathsf T}Ax_n+x_m^{\mathsf T}Ap_n\\
&\quad+p_m^{\mathsf T}Ax_n+p_m^{\mathsf T}Ap_n
\end{aligned}
$$

四项分别对应内容与内容、内容与位置、位置与内容，以及经过投影的位置与位置。输入处的一次加法，会沿着这四条关系进入匹配分数。

固定两份内容向量和同一组投影参数，分别放到 `(3,5)`、`(13,15)`，这次实验里的原始点积分数是：

| 内容所在的位置 | Q/K 原始点积 |
| --- | ---: |
| `(3,5)` | `6.607538` |
| `(13,15)` | `11.573339` |

看上面的展开式，就能找到差异来自哪里：整体平移位置，改变了内容与位置的组合，也改变了投影后的位置项。

接下来想换一种构造：让 Q/K 先表达内容，再用位置控制它们之间的相对转角。

![位置3和5、13和15的绝对坐标不同，相对距离都为2；下方对比输入加位置向量后投影，以及内容先投影成Q和K再按位置旋转的两条计算路径。](/images/position-encoding/absolute-relative.svg "从输入处相加，走到投影后旋转。位置进入计算的地方变了。")

## 旋转位置编码

RoPE 的全称是 Rotary Position Embedding。它把位置编号变成旋转角度，在计算匹配分数之前，对每个位置的 Q、K 做旋转。[RoFormer 论文 §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2)从二维情况推导了这个构造。

### 先看二维旋转

二维向量旋转角度 \(\alpha\)，可以乘一个矩阵：

$$
R(\alpha)=
\begin{pmatrix}
\cos\alpha&-\sin\alpha\\
\sin\alpha&\cos\alpha
\end{pmatrix}
$$

假设向量是 `[a,b]`，旋转后的两个分量为：

$$
\begin{aligned}
a'&=a\cos\alpha-b\sin\alpha\\
b'&=a\sin\alpha+b\cos\alpha
\end{aligned}
$$

旋转保留向量的长度，改变的是方向。把位置 \(m\) 对应的角度设为 \(m\theta\)，记这个旋转为 \(R_m\)，就有：

$$
\begin{aligned}
q'_m&=R_mq_m\\
k'_n&=R_nk_n
\end{aligned}
$$

这里的 Q、K 已经经过内容投影。现在固定它们，看旋转怎样影响点积：

$$
\begin{aligned}
(q'_m)^{\mathsf T}k'_n
&=q_m^{\mathsf T}R_m^{\mathsf T}R_nk_n\\
&=q_m^{\mathsf T}R_{n-m}k_n
\end{aligned}
$$

Q 的转置把旋转方向反过来，两次旋转相乘，留下的是两者的角度差 \((n-m)\theta\)。位置分别作用在两个向量上，匹配时组合出了相对距离。

![固定原始Q和K时，按位置m和n分别旋转两个二维向量；共同平移位置后，相对转角保持不变，点积中的旋转由n减m决定。](/images/position-encoding/rope-rotation.svg "固定旋转前的 Q/K，把两个位置一起向后挪，相对转角保持一致。")

这次固定一组 Q、K，取第一个 Head，记录点积：

| Q、K 的位置 | RoPE 后的原始点积 |
| --- | ---: |
| `(3,5)` | `1.359859` |
| `(13,15)` | `1.359859` |
| `(3,6)` | `0.485146` |

前两行共同平移了十个位置，分数保持一致；第三行把相对距离从 2 改成 3，分数也随之变化。

### 从二维铺到每个 Head

一个 Head 的特征宽度记为偶数 \(d\)。把它按相邻维度分成二维小块：

```text
(0,1)  (2,3)  (4,5)  ...  (d-2,d-1)
```

每一对各自旋转，频率仍然按几何级数排列：

$$
\theta_j=10000^{-2j/d}
$$

这里用的是**每个 Head 的宽度 `d`**。如果 `D=32, H=4`，拆分后是 `[B,4,T,8]`，每个 Head 内有四对特征，频率就是 `[1,0.1,0.01,0.001]`。

各对旋转完成，再把坐标按原顺序拼回去，Q、K 的形状都保持 `[B,H,T,d]`。V 沿原来的路径参与加权求和：

```python
q = apply_rope(q, positions)
k = apply_rope(k, positions)

scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
scores = scores.masked_fill(blocked, float("-inf"))
weights = torch.softmax(scores, dim=-1)
output = weights @ v
```

这样看，RoPE 的插入点很明确：Q/K 投影、拆 Head 之后，矩阵点积之前。每一层都用自己的 Q/K 投影，再应用对应的位置旋转。

### PyTorch 实现

先把位置 ID 写成 `[T]`，输入按 `[B,H,T,d]` 排好。相邻配对的旋转可以直接写成：

```python
def apply_rope(x, positions):
    d = x.shape[-1]
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    theta = 10000.0 ** (-torch.arange(
        0, d, 2, device=x.device, dtype=work_dtype
    ) / d)
    positions = positions.to(device=x.device, dtype=work_dtype)
    angles = positions[:, None] * theta[None, :]  # [T, d/2]

    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)       # [1,1,T,d]
    sin = sin[None, None, :, :].to(x.dtype)

    rotated = torch.stack(
        (-x[..., 1::2], x[..., 0::2]), dim=-1
    ).flatten(-2)
    return x * cos + rotated * sin
```

`repeat_interleave(2)` 让一对特征共用同一份 cos、sin。`rotated` 把 `[a,b]` 变成 `[-b,a]`，最后一行就得到：

```text
[a,b] × cos + [-b,a] × sin
= [a cos - b sin, b cos + a sin]
```

代码正好对应前面的二维旋转公式。

核验时，先检查每对向量的范数，再检查同位置旋转的点积，以及共同平移位置后的点积。float64 下，共同平移十个位置的最大误差约为 `1.55e-15`；float32 下约为 `4.77e-7`。数值结果和推导对上了。

## 生成时的位置编号

位置表示走到生成代码里，还有一个特别容易忽略的细节：输入长度会缩短，位置编号却要接着历史往下数。

### 一个新 Token 在第几位

假设已经处理完四个 Token，位置是 `[0,1,2,3]`。接下来使用缓存，只把一个新 Token 送进模型，输入形状是 `[B,1]`，它的位置应当是 4。

再送进一个新 Token，位置就是 5。

现在的 Decoder 从缓存长度取出 `past_len`，然后生成这一轮的位置：

```python
positions = torch.arange(
    past_len,
    past_len + T,
    device=input_ids.device,
)
```

完整前向时 `past_len=0`；增量前向时，动态缓存从历史 K 的长度读取它，预分配缓存从自己的有效长度读取它。于是，同一段序列无论怎样切成几次输入，各个 Token 都能找到原来对应的位置。

![预填充四个Token使用位置0到3，缓存长度为4；后续单Token输入依次使用位置4和5。图中同时标出位置表示和因果Mask使用相同的历史偏移，并与位置重置为0的对照排列。](/images/position-encoding/decode-positions.svg "这一轮输入长度 T=1，当前位置仍然是 past_len。")

位置还会用在因果 Mask 里。当前 Query 的绝对位置是 `past_len + arange(T)`，Key 的位置从 0 排到历史加当前的总长度：

```python
q_pos = past_len + torch.arange(T, device=x.device)
k_pos = torch.arange(past_len + T, device=x.device)
blocked = k_pos[None, :] > q_pos[:, None]
```

位置表示负责选向量或转角，Mask 负责确定读取范围。两处接着同一段历史，各自完成这一轮计算。

### 把编号重置会怎样

用已有的 `TinyDecoderOnly` 做一个对照。先处理四个 Token，再一起送入接下来的两个 Token。这一轮正确的位置编号是 `[4,5]`。

把增量前向的 logits，与六个 Token 一起完整前向得到的后两个位置比较；再用同一份模型，单独把位置表查找改成 `[0,1]`，保留原来的缓存和因果 Mask：

| 计算方式 | 与完整前向相比的最大绝对差异 |
| --- | ---: |
| 增量输入，位置 `[4,5]` | `3.33e-16` |
| 增量输入，位置重置为 `[0,1]` | `1.171331` |

前一行能接回完整前向；后一行给相同内容换了位置向量，经过投影和后续各层，最终 logits 已经明显不同。

逐个 Token 输入也跑了同样的比较，正确延续位置时，最大误差约为 `3.89e-16`。

这里的 Decoder 使用的是可学习位置表。如果接入 RoPE，就在每一层为本轮的新 Q/K 使用这些连续的位置编号，旋转后再把新 K 写入缓存。历史 K 保留此前位置上的旋转结果，下一轮继续追加。

从输入处加一个向量，到点积前转动 Q/K，再到生成时延续编号，位置始终在参与同一件事：让当前内容带着所在的位置，进入后面的计算。把这些小实验摆在一起，原来那句 `token_emb + pos_emb`，也就能顺着往里再看一层了。

## 源码与复现

这篇沿用原来的正弦编码实验和 Decoder 实现，另外补了位置置换、点积展开、RoPE 以及缓存位置对照。代码与运行记录放在一起：

- [完整实验包](../../downloads/position-encoding/experiments.zip)
- [运行说明](../../downloads/position-encoding/README.md)
- [本次结果](../../downloads/position-encoding/results.json)
- [位置编码实现](../../downloads/position-encoding/position_encoding.py)

解压后，在已安装 PyTorch 的环境中运行：

```bash
python -B verify_positions.py --output-dir ./my-results
```

复现使用 PyTorch `2.14.1+cu130`、CPU、单线程。原来的三个相同 Token 实验保留 float32，新增数学核验主要用 float64；RoPE 另外记录了一组 float32 误差。模型从固定种子初始化，完整脚本中的配置、矩阵和参数都写进了结果文件。

正文的 RoPE 按相邻维度配对；附件接口还支持 `[B,T]` 的位置 ID。位置重置对照通过 hook 改变位置 Embedding 的查表编号，因果 Mask 保留实际历史偏移。原 Decoder 文件作为模型定义导入，复现运行的是这里列出的实验。
