# 完整 Seq2Seq Transformer：反转序列实验

这份附件保留实际学习脚本 `seq2seq_transformer.py` 的原始内容，额外用 `verify_seq2seq.py` 运行并核验它的训练过程。没有预训练权重，也不需要下载数据集。

## 运行

需要 Python 3.10 以上和 PyTorch，记录环境为 PyTorch **2.14.1+cu130**。下载并解压 `experiments.zip`，在同一目录运行：

```bash
python -B verify_seq2seq.py --output-dir ./my-results
```

脚本按原文件入口训练 1000 步，执行原来的随机抽样评估和源序列扰动，再对**同一份训练后的模型**追加 Mask、因果依赖等检查，保存结果。`-B` 用于避免在源码旁边生成字节码缓存。

也可以只运行原来的实验：

```bash
python -B seq2seq_transformer.py
```

原文件同时固定 Python `random` 和 PyTorch 的种子为 7，自动选择 CUDA 或 CPU。记录结果来自 **NVIDIA GeForce RTX 4070 Laptop GPU**，float32，CPU 线程设为 1；其他设备或版本可能产生不同训练数值。

## 任务与模型

有效 Token ID 为 `3..14`，源序列长度随机取 `1..5`，右侧用 `PAD=0` 补齐。目标是将有效源 Token 反转，再输出 `EOS=2`；`BOS=1` 用来开始 Decoder 输入。

例如：

```text
source:       3  4  5  PAD PAD
target_full: BOS 5  4   3  EOS PAD PAD
decoder_in:  BOS 5  4   3  EOS PAD
targets:      5  4  3  EOS PAD PAD
```

这是一个带条件输入的完整 Encoder–Decoder：两个 Encoder Block、两个 Decoder Block，模型宽 32、4 个 Head、FFN 中间宽 64，词表大小 15。Decoder Block 包含 Causal Self-Attention、Cross-Attention 和 FFN。所有 Block 都是 **Post-LN**，FFN 用 GELU，位置表示可学习；这份配置不是对 2017 年原论文的逐项复刻。

源 PAD Key 在 Encoder Self-Attention 和 Decoder Cross-Attention 中屏蔽；Decoder Self-Attention 同时屏蔽目标 PAD Key 和未来 Key。交叉熵通过 `ignore_index=PAD` 排除 PAD **标签**。这三类 Attention Mask 与损失忽略 PAD 处理的是不同位置的计算，不能相互替代。

原模型的 Token Embedding 没有设置 `padding_idx`。PAD 的向量和中间隐藏状态可以非零；本次检查的是它们作为 Key 不被读取，不是强制 PAD Query 输出归零。

## 本次记录

原入口先用四个样本检查 shape：

| Tensor | 形状 |
| --- | --- |
| source | `[4,5]` |
| decoder input、targets | `[4,6]` |
| logits | `[4,6,15]` |

之后使用 Adam，学习率 `0.003`，batch size 为 64，每步重新随机生成一批数据。

| 训练步 | 当前 batch 的 loss（更新前） |
| ---: | ---: |
| 1 | `2.82663583755` |
| 100 | `0.02299645171` |
| 200 | `0.00390829798` |
| 500 | `0.00096427766` |
| 1000 | `0.00029541893` |

`loss.csv` 保存全部 1000 步。每行对应不同的随机 batch，末行不是固定验证集 loss，也不是整个训练集的汇总。

首次参数更新时，92 份参数 Tensor 都有有限梯度，并且都发生变化，合计 44,623 个标量参数；最大绝对变化为 `0.00300002098`。

训练后新抽取 64 个随机源序列，贪心生成的整条输出（包括 EOS）与目标完全一致，得到 `64/64`。例如：

```text
source:     [8, 7, 13, 4]
expected:   [4, 13, 7, 8, 2]
prediction: [4, 13, 7, 8, 2]
```

固定参数，把这 64 条源序列的每个有效 Token 循环替换为下一个 ID，保留 PAD 位置，再按新的源序列重新计算反转目标。新的整序列准确数仍是 `64/64`，64 条输出也都改变。例如：

```text
original source: [5, 7, 10, 6, 13]
modified source: [6, 8, 11, 7, 14]
new target:      [14, 7, 11, 8, 6, 2]
new prediction:  [14, 7, 11, 8, 6, 2]
```

原代码的注释把评估样本称为“未参与训练”，但这里只是**训练后新抽取的随机样本**。短序列很可能重复，未检查它们与训练序列的交集，因此不能将这 64 条称为严格去重的独立测试集。这个小任务的成功也不代表模型已经掌握真实语言任务，或能够外推到更长输入。

## 补充核验

对同一份训练后模型，使用两条固定、右侧补 PAD 的样本，通过 forward hook 直接观察各 Attention 模块返回的权重，没有改写模型的前向计算：

| Attention | 每层权重形状 | 验证 |
| --- | --- | --- |
| Encoder Self-Attention | `[2,4,5,5]` | 两层源 PAD Key 权重最大值均为 0 |
| Decoder Self-Attention | `[2,4,6,6]` | 两层目标 PAD Key、未来 Key 权重最大值均为 0 |
| Decoder Cross-Attention | `[2,4,6,5]` | 两层源 PAD Key 权重最大值均为 0 |

所有权重行和接近 1，最大误差为 `1.1920929e-7`。输出 logits 为 `[2,6,15]`，全部有限。

只修改最后一个 Decoder 输入 Token（位置 5），固定源序列及模型参数，较早五个位置的 logits 最大变化为 `0.0`，最后位置则变化 `0.57742106915`，符合因果依赖。

交叉熵的 `ignore_index=PAD` 结果为 `0.00029812852`，与手动筛出 10 个非 PAD 标签再算 loss 的结果一致。这是固定补充样本的 loss，与上表最后一个随机训练 batch 的 loss 不同。

这些输入都满足源序列非空、目标以 BOS 开始，至少存在一个可见 Key；未验证全 PAD 输入或任意左侧补 PAD。全被屏蔽的行直接进入本实现的 Softmax 会产生 NaN，不能照搬成功样本的结论。

## 附件与记录方法

- `seq2seq_transformer.py`：原脚本，SHA-256 在 `results.json` 中。
- `verify_seq2seq.py`：通过 `runpy` 执行原入口，收集 1000 步 loss 和首次优化器更新，之后补充核验。
- `results.json`：环境、参数更新、全部 64 条原始/扰动样本及输出、Attention 权重和补充检查。
- `original_stdout.txt`：原入口的真实控制台输出。
- `loss.csv`：全部训练步的更新前 loss。

记录 loss 与首次参数更新需要额外同步，会增加运行耗时；记录的墙钟时间不能当成训练性能基准。观察器不更改随机数序列、参数或优化器公式。

当前贪心生成只计算一次 Encoder memory，但每生成一步都会重新处理整个 Decoder 前缀，**尚未使用 KV Cache**。本附件关注完整 Seq2Seq 的结构、训练和输入依赖，缓存效率留待后面的实验。
