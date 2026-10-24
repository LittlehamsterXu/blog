# Attention 与 Transformer：实验附件

这份附件整理自学习时实际使用的九个脚本。原脚本按原内容保存，没有改写模型或训练任务；`verify_experiments.py` 是写文章时额外补的核验脚本。

## 运行

需要 Python 3.10 以上和 PyTorch。记录环境为 PyTorch **2.14.1+cu130**，没有额外数据集或预训练权重依赖。

将同一目录里的文件一起保存，或下载并解压 `experiments.zip`，然后在解压目录运行：

```bash
python verify_experiments.py
```

脚本先执行九份原始实验，再运行补充检查，将结果写入当前附件目录的 `results.json` 和 `original_stdout.txt`。也可以指定单独的输出目录：

```bash
python verify_experiments.py --output-dir ./my-results
```

`results.json` 中 `original_script_runs` 是原脚本的结果，`additional_checks` 是补充检查。原始控制台输出另存为 `original_stdout.txt`。文件里也记录了原脚本的 SHA-256，便于对照源码版本。

额外检查使用 CPU、float32、随机种子 7、单线程。原始 `single_attention.py` 会自动选择 CUDA；其他原始脚本没有迁移到 GPU。`multi_head_attention.py` 的原始入口没有固定种子，文章采用的是额外检查中明确设置种子 7 的矩阵。不同 PyTorch 版本和设备可能产生少量浮点差异。

## 文件与实验

| 文件 | 内容 |
| --- | --- |
| `single_attention.py` | 单头 Attention、Causal Mask、权重行和 |
| `multi_head_attention.py` | 拆分 Head、合并 Head、同参数对比 Causal 与 Full Attention |
| `positional_encoding_experiment.py` | 固定正弦位置编码；恒等 Q/K/V 投影；相同 Token 的位置对照 |
| `transformer_encoder.py` | 六层 Encoder、Padding Mask、逐层隐藏状态 |
| `cross_attention.py` | Decoder Query 读取 Encoder Key/Value |
| `teacher_forcing_loss.py` | 目标右移、交叉熵、一次 backward 与 Adam 更新 |
| `decoder_kv_cache.py` | 上一个脚本依赖的两层 TinyDecoder；还保留原有 KV Cache 对照入口 |
| `train_attention.py` | 两条固定交替序列，训练 300 步，观察 Q/K/V 梯度与参数变化 |
| `attention_ablation.py` | 随机二进制序列预测首 Token；比较有无 Attention 的逐位置准确率 |
| `verify_experiments.py` | 执行以上原实验，补充断言与可读的结构化结果 |

## 这次得到的结果

- `train_attention.py`：初始 loss 为 `0.6854668259620667`，300 步后的 float32 loss 为 `0.0`。首次反向时 Q/K/V 投影权重的梯度范数分别为 `0.0593109056`、`0.0741761550`、`0.1393321604`；Q 权重最大变化为 `0.1798643321`。这证明两条固定序列可以被拟合，不能据此判断模型会不会泛化。
- `attention_ablation.py`：每一步生成新的随机序列，训练 1200 步；另取 4096 条随机序列评估。没有 Attention 时，五个位置的准确率约为 `[1.0000, 0.4985, 0.4973, 0.4961, 0.5042]`；加入 Attention 后为 `[1, 1, 1, 1, 1]`。两个模型的随机数消耗不同，各自评估的随机样本并非同一批。这是一个受控的简单任务，不能外推到真实语言任务。
- 六层 Encoder 的全部 PAD Key 权重为零；每个样本仍有至少一个有效 Key，`dropout=0`，因此各行权重和为 1，输出均为有限数。Key Mask 不会把 PAD Query 的输出清零。
- Cross-Attention 的权重形状为 `[2, 4, 3, 5]`，输出形状为 `[2, 3, 16]`。
- 两层 TinyDecoder 的训练输入为 `[2, 4]`，输出 logits 为 `[2, 4, 6]`；一次更新前后 loss 为 `2.07197284698 → 2.01083016396`，LM head 梯度范数为 `1.57847821712`，权重最大变化为 `0.00100000203`。38 份参数 Tensor 均发生变化。这只核验了一次训练更新，并未训练出语言模型。
- 修改输入的最后一个 Token 后，较早三个位置的 logits 最大变化为 `0.0`，最后位置则发生变化，符合 Causal Mask 的依赖范围。

## 对照时要分清的几件事

1. 正弦位置编码是一个独立实验。六层 Encoder 和 TinyDecoder 都使用可学习的绝对位置表示。
2. Encoder 使用 **Post-LN、ReLU**；TinyDecoder 使用 **Pre-LN、GELU**。TinyDecoder 是 decoder-only 模型，没有 Cross-Attention。独立的 Cross-Attention 实验不等于已经实现并训练了完整 Encoder–Decoder。
3. 除了两个明确执行训练的实验，图中的 Attention 权重和 Encoder 隐藏状态来自随机初始化模型。Head 不同、隐藏状态发生变化，不代表模型已经学会语义。
4. 教学实现用 `masked_fill(-inf)` 再 Softmax。若一行所有 Key 都被屏蔽，会得到 NaN；全 PAD 输入不在这里验证成功的范围内。需要在真实数据处理时排除或单独处理这种情况。
5. Encoder 的 `padding_idx` 和 Key Mask 不能代替损失函数忽略 PAD。当前 teacher-forcing 两个样本等长且没有 PAD；TinyDecoder 也未实现 Padding Mask。不能直接将这个训练脚本用于任意补齐后的 batch。
6. 权重和为 1 指 Softmax 输出，或关闭 Dropout 时的返回值。Attention Dropout 后的单次采样不保证仍然逐行归一。
7. `all_hidden_states` 保存的是 `detach().clone()` 后的观察副本；它不会截断继续参与计算的 `x` 的梯度。

权重矩阵与中间结果完整保存在 `results.json`，可用于重画文章中的对照图。`experiments.zip` 包含本目录的源码、说明与本次结果；运行脚本不依赖这个压缩包。
