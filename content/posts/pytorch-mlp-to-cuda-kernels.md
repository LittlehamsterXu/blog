---
title: "PyTorch 训练与执行基础"
date: 2026-10-22
draft: true
tags: [PyTorch, CUDA, Autograd, Profiler, AI Infra]
categories: [AI Infra]
description: "从四个 XOR 样本开始，弄清模型、梯度和参数更新，再沿着 Profiler 与 Perfetto，找到一次训练迭代里的 24 次 CUDA kernel 启动。"
summary: "以 XOR MLP 为例，整理模型与参数、损失函数、Autograd 和优化器，再用 Profiler、Perfetto 分析一次训练的 24 次 CUDA kernel 启动，接着看异步执行、Stream 依赖和 Event 计时。"
math: true
ShowToc: true
TocOpen: true
---

前面写 CUDA、CuTe 和 Triton 时，一直在看一个 kernel 里面发生了什么：线程拿哪些数，矩阵怎样分块，数据又怎样经过 shared memory。

但把视线往上挪一点，反而有不少地方还没接起来。

PyTorch 里定义一个模型，算出 loss，再写一句 `loss.backward()`。梯度就有了。接着 `optimizer.step()`，参数也变了。这几行太常见了，看起来好像没什么好问的。

直到把一个很小的 MLP 放进 Profiler，发现一次训练迭代里，GPU 居然执行了 24 次 kernel。

网络只有两层 Linear，中间一个 Tanh。那剩下的工作都是哪来的？

先从这个网络本身说起。

## 神经网络与 PyTorch 基础

XOR 只有四种输入：

| 输入 \(x_1\) | 输入 \(x_2\) | 标签 |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

两个输入不同，输出 1；相同，输出 0。把它当成二分类任务，模型最后要在类别 0 和类别 1 之间做选择。

这次用的 MLP 是 `2 → 8 → 2`：输入有两个特征，隐藏层有八个单元，输出有两个类别分数。四个样本一起放进一个 batch，所以输入 tensor 的形状是 `[4, 2]`。

```python
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 2),
).to(device)
```

所谓神经网络，在这个例子里就是一个带参数的函数。输入数据经过两次仿射变换和一次非线性变换，得到两个分数：

$$
\begin{aligned}
H&=\tanh(XW_1^{\mathsf T}+b_1)\\
Z&=HW_2^{\mathsf T}+b_2
\end{aligned}
$$

训练要做的，是根据样本和标签，逐步调整这里的 \(W_1,b_1,W_2,b_2\)。

![XOR网络的输入、隐藏层和输出层，结构为2到8到2；四个样本组成的输入形状4乘2，经过隐藏层得到4乘8，再输出4乘2的logits，与四个类别标签计算交叉熵。](/images/pytorch/training-xor-mlp.svg "2、8、2 是每个样本在各层的宽度。batch 里的四个样本，不会因此变成四套网络参数。")

### 模型与参数

`nn.Linear(2, 8)` 的权重形状是 `[8, 2]`，bias 是 `[8]`。PyTorch 存的是 `[out_features, in_features]`，所以前向公式写成 \(XW^{\mathsf T}+b\)。这里转置的是权重，别把 shape 反过来。

这两层的参数可以直接列出来：

| 参数 | 形状 | 标量数量 |
| --- | --- | ---: |
| 第一层 `weight` | `[8, 2]` | 16 |
| 第一层 `bias` | `[8]` | 8 |
| 第二层 `weight` | `[2, 8]` | 16 |
| 第二层 `bias` | `[2]` | 2 |

合起来是 **42 个标量，分在四个 Parameter tensor 里**。Tanh 没有可训练参数，输入层也不需要再额外存一套权重。

`nn.Module` 负责组织这些子模块、参数和状态。自定义模型时，在 `__init__` 里放层，在 `forward` 里描述怎样计算；这里用 `nn.Sequential`，只是因为执行顺序正好就是从前到后。

Linear 的 `weight`、`bias` 都是 `nn.Parameter`。把 Parameter 赋给 Module 的属性，会把它登记到模块参数里，因此 `model.parameters()` 能找到它，优化器也可以拿到它。

普通 tensor 即使设置了 `requires_grad=True`，也不会因为这一个标记就自动成为 Module 注册的参数。能求梯度，和会被这份优化器管理，还隔着一步。可以对照 [Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html) 和 [Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) 的接口看这里的约定。

### 激活函数

如果把 Tanh 拿掉，两次仿射变换可以合并成一次：

$$
\begin{aligned}
&(XW_1^{\mathsf T}+b_1)W_2^{\mathsf T}+b_2\\
&\quad=X(W_2W_1)^{\mathsf T}+b_1W_2^{\mathsf T}+b_2
\end{aligned}
$$

层数看起来多了一层，表达出来的仍然是仿射函数。对这四个 XOR 点，两类分在对角线上，单条线性分类边界没法把它们分开。

激活函数在这里加入非线性。Tanh 把值压到 `(-1, 1)`，ReLU 则取 `max(0, x)`。它们改变了网络能表示的函数，也改变了反向时传递梯度的方式。

这次沿用 Tanh。后面在 trace 里，还会看到它的前向和反向各自对应了一次 kernel。

### 损失函数

网络输出的 `[4, 2]` 是 logits，也就是每个样本对两个类别的原始分数。标签则是 `[4]` 的整数 tensor：

```python
y = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=device)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, y)
```

交叉熵会把正确类别的分数，与所有类别的分数一起比较。对单个样本，可以写成：

$$
\ell=-\log\frac{\exp(z_y)}{\sum_c\exp(z_c)}
$$

实际实现不需要我们先算一遍 Softmax，再把概率交给 `CrossEntropyLoss`。它直接接 logits，并以数值稳定的方式完成相应计算。这一点后面能从 `_log_softmax` 和 `nll_loss_forward` 的调用里看到。[这里使用的是整数类别标签这一种接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)。

之前的 Linear 回归小实验，用的则是 MSE：

```python
loss = ((pred - target) ** 2).mean()
```

那时预测值和目标值都是连续数值，比较的是数值差距。换成分类任务，就要重新看输出和标签的含义，不能只把损失函数的名字替换一下。

| 常见情况 | 模型输出 | 标签 | 常用损失 |
| --- | --- | --- | --- |
| 回归 | 连续预测值 `[N, D]` | 同形状浮点数 | `MSELoss` |
| 本文的二分类 | 两个 logits `[N, 2]` | 类别编号 `[N]`，`long` | `CrossEntropyLoss` |
| 单 logit 二分类 | 一个 logit `[N, 1]` | 同形状的 0/1 浮点标签 | `BCEWithLogitsLoss` |

最后一种把 Sigmoid 和 BCE 合在一起，训练时也不用在前面再加 Sigmoid。二分类用一个输出还是两个输出，都有对应的写法；这次选两个输出，是为了和已有实验保持一致。

shape 也值得顺手检查。MSE 的 `[N, 1]` 预测如果误配 `[N]` 标签，可能通过广播变成另一种计算，不能只看“程序没报错”。

## Autograd 与模型训练

![PyTorch一轮训练的全流程：输入经模型得到logits，与标签计算loss，backward产生参数梯度，优化器更新参数供下一轮使用；zero_grad清理旧梯度。下方展示相关计算从PyTorch与ATen进入CUDA实现，再经Stream提交到GPU的执行层次。](/images/pytorch/training-execution-flow.svg "上面看训练中数据、梯度和参数的关系，下面看这些操作怎样走到 GPU。zero_grad 清梯度，step 才更新参数。")

先把完整训练循环放在这里。数据就用上面四个点，种子是 42，Adam 的学习率是 `0.05`：

```python
import torch
from torch import nn

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor(
    [[0., 0.], [0., 1.], [1., 0.], [1., 1.]],
    device=device,
)
y = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=device)

model = nn.Sequential(
    nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2),
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

model.train()
for epoch in range(301):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

model.eval()
with torch.inference_mode():
    pred = model(x).argmax(dim=1)
print(pred.tolist())
```

`range(301)` 是 301 次更新，epoch 编号从 0 到 300。这里每次都使用全部四个样本，没有再划分小 batch。

看这几行时，我最想先分开的，是“计算梯度”和“改变参数”。

### 计算图与梯度

在普通 eager 执行里，前向运算会在需要求导时记录依赖关系：这个结果由哪些输入算出，反向时又该怎样把梯度传回去。下一次重新 forward，又会建立对应这次计算的求导图。

这个图描述的是链式法则。loss 先对输出求导，再经过第二层、Tanh、第一层，算出 loss 对各个参数的导数。它和后面要看的 CUDA kernel 时间线不是同一张图，也不是 CUDA Graph。

拿学习代码里那个更小的标量实验看，会清楚一些：

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

v = w * x
loss = v ** 2
loss.backward()
print(w.grad.item())  # 36.0
```

这里 \(\mathrm d\ell/\mathrm dw=2(wx)x\)，代入 `w=2`、`x=3`，就是 36。Autograd 自动完成了沿运算求导和相乘的过程。

![标量Autograd计算图：叶子w等于2、输入x等于3，相乘得到v等于6，再平方得到loss等于36；反向先得到loss对v的梯度12，再乘x得到w的梯度36。x不要求梯度，v参与求导但默认不保留grad。](/images/pytorch/training-autograd-graph.svg "前向箭头计算数值，反向箭头传递导数。参与反向传播和默认保留 .grad，是两件事。")

`w` 是需要梯度的叶子 tensor，`grad_fn` 是 `None`；`v` 和 loss 是运算产生的非叶子，可以通过 `grad_fn` 找到相应的反向节点。`x` 没有要求梯度，所以不会因为参与了计算就自动得到 `x.grad`。

非叶子仍然参与链式求导，只是默认不把自己的梯度留在 `.grad` 中。真想观察中间结果，可以在 backward 前调用 `v.retain_grad()`。它和 `retain_graph=True` 不是一回事：前者保留这个中间值的梯度，后者保留同一次前向的图及反向所需缓存，供再次反向使用。

### 梯度累积

接着重新计算一次 loss：

```python
loss2 = (w * x) ** 2
loss2.backward()
print(w.grad.item())  # 72.0

w.grad = None
loss3 = (w * x) ** 2
loss3.backward()
print(w.grad.item())  # 36.0
```

两次 forward 建的是新图，但叶子上的 `.grad` 还在。默认的 backward 会往里面累加，所以先有 `36 + 36 = 72`，清空以后才重新从 36 开始。

![同一个未更新的参数w在三次新建计算图上的梯度累积：第一次backward将grad从None写成36，第二次不清空而累加为72，设grad为None后第三次得到36。](/images/pytorch/training-gradient-accumulation.svg "三次都重新 forward，w 一直是 2。这个例子没有调用 optimizer.step，只观察梯度怎样保留下来。")

这就是训练循环每次调用 `zero_grad` 的原因。它清的是优化器所管理参数的梯度，没有清零权重，也没有替我们把所有输入 tensor 的梯度都清掉。[backward](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html) 和 [zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html) 的分工要分开看。

`set_to_none=True` 让这些 `.grad` 回到 `None`。`None` 和一个全零梯度也不完全一样：优化器通常会跳过前者，而一个零梯度参数仍可能受到历史动量或 weight decay 的影响。

### 优化器

`loss.backward()` 结束以后，参数仍然是那份参数，只是旁边多了梯度。调用 `optimizer.step()`，优化器才根据梯度和自己的状态更新它们。

最简单的 SGD 可以写成：

$$
\theta\leftarrow\theta-\eta\nabla_\theta\ell
$$

学习率 \(\eta\) 决定这一步沿负梯度方向走多远。SGD 还可以带动量；Adam 则额外维护梯度的一阶、二阶矩估计，更新时做偏差修正和缩放。后面那七个 optimizer kernel，就在处理这些状态。

AdamW 把 weight decay 与基于梯度的自适应更新解耦。不要把“优化器都叫 Adam”理解成参数完全相同：这次版本的 Adam 默认 `weight_decay=0`，AdamW 默认 `0.01`；Adam 还提供 `decoupled_weight_decay=True` 来采用 AdamW 式更新。比较时要把这些配置一起对齐。[Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html) 和 [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) 的参数表可以直接对照。

这次 XOR 选择普通 Adam，暂时没有做优化器优劣对比。复跑后，四个样本的预测是 `[0, 1, 1, 0]`。CPU 参考运行中，更新前的 loss 从第 0 轮的约 `0.7392`，降到第 50 轮的 `0.0098`、第 300 轮的 `0.000407`；301 次更新后约为 `0.000405`。

### 训练与推理模式

训练代码最后还有两行经常一起出现：

```python
model.eval()
with torch.inference_mode():
    logits = model(x)
```

它们处理的是不同的事情：

| 调用 | 改变什么 |
| --- | --- |
| `model.train()` | 切换模块的 training 状态，例如启用训练时的 Dropout、BatchNorm 行为 |
| `model.eval()` | 切换到评估行为，不自动关闭 Autograd |
| `torch.no_grad()` | 作用域中的常规运算不记录反向求导图，不替模型调用 eval |
| `torch.inference_mode()` | 进一步省去 view、version 等跟踪，适合纯推理作用域 |

本文的模型只有 Linear 和 Tanh，train/eval 不会改变它的前向公式。换成有 Dropout 或 BatchNorm 的模型，这个开关才会带来相应的行为变化。

`inference_mode` 也比 `no_grad` 多一些限制。它创建的 inference tensor，不能像普通 no-grad 结果一样随意拿回后续 Autograd 计算，尤其涉及保存给 backward 的值时。这里输出只用来做分类预测，使用它很合适。[Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html) 把模块模式和梯度模式放在一起作了区分。

## 从 PyTorch 算子到 CUDA Kernel

模型能训练以后，再把一次迭代拆开看。

这次环境是 PyTorch `2.14.1+cu130`、RTX 4070 Laptop。使用普通 eager 执行，没有 `torch.compile`，也没有显式选择 fused Adam。模型和数据都在 CUDA 上。

采样仍然使用前面四个 XOR 输入，记录的是一次完整训练 step。

### Profiler 采样

Profiler 实验会重新初始化模型和优化器，先预热五步，再只记录一步。核心写法是：

```python
from torch.profiler import profile, ProfilerActivity, record_function

def train_step():
    optimizer.zero_grad(set_to_none=True)

    with record_function("FORWARD"):
        logits = model(x)
        loss = criterion(logits, y)

    with record_function("BACKWARD"):
        loss.backward()

    with record_function("OPTIMIZER"):
        optimizer.step()

for _ in range(5):
    train_step()
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    train_step()
    torch.cuda.synchronize()
```

预热让一些首次初始化工作先发生，记录结束前的同步则保证这一步提交的 GPU 工作已经完成。这里也把模型的 forward 和 loss 一起放进 `FORWARD`，因此它的计数包含损失函数。

导出 trace 后，筛选 `cat="kernel"`、`ph="X"` 的事件，得到：

| 阶段 | CUDA kernel 启动次数 |
| --- | ---: |
| Forward，包含 loss | 5 |
| Backward | 12 |
| Adam 参数更新 | 7 |
| 合计 | 24 |

24 是这次预热后、**单次训练 step 的动态事件数**。同一个 kernel 函数可以被启动多次，所以这里也不是“24 种不同函数”。Profiler 聚合表的行数、CPU 算子数量，都不能直接拿来替代这个数字。

完整的[逐条记录](../../downloads/pytorch-training/profile/kernel_events.csv)保留了 kernel 全名、CPU 算子、输入 shape、启动关联和相对时间。正文把过长的模板符号缩短了，先看它们在做什么。

### 前向传播

这一小节先只看前向。在本次二维输入、带 bias 的 Linear 路径里，可以找到 `aten::addmm`：

```text
第一层：[4, 2] @ [2, 8] + bias[8] → [4, 8]
第二层：[4, 8] @ [8, 2] + bias[2] → [4, 2]
```

`addmm` 对应 bias 加矩阵乘。这份 trace 里，两次都交给了 cuBLAS 的 `gemmSN_TN` kernel，bias 也在该调用里完成，没有另外多出一次逐元素加法启动。

但整个 Forward 还包含 Tanh 和交叉熵：

| 序号 | CPU 侧操作 | 对应 GPU 工作 |
| ---: | --- | --- |
| 1 | `aten::addmm` | 第一层 Linear，`gemmSN_TN` |
| 2 | `aten::tanh` | Tanh 逐元素计算 |
| 3 | `aten::addmm` | 第二层 Linear，`gemmSN_TN` |
| 4 | `aten::_log_softmax` | 计算 log-softmax |
| 5 | `aten::nll_loss_forward` | 根据标签取值并归约 loss |

所以前向是五次。这也能反过来解释前面的 API：CrossEntropy 直接接 logits，内部已经做了 log-softmax 和负对数似然损失的工作。

CPU 轨道上还会出现 `aten::t`、`aten::transpose`。这次它们只是构造转置视图，调整 shape、stride 等元数据，没有为每个转置都启动一个 CUDA kernel。矩阵乘会使用这些信息读取权重。

因此，一行 Python、一个 ATen 算子、一次 kernel 启动，这三者之间没有固定的一对一关系。

![本次trace中的算子与kernel对照：转置视图操作只改元数据，对应零次GPU启动；Linear的addmm对应一次GEMM启动；权重梯度的一次mm对应GEMM和splitKreduce两次启动。完整训练step为前向5次、反向12次、Adam更新7次，共24次。](/images/pytorch/training-operator-kernels.svg "先关联 CPU 算子和 CUDA launch，再数 GPU 事件。同样一条上层操作，在本次记录里可能对应零次、一次或两次启动。")

### 反向传播

前向时看两次矩阵乘，反向时却不能照抄这两次。

设 \(G=\partial\ell/\partial Z\)，第二层要算的至少有：

$$
\begin{aligned}
\frac{\partial\ell}{\partial H}&=GW_2\\
\frac{\partial\ell}{\partial W_2}&=G^{\mathsf T}H\\
\frac{\partial\ell}{\partial b_2}&=\sum_{\text{batch}}G
\end{aligned}
$$

一份梯度往前传，一份用于权重，一份沿 batch 求和得到 bias 梯度。经过 Tanh，还要乘上对应的导数，再算第一层的权重和 bias 梯度。

这次实际的执行次序如下：

| 序号 | CPU 侧操作 | 对应 GPU 工作 |
| ---: | --- | --- |
| 6 | `aten::fill_` | 为标量 loss 准备初始梯度 1 |
| 7 | `aten::fill_` | 初始化 NLL 反向的梯度缓冲 |
| 8 | `aten::nll_loss_backward` | NLL 反向 |
| 9 | `aten::_log_softmax_backward_data` | log-softmax 反向 |
| 10 | `aten::mm` | \(\partial\ell/\partial H\)：`[4,2] @ [2,8]` |
| 11 | `aten::mm` | \(\partial\ell/\partial W_2\)：`[2,4] @ [4,8]`，GEMM 部分 |
| 12 | 同一次 `aten::mm` | `splitKreduce_kernel`，合并该 GEMM 的部分结果 |
| 13 | `aten::sum` | 第二层 bias 梯度 |
| 14 | `aten::tanh_backward` | 经过 Tanh 传回梯度 |
| 15 | `aten::mm` | \(\partial\ell/\partial W_1\)：`[8,4] @ [4,2]`，GEMM 部分 |
| 16 | 同一次 `aten::mm` | `splitKreduce_kernel`，合并该 GEMM 的部分结果 |
| 17 | `aten::sum` | 第一层 bias 梯度 |

这里最容易看走眼的是两个 `splitKreduce_kernel`。

只凭名字，会想把所有 reduce 都归到 bias 的求和。但通过 CPU 算子与 launch 的关联，它们分别属于第 11、15 项的 `aten::mm`。这两次矩阵乘选用的库实现，把计算和部分结果归并分成了两次启动。

真正的 bias 求和是第 13、17 项。kernel 都有“归约”，归约的对象和上层来源却不同。

还有两件小事。第一，标量 loss 的 backward 也要从初始梯度 1 开始，第 6 项就是它留下的工作。第二，输入 `x.requires_grad=False`，这次没有继续计算 \(\partial\ell/\partial X\) 的矩阵乘；不能给每个 Linear 无条件套上固定的反向 kernel 数。

这些 shape 能和前面的链式法则对上以后，Backward 就不只是一个很长的灰色框了。

### 参数更新

更新参数时，除了当前梯度 \(g_t\)，Adam 还维护：

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\end{aligned}
$$

对这两个状态做偏差修正，再用它们更新参数。trace 里的七次启动，刚好可以沿着这组计算读下来：

| 序号 | CPU 侧操作 | 这次承担的工作 |
| ---: | --- | --- |
| 18 | `aten::_foreach_lerp_` | 更新一阶矩 \(m\) |
| 19 | `aten::_foreach_mul_` | 对二阶矩 \(v\) 乘衰减系数 |
| 20 | `aten::_foreach_addcmul_` | 向 \(v\) 加入梯度平方项 |
| 21 | `aten::_foreach_sqrt` | 计算二阶矩的平方根 |
| 22 | `aten::_foreach_div_` | 对分母做相应偏差修正 |
| 23 | `aten::_foreach_add_` | 向分母加入 \(\epsilon\) |
| 24 | `aten::_foreach_addcdiv_` | 用一阶矩和分母更新参数 |

这里的 foreach 会对一组参数 tensor 进行处理，所以并没有把这七项再机械地乘以“四个 Parameter”。GPU 侧可以看到 `multi_tensor_apply_kernel` 一类实现。

偏差修正所用的一些标量计算则发生在 CPU 上，CPU 轨道里的每条操作也不一定都要对应一次 GPU 启动。

这次没有显式设置 `foreach`、`fused`，CUDA 上的 Adam 走到了 foreach 路径。换成其他实现、参数分组或编译后的训练步，启动数量都可能变化。24 是这份记录的结果，不能贴到所有两层 MLP 上。

### CPU 与 GPU 时间线

表格把执行顺序排清楚了，时间关系还得回到 trace 看。

下载这份已整理的 [mlp-step.trace.json](../../downloads/pytorch-training/profile/mlp-step.trace.json)，打开 [Perfetto](https://ui.perfetto.dev/)，选择 **Open trace file**，或者直接把 JSON 拖进去。它支持 PyTorch 导出的这类 Chrome JSON trace，[官方格式说明](https://perfetto.dev/docs/getting-started/other-formats)也有介绍。

可以先找 CPU 侧的 `FORWARD`、`BACKWARD`、`OPTIMIZER`，再找到 CUDA GPU 的 stream 轨道。选中一条 kernel，看它从哪次 launch 过来，再回到包住这次 launch 的 ATen 操作。

随文记录按 External id 关联 kernel 与 CPU 操作，按 correlation 关联对应的 CUDA runtime launch，并保留了 CPU 到 GPU 的 flow。原始进程、线程标识已重新编号，时间归零；这些清理不改变事件之间的依赖对应。

尤其别只拿 CPU 标签的时间框去套 GPU kernel。`record_function("BACKWARD")` 标的是 CPU 执行范围，GPU 可以在它退出以后继续完成此前提交的工作。反向也可能由 Autograd 工作线程执行，不能只盯着主 Python 线程。

Perfetto 里可以试着找第 11、12 次 kernel：它们两个都追到同一个 `aten::mm`，而不是旁边的 `aten::sum`。这样，“一个算子启动两个 kernel”就有了直接的对应关系。

![Perfetto真实界面：CPU工作线程中的aten mm及cudaLaunchKernelExC，沿flow连接到GPU上的splitKreduce；下方详情展示kernel时长和对应关联标识。](/images/pytorch/training-perfetto.png "第 12 次启动的实际记录。上面是 CPU 的算子和 launch，下面是 GPU 执行，红色箭头把两边接起来；点击图片可放大。")

这份采样为了观察 shape 和关联信息，启用了 Profiler，也会引入额外开销。适合用它解释这一步发生了什么，不能把其中一次 step 的墙钟时间直接当成模型的性能结论。[Profiler 文档](https://docs.pytorch.org/docs/2.14/profiler.html)也提醒了记录选项的开销。

## CUDA 异步执行与 Stream

### 异步提交

在时间线上继续看，会发现 CPU 和 GPU 经常没有对齐成“一条调用结束，一条计算也结束”。

CUDA 工作通常是异步提交的。Python 调用返回时，可能只是把任务放进执行队列，GPU 还没做完。先前计时实验里的两次读钟，量到的便是不同的东西：

```python
torch.cuda.synchronize()
t0 = time.perf_counter()

c = a @ b
t1 = time.perf_counter()

torch.cuda.synchronize()
t2 = time.perf_counter()
```

在这次预热后的 `2048×2048` FP32 GEMM 中，`t1-t0` 约 `0.056 ms`，`t2-t0` 约 `1.965 ms`。前者量的是调用返回所花的主机时间，后者等到了 GPU 完成，也包含同步本身的开销，并不是纯粹的 kernel 时长。

如果只量第一段，很容易觉得矩阵乘已经快到不太合理。实际上，计算还在后面跑。

### Stream 的顺序保证

Stream 可以先理解为 GPU 工作的提交队列。同一条 Stream 中，任务按顺序执行。比如先做 `c = a @ b`，再做 `d = c + 1`，只要它们在同一条流里，后面的操作就会接在前面之后。

这种顺序不要求 CPU 在两行代码中间等待 GPU。CPU 可以继续提交工作，由队列里的先后关系保证依赖。

普通训练循环大部分时候不需要每行手动 `synchronize()`。加上同步虽然能方便观察，却也会把原本可以异步提交的过程切开。打印 CUDA tensor 或调用 `.item()`，也可能因为主机需要实际结果而引入等待。

### 跨 Stream 依赖

换成两条 Stream，就要明确说明谁依赖谁。学习代码里的这个例子，第一条流计算矩阵乘，第二条流使用结果：

```python
default = torch.cuda.current_stream()
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

a = torch.ones(1024, 1024, device="cuda")
b = torch.full((1024, 1024), 2.0, device="cuda")

s1.wait_stream(default)
with torch.cuda.stream(s1):
    c = a @ b

s2.wait_stream(s1)
with torch.cuda.stream(s2):
    d = c + 1

c.record_stream(s2)
default.wait_stream(s2)
print(d[0, 0].item())  # 2049.0
```

`a`、`b` 在当前默认流上初始化，`s1` 先等这些准备工作。`s2` 要用 `c`，就等 `s1` 已提交的工作完成。最后默认流要使用 `d`，再等 `s2`。

`wait_stream` 建立的是设备端后续工作的等待关系，调用它的 CPU 线程不因此阻塞到所有计算结束。等待针对调用时，对方已经提交的工作，并不是以后永远和那条流绑定。

![default、s1、s2三条Stream之间的依赖：default准备输入，s1等待后计算c，s2等待s1后计算d，最后default等待s2使用结果；record_stream只记录存储使用。](/images/pytorch/training-stream-dependencies.svg "箭头表示执行依赖，不是等比例的计时图。这几步本身有数据依赖，不能靠分到不同 Stream 就并行完成。")

`c.record_stream(s2)` 管的是另一件事：告诉 PyTorch 的缓存显存分配器，`c` 的存储也在 `s2` 上使用，复用这块存储时要考虑那边尚未结束的访问。

它不保证 `c` 已经算好，也不能代替 `s2.wait_stream(s1)`。一边管先后顺序，一边管 allocator 对存储生命周期的跟踪，这两个问题需要分别处理。上面的例子保留了变量直到使用结束，也把跨流使用显式记下来。

### 同步与计时

如果想测设备执行时间，可以在同一条 Stream 上记录一对 CUDA Event：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
c = a @ b
end.record()

end.synchronize()
print(start.elapsed_time(end))  # 毫秒
```

记录 Event 也是往流里放一个标记。`end.synchronize()` 让 CPU 等到这个标记完成，之后才能安全地读出两者之间的时间差。

它量的是设备时间线上两个标记之间的间隔。如果中间存在等待或任务提交不及时产生的空隙，也会算进去，不能把这个间隔直接当成各条 kernel 时长的简单总和。

几种等待可以放在一起记：

| 操作 | 等待发生在哪里 | 这时在等什么 |
| --- | --- | --- |
| `stream.wait_stream(other)` | 给 stream 的后续任务建立依赖，CPU 不在此阻塞 | other 此前已提交的工作 |
| `stream.wait_event(event)` | 给 stream 的后续任务建立依赖，CPU 不在此阻塞 | 一个具体 Event |
| `event.synchronize()` | CPU 等待 | 这个 Event 完成 |
| `stream.synchronize()` | CPU 等待 | 这条 Stream 已提交的工作完成 |
| `torch.cuda.synchronize()` | CPU 等待 | 指定设备上所有流此前提交的工作完成 |

多流计时要再多想一步。如果两个独立 GEMM 分别提交到 `s1`、`s2`，只在默认流记一个 end，然后直接读时间，并没有保证 end 位于两条工作流之后。

本次实验把 start 放在主流，让两条工作流都等待它；两边提交完任务后，主流再等待它们，最后记录 end。这才把整段待测工作包进同一对 Event。

### 多 Stream 实验

另一个实验用两组互不依赖的输入，计算 `a @ b`、`x @ y`。矩阵都是 `1024×1024`，每种模式预热五次，再分别测 20 次，取 Event 时间的中位数：

| 提交方式 | 本次中位数 |
| --- | ---: |
| 同一条 Stream，依次提交 | `0.544 ms` |
| 两条 Stream，分别提交 | `0.606 ms` |

这次双流还稍慢一些。

不同 Stream 只让没有依赖的工作有机会重叠，并不保证硬件一定能同时跑，也不保证重叠后会更快。一个 GEMM 可能已经占用了很多 GPU 资源，再加一个并没有凭空增加算力；提交、同步和库的调度也都有成本。

这组数字只是这台机器、这组尺寸和本次采样的结果，不能推出“多流没用”。能留下来的经验更具体：先检查独立性和依赖，再把计时边界放对，最后看实际结果。[CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution) 里对异步执行和跨流同步的说明，可以和这个例子一起看。

## 后续学习

到这里，再回头看训练循环里的几行，终于能多接上几件事情。

这次落成的是 24 次启动：五次前向，十二次反向，七次 Adam 更新。它们有具体的 CPU 来源，也有在 Stream 上等待执行的位置。

以前看一个 kernel，会把注意力放在里面每条线程干了什么。现在开始觉得，kernel 之间那些空隙、先后关系，以及为什么要启动它，同样有不少东西值得看。

这个只有四个样本的小网络，当然解释不了一个真实模型的全部训练过程。但至少，`backward()` 和 `step()` 已经不再只是两句需要背下来的调用了。

接下来准备看 Transformer。PyTorch 的训练流程和 CUDA 执行基础，就先整理到这里。

## 源码与复现

这篇从原来的 Linear、标量求导、XOR、Profiler 和 Stream 实验整理而来，独立脚本都放在这里：

原来保存的两份 trace 都是 24 次启动，但使用随机 `[4, 2]` 输入和固定标签，只能说明这种形状的执行过程。本篇的公开记录重新换回四个 XOR 输入，独立初始化、预热五步后采样一步；不要把这一步采样和前面训练了 301 步后的状态混在一起。

| 脚本 | 运行内容 |
| --- | --- |
| [pytorch_training.py](../../downloads/pytorch_training.py) | XOR 训练、参数与梯度检查；默认可以在 CPU 运行 |
| [pytorch_mlp_profile.py](../../downloads/pytorch_mlp_profile.py) | CUDA 上预热五步、采样一步，导出脱敏 trace 和逐 kernel 关联 |
| [pytorch_cuda_execution.py](../../downloads/pytorch_cuda_execution.py) | 异步计时、跨 Stream 依赖、单双流 Event 计时 |

在安装了兼容 PyTorch 的环境中，进入下载脚本所在目录运行。后两个脚本需要可用的 NVIDIA GPU：

```bash
python pytorch_training.py --dump-dir ./training-results
python pytorch_mlp_profile.py --dump-dir ./profile-results
python pytorch_cuda_execution.py --dump-dir ./cuda-execution-results
```

训练的[结果与梯度检查](../../downloads/pytorch-training/results.json)、[CUDA 训练结果](../../downloads/pytorch-training/cuda/results.json)、[逐步 loss](../../downloads/pytorch-training/loss.csv)、[profile 配置和关联证据](../../downloads/pytorch-training/profile/profile_summary.json)、[异步与 Stream 计时](../../downloads/pytorch-training/cuda-execution/execution_results.json)都保留了实际输出。Perfetto 用的文件是 [mlp-step.trace.json](../../downloads/pytorch-training/profile/mlp-step.trace.json)。

公开记录已去除本地目录和主机信息，进程、线程及关联标识重新编号。本文的数量和时间来自 PyTorch `2.14.1+cu130`、RTX 4070 Laptop；换版本、shape、优化器实现或编译方式后，kernel 的选择和数量都可能变化。
