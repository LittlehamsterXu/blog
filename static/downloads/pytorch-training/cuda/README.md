# PyTorch XOR 训练实验

```bash
python pytorch_training.py --dump-dir ./pytorch-training
# 可选：使用 CUDA
python pytorch_training.py --device cuda --dump-dir ./pytorch-training-cuda
```

默认 seed=42，CPU，301 次完整 batch 更新。模型为 Linear(2,8) → Tanh → Linear(8,2)，CrossEntropyLoss 接收原始 logits；Adam 学习率 0.05。仅推理展示时计算 softmax。

`loss.csv` 每行记录该 epoch 更新前的 loss 和 backward 后的梯度范数；epoch 从 0 开始。`results.json` 的 final_loss 则是在全部更新完成后重新计算。固定 seed 不能保证跨版本、设备或平台的浮点结果完全相同。

额外 probe 对照了叶张量与非叶张量、默认不保留中间梯度、retain_grad、新图上的梯度累积、重复使用已释放图的异常，以及 SGD 的更新前后权重。最后用本模型验证 train/eval 与 grad mode 是两个独立开关；Linear 和 Tanh 在 train/eval 下的计算相同。

CUDA profiler 与 Stream 实验见独立脚本 `pytorch_mlp_profile.py`、`pytorch_cuda_execution.py`，CPU 训练不依赖它们。此目录不保存原始 profiler trace、进程 ID 或本地路径。
