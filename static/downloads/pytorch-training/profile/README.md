# 一次 MLP 训练步的 CUDA 记录

```bash
python pytorch_mlp_profile.py --dump-dir ./pytorch-training/profile
```

默认为 seed=42 的四点 XOR、5 步预热后记录 1 步。`--input random` 改为随机输入；它不能替代四点 XOR 的收敛验证。

将 `mlp-step.trace.json` 导入 Perfetto 可查看 CPU 算子、CUDA runtime 与 GPU kernel。每条 CPU launch → GPU flow 都由 correlation 关联，并用 External id 交叉验证 CPU 父操作。阶段按匹配 CPU 操作所属的 CPU record_function 区间归类，允许 autograd 工作线程；不根据 GPU 开始时间猜阶段。CSV 包含输入 shape 和同样的关联证据。

导出仅保留这些事件及少量参数；PID/TID、External id、correlation 已重编号，时间从第一条保留事件起归零。主机信息、真实进程号、本地路径、GPU UUID 和原始 trace 不发布。此记录是当前环境的单步动态事件，不是普遍固定的 kernel 数，也不作为性能基准。
