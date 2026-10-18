# Triton Layout Conversion 实验

由 `triton_layout_conversion.py` 生成。每组保存 TTGIR 和 PTX；`metadata.json` 记录实际运行环境和验证结果。

导出时仅移除本地源码位置/调试元数据，执行指令与布局不变。IR 中的源码位置仅保留文件名；PTX 移除 `.loc`、`.file` 和全部 `.debug_*` 段。

三组均把 `BlockedLayout([2], [32], [num_warps], [0])` 转为 `BlockedLayout([4], [32], [num_warps], [0])`。

`summary.txt` 中的指令数是 PTX 文本里的静态条数，不能当作运行时执行次数或性能测量。布局与生成的指令会随编译器版本和目标 GPU 改变。

```text
Environment: {"python": "3.12.15", "torch": "2.14.1+cu130", "triton": "3.8.0", "cuda": "13.0", "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "compute_capability": [8, 9], "target": "GPUTarget(backend='cuda', arch=89, warp_size=32)"}

convert_w1_n128: {"N": 128, "num_warps": 1, "correct": true, "shared_bytes": 0, "static_instructions": {"shfl.sync": 4, "st.shared": 0, "ld.shared": 0, "bar.sync": 0}}
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

convert_w4_n512: {"N": 512, "num_warps": 4, "correct": true, "shared_bytes": 2048, "static_instructions": {"shfl.sync": 0, "st.shared": 2, "ld.shared": 2, "bar.sync": 1}}
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

convert_w1_n512: {"N": 512, "num_warps": 1, "correct": true, "shared_bytes": 0, "static_instructions": {"shfl.sync": 16, "st.shared": 0, "ld.shared": 0, "bar.sync": 0}}
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
```
