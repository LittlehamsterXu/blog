# Triton 线程映射实验

由 `triton_thread_mapping.py` 生成。每组求和保存 TTIR、TTGIR、PTX；`metadata.json` 记录实际运行环境和验证结果。

导出时仅移除本地源码位置/调试元数据，执行指令与布局不变。IR 中的源码位置仅保留文件名；PTX 移除 `.loc`、`.file` 和全部 `.debug_*` 段。

`summary.txt` 中的指令数是 PTX 文本里的静态条数，不能当作运行时执行次数或性能测量。布局与生成的指令会随编译器版本和目标 GPU 改变。

```text
Environment: {"python": "3.12.15", "torch": "2.14.1+cu130", "triton": "3.8.0", "cuda": "13.0", "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "compute_capability": [8, 9], "target": "GPUTarget(backend='cuda', arch=89, warp_size=32)"}

add: {"N": 1003, "BLOCK_SIZE": 128, "grid": [8], "num_warps": 4, "correct": true}

reduce128_w1: {"num_warps": 1, "result": 8128.0, "correct": true, "shared_bytes": 0, "static_instructions": {"shfl.sync": 5, "st.shared": 0, "ld.shared": 0, "bar.sync": 0}}
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

reduce128_w4: {"num_warps": 4, "result": 8128.0, "correct": true, "shared_bytes": 16, "static_instructions": {"shfl.sync": 7, "st.shared": 1, "ld.shared": 1, "bar.sync": 1}}
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```
