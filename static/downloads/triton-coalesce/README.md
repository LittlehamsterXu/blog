# Triton Coalesce 转置实验

由 `triton_coalesce.py` 生成。默认参数为 `BLOCK=32`、`num_warps=4`、一个 program、float32。验证 `Y == X.T.contiguous()`，不做性能测试。

```bash
python triton_coalesce.py --dump-dir ./triton-coalesce
```

脚本自动设置 `MLIR_ENABLE_DUMP=1` 和 `TRITON_ALWAYS_COMPILE=1`。完整编译日志仅在内存中解析，不写入此目录。

所有原始 pass 标题均为 Before：`before-*.ttgir` 取目标 pass 前的状态；`after-*.ttgir` 取紧随该 pass 的下一条 Before 快照，即目标 pass 执行后的状态。这里只观察第一次 RemoveLayoutConversions，后续同名 pass 不混入统计。具体原始标记见 `metadata.json`。

`transpose.ttir`、`transpose.ttgir`、`transpose.ptx` 为相应完整编译阶段的结果。导出时仅移除本地源码位置/调试元数据，执行指令与布局不变。IR 中的源码位置仅保留文件名；PTX 移除 `.loc`、`.file` 和全部 `.debug_*` 段。

指令数均为 PTX 文本的静态条数，不代表运行时执行次数或性能。编译器版本、目标 GPU 或参数变化，都可能改变布局、pass 流程和指令。

```text
Environment: {"python": "3.12.15", "torch": "2.14.1+cu130", "triton": "3.8.0", "cuda": "13.0", "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "compute_capability": [8, 9], "target": "GPUTarget(backend='cuda', arch=89, warp_size=32)"}
Kernel: {"BLOCK": 32, "num_warps": 4, "grid": [1], "dtype": "float32", "correct": true, "shared_bytes": 4096, "static_instructions": {"ld.global.v4": 2, "st.global.v4": 2, "shfl.sync": 0, "st.shared": 8, "ld.shared": 8, "bar.sync": 1}, "memory_opcodes": {"ld.global.v4.b32": 2, "st.shared::cta.b32": 8, "bar.sync": 1, "ld.shared.b32": 8, "st.global.v4.b32": 2}, "final_ttgir": {"convert_layout_count": 1, "blocked_layouts": ["#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>", "#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>"]}}

Pass snapshots (all source markers say Before):
before-coalesce.ttgir: convert_layout=6; captured Before TritonGPUCoalesce
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
after-coalesce.ttgir: convert_layout=10; captured Before TritonGPUF32DotTC
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
before-remove-layout-conversions.ttgir: convert_layout=10; captured Before TritonGPURemoveLayoutConversions
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
after-remove-layout-conversions.ttgir: convert_layout=1; captured Before TritonGPUOptimizeThreadLocality
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
```
