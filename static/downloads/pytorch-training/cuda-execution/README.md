# CUDA 异步执行与 Stream 实验

```bash
python pytorch_cuda_execution.py --dump-dir ./pytorch-training/cuda-execution
```

异步例子记录 2048×2048 GEMM 的 Python 调用返回时间，以及同步到 GPU 完成的 CPU 时间。后者包含同步开销，不应标为纯 kernel 耗时。

依赖例子用两个 Stream 计算 c=a@b、d=c+1。wait_stream 建立执行顺序；record_stream 记录 allocator 的跨流存储使用，两者用途不同。所有 d 元素均应为 2049。

独立例子对两个 1024×1024 GEMM 比较单流与双流，每种先预热 5 次，再取 20 次中位数。start event 在主流，两工作流都等待它；主流等待两工作流后才记录 end event。这些设备 event 时间不能与 Python 提交时间混为一谈。多个 Stream 只提供并发的可能；本机此次结果不能保证换尺寸、硬件或负载后仍然更快。
