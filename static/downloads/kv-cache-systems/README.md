# KV Cache：从显存管理到高效推理系统

这份附件包含五个原始学习脚本、一次运行的输出，以及便于对照文章的结构化记录。五个原脚本保留原始字节，校验值见 `simulation_sources.json`、`gpu_sources.json`。

## 文件

| 文件 | 内容 |
| --- | --- |
| `paged_kv_sim.py` | 固定大小块的分配、追加、块表和槽位统计 |
| `prefix_cache_sim.py` | 完整前缀块的哈希匹配与复用计数 |
| `continuous_batching_sim.py` | 静态与连续组批的逐轮状态和请求指标 |
| `kv_cache_quant.py` | FP32、FP16、逐向量 INT8 的存储与重建误差 |
| `kv_cache_offload.py` | GPU 与 pinned CPU 之间复制、释放引用、恢复和一致性检查 |
| `record_simulations.py` | 运行前三个原脚本并记录 JSON、控制台输出和源码哈希 |
| `simulation_results.json`、`simulation_stdout.txt` | 本次三个模拟器的状态与原始输出 |
| `gpu_results.json`、`gpu_stdout.txt` | 本次 CUDA 实验的统计与原始输出 |
| `simulation_sources.json`、`gpu_sources.json` | 五个原脚本的文件名、字节数、SHA-256 |

## 运行

前三个脚本只依赖 Python 标准库。后两个需要 PyTorch；量化脚本自动选择 CUDA 或 CPU，卸载脚本需要 CUDA。

在解压目录执行：

```bash
python -B paged_kv_sim.py
python -B prefix_cache_sim.py
python -B continuous_batching_sim.py
python -B kv_cache_quant.py
python -B kv_cache_offload.py
```

要重新记录三个模拟器的结构化结果，可运行附加脚本：

```bash
python -B record_simulations.py --output-dir ./my-results
```

它导入原脚本、调用原来的 `main()`，通过附加记录接口收集块表、命中、时间线与指标。结果写入指定目录，包含对应原源码副本。CUDA 实验直接运行两个原脚本即可查看重建误差、存储字节、显存读数和复制时间；JSON 还保留了采集过程记录的完整数值。

## 本次设置

- Python 3.12.15、PyTorch 2.14.1+cu130、CUDA 13.0，WSL Linux。
- NVIDIA GeForce RTX 4070 Laptop GPU，8 GiB 级显存；PyTorch 报告的可见容量保存在 JSON。
- 两个 CUDA Tensor 实验固定随机种子 42。
- 分页：块大小 4，池内 32 块，固定预留对照长度 32；A/B/C/D 的提示词长度为 8/19/5/30，分别追加 3/5/2/0 个位置。
- 前缀：两条手写字符串 Token 序列，共同前缀为 8 个位置，只登记完整块；字符串是抽象 Token。
- 调度：最大批次 3，活动请求每轮推进一个 Token。到达时刻 0/1/2/4/5，对应生成长度 8/5/3/6/2。
- 量化：K/V 各 `[1,4,1024,64]`，原值 FP32。沿最后一维每 64 个元素保存一个 FP32 scale；对称 INT8 范围 `[-127,127]`，零点为 0。
- 卸载：4 个独立 FP16 Tensor，各 `[2,1,8,2048,128]`，每块 8 MiB。第 0 维表示 K/V。卸载块 2、3，再恢复。
- 复制：目标先分配，pinned CPU 内存、`non_blocking=True`；预热 2 次，CUDA Event 计时 7 次取中位数，结束 Event 同步。

## 数字对应的含义

分页与前缀脚本操作长度、块 ID、空闲列表与哈希映射。分页主函数保留四个请求直到结束；释放请求的接口保留在类中。前缀脚本聚焦命中与复用，引用计数和淘汰由真实引擎的管理器继续处理。

调度的时间单位是一轮等长的抽象 Decode 迭代。槽位利用率为 `总 Token / (最大批次大小 × 总轮数)`。活动列表退出后会压紧，图中的批次位置变化与物理缓存的位置分开理解。模拟设置单独观察调度，Prefill 执行时间与显存容量留给完整系统建模。

量化比较的是同一份随机 CUDA K/V 的重建值。存储表按所选表示的 payload 与 scale 计数：FP32 为 2 MiB，FP16 为 1 MiB，INT8 加 scale 为 0.53125 MiB。脚本为比较误差同时持有原值、其他格式与还原张量，这些临时数据另外占空间。完整模型中的输出质量和低精度 Attention 执行，需要结合模型与 Kernel 继续评估。

卸载记录 Tensor 的已分配 GPU 存储从 32 → 24 → 16 MiB，恢复后回到 32 MiB。复制计时区间之外包括目标分配、GC 与数据检查；两块恢复后 `torch.equal=True`。`memory_reserved()` 统计分配器管理的空间，最终记录为 40 MiB。复制与计算的重叠、整段生成吞吐由后续推理负载决定。

## 校验源码

以下命令只读取文件并比对清单中的 SHA-256：

```bash
python -B -c "import json,hashlib,pathlib; a=json.load(open('simulation_sources.json')); b=json.load(open('gpu_sources.json'))['files']; rows=a+b; assert all(hashlib.sha256(pathlib.Path(r.get('file',r.get('filename'))).read_bytes()).hexdigest()==r['sha256'] for r in rows); print('5 source hashes match')"
```

GPU 随机数与复制耗时会受到设备、软件版本和当时状态影响。复现时关注相同配置下的计算方式、存储计数、还原误差和数据一致性，逐轮模拟器的状态则可直接与记录对照。
