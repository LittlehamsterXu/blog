# nano-vLLM 学习记录

文章《nano-vLLM 学习笔记》的运行附件。

## 内容与来源

| 文件 | 内容 |
| --- | --- |
| `test_continuous_batching.py` | WSL 项目中的原学习脚本，原样复制，顶部保留较早的双请求实验注释。 |
| `continuous-batching.log` | 2026-10-27 实际运行上述脚本的完整标准输出，共 68 次调度，进程退出码为 0。 |
| `schedule.json` | 从本次完整日志逐行提取的模式、队列长度和选中请求，包含运行设置。 |
| `prefill-decode-excerpt.txt` | 2026-10-26 保存的 `run_logs/example.log` 中前两次完整 step 的摘录，包含最初的编译提示与进度条。 |
| `sources.json` | 阅读版本、本地源码与原日志的字节数和 SHA-256。 |
| `record_schedule.py` | 附加记录工具，临时包装调度器，无需修改 nano-vLLM 源文件。 |

源码阅读版本：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`。

上游：[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)

本地在 `scheduler.py` 增加了队列打印，在 `llm_engine.py` 保留了详细状态打印的注释版本和退出保护。正文链接指向上游；`sources.json` 的哈希对应实际阅读的本地文件。

## 环境

WSL Ubuntu 26.04，Python 3.12.15、PyTorch 2.8.0+cu128、Triton 3.4.0、FlashAttention 2.8.3.post2、Transformers 4.57.6。单张 RTX 4070 Laptop GPU（8 GiB），模型 Qwen3-0.6B。

脚本从 `~/huggingface/Qwen3-0.6B/` 加载模型，使用项目 `.venv`。运行设置：

```text
enforce_eager=True
tensor_parallel_size=1
max_model_len=1024
max_num_batched_tokens=1024
max_num_seqs=4
gpu_memory_utilization=0.7
kvcache_block_size=256（配置默认值）
temperature=0.6
max_tokens=64
```

## 重新观察调度

在配好上述环境的 nano-vLLM 项目中，将两个 `.py` 附件放进项目目录，确认模型路径后运行：

```bash
.venv/bin/python record_schedule.py --output-dir ./my-schedule
```

记录工具会执行 `test_continuous_batching.py`，并保存完整调度输出和 JSON。上游没有本地调度打印也可以使用，记录工具会输出同样格式的调度行。它在当前进程临时包装 `Scheduler.schedule`，读取返回的请求和队列状态，结束时恢复方法。

单独运行原脚本：

```bash
.venv/bin/python test_continuous_batching.py
```

脚本先让 A 执行两次 step，再依次加入 B、C、D。前三个到达步骤各运行一次 Prefill，随后四条请求共同 Decode。前六次调度可以直接与图对照。采样没有固定种子，后续输出与结束时刻可能随运行变化；每轮耗时也各不相同。

## 对照文章

- 序列长度与缓存进度的数值，取自 10 月 26 日双请求日志；该次示例的输入和生成上限与连续组批脚本分别设置。
- 四请求到达时间线来自 10 月 27 日重新运行的调度日志，每列对应一次 step。
- 地址图使用每块 4 个位置的教学例子，实际运行每块 256 个位置。
- 前缀缓存、引用计数、抢占和 CUDA Graph 的机制按上述 commit 阅读；本次 GPU 运行使用 eager 模式。

`schedule.json` 的最后一行记录最后一批被选中时的状态，之后模型执行和后处理让这些请求结束。进程正常退出说明脚本已排空全部请求。
