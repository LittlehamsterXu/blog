"""观察 CUDA 异步提交、Stream 依赖和两个独立 GEMM 的 event 计时。

python pytorch_cuda_execution.py --dump-dir ./pytorch-training/cuda-execution
需要 CUDA GPU。数字只用于这次实验，不据此保证多个 Stream 更快。
"""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import torch


@torch.no_grad()
def async_call(size, warmup):
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    t0 = perf_counter()
    c = a @ b
    t1 = perf_counter()
    torch.cuda.synchronize()
    t2 = perf_counter()
    return {"size": size, "python_call_ms": (t1 - t0) * 1000,
            "until_gpu_completed_ms": (t2 - t0) * 1000,
            "note": "CPU wall time; second value includes synchronization overhead, not a pure CUDA kernel duration"}


@torch.no_grad()
def dependent_streams(size):
    default = torch.cuda.current_stream()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    a = torch.ones(size, size, device="cuda")
    b = torch.full((size, size), 2.0, device="cuda")
    s1.wait_stream(default)  # 等输入初始化；只建立设备端依赖，不阻塞 CPU。
    with torch.cuda.stream(s1):
        c = a @ b
    s2.wait_stream(s1)  # d 依赖 c 的计算完成。
    with torch.cuda.stream(s2):
        d = c + 1
    c.record_stream(s2)  # 告诉 allocator：s2 也在使用这块存储。
    default.wait_stream(s2)
    expected = 2.0 * size + 1
    correct = torch.equal(d, torch.full_like(d, expected))
    assert correct
    return {"size": size, "expected_each_element": expected, "actual_first_element": d[0, 0].item(),
            "correct": correct, "ordering": "default -> s1 -> s2 -> default",
            "allocator_lifetime": "c.record_stream(s2) records storage use; it does not establish execution ordering"}


@torch.no_grad()
def independent_streams(size, warmup, repeats):
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    x = torch.randn(size, size, device="cuda")
    y = torch.randn(size, size, device="cuda")
    main = torch.cuda.current_stream()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def one_sample(two_streams):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(main)
        if two_streams:
            s1.wait_stream(main)
            s2.wait_stream(main)
            with torch.cuda.stream(s1):
                c = a @ b
            with torch.cuda.stream(s2):
                d = x @ y
            main.wait_stream(s1)
            main.wait_stream(s2)
        else:
            c = a @ b
            d = x @ y
        end.record(main)
        end.synchronize()  # end 位于两条工作流之后；保留输出直到完成。
        return start.elapsed_time(end)

    for _ in range(warmup):
        one_sample(False)
        one_sample(True)
    single = [one_sample(False) for _ in range(repeats)]
    double = [one_sample(True) for _ in range(repeats)]
    return {"size": size, "warmup_per_mode": warmup, "repeats_per_mode": repeats,
            "one_stream_event_ms": single, "two_streams_event_ms": double,
            "one_stream_median_ms": median(single), "two_streams_median_ms": median(double),
            "timer": "CUDA events on main stream; both work streams wait for start, main joins both before end",
            "note": "Two independent GEMMs; ordering modes, GPU clock, load and size can change the observed result"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=Path(__file__).parent / "pytorch-training" / "cuda-execution")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("此实验需要 CUDA GPU")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats 需要大于 0，warmup 不能为负数")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    report = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                        "seed": args.seed, "dtype": "float32", "allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "async_call": async_call(2048, args.warmup),
        "dependent_streams": dependent_streams(1024),
        "independent_streams": independent_streams(1024, args.warmup, args.repeats),
    }
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    (args.dump_dir / "execution_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.dump_dir / "README.md").write_text(
        "# CUDA 异步执行与 Stream 实验\n\n"
        "```bash\npython pytorch_cuda_execution.py --dump-dir ./pytorch-training/cuda-execution\n```\n\n"
        "异步例子记录 2048×2048 GEMM 的 Python 调用返回时间，以及同步到 GPU 完成的 CPU 时间。"
        "后者包含同步开销，不应标为纯 kernel 耗时。\n\n"
        "依赖例子用两个 Stream 计算 c=a@b、d=c+1。wait_stream 建立执行顺序；"
        "record_stream 记录 allocator 的跨流存储使用，两者用途不同。所有 d 元素均应为 2049。\n\n"
        "独立例子对两个 1024×1024 GEMM 比较单流与双流，每种先预热 5 次，再取 20 次中位数。"
        "start event 在主流，两工作流都等待它；主流等待两工作流后才记录 end event。"
        "这些设备 event 时间不能与 Python 提交时间混为一谈。"
        "多个 Stream 只提供并发的可能；本机此次结果不能保证换尺寸、硬件或负载后仍然更快。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
