"""记录一次 eager MLP 训练步，导出可在 Perfetto 打开的脱敏 trace。

python pytorch_mlp_profile.py --dump-dir ./pytorch-training/profile
需要 CUDA GPU。默认使用四点 XOR；--input random 可复现随机输入的观察方式。
原始 trace 只在临时目录中短暂存在，不发布主机信息、真实 PID/TID 或路径。
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import platform
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


PHASES = ("FORWARD", "BACKWARD", "OPTIMIZER")


def short_name(name):
    for token in ("gemmSN_TN", "gemmSN_NN", "ampere_sgemm_32x32_sliced1x4_nt",
                  "splitKreduce_kernel", "softmax_warp_forward", "softmax_warp_backward",
                  "nll_loss_forward_reduce_cuda_kernel_2d", "nll_loss_backward_reduce_cuda_kernel_2d",
                  "FillFunctor", "tanh_backward", "tanh_kernel", "LerpFunctor", "Sqrt"):
        if token in name:
            return token
    return name.split("<", 1)[0].removeprefix("void ")[:100]


def analyze(trace):
    events = trace["traceEvents"]
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op" and e.get("ph") == "X"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"), key=lambda e: e["ts"])
    phases = [e for e in events if e.get("cat") == "user_annotation" and e.get("ph") == "X"
              and e["name"].removeprefix("MY_") in PHASES]
    by_external = defaultdict(list)
    by_correlation = defaultdict(list)
    for event in cpu_ops:
        by_external[event.get("args", {}).get("External id")].append(event)
    for event in events:
        if event.get("cat") == "cuda_runtime" and event.get("ph") == "X":
            by_correlation[event.get("args", {}).get("correlation")].append(event)
    links = []
    for kernel in kernels:
        external = kernel["args"]["External id"]
        correlation = kernel["args"]["correlation"]
        candidates = by_external[external]
        launches = by_correlation[correlation]
        if len(candidates) != 1 or len(launches) != 1:
            raise ValueError("无法唯一关联 kernel、CPU 操作和 CUDA launch")
        cpu, launch = candidates[0], launches[0]
        assert launch["args"]["External id"] == external
        assert (cpu["pid"], cpu["tid"]) == (launch["pid"], launch["tid"])
        assert cpu["ts"] <= launch["ts"] and launch["ts"] + launch["dur"] <= cpu["ts"] + cpu["dur"] + 0.01
        containers = [p for p in phases if p["pid"] == cpu["pid"] and p["ts"] <= cpu["ts"]
                      and cpu["ts"] + cpu["dur"] <= p["ts"] + p["dur"] + 0.01]
        if len(containers) != 1:
            raise ValueError("CPU 操作没有唯一的训练阶段")
        links.append((kernel, cpu, launch, containers[0]["name"].removeprefix("MY_")))

    selected = [e for e in events if e.get("ph") == "X" and
                (e.get("cat") in ("cpu_op", "cuda_runtime", "kernel") or e in phases)]
    origin = min(e["ts"] for e in selected)
    externals = sorted({e.get("args", {}).get("External id") for e in selected
                        if isinstance(e.get("args", {}).get("External id"), int)})
    correlations = sorted({e.get("args", {}).get("correlation") for e in selected
                           if isinstance(e.get("args", {}).get("correlation"), int)})
    external_map = {value: index + 1 for index, value in enumerate(externals)}
    correlation_map = {value: index + 1 for index, value in enumerate(correlations)}
    cpu_threads = sorted({e["tid"] for e in selected if e.get("cat") != "kernel"}, key=str)
    gpu_threads = sorted({e["tid"] for e in kernels}, key=str)
    cpu_thread_map = {value: index + 1 for index, value in enumerate(cpu_threads)}
    gpu_thread_map = {value: index + 1 for index, value in enumerate(gpu_threads)}
    clean_events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "CPU"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "CUDA GPU"}},
    ]
    for pid, mapping, label in ((1, cpu_thread_map, "CPU thread"), (2, gpu_thread_map, "CUDA stream")):
        for tid in mapping.values():
            clean_events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                                 "args": {"name": f"{label} {tid}"}})
    for event in selected:
        gpu = event.get("cat") == "kernel"
        clean_args = {}
        source_args = event.get("args", {})
        if "External id" in source_args:
            clean_args["external_id"] = external_map[source_args["External id"]]
        if "correlation" in source_args:
            clean_args["correlation"] = correlation_map[source_args["correlation"]]
        if "Input Dims" in source_args:
            clean_args["input_shapes"] = source_args["Input Dims"]
        clean_events.append({"name": event["name"].removeprefix("MY_"), "cat": event["cat"], "ph": "X",
                             "pid": 2 if gpu else 1,
                             "tid": gpu_thread_map[event["tid"]] if gpu else cpu_thread_map[event["tid"]],
                             "ts": round(event["ts"] - origin, 3), "dur": event["dur"], "args": clean_args})
    rows = []
    for index, (kernel, cpu, launch, phase) in enumerate(links, 1):
        external = external_map[kernel["args"]["External id"]]
        correlation = correlation_map[kernel["args"]["correlation"]]
        rows.append({"index": index, "stage": phase, "cpu_op": cpu["name"],
                     "cpu_input_shapes": json.dumps(cpu.get("args", {}).get("Input Dims", [])),
                     "kernel_short_name": short_name(kernel["name"]), "kernel_name": kernel["name"],
                     "start_us": round(kernel["ts"] - origin, 3), "duration_us": kernel["dur"],
                     "external_id": external, "correlation": correlation, "runtime_name": launch["name"],
                     "match_evidence": "CPU/kernel/launch external_id equal; kernel/launch correlation equal"})
        clean_events.extend([
            {"name": "CPU launch to CUDA kernel", "cat": "cpu_to_gpu", "ph": "s", "id": index,
             "pid": 1, "tid": cpu_thread_map[launch["tid"]],
             "ts": round(launch["ts"] - origin, 3), "args": {"external_id": external, "correlation": correlation}},
            {"name": "CPU launch to CUDA kernel", "cat": "cpu_to_gpu", "ph": "f", "id": index, "bp": "e",
             "pid": 2, "tid": gpu_thread_map[kernel["tid"]], "ts": round(kernel["ts"] - origin, 3),
             "args": {"external_id": external, "correlation": correlation}},
        ])
    launched_external = {cpu["args"]["External id"] for _, cpu, _, _ in links}
    metadata_examples = [
        {"name": e["name"], "input_shapes": e.get("args", {}).get("Input Dims", []),
         "matched_gpu_kernels": 0}
        for e in cpu_ops if e["name"] in ("aten::t", "aten::transpose", "aten::view")
        and e.get("args", {}).get("External id") not in launched_external
    ]
    summary = {
        "kernel_events": len(rows), "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "all_links_verified_by_external_id_and_correlation": True,
        "cpu_stage_assignment": "matched CPU op contained in CPU user_annotation; autograd worker threads allowed",
        "ids": "PID/TID/external_id/correlation remapped; timestamps normalized to first retained event",
        "metadata_only_examples": metadata_examples,
        "warning": "Dynamic events from one profiled step on this machine, not a universal kernel count or benchmark",
    }
    return rows, summary, {"displayTimeUnit": "us", "traceEvents": clean_events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=Path(__file__).parent / "pytorch-training" / "profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--input", choices=("xor", "random"), default="xor")
    parser.add_argument("--analyze-trace", type=Path, help="只分析已有 trace；不会复制原始文件")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.analyze_trace:
        trace = json.loads(args.analyze_trace.read_text())
        run = {"source": "provided trace", "input": "not inferred from trace", "seed": None}
    else:
        if not torch.cuda.is_available():
            parser.error("此实验需要 CUDA GPU；CPU 训练请运行 pytorch_training.py")
        torch.manual_seed(args.seed)
        model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)).cuda()
        x = (torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device="cuda")
             if args.input == "xor" else torch.randn(4, 2, device="cuda"))
        y = torch.tensor([0, 1, 1, 0], device="cuda")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with record_function("FORWARD"):
                logits = model(x)
                loss = criterion(logits, y)
            with record_function("BACKWARD"):
                loss.backward()
            with record_function("OPTIMIZER"):
                optimizer.step()
            return loss

        model.train()
        for _ in range(args.warmup):
            train_step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            loss = train_step()
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory(prefix="pytorch-mlp-profile-") as temporary:
            raw = Path(temporary) / "raw.trace.json"
            prof.export_chrome_trace(str(raw))
            trace = json.loads(raw.read_text())
        run = {"source": "fresh CUDA eager step", "python": platform.python_version(), "torch": torch.__version__,
               "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
               "seed": args.seed, "input": args.input, "input_values": x.cpu().tolist(),
               "target": y.cpu().tolist(), "warmup_steps": args.warmup, "profiled_steps": 1,
               "model": "Linear(2,8) -> Tanh -> Linear(8,2)", "parameter_count": 42,
               "optimizer": "Adam", "lr": 0.05, "loss": "CrossEntropyLoss",
               "profiled_step_loss_before_update": loss.item()}
    rows, summary, clean_trace = analyze(trace)
    summary["run"] = run
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    with (args.dump_dir / "kernel_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for filename, value in (("profile_summary.json", summary), ("mlp-step.trace.json", clean_trace)):
        (args.dump_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.dump_dir / "README.md").write_text(
        "# 一次 MLP 训练步的 CUDA 记录\n\n"
        "```bash\npython pytorch_mlp_profile.py --dump-dir ./pytorch-training/profile\n```\n\n"
        "默认为 seed=42 的四点 XOR、5 步预热后记录 1 步。"
        "`--input random` 改为随机输入；它不能替代四点 XOR 的收敛验证。\n\n"
        "将 `mlp-step.trace.json` 导入 Perfetto 可查看 CPU 算子、CUDA runtime 与 GPU kernel。"
        "每条 CPU launch → GPU flow 都由 correlation 关联，并用 External id 交叉验证 CPU 父操作。"
        "阶段按匹配 CPU 操作所属的 CPU record_function 区间归类，允许 autograd 工作线程；"
        "不根据 GPU 开始时间猜阶段。CSV 包含输入 shape 和同样的关联证据。\n\n"
        "导出仅保留这些事件及少量参数；PID/TID、External id、correlation 已重编号，"
        "时间从第一条保留事件起归零。主机信息、真实进程号、本地路径、GPU UUID 和原始 trace 不发布。"
        "此记录是当前环境的单步动态事件，不是普遍固定的 kernel 数，也不作为性能基准。\n",
        encoding="utf-8",
    )
    print(json.dumps({"kernel_events": summary["kernel_events"], "stage_counts": summary["stage_counts"],
                      "run": run}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
