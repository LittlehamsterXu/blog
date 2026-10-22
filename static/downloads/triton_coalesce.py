"""32×32 转置：观察 Coalesce 和 RemoveLayoutConversions。

运行：python triton_coalesce.py --dump-dir ./triton-coalesce
需要可用的 NVIDIA GPU、PyTorch 和 Triton；不做性能测试。
脚本自动重新编译，在内存中截取关键 pass 快照，不保存完整编译日志。
"""

import argparse
from collections import Counter
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(X, Y, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    x_ptrs = X + rows[:, None] * BLOCK + cols[None, :]
    values = tl.load(x_ptrs)
    y_ptrs = Y + cols[None, :] * BLOCK + rows[:, None]
    tl.store(y_ptrs, values)


def environment():
    target = triton.runtime.driver.active.get_current_target()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "target": str(target),
    }


def static_instruction_counts(ptx):
    # 统计文本里的指令条数，不代表运行时执行次数。
    return {
        op: len(re.findall(r"(?m)^\s*(?:@!?%\w+\s+)?" + re.escape(op) + r"[:.\s]", ptx))
        for op in ("ld.global.v4", "st.global.v4", "shfl.sync", "st.shared", "ld.shared", "bar.sync")
    }


def memory_opcodes(ptx):
    opcodes = re.findall(r"(?m)^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_.:]*)(?:\s|;)", ptx)
    return dict(Counter(op for op in opcodes if op.startswith(
        ("ld.global", "st.global", "ld.shared", "st.shared", "shfl.sync", "bar.sync")
    )))


def sanitize_ir(ir):
    # 保留行列号，源码位置只留下文件名。
    return re.sub(
        r'(loc\(")([^"\n]+)(?=":\d)',
        lambda match: match[1] + re.split(r"[/\\]", match[2])[-1],
        ir,
    )


def sanitize_ptx(ptx):
    # 调试段也可能用字节编码目录名，因此整段移除。
    lines = []
    skipping_debug = False
    opened = False
    depth = 0
    for line in ptx.splitlines():
        if not skipping_debug and re.match(r"\s*\.section\s+\.debug_", line):
            skipping_debug = True
            opened = False
            depth = 0
        if skipping_debug:
            opened = opened or "{" in line
            depth += line.count("{") - line.count("}")
            if opened and depth == 0:
                skipping_debug = False
            continue
        if re.match(r"\s*\.(?:loc|file)\b", line):
            continue
        lines.append(line)
    if skipping_debug:
        raise ValueError("PTX 调试段没有闭合")
    cleaned = "\n".join(lines) + "\n"
    if re.search(r"\.debug_\w+", cleaned):
        raise ValueError("PTX 中仍有调试段引用")
    assert static_instruction_counts(cleaned) == static_instruction_counts(ptx)
    return cleaned


def ir_summary(ir):
    return {
        "convert_layout_count": ir.count("ttg.convert_layout"),
        "blocked_layouts": [
            line.strip() for line in ir.splitlines()
            if line.lstrip().startswith("#") and "ttg.blocked" in line
        ],
    }


def run_kernel(args):
    x = torch.arange(args.block * args.block, device="cuda", dtype=torch.float32).reshape(args.block, args.block)
    y = torch.empty_like(x)
    kernel = transpose_kernel[(1,)](x, y, args.block, num_warps=args.num_warps)
    correct = torch.equal(y, x.T.contiguous())
    assert correct, "转置结果不一致"
    for stage in ("ttir", "ttgir", "ptx"):
        content = sanitize_ptx(kernel.asm[stage]) if stage == "ptx" else sanitize_ir(kernel.asm[stage])
        (args.dump_dir / f"transpose.{stage}").write_text(content, encoding="utf-8")
    return {
        "environment": environment(),
        "kernel": {
            "BLOCK": args.block, "num_warps": args.num_warps, "grid": [1], "dtype": "float32",
            "correct": correct, "shared_bytes": kernel.metadata.shared,
            "static_instructions": static_instruction_counts(kernel.asm["ptx"]),
            "memory_opcodes": memory_opcodes(kernel.asm["ptx"]),
            "final_ttgir": ir_summary(kernel.asm["ttgir"]),
        },
    }


def extract_snapshots(log):
    # MLIR_ENABLE_DUMP 的标题是 Before。下一条 Before 才是当前 pass 执行后的状态。
    headers = list(re.finditer(r"^// -----// IR Dump Before ([^:]+):[^\n]*$", log, re.M))
    stages = []
    for name, stem in (
        ("TritonGPUCoalesce", "coalesce"),
        ("TritonGPURemoveLayoutConversions", "remove-layout-conversions"),
    ):
        index = next((i for i, header in enumerate(headers) if header[1] == name), None)
        if index is None or index + 2 >= len(headers):
            raise RuntimeError(f"没有找到 {name} 的前后快照，请检查当前 Triton 的 pass 输出")
        for offset, relation in ((0, "before"), (1, "after")):
            current = index + offset
            ir = log[headers[current].end():headers[current + 1].start()].strip() + "\n"
            stages.append({
                "filename": f"{relation}-{stem}.ttgir",
                "pass": name,
                "relation": relation,
                "captured_before": headers[current][1],
                "ir": sanitize_ir(ir),
                **ir_summary(ir),
            })
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=Path(__file__).parent / "triton-coalesce")
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.block <= 0 or args.block & (args.block - 1):
        parser.error("--block 需要是正的 2 的幂")
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        print(json.dumps(run_kernel(args), ensure_ascii=False))
        return

    env = dict(os.environ, MLIR_ENABLE_DUMP="1", TRITON_ALWAYS_COMPILE="1")
    env.pop("MLIR_DUMP_PATH", None)
    with tempfile.TemporaryDirectory(prefix="triton-coalesce-") as temporary:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "--dump-dir", temporary,
             "--block", str(args.block), "--num-warps", str(args.num_warps)],
            env=env, check=True, capture_output=True, text=True,
        )
        report = json.loads(result.stdout)
        stages = extract_snapshots(result.stderr)
        report["snapshots"] = [{k: v for k, v in stage.items() if k != "ir"} for stage in stages]
        for stage in stages:
            (args.dump_dir / stage["filename"]).write_text(stage["ir"], encoding="utf-8")
        for path in Path(temporary).glob("transpose.*"):
            (args.dump_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        "Environment: " + json.dumps(report["environment"], ensure_ascii=False),
        "Kernel: " + json.dumps(report["kernel"], ensure_ascii=False),
        "\nPass snapshots (all source markers say Before):",
    ]
    for stage in report["snapshots"]:
        lines.append(f"{stage['filename']}: convert_layout={stage['convert_layout_count']}; "
                     f"captured Before {stage['captured_before']}")
        lines.extend(stage["blocked_layouts"])
    summary = "\n".join(lines) + "\n"
    (args.dump_dir / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.dump_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (args.dump_dir / "README.md").write_text(
        "# Triton Coalesce 转置实验\n\n"
        "由 `triton_coalesce.py` 生成。默认参数为 `BLOCK=32`、`num_warps=4`、"
        "一个 program、float32。验证 `Y == X.T.contiguous()`，不做性能测试。\n\n"
        "```bash\npython triton_coalesce.py --dump-dir ./triton-coalesce\n```\n\n"
        "脚本自动设置 `MLIR_ENABLE_DUMP=1` 和 `TRITON_ALWAYS_COMPILE=1`。"
        "完整编译日志仅在内存中解析，不写入此目录。\n\n"
        "所有原始 pass 标题均为 Before：`before-*.ttgir` 取目标 pass 前的状态；"
        "`after-*.ttgir` 取紧随该 pass 的下一条 Before 快照，即目标 pass 执行后的状态。"
        "这里只观察第一次 RemoveLayoutConversions，后续同名 pass 不混入统计。"
        "具体原始标记见 `metadata.json`。\n\n"
        "`transpose.ttir`、`transpose.ttgir`、`transpose.ptx` 为相应完整编译阶段的结果。"
        "导出时仅移除本地源码位置/调试元数据，执行指令与布局不变。"
        "IR 中的源码位置仅保留文件名；PTX 移除 `.loc`、`.file` 和全部 `.debug_*` 段。\n\n"
        "指令数均为 PTX 文本的静态条数，不代表运行时执行次数或性能。"
        "编译器版本、目标 GPU 或参数变化，都可能改变布局、pass 流程和指令。\n\n"
        "```text\n" + summary + "```\n", encoding="utf-8",
    )
    print(summary, end="")
    print("\nSaved sanitized snapshots and compilation stages.")


if __name__ == "__main__":
    main()
