"""配合《看懂 Triton：当线程映射交给编译器》的小实验。

运行：python triton_thread_mapping.py --dump-dir ./triton-thread-mapping
需要可用的 NVIDIA GPU、PyTorch 和 Triton；不做性能测试。
"""

import argparse
import json
import platform
import re
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = tl.load(Y + offsets, mask=mask, other=0.0)
    tl.store(Out + offsets, x + y, mask=mask)


@triton.jit
def reduce128(X, Y):
    offsets = tl.arange(0, 128)
    x = tl.load(X + offsets)
    s = tl.sum(x, axis=0)
    tl.store(Y, s)


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
        for op in ("shfl.sync", "st.shared", "ld.shared", "bar.sync")
    }


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


def save_kernel(dump_dir, name, kernel):
    for stage in ("ttir", "ttgir", "ptx"):
        content = sanitize_ptx(kernel.asm[stage]) if stage == "ptx" else sanitize_ir(kernel.asm[stage])
        (dump_dir / f"{name}.{stage}").write_text(content, encoding="utf-8")
    layouts = [
        line.strip()
        for line in kernel.asm["ttgir"].splitlines()
        if line.lstrip().startswith("#") and "ttg.blocked" in line
    ]
    return {
        "shared_bytes": kernel.metadata.shared,
        "blocked_layouts": layouts,
        "static_instructions": static_instruction_counts(kernel.asm["ptx"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=Path(__file__).parent / "triton-thread-mapping")
    args = parser.parse_args()
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    report = {"environment": environment(), "cases": {}}

    # 1003 不能被 128 整除，用最后一个 program 检查 mask。
    n, block_size = 1003, 128
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, 2.0)
    z = torch.empty_like(x)
    grid = (triton.cdiv(n, block_size),)
    add_kernel[grid](x, y, z, n, block_size, num_warps=4)
    correct = torch.equal(z, x + y)
    assert correct, "向量加法结果不一致"
    report["cases"]["add"] = {
        "N": n, "BLOCK_SIZE": block_size, "grid": list(grid), "num_warps": 4, "correct": correct
    }

    x = torch.arange(128, device="cuda", dtype=torch.float32)
    for nw in (1, 4):
        y = torch.empty((1,), device="cuda", dtype=torch.float32)
        kernel = reduce128[(1,)](x, y, num_warps=nw)
        result = y.item()
        correct = result == 8128.0
        assert correct, f"num_warps={nw} 的求和结果不一致：{result}"
        name = f"reduce128_w{nw}"
        report["cases"][name] = {
            "num_warps": nw, "result": result, "correct": correct,
            **save_kernel(args.dump_dir, name, kernel),
        }

    lines = ["Environment: " + json.dumps(report["environment"], ensure_ascii=False)]
    for name, case in report["cases"].items():
        lines.append(f"\n{name}: " + json.dumps({k: v for k, v in case.items() if k != "blocked_layouts"}, ensure_ascii=False))
        lines.extend(case.get("blocked_layouts", []))
    summary = "\n".join(lines) + "\n"
    (args.dump_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (args.dump_dir / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.dump_dir / "README.md").write_text(
        "# Triton 线程映射实验\n\n"
        "由 `triton_thread_mapping.py` 生成。每组求和保存 TTIR、TTGIR、PTX；"
        "`metadata.json` 记录实际运行环境和验证结果。\n\n"
        "导出时仅移除本地源码位置/调试元数据，执行指令与布局不变。"
        "IR 中的源码位置仅保留文件名；PTX 移除 `.loc`、`.file` 和全部 `.debug_*` 段。\n\n"
        "`summary.txt` 中的指令数是 PTX 文本里的静态条数，不能当作运行时执行次数或性能测量。"
        "布局与生成的指令会随编译器版本和目标 GPU 改变。\n\n"
        "```text\n" + summary + "```\n", encoding="utf-8"
    )
    print(summary, end="")
    print(f"\nSaved to {args.dump_dir.resolve()}")


if __name__ == "__main__":
    main()
