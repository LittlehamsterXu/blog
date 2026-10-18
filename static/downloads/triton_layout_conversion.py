"""配合《Triton Layout：一次转换，到底搬了什么》的小实验。

运行：python triton_layout_conversion.py --dump-dir ./triton-layout-conversion
需要支持 Gluon 的 Triton、PyTorch 和 NVIDIA GPU；不做性能测试。
"""

import argparse
import json
import platform
import re
from pathlib import Path

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def convert_demo(X, Y, N: gl.constexpr, SRC: gl.constexpr, DST: gl.constexpr):
    src_idx = gl.arange(0, N, layout=SRC)
    x = gl.load(X + src_idx)
    y = gl.convert_layout(x, DST)
    dst_idx = gl.arange(0, N, layout=DST)
    gl.store(Y + dst_idx, y)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=Path(__file__).parent / "triton-layout-conversion")
    args = parser.parse_args()
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    report = {"environment": environment(), "cases": {}}

    # 前两组对应原实验，第三组固定 N，避免同时改变 warp 数和元素数。
    for nw, n in ((1, 128), (4, 512), (1, 512)):
        src = gl.BlockedLayout([2], [32], [nw], [0])
        dst = gl.BlockedLayout([4], [32], [nw], [0])
        x = torch.arange(n, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        kernel = convert_demo[(1,)](x, y, n, src, dst, num_warps=nw)
        correct = torch.equal(x, y)
        assert correct, f"num_warps={nw}, N={n} 的转换结果不一致"
        name = f"convert_w{nw}_n{n}"
        for stage in ("ttgir", "ptx"):
            content = sanitize_ptx(kernel.asm[stage]) if stage == "ptx" else sanitize_ir(kernel.asm[stage])
            (args.dump_dir / f"{name}.{stage}").write_text(content, encoding="utf-8")
        layouts = [
            line.strip()
            for line in kernel.asm["ttgir"].splitlines()
            if line.lstrip().startswith("#") and "ttg.blocked" in line
        ]
        report["cases"][name] = {
            "N": n, "num_warps": nw, "correct": correct,
            "shared_bytes": kernel.metadata.shared,
            "blocked_layouts": layouts,
            "static_instructions": static_instruction_counts(kernel.asm["ptx"]),
        }

    lines = ["Environment: " + json.dumps(report["environment"], ensure_ascii=False)]
    for name, case in report["cases"].items():
        lines.append(f"\n{name}: " + json.dumps({k: v for k, v in case.items() if k != "blocked_layouts"}, ensure_ascii=False))
        lines.extend(case["blocked_layouts"])
    summary = "\n".join(lines) + "\n"
    (args.dump_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (args.dump_dir / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.dump_dir / "README.md").write_text(
        "# Triton Layout Conversion 实验\n\n"
        "由 `triton_layout_conversion.py` 生成。每组保存 TTGIR 和 PTX；"
        "`metadata.json` 记录实际运行环境和验证结果。\n\n"
        "导出时仅移除本地源码位置/调试元数据，执行指令与布局不变。"
        "IR 中的源码位置仅保留文件名；PTX 移除 `.loc`、`.file` 和全部 `.debug_*` 段。\n\n"
        "三组均把 `BlockedLayout([2], [32], [num_warps], [0])` 转为"
        " `BlockedLayout([4], [32], [num_warps], [0])`。\n\n"
        "`summary.txt` 中的指令数是 PTX 文本里的静态条数，不能当作运行时执行次数或性能测量。"
        "布局与生成的指令会随编译器版本和目标 GPU 改变。\n\n"
        "```text\n" + summary + "```\n", encoding="utf-8"
    )
    print(summary, end="")
    print(f"\nSaved to {args.dump_dir.resolve()}")


if __name__ == "__main__":
    main()
