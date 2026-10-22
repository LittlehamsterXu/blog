"""用 eager 模式训练四个 XOR 样本，并检查梯度与参数更新。

python pytorch_training.py --dump-dir ./pytorch-training
默认只用 CPU。可选 --device cuda；不需要 Triton 或 torch.compile。
"""

import argparse
import csv
import json
import platform
import warnings
from pathlib import Path

import torch
from torch import nn


def gradient_probe(device):
    w = torch.tensor(2.0, device=device, requires_grad=True)
    x = torch.tensor(3.0, device=device)
    intermediate = w * x
    loss = intermediate.square()
    loss.backward()
    first = w.grad.item()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        intermediate_grad = intermediate.grad
    (w * x).square().backward()  # 新建计算图，梯度仍加到同一个 w.grad。
    accumulated = w.grad.item()
    w.grad = None
    fresh_loss = (w * x).square()
    fresh_loss.backward()
    cleared = w.grad.item()
    try:
        fresh_loss.backward()
    except RuntimeError:
        same_graph_reuse_raises = True
    else:
        same_graph_reuse_raises = False
    w.grad = None
    retained = w * x
    retained.retain_grad()
    retained.square().backward()
    assert (first, accumulated, cleared, retained.grad.item()) == (36.0, 72.0, 36.0, 12.0)
    assert same_graph_reuse_raises
    return {
        "w_is_leaf": w.is_leaf, "intermediate_is_leaf": intermediate.is_leaf,
        "intermediate_requires_grad": intermediate.requires_grad,
        "intermediate_grad_fn": type(intermediate.grad_fn).__name__,
        "input_grad_is_none": x.grad is None,
        "intermediate_grad_is_none_by_default": intermediate_grad is None,
        "first_backward_w_grad": first, "new_graph_accumulated_w_grad": accumulated,
        "clear_then_new_graph_w_grad": cleared,
        "same_graph_second_backward_raises": same_graph_reuse_raises,
        "retained_intermediate_grad": retained.grad.item(),
    }


def update_probe(device, seed):
    torch.manual_seed(seed)
    model = nn.Linear(2, 1).to(device)
    x = torch.tensor([[1., 2.], [2., 1.], [3., 2.]], device=device)
    y = torch.tensor([[4.], [5.], [8.]], device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = model.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    unchanged_after_backward = torch.equal(before, model.weight)
    grad = model.weight.grad.detach().clone()
    optimizer.step()
    after = model.weight.detach().clone()
    assert unchanged_after_backward and not torch.equal(before, after)
    assert torch.allclose(after, before - 0.05 * grad)
    return {
        "loss": loss.item(), "weight_before": before.cpu().tolist(),
        "weight_grad": grad.cpu().tolist(), "weight_after": after.cpu().tolist(),
        "backward_leaves_weight_unchanged": unchanged_after_backward,
        "step_matches_weight_minus_lr_times_grad": True,
    }


def mode_probe(model, x):
    rows = []
    for training in (True, False):
        model.train(training)
        for grad_enabled in (True, False):
            with torch.set_grad_enabled(grad_enabled):
                out = model(x)
                rows.append({"training": model.training, "grad_enabled": torch.is_grad_enabled(),
                             "output_requires_grad": out.requires_grad})
    model.train()
    train_logits = model(x)
    model.eval()
    eval_logits = model(x)
    with torch.inference_mode():
        out = model(x)
        inference = {"training": model.training, "grad_enabled": torch.is_grad_enabled(),
                     "output_requires_grad": out.requires_grad, "output_is_inference": out.is_inference()}
    assert torch.equal(train_logits, eval_logits)
    return {"cases": rows, "inference_mode": inference,
            "train_eval_logits_equal_for_this_model": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=301)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--dump-dir", type=Path, default=Path(__file__).parent / "pytorch-training")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 需要大于 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前 PyTorch 环境没有可用的 CUDA GPU")
    torch.set_num_threads(1)  # 四个样本很小，固定 CPU 线程数以便复现。
    probes = {"gradient": gradient_probe(args.device), "sgd_update": update_probe(args.device, args.seed)}

    torch.manual_seed(args.seed)
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=args.device)
    y = torch.tensor([0, 1, 1, 0], device=args.device)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        history.append({"epoch": epoch, "loss_before_update": loss.item(),
                        "first_layer_weight_grad_norm": model[0].weight.grad.norm().item()})
        optimizer.step()
    probes["modes"] = mode_probe(model, x)
    with torch.inference_mode():
        logits = model(x)
        final_loss = criterion(logits, y).item()
        prediction = logits.argmax(dim=1)
        probabilities = logits.softmax(dim=1)
    metadata = {
        "python": platform.python_version(), "torch": torch.__version__, "device": args.device,
        "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "seed": args.seed, "epochs": args.epochs, "updates": args.epochs, "cpu_threads": torch.get_num_threads(),
        "execution": "eager", "model": "Linear(2,8) -> Tanh -> Linear(8,2)",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "loss": "CrossEntropyLoss", "optimizer": "Adam", "lr": args.lr,
    }
    results = {
        "input": x.cpu().tolist(), "target": y.cpu().tolist(),
        "prediction": prediction.cpu().tolist(), "logits": logits.cpu().tolist(),
        "probabilities": probabilities.cpu().tolist(), "final_loss_after_all_updates": final_loss,
        "accuracy": (prediction == y).float().mean().item(),
        "checkpoints_before_update": [row for row in history if row["epoch"] % 50 == 0],
        "probes": probes,
    }
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("metadata.json", metadata), ("results.json", results)):
        (args.dump_dir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.dump_dir / "loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.dump_dir / "README.md").write_text(
        "# PyTorch XOR 训练实验\n\n"
        "```bash\npython pytorch_training.py --dump-dir ./pytorch-training\n"
        "# 可选：使用 CUDA\npython pytorch_training.py --device cuda --dump-dir ./pytorch-training-cuda\n```\n\n"
        "默认 seed=42，CPU，301 次完整 batch 更新。模型为 Linear(2,8) → Tanh → Linear(8,2)，"
        "CrossEntropyLoss 接收原始 logits；Adam 学习率 0.05。仅推理展示时计算 softmax。\n\n"
        "`loss.csv` 每行记录该 epoch 更新前的 loss 和 backward 后的梯度范数；epoch 从 0 开始。"
        "`results.json` 的 final_loss 则是在全部更新完成后重新计算。"
        "固定 seed 不能保证跨版本、设备或平台的浮点结果完全相同。\n\n"
        "额外 probe 对照了叶张量与非叶张量、默认不保留中间梯度、retain_grad、"
        "新图上的梯度累积、重复使用已释放图的异常，以及 SGD 的更新前后权重。"
        "最后用本模型验证 train/eval 与 grad mode 是两个独立开关；"
        "Linear 和 Tanh 在 train/eval 下的计算相同。\n\n"
        "CUDA profiler 与 Stream 实验见独立脚本 `pytorch_mlp_profile.py`、"
        "`pytorch_cuda_execution.py`，CPU 训练不依赖它们。"
        "此目录不保存原始 profiler trace、进程 ID 或本地路径。\n", encoding="utf-8",
    )
    print(json.dumps({"metadata": metadata, "prediction": results["prediction"],
                      "final_loss": final_loss, "accuracy": results["accuracy"],
                      "checkpoints": results["checkpoints_before_update"], "probes": probes},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
