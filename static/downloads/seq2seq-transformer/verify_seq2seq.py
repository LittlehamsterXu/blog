"""Run the unmodified Seq2Seq experiment and record extra checks.

Requires PyTorch; no data or pretrained model downloads. The original entry
point sets seed 7 and chooses CUDA when available. Added checks inspect the
same trained model. Use python -B to avoid writing bytecode beside the files.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "seq2seq_transformer.py"


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def as_list(value):
    return value.detach().cpu().tolist()


def run_original():
    """Observe losses/first update without changing inputs, parameters, or RNG."""
    losses = []
    first_update = {}
    original_ce = F.cross_entropy
    original_step = torch.optim.Adam.step

    def observed_ce(*args, **kwargs):
        value = original_ce(*args, **kwargs)
        losses.append(float(value.detach()))
        return value

    def observed_step(optimizer, *args, **kwargs):
        if first_update:
            return original_step(optimizer, *args, **kwargs)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in parameters]
        grads = [p.grad for p in parameters]
        gradient_present = sum(g is not None for g in grads)
        gradients_finite = all(g is not None and torch.isfinite(g).all() for g in grads)
        grad_norm = sum(float(g.detach().square().sum()) for g in grads if g is not None) ** .5
        result = original_step(optimizer, *args, **kwargs)
        changes = [float((p.detach() - old).abs().max()) for p, old in zip(parameters, before)]
        first_update.update({
            "parameter_tensors": len(parameters),
            "parameter_scalars": sum(p.numel() for p in parameters),
            "gradient_tensors_present": gradient_present,
            "all_gradients_finite": bool(gradients_finite),
            "combined_gradient_l2_norm": grad_norm,
            "parameter_tensors_changed": sum(change > 0 for change in changes),
            "maximum_absolute_parameter_change": max(changes),
        })
        assert gradients_finite and gradient_present == len(parameters)
        assert any(change > 0 for change in changes)
        return result

    console = io.StringIO()
    start = time.perf_counter()
    try:
        F.cross_entropy = observed_ce
        torch.optim.Adam.step = observed_step
        with contextlib.redirect_stdout(Tee(sys.stdout, console)):
            namespace = runpy.run_path(str(SOURCE), run_name="__main__")
    finally:
        F.cross_entropy = original_ce
        torch.optim.Adam.step = original_step
    assert len(losses) == 1000
    return namespace, losses, first_update, console.getvalue(), time.perf_counter() - start


def evaluation_results(ns):
    src = ns["test_src"]
    modified = ns["changed_src"]
    predictions = ns["predictions"]
    changed_predictions = ns["changed_predictions"]
    pad, eos = ns["PAD"], ns["EOS"]
    trim = ns["trim_at_eos"]
    examples = []
    for source, changed, output, changed_output in zip(src, modified, predictions, changed_predictions):
        source_tokens = as_list(source[source.ne(pad)])
        modified_tokens = as_list(changed[changed.ne(pad)])
        actual = trim(as_list(output))
        changed_actual = trim(as_list(changed_output))
        expected = list(reversed(source_tokens)) + [eos]
        modified_expected = list(reversed(modified_tokens)) + [eos]
        examples.append({
            "source": source_tokens, "expected": expected, "prediction": actual,
            "source_modified": modified_tokens,
            "modified_expected": modified_expected, "modified_prediction": changed_actual,
            "exact": actual == expected,
            "modified_exact": changed_actual == modified_expected,
            "output_changed": actual != changed_actual,
        })
    correct = sum(example["exact"] for example in examples)
    changed_correct = sum(example["modified_exact"] for example in examples)
    changed_count = sum(example["output_changed"] for example in examples)
    assert correct == ns["correct"]
    assert changed_correct == ns["changed_correct"]
    assert changed_count == ns["output_changed"]
    return {
        "sample_count": len(examples), "exact_correct": correct,
        "modified_exact_correct": changed_correct, "outputs_changed": changed_count,
        "train_eval_overlap_checked": False,
        "sampling": "New random draws after training; overlap with training sequences not checked.",
        "examples": examples,
    }


def diagnostic_checks(ns):
    model = ns["model"].eval()
    device = ns["device"]
    # Fixed, right-padded examples. Every source has valid Keys; every target
    # begins with BOS, so its causal rows always have at least one visible Key.
    src = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]], device=device)
    target_full = torch.tensor([[1, 5, 4, 3, 2, 0, 0], [1, 10, 9, 8, 7, 6, 2]], device=device)
    decoder_input = target_full[:, :-1]
    targets = target_full[:, 1:]
    captured = []
    handles = []

    def capture(name, attention_kind):
        def hook(module, args, kwargs, output):
            weights = output[1].detach()
            key_mask = kwargs.get("key_padding_mask")
            assert key_mask is not None
            pad_keys = key_mask[:, None, None, :].expand_as(weights)
            pad_values = weights.masked_select(pad_keys)
            maximum_pad = float(pad_values.abs().max()) if pad_values.numel() else 0.
            assert maximum_pad == 0.
            row_error = float((weights.sum(-1) - 1).abs().max())
            assert row_error < 1e-5
            max_future = None
            if kwargs.get("causal", False):
                max_future = float(weights.triu(1).abs().max())
                assert max_future == 0.
            captured.append({
                "module": name, "kind": attention_kind,
                "weights_shape": list(weights.shape),
                "pad_key_max_weight": maximum_pad,
                "future_key_max_weight": max_future,
                "maximum_row_sum_error": row_error,
                "first_sample_first_head_weights": as_list(weights[0, 0]),
            })
        return hook

    for i, layer in enumerate(model.encoder):
        handles.append(layer.attn.register_forward_hook(capture(f"encoder.{i}.attn", "encoder_self"), with_kwargs=True))
    for i, layer in enumerate(model.decoder):
        handles.append(layer.self_attn.register_forward_hook(capture(f"decoder.{i}.self_attn", "decoder_self"), with_kwargs=True))
        handles.append(layer.cross_attn.register_forward_hook(capture(f"decoder.{i}.cross_attn", "cross"), with_kwargs=True))
    try:
        with torch.no_grad():
            logits = model(src, decoder_input)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured) == 6
    assert torch.isfinite(logits).all()
    with torch.no_grad():
        memory, src_pad_mask = model.encode(src)
        # Mutate only the final input position. Earlier target Queries must not
        # change; the source remains exactly the same in this comparison.
        changed = decoder_input.clone()
        changed[:, -1] = 14
        changed_logits = model.decode(changed, memory, src_pad_mask)
        original_logits = model.decode(decoder_input, memory, src_pad_mask)
        early_diff = float((changed_logits[:, :-1] - original_logits[:, :-1]).abs().max())
        last_diff = float((changed_logits[:, -1] - original_logits[:, -1]).abs().max())
        assert early_diff < 1e-5
        assert last_diff > 0

        ignored_loss = F.cross_entropy(logits.reshape(-1, ns["VOCAB_SIZE"]), targets.reshape(-1), ignore_index=ns["PAD"])
        valid_targets = targets.ne(ns["PAD"])
        selected_loss = F.cross_entropy(logits[valid_targets], targets[valid_targets])
        torch.testing.assert_close(ignored_loss, selected_loss)

        # Key masking removes PAD columns, not PAD Query outputs. The source
        # embedding has no padding_idx in this original model.
        source_positions = torch.arange(src.size(1), device=device)
        input_repr = model.src_emb(src) + model.src_pos(source_positions)[None]
        pad_repr_max = float(input_repr[src_pad_mask].abs().max())
        pad_memory_max = float(memory[src_pad_mask].abs().max())
        assert pad_repr_max > 0 and pad_memory_max > 0

    return {
        "source_ids": as_list(src), "decoder_input_ids": as_list(decoder_input),
        "target_ids": as_list(targets), "source_shape": list(src.shape),
        "decoder_input_shape": list(decoder_input.shape), "memory_shape": list(memory.shape),
        "logits_shape": list(logits.shape), "all_logits_finite": True,
        "attention_layers": captured,
        "future_target_perturbation": {
            "changed_target_input_position": int(changed.size(1) - 1),
            "earlier_logits_max_difference": early_diff,
            "final_position_logits_max_difference": last_diff,
        },
        "loss_padding_check": {
            "non_pad_target_count": int(valid_targets.sum()),
            "ignore_pad_loss": float(ignored_loss),
            "manually_selected_non_pad_loss": float(selected_loss),
        },
        "source_padding": {
            "embedding_padding_idx": model.src_emb.padding_idx,
            "pad_input_representation_max_absolute_value": pad_repr_max,
            "pad_memory_max_absolute_value": pad_memory_max,
            "pad_key_weights_are_zero": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ns, losses, first_update, stdout, elapsed = run_original()
    checks = diagnostic_checks(ns)
    evaluation = evaluation_results(ns)
    device = ns["device"]
    results = {
        "schema_version": 1, "run_date": "2026-10-24",
        "torch_version": torch.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cpu_threads": 1, "dtype": "float32", "seed": 7,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "original_entrypoint": {
            "steps": ns["steps"], "batch_size": ns["batch_size"],
            "optimizer": "Adam", "learning_rate": 3e-3,
            "initial_shape_check": {
                label: ast.literal_eval(re.search(
                    rf"^{label}: torch.Size\((\[[^\n]+\])\)$", stdout, re.MULTILINE
                ).group(1))
                for label in ("Source", "Decoder input", "Targets", "Logits")
            },
            "first_loss_before_update": losses[0],
            "final_training_batch_loss_before_update": losses[-1],
            "loss_checkpoints": {str(i): losses[i - 1] for i in [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
            "elapsed_seconds_including_observation": elapsed,
            "first_optimizer_step": first_update,
        },
        "evaluation": evaluation, "additional_checks": checks,
        "notes": [
            "The original source is unchanged; runpy executes its main entrypoint.",
            "Loss observation and the first-step optimizer observer add synchronization overhead but do not change random draws or optimizer formulas.",
            "The original auto-device choice selected CUDA for this record. CPU runs may yield different training results.",
            "The evaluation samples are new random draws; training/evaluation sequence overlap is not checked.",
            "All additional diagnostics use the same trained model, not an independently trained replacement.",
            "The model is Post-LN/GELU with learned absolute positions and two Encoder/two Decoder blocks; it is not a literal original-paper reproduction.",
            "Inputs are right-padded, source length is at least one, and target inputs begin with BOS. All-masked rows and arbitrary left padding are not validated.",
            "Greedy decoding reuses Encoder memory but recomputes the entire Decoder prefix each step; it has no KV Cache.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "original_stdout.txt").write_text(stdout, encoding="utf-8")
    with (args.output_dir / "loss.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "batch_cross_entropy_before_update"])
        writer.writerows(enumerate(losses, 1))
    print("\nADDED CHECKS PASSED")
    print(json.dumps({
        "device": str(device), "final_batch_loss": losses[-1],
        "exact": evaluation["exact_correct"],
        "modified_exact": evaluation["modified_exact_correct"],
        "outputs_changed": evaluation["outputs_changed"],
        "sample_count": evaluation["sample_count"],
        "causal_check": checks["future_target_perturbation"],
    }, indent=2))


if __name__ == "__main__":
    main()
