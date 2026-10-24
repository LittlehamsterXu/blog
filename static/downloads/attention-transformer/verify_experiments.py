"""Re-run the learning scripts, then make the article's extra checks explicit.

Requires Python 3.10+ and PyTorch. The original scripts are unchanged copies.
The additional checks use CPU/float32, seed 7, and one CPU thread.
No dataset or model download is needed. Writes results.json and original_stdout.txt.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import runpy

import torch
import torch.nn.functional as F

from cross_attention import CrossAttention
from decoder_kv_cache import TinyDecoder
from multi_head_attention import MultiHeadSelfAttention
from transformer_encoder import TransformerEncoder


ROOT = Path(__file__).resolve().parent
SOURCES = [
    "single_attention.py", "multi_head_attention.py",
    "positional_encoding_experiment.py", "transformer_encoder.py",
    "cross_attention.py", "teacher_forcing_loss.py", "decoder_kv_cache.py",
    "train_attention.py", "attention_ablation.py",
]


def as_list(tensor):
    return tensor.detach().cpu().tolist()


def reset():
    torch.manual_seed(7)


def original_runs():
    logs = []
    original = {}
    for name in SOURCES:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            ns = runpy.run_path(str(ROOT / name), run_name="__main__")
        stdout = stream.getvalue()
        logs.append(f"=== {name} ===\n{stdout}")
        original[name] = {"completed": True}
        if name == "single_attention.py":
            original[name]["device"] = ns["device"]
        if name == "train_attention.py":
            original[name].update({
                "initial_loss": ns["initial_loss"],
                "final_loss": ns["final_loss"],
                "predictions": as_list(ns["final_logits"].argmax(-1)),
                "targets": as_list(ns["y"]),
                "q_weight_max_change": float(
                    (ns["model"].attn.q_proj.weight.detach() - ns["initial_q"]).abs().max()
                ),
                "head_0_weights_before": as_list(ns["initial_weights"]),
                "head_0_weights_after": as_list(ns["final_weights"][0, 0]),
                "first_step_projection_gradient_norms": {
                    letter: float(re.search(
                        rf"{letter} projection gradient norm: ([^\n]+)", stdout
                    ).group(1)) for letter in ("Q", "K", "V")
                },
            })
        if name == "attention_ablation.py":
            values = re.findall(
                r"(Without Attention|With Attention)\nTest loss: ([\d.]+)"
                r"\nAccuracy by position:\s*(\[[^\n]+\])", stdout
            )
            assert len(values) == 2
            original[name]["reported_results"] = {
                label: {
                    "test_loss_rounded": float(loss),
                    "accuracy_by_position": ast.literal_eval(acc),
                }
                for label, loss, acc in values
            }
        del ns
    return original, "\n".join(logs)


def attention_check():
    # A small explicit example: three Queries/Keys, two Value features.
    q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    k = q.clone()
    v = torch.tensor([[1., 10.], [2., 20.], [4., 40.]])
    scores = q @ k.T / math.sqrt(2)
    weights = scores.softmax(-1)
    output = weights @ v
    torch.testing.assert_close(weights.sum(-1), torch.ones(3))
    assert weights.shape == (3, 3) and output.shape == (3, 2)
    changed_output = weights @ (v + 1)
    torch.testing.assert_close(changed_output, output + 1)
    return {
        "q": as_list(q), "k": as_list(k), "v": as_list(v),
        "scores": as_list(scores), "weights": as_list(weights),
        "output": as_list(output),
        "row_sums": as_list(weights.sum(-1)),
        "v_plus_one_output_max_change": float((changed_output - output).abs().max()),
    }


def multihead_check():
    reset()
    model = MultiHeadSelfAttention(d_model=4, n_heads=2, causal=True).eval()
    x = torch.tensor([[[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]]])
    with torch.no_grad():
        output_causal, causal = model(x)
        model.causal = False
        output_full, full = model(x)
    torch.testing.assert_close(causal.sum(-1), torch.ones(1, 2, 3))
    assert torch.count_nonzero(causal.triu(1)) == 0
    torch.testing.assert_close(causal[:, :, -1], full[:, :, -1])
    assert float((full[:, 0] - full[:, 1]).abs().max()) > 0
    return {
        "input_shape": list(x.shape), "weights_shape": list(full.shape),
        "output_shape": list(output_full.shape),
        "head_0_full": as_list(full[0, 0]), "head_1_full": as_list(full[0, 1]),
        "head_0_causal": as_list(causal[0, 0]),
        "head_1_causal": as_list(causal[0, 1]),
        "last_query_causal_full_max_difference": float((causal[:, :, -1] - full[:, :, -1]).abs().max()),
        "untrained_head_max_difference": float((full[:, 0] - full[:, 1]).abs().max()),
        "causal_first_output": as_list(output_causal[0, 0]),
    }


def position_check():
    # Load the original fixed-position functions without retaining console output.
    with contextlib.redirect_stdout(io.StringIO()):
        ns = runpy.run_path(str(ROOT / "positional_encoding_experiment.py"))
    pe = ns["pe"]
    without = ns["weights0"]
    with_pe = ns["weights1"]
    torch.testing.assert_close(without, torch.full_like(without, 1 / 3))
    assert float((with_pe - without).abs().max()) > 0
    return {
        "positional_encoding": as_list(pe[0]),
        "weights_without_position": as_list(without[0]),
        "weights_with_position": as_list(with_pe[0]),
        "weights_max_change": float((with_pe - without).abs().max()),
        "identity_qkv_projections": True,
    }


def encoder_check():
    reset()
    model = TransformerEncoder(
        vocab_size=100, max_len=32, d_model=16, n_heads=4,
        d_ff=64, n_layers=6, dropout=0., pad_token_id=0,
    ).eval()
    ids = torch.tensor([[11, 25, 70, 0, 0], [9, 3, 2, 4, 1]])
    padding = ids.eq(0)
    with torch.no_grad():
        x = model.token_emb(ids) + model.pos_emb[:, :5]
        _, without = model.layers[0].attn(x)
        _, masked = model.layers[0].attn(x, padding_mask=padding)
        output, weights, hidden = model(ids)
        maxima = []
        for layer_weights in weights:
            pad_keys = padding[:, None, None, :].expand_as(layer_weights)
            pad_values = layer_weights.masked_select(pad_keys)
            maxima.append(float(pad_values.abs().max()))
            assert torch.count_nonzero(pad_values) == 0
            torch.testing.assert_close(layer_weights.sum(-1), torch.ones(2, 4, 5))
        # This is a limitation of these educational masked_fill + softmax scripts:
        # a row consisting entirely of -inf has no valid Softmax distribution.
        all_pad_ids = torch.zeros(1, 5, dtype=torch.long)
        all_pad_output, all_pad_weights, _ = model(all_pad_ids)
    assert torch.isfinite(output).all()
    assert len(hidden) == 7 and len(weights) == 6
    assert float(output[0, 3:].abs().max()) > 0
    assert torch.isnan(all_pad_weights[0]).all()
    initial = hidden[0][0, 2]
    return {
        "token_ids": as_list(ids), "padding_mask": as_list(padding),
        "output_shape": list(output.shape),
        "hidden_states_count": len(hidden), "blocks_count": len(weights),
        "first_query_unmasked": as_list(without[0, 0, 0]),
        "first_query_masked": as_list(masked[0, 0, 0]),
        "first_head_masked": as_list(masked[0, 0]),
        "per_layer_pad_key_max_weight": maxima,
        "pad_query_output_max_absolute_value": float(output[0, 3:].abs().max()),
        "output_finite_for_mixed_valid_and_pad": bool(torch.isfinite(output).all()),
        "all_pad_produces_nan_in_this_implementation": bool(torch.isnan(all_pad_output).any()),
        "tracked_position": 2, "tracked_token_id": 70,
        "untrained_hidden_states": [
            {
                "block": i,
                "first_four_features": as_list(state[0, 2, :4]),
                "distance_from_input": float((state[0, 2] - initial).norm()),
                "cosine_to_input": float(F.cosine_similarity(initial, state[0, 2], dim=0)),
            }
            for i, state in enumerate(hidden)
        ],
    }


def cross_check():
    reset()
    # Keep exactly the random-number consumption order in cross_attention.py.
    encoder_output = torch.randn(2, 5, 16)
    decoder_hidden = torch.randn(2, 3, 16)
    model = CrossAttention(d_model=16, n_heads=4).eval()
    with torch.no_grad():
        output, weights = model(decoder_hidden, encoder_output)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 3))
    assert weights.shape == (2, 4, 3, 5)
    return {
        "encoder_shape": list(encoder_output.shape),
        "decoder_shape": list(decoder_hidden.shape),
        "weights_shape": list(weights.shape), "output_shape": list(output.shape),
        "first_head_weights": as_list(weights[0, 0]),
    }


def training_check():
    reset()
    sequences = torch.tensor([[1, 2, 3, 4, 5], [1, 3, 2, 4, 5]])
    decoder_input, targets = sequences[:, :-1], sequences[:, 1:]
    model = TinyDecoder(vocab_size=6, max_len=8, d_model=16,
                        n_heads=4, d_ff=32, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    original_weight = model.lm_head.weight.detach().clone()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    logits = model(decoder_input)
    loss = F.cross_entropy(logits.reshape(-1, 6), targets.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(model.lm_head.weight.grad.norm())
    assert grad_norm > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    change = float((model.lm_head.weight.detach() - original_weight).abs().max())
    assert change > 0
    with torch.no_grad():
        loss_after = F.cross_entropy(model(decoder_input).reshape(-1, 6), targets.reshape(-1))
        # Change the final input Token. Earlier positions cannot read it.
        changed = decoder_input.clone()
        changed[:, -1] = 5
        original_logits = model(decoder_input)
        changed_logits = model(changed)
    early_diff = float((original_logits[:, :-1] - changed_logits[:, :-1]).abs().max())
    final_diff = float((original_logits[:, -1] - changed_logits[:, -1]).abs().max())
    assert early_diff == 0 and final_diff > 0
    return {
        "model": "TinyDecoder: decoder-only, Pre-LN, GELU, no Cross-Attention",
        "decoder_input": as_list(decoder_input), "targets": as_list(targets),
        "logits_shape": list(logits.shape),
        "loss_before_step": float(loss.detach()),
        "lm_head_gradient_norm": grad_norm, "lm_head_weight_max_change": change,
        "loss_after_one_step": float(loss_after),
        "all_parameter_gradients_present_and_finite": True,
        "parameter_tensors_changed": sum(
            not torch.equal(p.detach(), before[name]) for name, p in model.named_parameters()
        ),
        "parameter_tensors_total": len(before),
        "future_token_check": {
            "changed_input_position": 3,
            "earlier_logits_max_change": early_diff,
            "final_position_logits_max_change": final_diff,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original, stdout = original_runs()
    with torch.no_grad():
        checks = {
            "attention": attention_check(), "multihead": multihead_check(),
            "positions": position_check(), "encoder": encoder_check(),
            "cross_attention": cross_check(),
        }
    checks["teacher_forcing"] = training_check()
    results = {
        "schema_version": 1, "run_date": "2026-10-24",
        "torch_version": torch.__version__,
        "verification_device": "cpu", "verification_dtype": "float32",
        "verification_seed": 7, "cpu_threads": 1,
        "notes": [
            "Original scripts are copied unchanged. single_attention.py selects CUDA if available; other originals use CPU.",
            "Additional verification functions use CPU. All attention maps except explicitly trained examples are untrained.",
            "Encoder uses Post-LN/ReLU/learned absolute positions. TinyDecoder uses Pre-LN/GELU/learned absolute positions.",
            "The teacher-forcing experiment performs one optimizer step, not a successfully trained language model.",
            "An all-masked row produces NaN with these educational implementations; no all-PAD input is claimed supported.",
            "Weight row sums are checked with dropout disabled. Post-dropout attention weights need not sum to one.",
        ],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
        "original_script_runs": original, "additional_checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "original_stdout.txt").write_text(stdout, encoding="utf-8")
    print(json.dumps({
        "status": "all assertions passed", "original_scripts": len(original),
        "torch_version": torch.__version__,
        "training": checks["teacher_forcing"],
    }, indent=2))


if __name__ == "__main__":
    main()
