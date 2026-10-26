"""Reproduce the position-encoding article's small CPU experiments.

Run with Python 3.10+ and PyTorch: python -B verify_positions.py
No training, dataset or pretrained weight download is involved.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import runpy

import torch

from decoder_only import TinyDecoderOnly
from position_encoding import apply_rope, rotate_pairs, sinusoidal_positions

ROOT = Path(__file__).resolve().parent


def arr(x):
    return x.detach().cpu().tolist()


def error(a, b):
    return float((a - b).abs().max())


def original_sinusoidal():
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        ns = runpy.run_path(str(ROOT / "positional_encoding_experiment.py"))
    return {
        "dtype": "float32",
        "identity_qkv": True,
        "token_embedding": arr(ns["token_embedding"]),
        "positional_encoding": arr(ns["pe"][0]),
        "weights_without_position": arr(ns["weights0"][0]),
        "weights_with_position": arr(ns["weights1"][0]),
        "context_without_position": arr(ns["context0"][0]),
        "context_with_position": arr(ns["context1"][0]),
        "maximum_weight_change": error(ns["weights0"], ns["weights1"]),
    }, capture.getvalue()


def permutation_check():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, 4, 4, generator=generator, dtype=torch.float64)
    wq, wk, wv = [torch.randn(4, 4, generator=generator, dtype=torch.float64) / 2
                   for _ in range(3)]
    perm = torch.tensor([2, 0, 3, 1])
    position_table = sinusoidal_positions(torch.arange(4), 4)

    def attention(value):
        q, k, v = value @ wq, value @ wk, value @ wv
        weights = (q @ k.transpose(-2, -1) / 2).softmax(-1)
        return weights, weights @ v

    weights, y = attention(x)
    shuffled_weights, shuffled_y = attention(x[:, perm])
    torch.testing.assert_close(shuffled_y, y[:, perm], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(shuffled_weights, weights[:, perm][:, :, perm],
                               atol=1e-12, rtol=1e-12)
    positioned_weights, positioned_y = attention(x + position_table)
    permuted_fixed_weights, permuted_fixed_y = attention(x[:, perm] + position_table)
    changed = error(permuted_fixed_y, positioned_y[:, perm])
    assert changed > 1e-3
    return {
        "seed": 7, "input_shape": list(x.shape), "mask": None,
        "permutation": arr(perm), "weights": arr(weights[0]),
        "permuted_weights": arr(shuffled_weights[0]),
        "without_position_output_max_error": error(shuffled_y, y[:, perm]),
        "without_position_weights_max_error": error(
            shuffled_weights, weights[:, perm][:, :, perm]),
        "fixed_position_output_max_difference": changed,
        "fixed_position_weights": arr(positioned_weights[0]),
        "tokens_permuted_positions_fixed_weights": arr(permuted_fixed_weights[0]),
    }


def additive_check():
    d, base = 8, 10000.
    inv_freq = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    pairs = [(3, 5), (13, 15), (3, 6)]
    records = []
    for m, n in pairs:
        pe = sinusoidal_positions(torch.tensor([m, n]), d)
        dot = pe[0] @ pe[1]
        expected = ((m - n) * inv_freq).cos().sum()
        torch.testing.assert_close(dot, expected, atol=1e-12, rtol=1e-12)
        records.append({"m": m, "n": n, "pe_dot": float(dot),
                        "cosine_sum": float(expected)})
    g = torch.Generator().manual_seed(11)
    xm, xn = [torch.randn(d, generator=g, dtype=torch.float64) for _ in range(2)]
    wq, wk = [torch.randn(d, d, generator=g, dtype=torch.float64) / math.sqrt(d)
              for _ in range(2)]
    terms = []
    for m, n in [(3, 5), (13, 15)]:
        pm, pn = sinusoidal_positions(torch.tensor([m, n]), d)
        xq, pk = xm @ wq, pn @ wk
        pq, xk = pm @ wq, xn @ wk
        parts = torch.stack((xq @ xk, xq @ pk, pq @ xk, pq @ pk))
        score = ((xm + pm) @ wq) @ ((xn + pn) @ wk)
        torch.testing.assert_close(parts.sum(), score, atol=1e-12, rtol=1e-12)
        terms.append({"m": m, "n": n, "content_content": float(parts[0]),
                      "content_position": float(parts[1]),
                      "position_content": float(parts[2]),
                      "position_position_after_projection": float(parts[3]),
                      "total_raw_dot": float(score),
                      "sum_error": float((parts.sum() - score).abs())})
    assert abs(terms[0]["total_raw_dot"] - terms[1]["total_raw_dot"]) > 1e-3
    return {"d_model": d, "base": base, "inverse_frequencies": arr(inv_freq),
            "pe_inner_product": records, "additive_dot_terms": terms,
            "term_order": ["content-content", "content-position",
                           "position-content", "position-position"]}


def rope_check():
    g = torch.Generator().manual_seed(19)
    q = torch.randn(1, 2, 4, 8, generator=g, dtype=torch.float64)
    k = torch.randn(1, 2, 4, 8, generator=g, dtype=torch.float64)
    ids = torch.arange(4)
    rq, rk = apply_rope(q, ids), apply_rope(k, ids)
    raw_scores = rq @ rk.transpose(-2, -1)
    shifted = apply_rope(q, ids + 10) @ apply_rope(k, ids + 10).transpose(-2, -1)
    torch.testing.assert_close(q.norm(dim=-1), rq.norm(dim=-1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((q * k).sum(-1), (rq * rk).sum(-1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(raw_scores, shifted, atol=1e-12, rtol=1e-12)
    # Isolate two fixed content vectors and change only their positions.
    q_one, k_one = q[:, :, :1], k[:, :, :1]
    dots = []
    for m, n in [(3, 5), (13, 15), (3, 6)]:
        dot = (apply_rope(q_one, torch.tensor([m])) *
               apply_rope(k_one, torch.tensor([n]))).sum(-1)
        relative_dot = (q_one * apply_rope(k_one, torch.tensor([n - m]))).sum(-1)
        torch.testing.assert_close(dot, relative_dot, atol=1e-12, rtol=1e-12)
        dots.append({"m": m, "n": n, "head_0_raw_dot": float(dot[0, 0, 0]),
                     "relative_formula_error": error(dot, relative_dot)})
    assert abs(dots[0]["head_0_raw_dot"] - dots[2]["head_0_raw_dot"]) > 1e-3
    float_scores = apply_rope(q.float(), ids) @ apply_rope(k.float(), ids).transpose(-2, -1)
    float_shifted = apply_rope(q.float(), ids + 10) @ apply_rope(k.float(), ids + 10).transpose(-2, -1)
    torch.testing.assert_close(float_scores, float_shifted, atol=5e-6, rtol=5e-6)
    batched_ids = torch.stack((ids, ids + 10))
    expanded_q = q.expand(2, -1, -1, -1)
    batched_result = apply_rope(expanded_q, batched_ids)
    torch.testing.assert_close(batched_result[0], rq[0])
    torch.testing.assert_close(batched_result[1], apply_rope(q, ids + 10)[0])
    for bad in [torch.randn(1, 2, 4, 7), torch.randn(1, 4, 8)]:
        try:
            apply_rope(bad, ids)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid RoPE input was accepted")
    return {
        "pair_layout": "adjacent even/odd pairs", "base": 10000.,
        "q_k_shape": list(q.shape), "v_rotated": False,
        "norm_max_error_float64": error(q.norm(dim=-1), rq.norm(dim=-1)),
        "same_position_dot_max_error_float64": error((q*k).sum(-1), (rq*rk).sum(-1)),
        "shared_shift_score_max_error_float64": error(raw_scores, shifted),
        "shared_shift_score_max_error_float32": error(float_scores, float_shifted),
        "float32_vs_float64_score_max_error": error(float_scores.double(), raw_scores),
        "head_0_scores": arr(raw_scores[0, 0]),
        "head_0_scores_shared_shift_10": arr(shifted[0, 0]),
        "fixed_content_position_pairs": dots,
        "batched_position_shape_checked": list(batched_ids.shape),
    }


def cache_position_check():
    torch.manual_seed(7)
    model = TinyDecoderOnly(vocab_size=20, d_model=16, n_heads=4,
                            d_ff=32, n_layers=2, max_seq_len=16).double().eval()
    ids = torch.tensor([[2, 7, 3, 11, 5, 9]])
    prefix_len = 4
    with torch.inference_mode():
        full_logits = model(ids)
        prefill_logits, prefix_cache = model(ids[:, :prefix_len], use_cache=True)
        torch.testing.assert_close(prefill_logits, full_logits[:, :prefix_len],
                                   atol=1e-12, rtol=1e-12)
        new_logits, new_cache = model(ids[:, prefix_len:],
                                      past_key_values=prefix_cache, use_cache=True)
        torch.testing.assert_close(new_logits, full_logits[:, prefix_len:],
                                   atol=1e-12, rtol=1e-12)
        # Change only the position-table lookup. The model and each Attention
        # still see the real cache length 4, so the causal Mask is unchanged.
        seen_ids = []
        def reset_embedding_positions(module, args):
            seen_ids.append(args[0].clone())
            return (args[0] - prefix_len,)
        handle = model.pos_emb.register_forward_pre_hook(reset_embedding_positions)
        try:
            wrong_logits, _ = model(ids[:, prefix_len:],
                                     past_key_values=prefix_cache, use_cache=True)
        finally:
            handle.remove()
        wrong_error = error(wrong_logits, full_logits[:, prefix_len:])
        assert wrong_error > 1e-3
        # One-token decoding repeats this offset rule over the whole sequence.
        cache = None
        pieces = []
        for t in range(ids.size(1)):
            piece, cache = model(ids[:, t:t+1], past_key_values=cache, use_cache=True)
            pieces.append(piece)
        tokenwise = torch.cat(pieces, dim=1)
        torch.testing.assert_close(tokenwise, full_logits, atol=1e-12, rtol=1e-12)
    return {
        "model": "Original TinyDecoderOnly, learned absolute position table",
        "untrained_model": True, "dtype": "float64", "seed": 7,
        "config": {"vocab_size": 20, "d_model": 16, "n_heads": 4,
                   "d_ff": 32, "n_layers": 2, "max_seq_len": 16},
        "input_ids": arr(ids), "prefill_length": prefix_len,
        "full_position_ids": list(range(6)),
        "correct_cached_position_ids": list(range(4, 6)),
        "reset_position_ids": [0, 1],
        "prefill_logits_max_error": error(prefill_logits, full_logits[:, :prefix_len]),
        "correct_cached_logits_max_error": error(new_logits, full_logits[:, prefix_len:]),
        "reset_embedding_logits_max_difference": wrong_error,
        "one_token_cached_logits_max_error": error(tokenwise, full_logits),
        "correct_attention_query_positions": [4, 5],
        "correct_attention_key_positions": list(range(6)),
        "full_last_token_logits": arr(full_logits[0, -1]),
        "correct_cached_last_token_logits": arr(new_logits[0, -1]),
        "wrong_cached_last_token_logits": arr(wrong_logits[0, -1]),
        "final_cache_shape_per_layer": list(new_cache[0][0].shape),
        "intervention": "pre-hook changes only pos_emb lookup [4,5] to [0,1]; cache length and causal Mask remain 4-based",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    original, stdout = original_sinusoidal()
    result = {
        "run_date": "2026-10-26", "torch_version": torch.__version__,
        "device": "cpu", "dtype_for_added_checks": "float64",
        "cpu_threads": 1,
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ['positional_encoding_experiment.py', 'decoder_only.py']},
        "original_sinusoidal_experiment": original,
        "permutation": permutation_check(),
        "additive_positions": additive_check(),
        "rope": rope_check(),
        "cache_positions": cache_position_check(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (args.output_dir/'original_stdout.txt').write_text(stdout, encoding='utf-8')
    print(json.dumps({
        "permutation_error": result['permutation']['without_position_output_max_error'],
        "fixed_position_difference": result['permutation']['fixed_position_output_max_difference'],
        "rope_shift_error": result['rope']['shared_shift_score_max_error_float64'],
        "cache_correct_error": result['cache_positions']['correct_cached_logits_max_error'],
        "cache_position_reset_difference": result['cache_positions']['reset_embedding_logits_max_difference'],
    }, indent=2))
    print('PASS: original sine experiment, permutation, additive terms, RoPE, and cache positions')


if __name__ == '__main__':
    main()
