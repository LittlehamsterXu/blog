import math
import time
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Causal Self-Attention
#    支持无缓存、torch.cat 缓存、预分配缓存
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x,
        past_kv=None,
        static_cache=None,
        cache_position=0,
    ):
        # x: [B, T, D]
        B, T, D = x.shape
        H = self.n_heads
        d = self.d_head

        # [B, T, D] -> [B, H, T, d]
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k_new = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v_new = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # ----------------------------------------------------
        # A. 预分配缓存：将新 K/V 直接写入指定位置
        # ----------------------------------------------------
        if static_cache is not None:
            k_cache, v_cache = static_cache

            end = cache_position + T

            if end > k_cache.size(-2):
                raise ValueError("KV cache capacity exceeded")

            k_cache[:, :, cache_position:end, :].copy_(k_new)
            v_cache[:, :, cache_position:end, :].copy_(v_new)

            # 只读取有效前缀
            k = k_cache[:, :, :end, :]
            v = v_cache[:, :, :end, :]

            past_len = cache_position

        # ----------------------------------------------------
        # B. 无缓存：直接使用当前输入计算的 K/V
        # ----------------------------------------------------
        elif past_kv is None:
            past_len = 0
            k, v = k_new, v_new

        # ----------------------------------------------------
        # C. 动态缓存：通过 torch.cat 追加历史 K/V
        # ----------------------------------------------------
        else:
            past_k, past_v = past_kv
            past_len = past_k.size(-2)

            k = torch.cat([past_k, k_new], dim=-2)
            v = torch.cat([past_v, v_new], dim=-2)

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        # q: [B, H, T, d]
        # k: [B, H, total_len, d]
        # scores: [B, H, T, total_len]

        total_len = k.size(-2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

        # Query 在完整序列中的位置
        q_pos = torch.arange(
            past_len,
            past_len + T,
            device=x.device,
        )

        # Key 在完整序列中的位置
        k_pos = torch.arange(
            total_len,
            device=x.device,
        )

        # 屏蔽未来 Key
        mask = k_pos[None, :] > q_pos[:, None]

        scores = scores.masked_fill(
            mask[None, None, :, :],
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)

        context = weights @ v

        # [B, H, T, d] -> [B, T, D]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(B, T, D)
        )

        output = self.out_proj(context)

        if static_cache is not None:
            # 预分配缓存由外部对象管理
            return output, None

        # 无缓存和动态缓存都返回本轮完整 K/V
        return output, (k, v)


# ============================================================
# 2. Feed Forward Network
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. Decoder Block
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        x,
        past_kv=None,
        static_cache=None,
        cache_position=0,
    ):
        attn_out, present_kv = self.attn(
            self.norm1(x),
            past_kv=past_kv,
            static_cache=static_cache,
            cache_position=cache_position,
        )

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x, present_kv


# ============================================================
# 4. Tiny Decoder-only Transformer
# ============================================================

class TinyDecoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size=100,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        max_seq_len=64,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=False,
        static_cache=None,
    ):
        B, T = input_ids.shape

        # 不允许混用两种缓存接口
        if static_cache is not None:
            if past_key_values is not None or use_cache:
                raise ValueError(
                    "Do not combine static_cache with "
                    "past_key_values/use_cache"
                )

            if len(static_cache.keys) != len(self.blocks):
                raise ValueError("Incorrect number of cache layers")

            if static_cache.keys[0].size(0) != B:
                raise ValueError("Cache batch size mismatch")

            past_len = static_cache.length

        elif past_key_values is None:
            past_len = 0

        else:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("Incorrect number of KV cache layers")

            past_len = past_key_values[0][0].size(-2)

        if past_len + T > self.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")

        # 位置编码必须延续历史位置
        positions = torch.arange(
            past_len,
            past_len + T,
            device=input_ids.device,
        )

        x = (
            self.token_emb(input_ids)
            + self.pos_emb(positions)[None, :, :]
        )

        presents = []

        for i, block in enumerate(self.blocks):

            # 预分配缓存路径
            if static_cache is not None:
                x, _ = block(
                    x,
                    static_cache=(
                        static_cache.keys[i],
                        static_cache.values[i],
                    ),
                    cache_position=past_len,
                )

            # 无缓存或 torch.cat 缓存路径
            else:
                past_kv = (
                    None
                    if past_key_values is None
                    else past_key_values[i]
                )

                x, present_kv = block(
                    x,
                    past_kv=past_kv,
                )

                if use_cache:
                    presents.append(present_kv)

        logits = self.lm_head(self.final_norm(x))

        if static_cache is not None:
            # 所有层完成写入后，统一更新有效长度
            static_cache.length += T
            return logits

        if use_cache:
            return logits, presents

        return logits


# ============================================================
# 5. 预分配 KV Cache
# ============================================================

class StaticKVCache:
    def __init__(self, model, batch_size, device):
        self.length = 0
        self.keys = []
        self.values = []

        dtype = next(model.parameters()).dtype

        for block in model.blocks:
            attn = block.attn

            shape = (
                batch_size,
                attn.n_heads,
                model.max_seq_len,
                attn.d_head,
            )

            self.keys.append(
                torch.empty(
                    shape,
                    device=device,
                    dtype=dtype,
                )
            )

            self.values.append(
                torch.empty(
                    shape,
                    device=device,
                    dtype=dtype,
                )
            )


# ============================================================
# 6. 三种生成方式
# ============================================================

@torch.no_grad()
def generate_no_cache(
    model,
    input_ids,
    max_new_tokens,
):
    """每轮重新计算完整前缀。"""

    model.eval()

    for _ in range(max_new_tokens):
        logits = model(input_ids)

        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    return input_ids


@torch.no_grad()
def generate_with_cache(
    model,
    input_ids,
    max_new_tokens,
):
    """使用 torch.cat 追加 K/V 的动态缓存。"""

    model.eval()

    if max_new_tokens <= 0:
        return input_ids

    # Prefill：处理完整 Prompt
    logits, past_key_values = model(
        input_ids,
        use_cache=True,
    )

    for step in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        # 最后一个 Token 无需再次送入模型
        if step + 1 < max_new_tokens:
            logits, past_key_values = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

    return input_ids


@torch.no_grad()
def generate_with_preallocated_cache(
    model,
    input_ids,
    max_new_tokens,
):
    """使用预分配 K/V 缓冲区的静态缓存。"""

    model.eval()

    if max_new_tokens <= 0:
        return input_ids

    if input_ids.size(1) + max_new_tokens > model.max_seq_len:
        raise ValueError("Prompt + generation exceeds max_seq_len")

    cache = StaticKVCache(
        model,
        batch_size=input_ids.size(0),
        device=input_ids.device,
    )

    # Prefill：处理完整 Prompt
    logits = model(
        input_ids,
        static_cache=cache,
    )

    for step in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        if step + 1 < max_new_tokens:
            # Decode：只输入新 Token
            logits = model(
                next_token,
                static_cache=cache,
            )

    return input_ids


# ============================================================
# 7. 正确性验证
# ============================================================

@torch.no_grad()
def check_cache_equivalence(
    model,
    prompt,
    cache_mode="cat",
    steps=3,
):
    """
    逐步比较：
    无缓存完整前缀计算 vs. 增量 KV Cache 计算。
    """

    model.eval()

    full_prefix = prompt.clone()

    full_logits = model(full_prefix)

    if cache_mode == "cat":
        cached_logits, cache = model(
            prompt,
            use_cache=True,
        )

    elif cache_mode == "static":
        cache = StaticKVCache(
            model,
            batch_size=prompt.size(0),
            device=prompt.device,
        )

        cached_logits = model(
            prompt,
            static_cache=cache,
        )

    else:
        raise ValueError("cache_mode must be 'cat' or 'static'")

    diff = (
        full_logits - cached_logits
    ).abs().max().item()

    print(
        f"[{cache_mode}] Prefill max diff: "
        f"{diff:.8e}"
    )

    for step in range(steps):
        next_token = cached_logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        full_prefix = torch.cat(
            [full_prefix, next_token],
            dim=1,
        )

        # 重新计算完整前缀作为参考
        full_logits = model(full_prefix)

        # 增量缓存路径
        if cache_mode == "cat":
            cached_logits, cache = model(
                next_token,
                past_key_values=cache,
                use_cache=True,
            )

        else:
            cached_logits = model(
                next_token,
                static_cache=cache,
            )

        diff = (
            full_logits[:, -1:, :] - cached_logits
        ).abs().max().item()

        print(
            f"[{cache_mode}] Decode {step + 1}: "
            f"max diff = {diff:.8e}"
        )


# ============================================================
# 8. 性能基准
# ============================================================

def benchmark_case(
    model,
    prompt,
    max_new_tokens,
    warmup=3,
    repeats=10,
):
    model.eval()

    def run_no_cache():
        return generate_no_cache(
            model,
            prompt.clone(),
            max_new_tokens,
        )

    def run_cat_cache():
        return generate_with_cache(
            model,
            prompt.clone(),
            max_new_tokens,
        )

    def run_static_cache():
        return generate_with_preallocated_cache(
            model,
            prompt.clone(),
            max_new_tokens,
        )

    methods = {
        "no_cache": run_no_cache,
        "cat_cache": run_cat_cache,
        "static_cache": run_static_cache,
    }

    def synchronize():
        if prompt.device.type == "cuda":
            torch.cuda.synchronize(prompt.device)

    # --------------------------------------------------------
    # 首先验证三种生成结果一致
    # --------------------------------------------------------

    outputs = {
        name: fn()
        for name, fn in methods.items()
    }

    reference = outputs["no_cache"]

    for name, output in outputs.items():
        if not torch.equal(reference, output):
            raise AssertionError(
                f"Generation mismatch: {name}"
            )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(warmup):
        for fn in methods.values():
            fn()

    synchronize()

    samples = {
        name: []
        for name in methods
    }

    # --------------------------------------------------------
    # 多次测量，交替执行顺序
    # --------------------------------------------------------

    for i in range(repeats):
        order = list(methods.items())

        if i % 2:
            order.reverse()

        for name, fn in order:
            synchronize()
            start = time.perf_counter()

            fn()

            synchronize()

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            samples[name].append(elapsed_ms)

    # --------------------------------------------------------
    # 输出统计结果
    # --------------------------------------------------------

    medians = {}

    print(
        f"\nPrompt length={prompt.size(1)}, "
        f"new tokens={max_new_tokens}"
    )

    for name, times in samples.items():
        median_ms = statistics.median(times)
        mean_ms = statistics.mean(times)

        medians[name] = median_ms

        print(
            f"{name:12s} | "
            f"median={median_ms:.3f} ms | "
            f"mean={mean_ms:.3f} ms"
        )

    print(
        "Speedup (no_cache / cat_cache): "
        f"{medians['no_cache'] / medians['cat_cache']:.3f}x"
    )

    print(
        "Speedup (no_cache / static_cache): "
        f"{medians['no_cache'] / medians['static_cache']:.3f}x"
    )

    print(
        "Speedup (cat_cache / static_cache): "
        f"{medians['cat_cache'] / medians['static_cache']:.3f}x"
    )



@torch.no_grad()
def benchmark_prefill_and_decode(
    model,
    prompt_len,
    warmup=5,
    repeats=30,
):
    model.eval()

    device = next(model.parameters()).device
    assert prompt_len + 1 <= model.max_seq_len

    # 构造固定的 Prompt 和一个新 Token
    prefix = torch.arange(
        1, prompt_len + 1,
        device=device,
    ).unsqueeze(0)

    next_token = torch.tensor(
        [[prompt_len + 1]],
        dtype=torch.long,
        device=device,
    )

    full_input = torch.cat(
        [prefix, next_token],
        dim=1,
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure_methods(methods):
        """
        methods:
            name -> (prepare, run)

        prepare 在计时区间外执行；
        run 是本次真正测量的操作。
        """
        samples = {name: [] for name in methods}

        # Warm-up
        for _ in range(warmup):
            for prepare, run in methods.values():
                state = prepare()
                sync()
                run(state)
                sync()

        # 交替执行顺序，降低固定顺序的影响
        for i in range(repeats):
            items = list(methods.items())

            if i % 2:
                items.reverse()

            for name, (prepare, run) in items:
                state = prepare()
                sync()

                start = time.perf_counter()
                run(state)
                sync()

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                samples[name].append(elapsed_ms)

        results = {}

        for name, values in samples.items():
            median_ms = statistics.median(values)
            mean_ms = statistics.mean(values)
            results[name] = median_ms

            print(
                f"{name:25s} "
                f"median={median_ms:.4f} ms, "
                f"mean={mean_ms:.4f} ms"
            )

        return results

    # ========================================================
    # A. Prefill
    # ========================================================

    print(f"\n===== Prefill: prompt_len={prompt_len} =====")

    # 这份缓存会被重复使用；每次测量前重置有效长度
    reusable_cache = StaticKVCache(
        model,
        batch_size=1,
        device=device,
    )

    def prepare_static_prefill():
        reusable_cache.length = 0
        return reusable_cache

    def run_static_prefill(cache):
        return model(prefix, static_cache=cache)

    def run_static_alloc_and_prefill(_):
        # 此方法的计时包含缓存分配
        cache = StaticKVCache(
            model,
            batch_size=1,
            device=device,
        )
        return model(prefix, static_cache=cache)

    prefill_methods = {
        "no_cache_prefill": (
            lambda: None,
            lambda _: model(prefix),
        ),
        "cat_cache_prefill": (
            lambda: None,
            lambda _: model(prefix, use_cache=True),
        ),
        "static_preallocated": (
            prepare_static_prefill,
            run_static_prefill,
        ),
        "static_alloc_plus_prefill": (
            lambda: None,
            run_static_alloc_and_prefill,
        ),
    }

    measure_methods(prefill_methods)

    # ========================================================
    # B. 单步 Decode
    # ========================================================

    print(
        f"\n===== Decode: context_len={prompt_len} ====="
    )

    # 先检查三种路径的最后一个位置 Logits 是否一致
    reference = model(full_input)[:, -1:, :]

    _, cat_past = model(
        prefix,
        use_cache=True,
    )

    cat_logits, _ = model(
        next_token,
        past_key_values=cat_past,
        use_cache=True,
    )

    check_cache = StaticKVCache(
        model,
        batch_size=1,
        device=device,
    )

    model(prefix, static_cache=check_cache)

    static_logits = model(
        next_token,
        static_cache=check_cache,
    )

    cat_diff = (
        reference - cat_logits
    ).abs().max().item()

    static_diff = (
        reference - static_logits
    ).abs().max().item()

    print(f"Cat-cache max logit diff: {cat_diff:.8e}")
    print(f"Static-cache max logit diff: {static_diff:.8e}")

    # 为每次 Decode 测量准备同样长度的历史缓存
    def prepare_cat_decode():
        _, past = model(
            prefix,
            use_cache=True,
        )
        return past

    def prepare_static_decode():
        cache = StaticKVCache(
            model,
            batch_size=1,
            device=device,
        )
        model(prefix, static_cache=cache)
        return cache

    decode_methods = {
        "no_cache_full_prefix": (
            lambda: full_input,
            lambda x: model(x),
        ),
        "cat_cache_one_token": (
            prepare_cat_decode,
            lambda past: model(
                next_token,
                past_key_values=past,
                use_cache=True,
            ),
        ),
        "static_cache_one_token": (
            prepare_static_decode,
            lambda cache: model(
                next_token,
                static_cache=cache,
            ),
        ),
    }

    measure_methods(decode_methods)


# ============================================================
# 9. 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = TinyDecoderOnly(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        max_seq_len=64,
    ).to(device)

    prompt = torch.tensor(
        [[1, 15, 27]],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # A. 基础前向检查
    # --------------------------------------------------------

    with torch.no_grad():
        logits = model(prompt)

    print("\n===== Basic Forward =====")
    print("Prompt shape:", tuple(prompt.shape))
    print("Logits shape:", tuple(logits.shape))
    print(
        "All logits finite:",
        torch.isfinite(logits).all().item(),
    )

    # --------------------------------------------------------
    # B. 无缓存生成
    # --------------------------------------------------------

    print("\n===== No Cache Generation =====")

    no_cache_output = generate_no_cache(
        model,
        prompt.clone(),
        max_new_tokens=3,
    )

    print("Generated:", no_cache_output.tolist())

    # --------------------------------------------------------
    # C. 验证历史位置不受未来 Token 影响
    # --------------------------------------------------------

    print("\n===== Causal Mask Check =====")

    prefix = torch.tensor(
        [[1, 15, 27]],
        dtype=torch.long,
        device=device,
    )

    extended = torch.tensor(
        [[1, 15, 27, 56, 39]],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits_prefix = model(prefix)
        logits_extended = model(extended)

    max_diff = (
        logits_prefix - logits_extended[:, :3, :]
    ).abs().max().item()

    print("Historical logits max diff:", max_diff)

    print(
        "Historical logits unchanged:",
        torch.allclose(
            logits_prefix,
            logits_extended[:, :3, :],
            atol=1e-6,
            rtol=1e-5,
        ),
    )

    # --------------------------------------------------------
    # D. 检查动态缓存与预分配缓存的数值等价性
    # --------------------------------------------------------

    print("\n===== Cat Cache Equivalence =====")

    check_cache_equivalence(
        model,
        prompt,
        cache_mode="cat",
        steps=3,
    )

    print("\n===== Static Cache Equivalence =====")

    check_cache_equivalence(
        model,
        prompt,
        cache_mode="static",
        steps=3,
    )

    # --------------------------------------------------------
    # E. 比较三种生成结果
    # --------------------------------------------------------

    print("\n===== Generation Comparison =====")

    cat_output = generate_with_cache(
        model,
        prompt.clone(),
        max_new_tokens=3,
    )

    static_output = generate_with_preallocated_cache(
        model,
        prompt.clone(),
        max_new_tokens=3,
    )

    print("No cache:   ", no_cache_output.tolist())
    print("Cat cache:  ", cat_output.tolist())
    print("Static cache:", static_output.tolist())

    print(
        "Cat equals no-cache:",
        torch.equal(no_cache_output, cat_output),
    )

    print(
        "Static equals no-cache:",
        torch.equal(no_cache_output, static_output),
    )

    # --------------------------------------------------------
    # F. 性能基准
    # --------------------------------------------------------

    print("\n===== Generation Benchmark =====")

    for prompt_len, new_tokens in [
        (3, 3),
        (16, 8),
        (32, 16),
    ]:
        benchmark_prompt = torch.arange(
            1,
            prompt_len + 1,
            device=device,
        ).unsqueeze(0)

        benchmark_case(
            model,
            benchmark_prompt,
            max_new_tokens=new_tokens,
            warmup=3,
            repeats=10,
        )
    print("\n===== Prefill vs Decode =====")

    for prompt_len in [3, 16, 32, 48]:
        benchmark_prefill_and_decode(
            model,
            prompt_len=prompt_len,
            warmup=5,
            repeats=30,
        )