
import math
import torch
from torch import nn


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

    def forward(self, x, past_kv=None):
        B, T, D = x.shape
        H = self.n_heads
        d = self.d_head

        # 1. 投影并拆分 Head
        q = self.q_proj(x).reshape(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, d).transpose(1, 2)

        # 2. 读取历史缓存
        past_len = 0

        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # 当前 Query 长度为 T，Key 长度为 past_len + T
        total_len = k.size(2)

        # 3. 计算注意力分数
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)

        # 4. Causal Mask
        # Query 的绝对位置从 past_len 开始
        q_pos = past_len + torch.arange(T, device=x.device)
        k_pos = torch.arange(total_len, device=x.device)

        # 禁止读取位于 Query 之后的 Key
        future_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)

        scores = scores.masked_fill(
            future_mask[None, None, :, :],
            float("-inf"),
        )

        weights = torch.softmax(scores, dim=-1)

        # 5. 聚合 Value
        context = weights @ v
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(B, T, D)

        output = self.out_proj(context)

        # 返回本层完整缓存，包含历史和当前 Token
        return output, (k, v)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(
            self.norm1(x), past_kv
        )

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x, present_kv


class TinyDecoder(nn.Module):
    def __init__(
        self,
        vocab_size=20,
        max_len=32,
        d_model=16,
        n_heads=4,
        d_ff=64,
        n_layers=2,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        """完整序列计算：使用 Causal Mask。"""
        B, T = token_ids.shape

        positions = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)[None, :, :]

        for layer in self.layers:
            x, _ = layer(x)

        return self.lm_head(self.final_norm(x))

    def step(self, token_id, position, past_kvs=None):
        """增量解码：输入一个新 Token，更新各层缓存。"""
        assert token_id.size(1) == 1

        pos = torch.tensor([position], device=token_id.device)

        x = self.token_emb(token_id) + self.pos_emb(pos)[None, :, :]

        new_kvs = []

        for i, layer in enumerate(self.layers):
            past_kv = None if past_kvs is None else past_kvs[i]

            x, present_kv = layer(x, past_kv)
            new_kvs.append(present_kv)

        logits = self.lm_head(self.final_norm(x))

        return logits, new_kvs


if __name__ == "__main__":
    torch.manual_seed(7)

    model = TinyDecoder().eval()

    # 两个样本，每个样本包含 5 个 Token
    token_ids = torch.tensor([
        [1, 7, 3, 9, 2],
        [4, 6, 8, 5, 1],
    ])

    B, T = token_ids.shape

    with torch.inference_mode():
        # 方法 A：一次性处理整个序列
        full_logits = model(token_ids)

        # 方法 B：逐 Token 增量处理，使用 KV Cache
        past_kvs = None
        cached_logits = []

        print("Incremental decoding:")

        for t in range(T):
            current_token = token_ids[:, t:t + 1]

            logits, past_kvs = model.step(
                current_token,
                position=t,
                past_kvs=past_kvs,
            )

            cached_logits.append(logits)

            cache_len = past_kvs[0][0].size(2)

            print(
                f"Step {t + 1}: "
                f"input={tuple(current_token.shape)}, "
                f"logits={tuple(logits.shape)}, "
                f"cache_len={cache_len}"
            )

        cached_logits = torch.cat(cached_logits, dim=1)

        print("\nFull logits shape:", full_logits.shape)
        print("Cached logits shape:", cached_logits.shape)

        max_diff = (full_logits - cached_logits).abs().max().item()

        print("Maximum absolute difference:", max_diff)

        assert torch.allclose(
            full_logits,
            cached_logits,
            atol=1e-5,
            rtol=1e-5,
        ), "Full-sequence and cached decoding differ!"

        print("PASS: Full decoding and KV-cache decoding match.")

        # 观察每一层的最终缓存形状
        for i, (k, v) in enumerate(past_kvs):
            print(
                f"Layer {i + 1}: "
                f"K={tuple(k.shape)}, "
                f"V={tuple(v.shape)}"
            )
