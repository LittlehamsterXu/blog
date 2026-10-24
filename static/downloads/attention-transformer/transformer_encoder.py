
import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        B, T, D = x.shape
        H = self.n_heads
        d = self.d_head

        # 1. 投影得到 Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. [B, T, D] -> [B, H, T, d]
        q = q.reshape(B, T, H, d).transpose(1, 2)
        k = k.reshape(B, T, H, d).transpose(1, 2)
        v = v.reshape(B, T, H, d).transpose(1, 2)

        # 3. 计算注意力分数：[B, H, T, T]
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)

        # 4. Padding Mask：屏蔽无效的 Key
        if padding_mask is not None:
            if padding_mask.shape != (B, T):
                raise ValueError(
                    f"Expected padding_mask shape {(B, T)}, "
                    f"got {tuple(padding_mask.shape)}"
                )

            padding_mask = padding_mask.to(
                device=x.device, dtype=torch.bool
            )

            # [B, T] -> [B, 1, 1, T]
            # 广播到所有 Head 和 Query
            scores = scores.masked_fill(
                padding_mask[:, None, None, :],
                float("-inf"),
            )

        # 5. Softmax：被屏蔽位置的权重为 0
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # 6. 使用注意力权重聚合 V
        context = weights @ v

        # 7. 合并 Head：[B, H, T, d] -> [B, T, D]
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(B, T, D)

        # 8. 输出投影
        output = self.out_proj(context)

        return output, weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, dropout
        )
        self.ffn = FeedForward(
            d_model, d_ff, dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # 子层 1：Attention + Residual + LayerNorm
        attn_output, weights = self.attn(
            x, padding_mask=padding_mask
        )
        x = self.norm1(x + self.dropout1(attn_output))

        # 子层 2：FFN + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x, weights


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        dropout=0.0,
        pad_token_id=0,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id

        # PAD 的 Token Embedding 不参与梯度更新
        self.token_emb = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )

        # 可学习的绝对位置嵌入
        self.pos_emb = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, token_ids, padding_mask=None):
        B, T = token_ids.shape

        if T > self.pos_emb.size(1):
            raise ValueError("Sequence length exceeds max_len")

        # 如果调用者没有提供 Mask，就根据 PAD ID 自动生成
        if padding_mask is None:
            padding_mask = token_ids.eq(self.pad_token_id)
        else:
            if padding_mask.shape != (B, T):
                raise ValueError(
                    f"Expected padding_mask shape {(B, T)}, "
                    f"got {tuple(padding_mask.shape)}"
                )

            padding_mask = padding_mask.to(
                device=token_ids.device, dtype=torch.bool
            )

        # 1. Token Embedding + Position Embedding
        x = self.token_emb(token_ids)
        x = x + self.pos_emb[:, :T, :]

        # 2. 保存每一层的隐藏状态和注意力权重
        all_hidden_states = [x.detach().clone()]
        all_weights = []

        # 3. 依次经过所有 Encoder Block
        for layer in self.layers:
            x, weights = layer(
                x, padding_mask=padding_mask
            )

            all_weights.append(weights)
            all_hidden_states.append(x.detach().clone())

        return x, all_weights, all_hidden_states


if __name__ == "__main__":
    torch.manual_seed(7)

    model = TransformerEncoder(
        vocab_size=100,
        max_len=32,
        d_model=16,
        n_heads=4,
        d_ff=64,
        n_layers=6,
        dropout=0.0,
        pad_token_id=0,
    )

    model.eval()

    # 样本 1：三个有效 Token，两个 PAD
    # 样本 2：五个有效 Token
    token_ids = torch.tensor([
        [11, 25, 70, 0, 0],
        [9,   3,  2, 4, 1],
    ])

    padding_mask = token_ids.eq(0)

    print("Token IDs:")
    print(token_ids)

    print("\nPadding mask (True means PAD):")
    print(padding_mask)

    with torch.no_grad():
        # 先构造进入第一个 Block 前的表示
        x0 = model.token_emb(token_ids)
        x0 = x0 + model.pos_emb[:, :token_ids.size(1), :]

        # 对照：不使用 Padding Mask
        _, weights_unmasked = model.layers[0].attn(x0)

        # 对照：使用 Padding Mask
        _, weights_masked = model.layers[0].attn(
            x0, padding_mask=padding_mask
        )

        print("\nFirst sample, Head 1, Query 1:")
        print("Without mask:", weights_unmasked[0, 0, 0])
        print("With mask:   ", weights_masked[0, 0, 0])

        # 完整的六层 Encoder
        output, all_weights, all_hidden_states = model(
            token_ids, padding_mask=padding_mask
        )

    print("\nEncoder output shape:", output.shape)
    print("Hidden states:", len(all_hidden_states))
    print("Number of layers:", len(all_weights))

    print("\nFirst layer, first sample, Head 1:")
    print(all_weights[0][0, 0])

    # 检查所有层、所有样本都没有关注 PAD Key
    for layer_idx, weights in enumerate(all_weights, start=1):
        pad_key_mask = padding_mask[:, None, None, :].expand_as(
            weights
        )

        pad_weights = weights.masked_select(pad_key_mask)

        assert torch.allclose(
            pad_weights,
            torch.zeros_like(pad_weights),
            atol=1e-7,
        ), f"Layer {layer_idx}: PAD weights are not zero"

        # dropout=0.0，因此每一行权重和应为 1
        row_sums = weights.sum(dim=-1)

        assert torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-6,
        ), f"Layer {layer_idx}: attention row sums are not 1"

    print("\nAll layers: PAD key weights are zero.")
    print("All layers: attention row sums are 1.")

    print(
        "All outputs finite:",
        torch.isfinite(output).all().item(),
    )
    print("Padding mask test passed!")

    # 保留之前的隐藏状态追踪实验
    position = 2
    token_id = token_ids[0, position].item()

    print(f"\nTracked token ID: {token_id}")
    print(f"Position: {position}")

    initial = all_hidden_states[0][0, position]

    for i, states in enumerate(all_hidden_states):
        h = states[0, position]

        cosine = torch.nn.functional.cosine_similarity(
            initial, h, dim=0
        ).item()

        delta = (h - initial).norm().item()

        name = (
            "Embedding + Position"
            if i == 0 else f"Block {i}"
        )

        print(f"\n{name}")
        print("First 4 dimensions:", h[:4])
        print("Distance from initial:", round(delta, 4))
        print("Cosine similarity to initial:", round(cosine, 4))
