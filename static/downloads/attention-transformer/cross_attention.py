
import math
import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q 来自 Decoder
        self.q_proj = nn.Linear(d_model, d_model)

        # K、V 来自 Encoder
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, decoder_hidden, encoder_output):
        B, T_dec, D = decoder_hidden.shape
        B_enc, T_enc, D_enc = encoder_output.shape

        assert B == B_enc
        assert D == D_enc

        H = self.n_heads
        d = self.d_head

        # 1. 不同来源的输入分别生成 Q、K、V
        q = self.q_proj(decoder_hidden)
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)

        # 2. 拆分 Head
        q = q.reshape(B, T_dec, H, d).transpose(1, 2)
        k = k.reshape(B, T_enc, H, d).transpose(1, 2)
        v = v.reshape(B, T_enc, H, d).transpose(1, 2)

        # 3. [B,H,T_dec,d] @ [B,H,d,T_enc]
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)

        # 4. 对源序列位置做 Softmax
        weights = torch.softmax(scores, dim=-1)

        # 5. 聚合 Encoder 的 Value
        context = weights @ v

        # 6. 合并 Head
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(B, T_dec, D)

        output = self.out_proj(context)

        return output, weights


if __name__ == "__main__":
    torch.manual_seed(7)

    encoder_output = torch.randn(2, 5, 16)
    decoder_hidden = torch.randn(2, 3, 16)

    cross_attn = CrossAttention(d_model=16, n_heads=4)

    output, weights = cross_attn(
        decoder_hidden,
        encoder_output,
    )

    print("Encoder output:", encoder_output.shape)
    print("Decoder hidden:", decoder_hidden.shape)
    print("Attention weights:", weights.shape)
    print("Cross-attention output:", output.shape)
    print("Weight row sums:", weights.sum(dim=-1))
