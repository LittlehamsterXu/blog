import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 保存是否启用因果掩码
        self.causal = causal

        # 三组可学习的投影参数
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # 将所有 Head 的结果融合
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        d = self.d_head

        # 1. 生成 Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分 Head：[B,T,D] -> [B,H,T,d]
        q = q.reshape(B, T, H, d).transpose(1, 2)
        k = k.reshape(B, T, H, d).transpose(1, 2)
        v = v.reshape(B, T, H, d).transpose(1, 2)

        # 3. 计算每个 Head 的注意力分数
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)

        # 4. 可选的 Causal Mask
        if self.causal:
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        # 5. Softmax：将分数转换为注意力权重
        weights = torch.softmax(scores, dim=-1)

        # 6. 使用注意力权重聚合 V
        context = weights @ v

        # 7. 合并 Head：[B,H,T,d] -> [B,T,D]
        context = context.transpose(1, 2).reshape(B, T, D)

        # 8. 输出投影
        output = self.out_proj(context)

        return output, weights


if __name__ == "__main__":
    # # 一个样本，包含三个 Token，每个 Token 为 4 维
    # x = torch.tensor([[
    #     [1.0, 0.0, 1.0, 0.0],
    #     [0.0, 1.0, 0.0, 1.0],
    #     [1.0, 1.0, 0.0, 0.0],
    # ]])

    # model = MultiHeadSelfAttention(d_model=4, n_heads=2, causal=False)
    # output, weights = model(x)

    # print("Input shape: ", x.shape)
    # print("Weights shape:", weights.shape)
    # print("Output shape: ", output.shape)

    # print("\nHead 1 attention weights:")
    # print(weights[0, 0])

    # print("\nHead 2 attention weights:")
    # print(weights[0, 1])

    # print("\nEach row's weight sum:")
    # print(weights.sum(dim=-1))

    # print("\nDifference between the two heads:")
    # print((weights[0, 0] - weights[0, 1]).abs().max().item())

    model = MultiHeadSelfAttention(d_model=4, n_heads=2)
    x = torch.tensor([[
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]])

    # 同一组参数：开启 Causal Mask
    model.causal = True
    _, weights_causal = model(x)

    # 同一组参数：关闭 Causal Mask
    model.causal = False
    _, weights_full = model(x)

    print("Causal Attention:")
    print(weights_causal[0, 0])

    print("\nFull Self-Attention:")
    print(weights_full[0, 0])

    print("\n两种模式下，第三个 Token 的权重：")
    print("Causal:", weights_causal[0, 0, 2])
    print("Full:  ", weights_full[0, 0, 2])
