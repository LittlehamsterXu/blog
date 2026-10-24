import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_printoptions(precision=3, sci_mode=False)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=16, n_heads=2, causal=True):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head

        q = self.q_proj(x).reshape(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, d).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(d)

        if self.causal:
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(context), weights


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=2, d_model=16, n_heads=2, max_len=5):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=True)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        _, T = token_ids.shape

        # Token ID -> 向量，并加入可学习的位置表示
        x = self.token_emb(token_ids) + self.pos_emb[:, :T, :]

        # 多头注意力
        x, weights = self.attn(x)

        # 每个位置输出两个类别的 logits
        logits = self.lm_head(x)
        return logits, weights


# 两条交替序列
tokens = torch.tensor([
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
])

# 输入和预测目标错开一个位置
x = tokens[:, :-1]   # [2, 5]
y = tokens[:, 1:]    # [2, 5]

model = TinyLanguageModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def calculate_loss(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, 2),
        targets.reshape(-1),
    )


# 记录训练前的损失、注意力权重和 Q 投影参数
with torch.no_grad():
    initial_logits, initial_weights = model(x)
    initial_loss = calculate_loss(initial_logits, y).item()
    initial_weights = initial_weights[0, 0].clone()

initial_q = model.attn.q_proj.weight.detach().clone()

print(f"Initial loss: {initial_loss:.4f}")

# 训练
for step in range(300):
    optimizer.zero_grad(set_to_none=True)

    logits, weights = model(x)
    loss = calculate_loss(logits, y)

    # 反向传播：计算所有相关参数的梯度
    loss.backward()

    if step == 0:
        print(
            "Q projection gradient norm:",
            model.attn.q_proj.weight.grad.norm().item(),
        )
        print(
            "K projection gradient norm:",
            model.attn.k_proj.weight.grad.norm().item(),
        )
        print(
            "V projection gradient norm:",
            model.attn.v_proj.weight.grad.norm().item(),
        )

    # 更新参数
    optimizer.step()

# 训练后的评估
with torch.no_grad():
    final_logits, final_weights = model(x)
    final_loss = calculate_loss(final_logits, y).item()

print(f"\nFinal loss: {final_loss:.4f}")
print(
    "Q projection parameter change:",
    (model.attn.q_proj.weight - initial_q).abs().max().item(),
)

print("\nFirst sample, Head 1 attention before training:")
print(initial_weights)

print("\nFirst sample, Head 1 attention after training:")
print(final_weights[0, 0])

print("\nPredictions:")
print(final_logits.argmax(dim=-1))

print("\nTargets:")
print(y)