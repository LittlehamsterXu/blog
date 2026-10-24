
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_printoptions(precision=3, sci_mode=False)

T = 5
D = 16
H = 2


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

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

        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(context), weights


class TinyModel(nn.Module):
    def __init__(self, use_attention):
        super().__init__()
        self.use_attention = use_attention

        self.token_emb = nn.Embedding(2, D)
        self.pos_emb = nn.Parameter(torch.randn(1, T, D) * 0.02)

        if use_attention:
            self.attn = MultiHeadSelfAttention(D, H)

        self.lm_head = nn.Linear(D, 2)

    def forward(self, ids):
        # x = self.token_emb(ids) # 如果去掉位置编码
        x = self.token_emb(ids) + self.pos_emb[:, :ids.size(1)]

        weights = None
        if self.use_attention:
            x, weights = self.attn(x)

        return self.lm_head(x), weights


def make_batch(batch_size):
    # 每条序列都随机生成，避免只记住固定训练样本
    x = torch.randint(0, 2, (batch_size, T))

    # 每个位置都要预测该序列的第一个 Token
    y = x[:, :1].expand(-1, T)

    return x, y


def train_and_evaluate(use_attention):
    torch.manual_seed(7)

    model = TinyModel(use_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for step in range(1200):
        x, y = make_batch(128)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, 2),
            y.reshape(-1),
        )

        loss.backward()
        optimizer.step()

    # 使用新的随机序列进行评估
    model.eval()
    with torch.no_grad():
        x_test, y_test = make_batch(4096)
        logits, weights = model(x_test)

        loss = F.cross_entropy(
            logits.reshape(-1, 2),
            y_test.reshape(-1),
        ).item()

        accuracy_by_position = (
            (logits.argmax(dim=-1) == y_test)
            .float()
            .mean(dim=0)
        )

    return model, loss, accuracy_by_position, x_test, weights


for use_attention in [False, True]:
    model, loss, acc, x_test, weights = train_and_evaluate(
        use_attention
    )

    name = "With Attention" if use_attention else "Without Attention"

    print(f"\n{name}")
    print(f"Test loss: {loss:.4f}")
    print("Accuracy by position:", acc.tolist())

    if weights is not None:
        print("Example input:", x_test[0].tolist())
        print("Head 1 attention:")
        print(weights[0, 0])
