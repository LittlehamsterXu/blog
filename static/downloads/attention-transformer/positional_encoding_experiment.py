
import math
import torch

torch.set_printoptions(precision=3, sci_mode=False)


def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def attention_experiment(x):
    # x: [B, T, D]
    # 为了控制变量，Q、K、V 投影均设为恒等映射
    q = x
    k = x
    v = x

    scores = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    context = weights @ v

    return q, k, scores, weights, context


T, D = 3, 4

# 三个位置使用完全相同的 Token 内容向量
token_embedding = torch.tensor([1.0, 0.0, 1.0, 0.0])
embeddings = token_embedding.view(1, 1, D).expand(1, T, D)

# 固定的正弦位置编码
pe = sinusoidal_positional_encoding(T, D).unsqueeze(0)

print("Token embeddings:")
print(embeddings)

print("\nPositional encodings:")
print(pe)

# 实验 A：不加入位置编码
q0, k0, scores0, weights0, context0 = attention_experiment(
    embeddings
)

# 实验 B：加入位置编码
x1 = embeddings + pe
q1, k1, scores1, weights1, context1 = attention_experiment(x1)

print("\n========== Without positional encoding ==========")
print("Q:")
print(q0[0])
print("Attention scores:")
print(scores0[0])
print("Attention weights:")
print(weights0[0])
print("Context:")
print(context0[0])

print("\n========== With positional encoding ==========")
print("Q:")
print(q1[0])
print("Attention scores:")
print(scores1[0])
print("Attention weights:")
print(weights1[0])
print("Context:")
print(context1[0])

print("\nMaximum attention-weight difference:")
print((weights1 - weights0).abs().max().item())
