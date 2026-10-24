import math
import torch
from torch import nn

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# B: batch size
# T: sequence length
# D: input feature dimension
B, T, D = 2, 4, 8

# 为了方便理解，这里令 d_k = d_v = D
d_k = d_v = D

# 假设输入已经是 Token 的向量表示
X = torch.randn(B, T, D, device=device)

# 三个独立的、可学习的线性投影
W_Q = nn.Linear(D, d_k, bias=False).to(device)
W_K = nn.Linear(D, d_k, bias=False).to(device)
W_V = nn.Linear(D, d_v, bias=False).to(device)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

# 1. 计算匹配分数
scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(d_k)

# 2. 创建 Causal Mask
# True 表示屏蔽；只屏蔽主对角线右上方的未来位置
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool, device=device),
    diagonal=1
)

scores = scores.masked_fill(mask, float("-inf"))

# 3. 将分数转换为注意力权重
weights = torch.softmax(scores, dim=-1)

# 4. 对 Value 加权求和
output = weights @ V

print("X:      ", X.shape)
print("Q:      ", Q.shape)
print("Scores: ", scores.shape)
print("Weights:", weights.shape)
print("V:      ", V.shape)
print("Output: ", output.shape)

print("\nAttention weights of batch 0:")
print(weights[0].detach().cpu())

print("\nSum of each row:")
print(weights[0].sum(dim=-1))