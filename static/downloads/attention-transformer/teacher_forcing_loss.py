
import torch
import torch.nn.functional as F
from torch.optim import Adam

from decoder_kv_cache import TinyDecoder


torch.manual_seed(7)

# Token IDs:
# PAD=0, BOS=1, A=2, B=3, C=4, EOS=5

# 两个训练样本，包含完整目标序列
sequences = torch.tensor([
    [1, 2, 3, 4, 5],  # BOS A B C EOS
    [1, 3, 2, 4, 5],  # BOS B A C EOS
])

# 目标序列错开一个位置
decoder_input = sequences[:, :-1]
targets = sequences[:, 1:]

print("Full sequences:", sequences.shape)
print("Decoder input:", decoder_input.shape)
print("Targets:", targets.shape)

print("\nDecoder input:")
print(decoder_input)

print("\nTargets:")
print(targets)

# 复用之前实现的两层 Decoder
model = TinyDecoder(
    vocab_size=6,
    max_len=8,
    d_model=16,
    n_heads=4,
    d_ff=32,
    n_layers=2,
)

optimizer = Adam(model.parameters(), lr=1e-3)

# 1. Forward：预测每个位置的下一个 Token
logits = model(decoder_input)

print("\nLogits shape:", logits.shape)

# 2. Loss：将 [B,T,V] 和 [B,T] 转成 CE 所需的形状
B, T, V = logits.shape

loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T),
)

print("Cross-Entropy Loss:", round(loss.item(), 4))

# 3. Backward：计算梯度
optimizer.zero_grad(set_to_none=True)
loss.backward()

grad_norm = model.lm_head.weight.grad.norm().item()
print("LM head gradient norm:", round(grad_norm, 6))

# 4. Optimizer：更新参数
optimizer.step()

print("Optimizer step completed.")
