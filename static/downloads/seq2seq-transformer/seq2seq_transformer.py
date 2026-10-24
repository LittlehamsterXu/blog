
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Token IDs
PAD = 0
BOS = 1
EOS = 2

VOCAB_SIZE = 15
MAX_SRC_LEN = 5


class MultiHeadAttention(nn.Module):
    """支持 Self-Attention 和 Cross-Attention。"""

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
        query,
        key_value,
        key_padding_mask=None,
        causal=False,
    ):
        B, Tq, D = query.shape
        B_kv, Tk, D_kv = key_value.shape

        assert B == B_kv and D == D_kv

        H = self.n_heads
        d = self.d_head

        q = self.q_proj(query).reshape(B, Tq, H, d).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, Tk, H, d).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, Tk, H, d).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)

        # True 表示屏蔽未来位置
        if causal:
            future_mask = torch.triu(
                torch.ones(Tq, Tk, dtype=torch.bool, device=query.device),
                diagonal=1,
            )
            scores = scores.masked_fill(
                future_mask[None, None, :, :],
                float("-inf"),
            )

        # True 表示对应 Key 是 PAD
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        weights = torch.softmax(scores, dim=-1)
        context = weights @ v

        context = context.transpose(1, 2).contiguous().reshape(B, Tq, D)

        return self.out_proj(context), weights


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


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_pad_mask):
        # 普通 Self-Attention，不使用 Causal Mask
        attn_out, _ = self.attn(
            x, x,
            key_padding_mask=src_pad_mask,
        )
        x = self.norm1(x + attn_out)

        x = self.norm2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_pad_mask, src_pad_mask):
        # 1. Decoder 内部的因果 Self-Attention
        attn_out, _ = self.self_attn(
            x, x,
            key_padding_mask=tgt_pad_mask,
            causal=True,
        )
        x = self.norm1(x + attn_out)

        # 2. Q 来自 Decoder；K、V 来自 Encoder
        cross_out, _ = self.cross_attn(
            x, memory,
            key_padding_mask=src_pad_mask,
        )
        x = self.norm2(x + cross_out)

        # 3. FFN
        x = self.norm3(x + self.ffn(x))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        max_src_len,
        max_tgt_len,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
    ):
        super().__init__()

        self.pad_id = pad_id

        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)

        self.src_pos = nn.Embedding(max_src_len, d_model)
        self.tgt_pos = nn.Embedding(max_tgt_len, d_model)

        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        B, S = src.shape

        src_pad_mask = src.eq(self.pad_id)
        pos = torch.arange(S, device=src.device)

        x = self.src_emb(src) + self.src_pos(pos)[None, :, :]

        for layer in self.encoder:
            x = layer(x, src_pad_mask)

        return x, src_pad_mask

    def decode(self, tgt_input, memory, src_pad_mask):
        B, T = tgt_input.shape

        tgt_pad_mask = tgt_input.eq(self.pad_id)
        pos = torch.arange(T, device=tgt_input.device)

        x = self.tgt_emb(tgt_input) + self.tgt_pos(pos)[None, :, :]

        for layer in self.decoder:
            x = layer(x, memory, tgt_pad_mask, src_pad_mask)

        return self.lm_head(self.final_norm(x))

    def forward(self, src, tgt_input):
        memory, src_pad_mask = self.encode(src)
        return self.decode(tgt_input, memory, src_pad_mask)


def make_batch(batch_size, device):
    """生成随机变长序列，以及对应的反转目标。"""

    src = torch.full(
        (batch_size, MAX_SRC_LEN),
        PAD,
        dtype=torch.long,
    )

    # BOS + 最长反转序列 + EOS
    target_full = torch.full(
        (batch_size, MAX_SRC_LEN + 2),
        PAD,
        dtype=torch.long,
    )

    for i in range(batch_size):
        length = random.randint(1, MAX_SRC_LEN)

        tokens = torch.randint(3, VOCAB_SIZE, (length,))
        src[i, :length] = tokens

        reversed_tokens = tokens.flip(0).tolist()
        target = [BOS] + reversed_tokens + [EOS]

        target_full[i, :len(target)] = torch.tensor(target)

    # Teacher Forcing：输入与标签错开一个 Token
    decoder_input = target_full[:, :-1]
    targets = target_full[:, 1:]

    return (
        src.to(device),
        decoder_input.to(device),
        targets.to(device),
    )


@torch.no_grad()
def greedy_decode(model, src, max_new_tokens):
    model.eval()

    memory, src_pad_mask = model.encode(src)

    B = src.size(0)
    generated = torch.full(
        (B, 1),
        BOS,
        dtype=torch.long,
        device=src.device,
    )

    finished = torch.zeros(B, dtype=torch.bool, device=src.device)

    for _ in range(max_new_tokens):
        logits = model.decode(generated, memory, src_pad_mask)

        # 取最后一个位置对下一个 Token 的预测
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # 已经结束的样本后续填 PAD
        next_token = torch.where(
            finished[:, None],
            torch.full_like(next_token, PAD),
            next_token,
        )

        generated = torch.cat([generated, next_token], dim=1)
        finished |= next_token.squeeze(1).eq(EOS)

        if finished.all():
            break

    return generated[:, 1:]


def trim_at_eos(tokens):
    tokens = list(tokens)

    if EOS in tokens:
        tokens = tokens[:tokens.index(EOS) + 1]

    return tokens


if __name__ == "__main__":
    random.seed(7)
    torch.manual_seed(7)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    model = Seq2SeqTransformer(
        vocab_size=VOCAB_SIZE,
        pad_id=PAD,
        max_src_len=MAX_SRC_LEN,
        max_tgt_len=MAX_SRC_LEN + 1,
    ).to(device)

    # ---------- 1. 先检查完整模型的形状 ----------
    src, decoder_input, targets = make_batch(4, device)

    with torch.no_grad():
        logits = model(src, decoder_input)

    print("Source:", src.shape)
    print("Decoder input:", decoder_input.shape)
    print("Targets:", targets.shape)
    print("Logits:", logits.shape)
    print("All logits finite:", torch.isfinite(logits).all().item())

    # ---------- 2. 训练 ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    model.train()
    steps = 1000
    batch_size = 64

    for step in range(1, steps + 1):
        src, decoder_input, targets = make_batch(batch_size, device)

        logits = model(src, decoder_input)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=PAD,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    # ---------- 3. 检查未参与训练的随机样本 ----------
    model.eval()

    test_src, _, _ = make_batch(64, device)

    with torch.no_grad():
        predictions = greedy_decode(
            model,
            test_src,
            max_new_tokens=MAX_SRC_LEN + 1,
        )

    correct = 0

    for i in range(test_src.size(0)):
        source_tokens = test_src[i][test_src[i] != PAD].tolist()

        expected = list(reversed(source_tokens)) + [EOS]
        actual = trim_at_eos(predictions[i].tolist())

        if actual == expected:
            correct += 1

    print(f"\nExact sequence accuracy: {correct}/64 = {correct / 64:.2%}")

    # 展示一个样本
    source_tokens = test_src[0][test_src[0] != PAD].tolist()
    predicted = trim_at_eos(predictions[0].tolist())

    print("Example source:", source_tokens)
    print("Expected output:", list(reversed(source_tokens)) + [EOS])
    print("Predicted output:", predicted)

    
    # ---------- 4. 改变源序列，检查模型是否利用输入 ----------
    # 不修改模型参数，不重新训练
    # 只将有效 Token 循环替换为下一个 Token ID，保留 PAD 位置

    changed_src = test_src.clone()
    valid_mask = changed_src.ne(PAD)

    # 有效 Token 的 ID 范围是 [3, VOCAB_SIZE - 1]
    changed_src[valid_mask] = (
        3 + (changed_src[valid_mask] - 3 + 1) % (VOCAB_SIZE - 3)
    )

    with torch.no_grad():
        changed_predictions = greedy_decode(
            model,
            changed_src,
            max_new_tokens=MAX_SRC_LEN + 1,
        )

    changed_correct = 0
    output_changed = 0

    for i in range(test_src.size(0)):
        original_tokens = test_src[i][test_src[i] != PAD].tolist()
        modified_tokens = changed_src[i][changed_src[i] != PAD].tolist()

        original_output = trim_at_eos(predictions[i].tolist())
        modified_output = trim_at_eos(changed_predictions[i].tolist())

        # 新输入对应的新反转目标
        expected_modified = list(reversed(modified_tokens)) + [EOS]

        if modified_output == expected_modified:
            changed_correct += 1

        if modified_output != original_output:
            output_changed += 1

    n = test_src.size(0)

    print("\n===== Source Perturbation Test =====")
    print(f"Original exact accuracy: {correct}/{n}")
    print(f"Modified exact accuracy: {changed_correct}/{n}")
    print(f"Outputs changed: {output_changed}/{n}")

    print("\nOriginal source:", original_tokens)
    print("Modified source:", modified_tokens)
    print("Expected modified output:", expected_modified)
    print("Predicted modified output:", modified_output)

