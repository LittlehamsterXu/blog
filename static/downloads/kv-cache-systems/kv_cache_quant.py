
import torch


def quantize_int8_per_vector(x):
    """对最后一个维度的每个向量执行对称 INT8 量化。"""
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / 127.0).clamp(min=1e-8)

    q = torch.round(x / scale)
    q = q.clamp(-127, 127).to(torch.int8)

    return q, scale


def dequantize_int8(q, scale):
    """将 INT8 近似恢复为 FP32。"""
    return q.float() * scale


def evaluate(original, restored, name):
    """统一以原始 FP32 数据作为误差基准。"""
    error = restored.float() - original.float()

    mae = error.abs().mean().item()
    rmse = error.square().mean().sqrt().item()
    max_error = error.abs().max().item()

    reference_rms = original.float().square().mean().sqrt().item()
    relative_rmse = rmse / max(reference_rms, 1e-12)

    print(f"\n[{name}]")
    print(f"MAE:           {mae:.8f}")
    print(f"RMSE:          {rmse:.8f}")
    print(f"Max abs error: {max_error:.8f}")
    print(f"Relative RMSE: {relative_rmse:.6%}")


def tensor_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def mib(num_bytes):
    return num_bytes / (1024 ** 2)


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # 生成同一份 FP32 原始 K/V，作为所有实验的基准。
    shape = (1, 4, 1024, 64)

    k_fp32 = torch.randn(shape, device=device, dtype=torch.float32)
    v_fp32 = torch.randn(shape, device=device, dtype=torch.float32)

    print(f"Cache shape per tensor: {tuple(k_fp32.shape)}")

    # --------------------------------------------------
    # Experiment A: FP32 -> FP16 -> FP32
    # --------------------------------------------------
    k_fp16 = k_fp32.to(torch.float16)
    v_fp16 = v_fp32.to(torch.float16)

    k_fp16_restored = k_fp16.float()
    v_fp16_restored = v_fp16.float()

    evaluate(k_fp32, k_fp16_restored, "FP16 Key reconstruction")
    evaluate(v_fp32, v_fp16_restored, "FP16 Value reconstruction")

    # --------------------------------------------------
    # Experiment B: FP32 -> INT8 + FP32 scale
    # --------------------------------------------------
    qk, sk = quantize_int8_per_vector(k_fp32)
    qv, sv = quantize_int8_per_vector(v_fp32)

    k_int8_restored = dequantize_int8(qk, sk)
    v_int8_restored = dequantize_int8(qv, sv)

    evaluate(k_fp32, k_int8_restored, "INT8 Key reconstruction")
    evaluate(v_fp32, v_int8_restored, "INT8 Value reconstruction")

    # --------------------------------------------------
    # Storage comparison: count stored cache and metadata.
    # Temporary FP32 reconstruction tensors are not included.
    # --------------------------------------------------
    fp32_bytes = tensor_bytes(k_fp32, v_fp32)
    fp16_bytes = tensor_bytes(k_fp16, v_fp16)

    int8_payload_bytes = tensor_bytes(qk, qv)
    scale_bytes = tensor_bytes(sk, sv)
    int8_total_bytes = int8_payload_bytes + scale_bytes

    print("\n===== Storage Comparison =====")
    print(f"FP32 K/V:       {mib(fp32_bytes):.5f} MiB")
    print(f"FP16 K/V:       {mib(fp16_bytes):.5f} MiB")
    print(f"INT8 payload:   {mib(int8_payload_bytes):.5f} MiB")
    print(f"Scale metadata: {mib(scale_bytes):.5f} MiB")
    print(f"INT8 total:     {mib(int8_total_bytes):.5f} MiB")

    print("\n===== Storage Reduction =====")
    print(
        f"FP32 -> FP16: "
        f"{(1 - fp16_bytes / fp32_bytes):.2%}"
    )
    print(
        f"FP32 -> INT8: "
        f"{(1 - int8_total_bytes / fp32_bytes):.2%}"
    )
    print(
        f"FP16 -> INT8: "
        f"{(1 - int8_total_bytes / fp16_bytes):.2%}"
    )


if __name__ == "__main__":
    main()
