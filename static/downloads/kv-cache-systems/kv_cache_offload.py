
import gc
import statistics
import torch


def mib(num_bytes):
    return num_bytes / (1024 ** 2)


def measure_d2h(src, repeats=7):
    """GPU -> pinned CPU memory，使用 CUDA events 测量复制时间。"""
    dst = torch.empty(
        src.shape,
        dtype=src.dtype,
        device="cpu",
        pin_memory=True,
    )

    # Warm-up
    for _ in range(2):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        end.synchronize()

        times.append(start.elapsed_time(end))

    return dst, statistics.median(times)


def measure_h2d(src, repeats=7):
    """Pinned CPU memory -> GPU，使用 CUDA events 测量复制时间。"""
    dst = torch.empty(
        src.shape,
        dtype=src.dtype,
        device="cuda",
    )

    # Warm-up
    for _ in range(2):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        end.synchronize()

        times.append(start.elapsed_time(end))

    return dst, statistics.median(times)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA.")

    device = torch.device("cuda")
    dtype = torch.float16

    # 第 0 维表示 K/V 两份数据。
    # 每个块：2 * 1 * 8 * 2048 * 128 * 2 bytes = 8 MiB。
    chunk_shape = (2, 1, 8, 2048, 128)
    num_chunks = 4
    offload_indices = [2, 3]

    torch.manual_seed(42)
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated(device)

    # 每个块单独分配，才能真正释放被卸载块的 GPU 存储。
    chunks = [
        torch.randn(chunk_shape, device=device, dtype=dtype)
        for _ in range(num_chunks)
    ]

    torch.cuda.synchronize()

    def cache_allocated_mib():
        current = torch.cuda.memory_allocated(device)
        return mib(current - baseline_allocated)

    chunk_bytes = chunks[0].numel() * chunks[0].element_size()

    print(f"Device: {device}")
    print(f"Chunk size: {mib(chunk_bytes):.2f} MiB")
    print(f"Total cache: {mib(chunk_bytes * num_chunks):.2f} MiB")
    print(f"GPU cache allocated: {cache_allocated_mib():.2f} MiB")

    cpu_cache = {}

    # Offload：复制到 CPU，然后释放原来的 GPU Tensor。
    print("\n===== Offloading =====")

    for idx in offload_indices:
        cpu_tensor, d2h_ms = measure_d2h(chunks[idx])

        cpu_cache[idx] = cpu_tensor

        # 移除对 GPU Tensor 的引用，释放其已分配显存。
        chunks[idx] = None
        del cpu_tensor

        gc.collect()
        torch.cuda.synchronize()

        print(
            f"Block {idx}: D2H median={d2h_ms:.3f} ms | "
            f"GPU cache allocated={cache_allocated_mib():.2f} MiB"
        )

    print(
        f"CPU cache payload: "
        f"{sum(x.numel() * x.element_size() for x in cpu_cache.values()) / 1024**2:.2f} MiB"
    )

    # Restore：将卸载的块复制回 GPU，并检查数据是否一致。
    print("\n===== Restoring =====")

    for idx in offload_indices:
        cpu_tensor = cpu_cache.pop(idx)

        gpu_tensor, h2d_ms = measure_h2d(cpu_tensor)

        restored_cpu = gpu_tensor.cpu()
        same = torch.equal(restored_cpu, cpu_tensor)

        chunks[idx] = gpu_tensor

        print(
            f"Block {idx}: H2D median={h2d_ms:.3f} ms | "
            f"exact match={same}"
        )

        del restored_cpu, cpu_tensor

    torch.cuda.synchronize()

    print("\n===== Final State =====")
    print(f"GPU cache allocated: {cache_allocated_mib():.2f} MiB")
    print(f"All blocks restored: {all(x is not None for x in chunks)}")
    print(
        f"GPU allocator reserved: "
        f"{mib(torch.cuda.memory_reserved(device)):.2f} MiB"
    )


if __name__ == "__main__":
    main()
