"""Small position-encoding primitives used by the accompanying experiments.

RoPE pairs adjacent features (0,1), (2,3), ... and rotates with
R(a) = [[cos(a), -sin(a)], [sin(a), cos(a)]].
"""

from __future__ import annotations

import torch


def sinusoidal_positions(positions, d_model, *, base=10000.0,
                         dtype=torch.float64):
    """Return [..., d_model] fixed sin/cos embeddings for position IDs [...]."""
    if d_model <= 0 or d_model % 2:
        raise ValueError("d_model must be a positive even integer")
    positions = torch.as_tensor(positions)
    inv_freq = base ** (-torch.arange(
        0, d_model, 2, device=positions.device, dtype=dtype
    ) / d_model)
    angles = positions.to(dtype)[..., None] * inv_freq
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


def rotate_pairs(x):
    """Map each adjacent pair (a,b) to (-b,a), preserving tensor shape."""
    if x.shape[-1] % 2:
        raise ValueError("the last feature dimension must be even")
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


def apply_rope(x, positions, *, base=10000.0):
    """Rotate x [B,H,T,d] at positions [T] or [B,T]; return [B,H,T,d]."""
    if x.ndim != 4 or x.shape[-1] <= 0 or x.shape[-1] % 2:
        raise ValueError("x must have shape [B,H,T,d] with positive even d")
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")
    B, _, T, d = x.shape
    positions = torch.as_tensor(positions, device=x.device)
    if positions.shape == (T,):
        positions = positions[None, None, :, None]
    elif positions.shape == (B, T):
        positions = positions[:, None, :, None]
    else:
        raise ValueError("positions must have shape [T] or [B,T]")
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv_freq = base ** (-torch.arange(
        0, d, 2, device=x.device, dtype=work_dtype
    ) / d)
    angles = positions.to(work_dtype) * inv_freq
    cos = angles.cos().repeat_interleave(2, dim=-1).to(x.dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(x.dtype)
    return x * cos + rotate_pairs(x) * sin
