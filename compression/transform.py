"""
Incoherence / orthogonal preprocessing for weights (Phase 2): blocked Hadamard + Rademacher.

Avoids materializing a full ``n × n`` Hadamard by applying independent ``block_size`` Walsh–Hadamard
transforms along a chosen axis (``hadamard_dim``). Each block uses its own diagonal Rademacher ``D``;
within a block the transform is ``(1/sqrt(B)) * H_B @ D`` (same as the full-axis case, but ``B << n``).

Metadata versions:
  - ``incoherence_version == 1``: legacy single-axis padding to one power-of-two; signs stored explicitly.
  - ``incoherence_version == 2``: blocked; stores integer ``seed`` and block geometry (signs regenerated on decode).
"""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F

HadamardDim = Literal["in_features", "out_features"]

__all__ = [
    "HadamardDim",
    "apply_incoherence",
    "revert_incoherence",
    "resolve_hadamard_dim",
    "next_pow2",
]


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _normalize_block_size(block_size: int) -> int:
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")
    return block_size


def _fwht_unnormalized(x: torch.Tensor) -> torch.Tensor:
    """``x @ H_B`` with unnormalized Sylvester Hadamard on the last dimension (``B`` power of 2)."""
    n = x.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT last dim must be a positive power of 2, got {n}")
    if n == 1:
        return x.clone()
    a = x[..., : n // 2]
    b = x[..., n // 2 :]
    t1 = _fwht_unnormalized(a + b)
    t2 = _fwht_unnormalized(a - b)
    return torch.cat([t1, t2], dim=-1)


def resolve_hadamard_dim(tensor: torch.Tensor, hadamard_dim: HadamardDim | int) -> int:
    """Map ``in_features`` / ``out_features`` or explicit index to a dim index."""
    if isinstance(hadamard_dim, int):
        d = hadamard_dim
        return d if d >= 0 else tensor.dim() + d

    nd = tensor.dim()
    if nd < 1:
        raise ValueError("tensor must be at least 1-D")
    if nd == 1:
        return 0
    if hadamard_dim == "in_features":
        return nd - 1
    if hadamard_dim == "out_features":
        return 0
    raise ValueError(f"unknown hadamard_dim: {hadamard_dim!r}")


def _rademacher_blocks(
    num_blocks: int,
    block_size: int,
    *,
    generator: torch.Generator,
    out_device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    # Draw on CPU so the same Generator + seed replays identically on decode
    bits = torch.randint(
        0,
        2,
        (num_blocks, block_size),
        dtype=torch.int32,
        generator=generator,
        device=torch.device("cpu"),
    )
    return (bits.to(device=out_device, dtype=compute_dtype) * 2) - 1


def apply_incoherence(
    tensor: torch.Tensor,
    *,
    block_size: int = 1024,
    hadamard_dim: HadamardDim | int = "in_features",
    generator: torch.Generator | None = None,
    layer_seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Blocked orthogonal preprocessing before RTN.

    Args:
        tensor: Weight (or parameter) tensor.
        block_size: Hadamard block length (power of 2). Typical: 512, 1024, 2048.
        hadamard_dim: ``\"in_features\"`` (last axis for 2-D weights), ``\"out_features\"`` (first),
            or an explicit integer dim index.
        generator: Optional ``torch.Generator`` (CPU). If ``layer_seed`` is set, it is applied first.
        layer_seed: If set, ``generator.manual_seed(layer_seed)`` before drawing Rademacher entries.
    """
    if tensor.dim() == 0:
        raise ValueError("incoherence expects at least 1-D tensor")

    if tensor.numel() <= 1:
        dim = resolve_hadamard_dim(tensor, hadamard_dim)
        return tensor.clone(), {
            "incoherence_version": 2,
            "skip": True,
            "hadamard_dim": hadamard_dim if isinstance(hadamard_dim, str) else int(hadamard_dim),
            "dim": int(dim),
            "n_original": int(tensor.shape[dim]),
            "block_size": int(block_size),
            "seed": int(layer_seed) if layer_seed is not None else None,
        }

    if layer_seed is None:
        raise ValueError(
            "apply_incoherence requires layer_seed for reproducible blocked Rademacher draws on decode"
        )

    block_size = _normalize_block_size(block_size)
    ndim = tensor.dim()
    dim = resolve_hadamard_dim(tensor, hadamard_dim)
    if not 0 <= dim < ndim:
        raise ValueError(f"resolved dim {dim} out of range")

    device = tensor.device
    dtype = tensor.dtype
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    gen = generator if generator is not None else torch.Generator(device="cpu")
    gen.manual_seed(int(layer_seed))

    n_orig = int(tensor.shape[dim])
    n_padded = ((n_orig + block_size - 1) // block_size) * block_size
    num_blocks = n_padded // block_size

    perm = list(range(ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    x = tensor.permute(*perm).contiguous().to(compute_dtype)
    pad_amt = n_padded - n_orig
    if pad_amt > 0:
        x = F.pad(x, (0, pad_amt))

    signs = _rademacher_blocks(
        num_blocks,
        block_size,
        generator=gen,
        out_device=device,
        compute_dtype=compute_dtype,
    )
    flat = x.reshape(-1, num_blocks, block_size)
    tmp = flat * signs.unsqueeze(0)
    had = _fwht_unnormalized(tmp)
    scale = 1.0 / math.sqrt(float(block_size))
    y_flat = had * scale
    y = y_flat.reshape(*x.shape)

    out = y.permute(*inv_perm).contiguous().to(dtype)

    meta: dict[str, Any] = {
        "incoherence_version": 2,
        "block_size": block_size,
        "hadamard_dim": hadamard_dim if isinstance(hadamard_dim, str) else int(hadamard_dim),
        "dim": dim,
        "n_original": n_orig,
        "n_padded": n_padded,
        "num_blocks": num_blocks,
        "seed": int(layer_seed),
    }
    return out, meta


def revert_incoherence(tensor: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """Inverse orthogonal transform; supports v1 (full-axis) and v2 (blocked + seed)."""
    ver = meta.get("incoherence_version")
    if ver == 1:
        return _revert_v1(tensor, meta)
    if ver == 2:
        return _revert_v2(tensor, meta)
    raise ValueError(f"unsupported incoherence_version: {ver!r}")


def _revert_v1(tensor: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """Legacy: single padded power-of-two axis, signs stored in JSON."""
    if meta.get("skip"):
        return tensor

    dim = int(meta["dim"])
    n_orig = int(meta["n_original"])
    n_pad = int(meta["n_padded"])
    dtype = tensor.dtype
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    signs = torch.tensor(meta["rademacher_signs"], dtype=compute_dtype, device=tensor.device)

    ndim = tensor.dim()
    perm = list(range(ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    x = tensor.permute(*perm).contiguous().to(compute_dtype)
    if x.shape[-1] != n_pad:
        raise ValueError(f"incoherence v1 revert: expected last dim {n_pad}, got {x.shape[-1]}")

    flat = x.reshape(-1, n_pad)
    inv_h = _fwht_unnormalized(flat) * (1.0 / math.sqrt(float(n_pad)))
    tmp = inv_h * signs.unsqueeze(0)
    y = tmp.reshape(*x.shape)

    if n_pad > n_orig:
        y = y[..., :n_orig]

    return y.permute(*inv_perm).contiguous().to(dtype)


def _revert_v2(tensor: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    if meta.get("skip"):
        return tensor

    block_size = int(meta["block_size"])
    dim = int(meta["dim"])
    n_orig = int(meta["n_original"])
    n_padded = int(meta["n_padded"])
    num_blocks = int(meta["num_blocks"])
    seed = meta.get("seed")
    if seed is None:
        raise ValueError("incoherence v2 metadata missing integer seed")

    dtype = tensor.dtype
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    ndim = tensor.dim()

    perm = list(range(ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    x = tensor.permute(*perm).contiguous().to(compute_dtype)
    if x.shape[-1] != n_padded:
        raise ValueError(f"incoherence v2 revert: expected last dim {n_padded}, got {x.shape[-1]}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    signs = _rademacher_blocks(
        num_blocks,
        block_size,
        generator=gen,
        out_device=tensor.device,
        compute_dtype=compute_dtype,
    )

    flat = x.reshape(-1, num_blocks, block_size)
    inv_h = _fwht_unnormalized(flat) * (1.0 / math.sqrt(float(block_size)))
    tmp = inv_h * signs.unsqueeze(0)
    y = tmp.reshape(*x.shape)

    if n_padded > n_orig:
        y = y[..., :n_orig]

    return y.permute(*inv_perm).contiguous().to(dtype)
