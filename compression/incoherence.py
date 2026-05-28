"""
Backward-compatible imports for weight incoherence preprocessing.

The implementation lives in ``compression.transform`` (blocked Hadamard + Rademacher).
"""
from __future__ import annotations

from .transform import (
    HadamardDim,
    apply_incoherence,
    next_pow2,
    resolve_hadamard_dim,
    revert_incoherence,
)

__all__ = [
    "HadamardDim",
    "apply_incoherence",
    "next_pow2",
    "resolve_hadamard_dim",
    "revert_incoherence",
]
