"""
Architecture-aware routing for incoherence (Hadamard axis + shared rotation group key).

``group_key`` is hashed to a reproducible RNG seed. All tensors in the same group use the
same ``incoherence_block_size`` (from the pipeline) and the same seed, so they share the
same orthogonal transform P on the rotated axis (when shapes along that axis match after padding).

GPT-NeoX / Pythia (``model_type == "gpt_neox"``):
  - ``query_key_value``, ``dense_h_to_4h`` (fused up-proj): rotate **in_features** (last dim).
    Group: ``neoX:L{layer}:in`` — same P within a block as layernorms on the residual stream.
  - ``attention.dense`` (attn out), ``dense_4h_to_h`` (mlp down): rotate **out_features** (first dim).
    Group: ``neoX:L{layer}:out``.
  - ``embed_in`` / ``embed_out``: rotate **in_features**; group ``neoX:embed`` (tied weights share P).
  - Block layernorms: **in_features**, group ``neoX:L{layer}:in``.
  - ``final_layer_norm``: **in_features**, group ``neoX:final``.

Other architectures: conservative default — **in_features**, group ``{model_type}:global:{param_name}``
(one P per parameter; extend with explicit tables as needed).
"""
from __future__ import annotations

import hashlib
import re
from typing import Literal

import torch.nn as nn

from compression.transform import HadamardDim

RouteResult = tuple[HadamardDim | None, str | None]


def incoherence_seed_from_group(group_key: str) -> int:
    """Deterministic seed from rotation group (shared P ⇒ same key ⇒ same seed)."""
    digest = hashlib.md5(group_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def infer_model_type(model: nn.Module) -> str:
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None) if cfg is not None else None
    if isinstance(mt, str) and mt:
        return mt
    return type(model).__name__


def route_incoherence(param_name: str, model_type: str) -> RouteResult:
    """
    Returns:
        (hadamard_axis, group_key). ``(None, None)`` ⇒ skip incoherence for this tensor.
    """
    if model_type == "gpt_neox":
        return _route_gpt_neox(param_name)
    return _route_default(param_name, model_type)


def _route_default(param_name: str, model_type: str) -> RouteResult:
    # One independent transform per parameter unless a dedicated table is added.
    safe = param_name.replace(".", "_")
    return "in_features", f"{model_type}:param:{safe}"


def _route_gpt_neox(name: str) -> RouteResult:
    # Only weight/bias tensors participate; buffers skipped at caller if needed.
    if not (name.endswith(".weight") or name.endswith(".bias")):
        return None, None

    # Embeddings (often weight-tied): same rotation group
    if "embed_in" in name or "embed_out" in name:
        return "in_features", "neoX:embed"

    if "final_layer_norm" in name:
        return "in_features", "neoX:final"

    m = re.search(r"\.layers\.(\d+)\.", name)
    if not m:
        return "in_features", "neoX:other"

    li = m.group(1)

    # LayerNorms on the residual / block (share "in" side group with qkv / mlp up)
    if "layernorm" in name or "layer_norm" in name:
        return "in_features", f"neoX:L{li}:in"

    # Fused QKV and MLP up (input-facing fan-in)
    if "query_key_value" in name or "dense_h_to_4h" in name:
        return "in_features", f"neoX:L{li}:in"

    # Attention output projection (not QKV)
    if ".attention.dense" in name and "query_key_value" not in name:
        return "out_features", f"neoX:L{li}:out"

    # MLP down
    if "dense_4h_to_h" in name:
        return "out_features", f"neoX:L{li}:out"

    # Other block params (e.g. some bias layouts): align with input-side group
    return "in_features", f"neoX:L{li}:misc"
