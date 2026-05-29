"""
4D diffusion feature maps -> VcLLM-compatible 2D frame tiles.

Reuses ``codec.frame_mapper.tensor_to_frames`` / ``frames_to_tensor`` so DiffCodec
shares the same RTN + HEVC geometry as weight compression.

Layout strategy (spatial continuity first):
  - Input activation ``(B, C, H, W)``. SD CFG uses ``B == 2`` (uncond, cond);
    default ``cfg_batch="cond"`` keeps only the conditional slice ``(1, C, H, W)``.
  - Split channels into groups of ``channel_group_size`` (default 64).
  - Per group, view as ``(C_g, H * W)`` where each **row** is one channel and
    columns are raster-ordered spatial pixels — HEVC CTUs stay inside one channel's
    spatial neighborhood when ``H * W`` fits a frame row or is tiled row-major.
  - Each group's 2D matrix is independently passed through ``tensor_to_frames``.
  - Returned ``frames`` are concatenated along the temporal (frame-index) axis;
    ``FeatureMapMappingMeta`` records how to invert the split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from codec.frame_mapper import frames_to_tensor, tensor_to_frames


@dataclass
class ChannelGroupMapping:
    """One channel group's 2D matrix <-> frame tiles."""

    channel_start: int
    channel_end: int
    matrix_shape: tuple[int, int]
    frame_metadata: dict[str, Any]
    num_frames: int


@dataclass
class FeatureMapMappingMeta:
    """Invertible metadata for a single denoise-step feature map."""

    original_shape: tuple[int, ...]
    channel_group_size: int
    frame_size: int
    cfg_batch: str = "cond"
    raw_batch_size: int = 1
    groups: list[ChannelGroupMapping] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_shape": list(self.original_shape),
            "channel_group_size": self.channel_group_size,
            "frame_size": self.frame_size,
            "cfg_batch": self.cfg_batch,
            "raw_batch_size": self.raw_batch_size,
            "groups": [
                {
                    "channel_start": g.channel_start,
                    "channel_end": g.channel_end,
                    "matrix_shape": list(g.matrix_shape),
                    "frame_metadata": g.frame_metadata,
                    "num_frames": g.num_frames,
                }
                for g in self.groups
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureMapMappingMeta:
        groups = [
            ChannelGroupMapping(
                channel_start=int(g["channel_start"]),
                channel_end=int(g["channel_end"]),
                matrix_shape=tuple(g["matrix_shape"]),
                frame_metadata=dict(g["frame_metadata"]),
                num_frames=int(g["num_frames"]),
            )
            for g in data["groups"]
        ]
        return cls(
            original_shape=tuple(data["original_shape"]),
            channel_group_size=int(data["channel_group_size"]),
            frame_size=int(data["frame_size"]),
            cfg_batch=str(data.get("cfg_batch", "cond")),
            raw_batch_size=int(data.get("raw_batch_size", 1)),
            groups=groups,
        )


def select_cfg_batch(feature: Tensor, *, cfg_batch: str = "cond") -> Tensor:
    """
    Reduce ``(B, C, H, W)`` to ``(1, C, H, W)`` for SD classifier-free guidance.

    Diffusers concatenates ``[uncond, cond]`` when ``guidance_scale > 1`` (``B == 2``).
    """
    if feature.dim() != 4:
        raise ValueError(f"Expected 4D (B,C,H,W), got {tuple(feature.shape)}")

    b = int(feature.shape[0])
    if b == 1:
        return feature
    if b == 2:
        if cfg_batch == "cond":
            return feature[1:2]
        if cfg_batch == "uncond":
            return feature[0:1]
        raise ValueError(
            f"cfg_batch must be 'cond' or 'uncond' when B=2 (CFG), got {cfg_batch!r}"
        )
    raise ValueError(
        f"Unsupported feature batch size B={b}. Expected 1 or 2 (CFG); "
        f"set guidance_scale=1 to disable CFG doubling."
    )


def _feature_channels_first(feature: Tensor, *, cfg_batch: str = "cond") -> Tensor:
    """``(1,C,H,W)`` -> ``(C,H,W)`` after optional CFG slice."""
    x = select_cfg_batch(feature, cfg_batch=cfg_batch)
    return x[0]


def group_feature_map_for_mapping(
    feature: Tensor,
    *,
    channel_group_size: int = 64,
    cfg_batch: str = "cond",
) -> list[tuple[Tensor, int, int]]:
    """
    Split ``(B,C,H,W)`` into channel groups, each as a 2D matrix ``(C_g, H*W)``.

    Returns:
        List of ``(matrix_2d, channel_start, channel_end)`` on the same device/dtype
        as input (matrix is detached for mapping).
    """
    if channel_group_size <= 0:
        raise ValueError(f"channel_group_size must be > 0, got {channel_group_size}")

    x = _feature_channels_first(feature, cfg_batch=cfg_batch)
    c, h, w = x.shape
    spatial = h * w
    groups: list[tuple[Tensor, int, int]] = []

    for start in range(0, c, channel_group_size):
        end = min(start + channel_group_size, c)
        chunk = x[start:end]  # (C_g, H, W)
        matrix = chunk.reshape(end - start, spatial).contiguous()
        groups.append((matrix, start, end))

    return groups


def feature_map_to_frames(
    feature: Tensor,
    *,
    frame_size: int = 1024,
    channel_group_size: int = 64,
    cfg_batch: str = "cond",
) -> tuple[np.ndarray, FeatureMapMappingMeta]:
    """
    Map one feature map ``F_t`` to uint8-ready frame tiles via VcLLM ``tensor_to_frames``.

    Returns:
        ``frames``: ``(total_frames, H_enc, H_enc)`` numpy array (concatenated groups).
        ``meta``: invertible mapping metadata.
    """
    raw_batch_size = int(feature.shape[0])
    sliced = select_cfg_batch(feature, cfg_batch=cfg_batch)
    original_shape = tuple(sliced.shape)
    groups_2d = group_feature_map_for_mapping(
        sliced, channel_group_size=channel_group_size, cfg_batch=cfg_batch
    )

    all_frames: list[np.ndarray] = []
    meta_groups: list[ChannelGroupMapping] = []

    for matrix, ch_start, ch_end in groups_2d:
        frames, frame_meta = tensor_to_frames(matrix, frame_size=frame_size)
        meta_groups.append(
            ChannelGroupMapping(
                channel_start=ch_start,
                channel_end=ch_end,
                matrix_shape=tuple(matrix.shape),
                frame_metadata=frame_meta,
                num_frames=int(frames.shape[0]),
            )
        )
        all_frames.append(frames)

    if not all_frames:
        raise ValueError("Empty feature map — nothing to map")

    stacked = np.concatenate(all_frames, axis=0)
    meta = FeatureMapMappingMeta(
        original_shape=original_shape,
        channel_group_size=channel_group_size,
        frame_size=frame_size,
        cfg_batch=cfg_batch,
        raw_batch_size=raw_batch_size,
        groups=meta_groups,
    )
    return stacked, meta


def frames_to_feature_map(
    frames: np.ndarray,
    meta: FeatureMapMappingMeta | dict[str, Any],
    *,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Inverse of ``feature_map_to_frames``; restores ``(1, C, H, W)``."""
    if isinstance(meta, dict):
        meta = FeatureMapMappingMeta.from_dict(meta)

    offset = 0
    c, h, w = meta.original_shape[1], meta.original_shape[2], meta.original_shape[3]
    x = torch.zeros((c, h, w), dtype=dtype, device=device)
    spatial = h * w

    for group in meta.groups:
        n = group.num_frames
        chunk_frames = frames[offset : offset + n]
        offset += n

        matrix = frames_to_tensor(chunk_frames, group.frame_metadata).to(
            dtype=dtype, device=device
        )
        if tuple(matrix.shape) != group.matrix_shape:
            raise ValueError(
                f"Decoded matrix shape {tuple(matrix.shape)} != "
                f"expected {group.matrix_shape}"
            )

        x[group.channel_start : group.channel_end] = matrix.reshape(
            group.channel_end - group.channel_start, h, w
        )

    if offset != frames.shape[0]:
        raise ValueError(
            f"Frame count mismatch: consumed {offset}, array has {frames.shape[0]}"
        )

    return x.unsqueeze(0)
