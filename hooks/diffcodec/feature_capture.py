"""
Forward hooks on SD UNet blocks to capture per-denoise-step feature maps.

Phase 1: lossless capture + optional Tensor2Video mapping (no bypass yet).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

from .feature_map_mapper import (
    FeatureMapMappingMeta,
    feature_map_to_frames,
    select_cfg_batch,
)


class HookSite(str, Enum):
    """Canonical capture points in diffusers ``UNet2DConditionModel``."""

    MID_BLOCK = "mid_block"
    UP_BLOCK_0 = "up_blocks.0"
    UP_BLOCK_1 = "up_blocks.1"
    DOWN_BLOCK_2 = "down_blocks.2"


# Recommended defaults for DiffCodec experiments (see module docstring in eval script).
DEFAULT_HOOK_SITES: tuple[HookSite, ...] = (HookSite.MID_BLOCK,)


def _resolve_hook_module(unet: nn.Module, site: HookSite) -> nn.Module:
    """Return the ``nn.Module`` to register a forward hook on (module output)."""
    if site == HookSite.MID_BLOCK:
        return unet.mid_block
    if site == HookSite.UP_BLOCK_0:
        return unet.up_blocks[0]
    if site == HookSite.UP_BLOCK_1:
        return unet.up_blocks[1]
    if site == HookSite.DOWN_BLOCK_2:
        return unet.down_blocks[2]
    raise ValueError(f"Unknown HookSite: {site!r}")


@dataclass
class StepCapture:
    """One denoise step at one hook site."""

    timestep: int
    step_index: int
    site: str
    feature: Tensor
    frames_meta: FeatureMapMappingMeta | None = None
    frames: Any = None  # np.ndarray when map_to_frames=True


@dataclass
class DiffusionFeatureCapture:
    """
    Registers forward hooks and collects ``F_t`` across denoise steps.

    Usage::

        capture = DiffusionFeatureCapture(sites=[HookSite.MID_BLOCK])
        handles = attach_feature_hooks(pipe.unet, capture)
        pipe(..., callback_on_step_end=capture.on_step_end)
        handles.remove()
    """

    sites: tuple[HookSite, ...] = DEFAULT_HOOK_SITES
    map_to_frames: bool = False
    frame_size: int = 1024
    channel_group_size: int = 64
    store_device: str = "cpu"
    cfg_batch: str = "cond"  # SD CFG doubles batch: [uncond, cond]

    # Written by hooks / callback
    _pending_features: dict[str, Tensor] = field(default_factory=dict, repr=False)
    _records: list[StepCapture] = field(default_factory=list, repr=False)
    _step_index: int = field(default=0, repr=False)

    def clear(self) -> None:
        self._pending_features.clear()
        self._records.clear()
        self._step_index = 0

    @property
    def records(self) -> list[StepCapture]:
        return list(self._records)

    def _hook_fn(self, site: HookSite) -> Callable:
        site_name = site.value

        def _fn(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            del module, inputs
            # diffusers blocks may return tuple (hidden, ...) for gradient checkpointing
            feat = output[0] if isinstance(output, tuple) else output
            if not isinstance(feat, Tensor):
                return
            feat = select_cfg_batch(feat.detach(), cfg_batch=self.cfg_batch)
            self._pending_features[site_name] = feat.to(self.store_device)

        return _fn

    def on_step_end(
        self,
        pipeline: Any,
        step_index: int,
        timestep: int,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Pass to ``StableDiffusionPipeline(..., callback_on_step_end=...)``.

        Flushes hook outputs captured during this denoise step into ``records``.
        """
        del pipeline, callback_kwargs
        for site in self.sites:
            site_name = site.value
            feat = self._pending_features.get(site_name)
            if feat is None:
                continue

            frames = None
            meta = None
            if self.map_to_frames:
                frames, meta = feature_map_to_frames(
                    feat,
                    frame_size=self.frame_size,
                    channel_group_size=self.channel_group_size,
                    cfg_batch=self.cfg_batch,
                )

            self._records.append(
                StepCapture(
                    timestep=int(timestep),
                    step_index=int(step_index),
                    site=site_name,
                    feature=feat,
                    frames_meta=meta,
                    frames=frames,
                )
            )
            self._pending_features.pop(site_name, None)

        self._step_index = step_index + 1
        return {}


def attach_feature_hooks(
    unet: nn.Module,
    capture: DiffusionFeatureCapture,
) -> nn.Module:
    """
    Register forward hooks on selected UNet submodules.

    Returns an ``nn.Module`` (``_HookHandleBag``) whose ``.remove()`` drops all hooks.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []

    class _HookHandleBag(nn.Module):
        def remove(self) -> None:
            for h in handles:
                h.remove()

    for site in capture.sites:
        target = _resolve_hook_module(unet, site)
        h = target.register_forward_hook(capture._hook_fn(site))
        handles.append(h)

    return _HookHandleBag()
