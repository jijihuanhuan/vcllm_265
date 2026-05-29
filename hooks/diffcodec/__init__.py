"""DiffCodec: temporal redundancy elimination for Diffusion feature maps."""

from .feature_capture import DiffusionFeatureCapture, HookSite, attach_feature_hooks
from .feature_map_mapper import (
    FeatureMapMappingMeta,
    feature_map_to_frames,
    frames_to_feature_map,
    group_feature_map_for_mapping,
    select_cfg_batch,
)

__all__ = [
    "DiffusionFeatureCapture",
    "FeatureMapMappingMeta",
    "HookSite",
    "attach_feature_hooks",
    "feature_map_to_frames",
    "frames_to_feature_map",
    "select_cfg_batch",
    "group_feature_map_for_mapping",
]
