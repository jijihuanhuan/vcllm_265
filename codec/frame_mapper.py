"""
Tensor ↔ fixed-size grayscale frame tiles for the RTN + HEVC pipeline.

**Value range:** No luma scaling or TV-range conversion. Padding uses constant ``0``.

**NVENC minimum geometry:** HEVC NVENC typically requires frames ≥256×256. When the logical
tile ``frame_size`` is smaller, each tile is zero-padded to ``hardware_frame_size`` before
encode; ``frames_to_tensor`` crops back using ``content_frame_size``.
"""
from __future__ import annotations

import numpy as np
import torch

# HEVC NVENC safe minimum (covers 128/144 SKU variance; user requirement 256×256).
NVENC_MIN_FRAME_HW = 256


def hardware_encode_frame_size(logical_frame_size: int) -> int:
    """Edge length of each stored frame (NVENC-safe)."""
    return max(int(logical_frame_size), NVENC_MIN_FRAME_HW)


def tensor_to_frames(tensor, frame_size=1024):
    """
    Tile ``tensor`` into ``(num_frames, H, H)`` uint8/float frames for ffmpeg.

    - First pads the flattened vector to a multiple of ``L*L`` where ``L=frame_size``
      (logical content tile).
    - If ``L < NVENC_MIN_FRAME_HW``, each ``L×L`` tile is embedded top-left in a
      ``H×H`` canvas with ``H=hardware_encode_frame_size(L)``, rest zeros.
    """
    L = int(frame_size)
    if L <= 0:
        raise ValueError(f"frame_size must be positive, got {L}")

    H_enc = hardware_encode_frame_size(L)
    flat = tensor.flatten().detach().cpu().numpy()

    tile = L * L
    padding_len = (tile - len(flat) % tile) % tile
    if padding_len > 0:
        flat = np.pad(flat, (0, padding_len), mode="constant")

    num_frames = len(flat) // tile
    small = flat.reshape(num_frames, L, L)

    if H_enc > L:
        frames = np.zeros((num_frames, H_enc, H_enc), dtype=small.dtype)
        frames[:, :L, :L] = small
        padded_to = [H_enc, H_enc]
    else:
        frames = small
        padded_to = None

    metadata = {
        "original_shape": list(tensor.shape),
        "content_frame_size": L,
        "logical_frame_size": L,
        "hardware_frame_size": H_enc,
        "padded_to": padded_to,
        "frame_size": H_enc,
        "width": H_enc,
        "height": H_enc,
        "num_frames": num_frames,
        "padding_len": int(padding_len),
        "dtype": str(tensor.dtype),
    }

    return frames, metadata


def frames_to_tensor(frames, metadata):
    """
    Inverse of ``tensor_to_frames``: crop NVENC padding, drop 1D tail ``padding_len``,
    reshape to ``original_shape``.
    """
    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"frames must be (N, H, W), got shape {arr.shape}")

    L = int(metadata.get("content_frame_size", metadata.get("logical_frame_size", 0)))
    if L <= 0:
        L = int(metadata["frame_size"])

    H_enc = int(metadata.get("hardware_frame_size", metadata["frame_size"]))
    if arr.shape[1] != H_enc or arr.shape[2] != H_enc:
        raise ValueError(
            f"Decoded frame size {arr.shape[1:]} does not match metadata hardware_frame_size "
            f"{H_enc}x{H_enc}"
        )

    cropped = arr[:, :L, :L]
    flat = cropped.reshape(-1)

    pl = int(metadata.get("padding_len", 0))
    if pl > 0:
        flat = flat[:-pl]

    return torch.tensor(flat.reshape(metadata["original_shape"]))
