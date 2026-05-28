import json
import os
from typing import Literal

import numpy as np
import torch

from .rtn import rtn_quantize_tensor, rtn_dequantize_tensor
from codec.codec_job import CodecJob
from codec.frame_mapper import tensor_to_frames, frames_to_tensor
from codec.hevc_backend import encode_frames_to_bitstream, decode_bitstream_to_frames
from codec.metadata import (
    save_metadata,
    load_metadata,
    build_pipeline_metadata,
    build_rtn_only_metadata,
    codec_job_from_metadata,
)

SEPARATOR = "___DOT___"

WeightCompressionMode = Literal["rtn_only", "rtn_lossless_hevc", "rtn_lossy_hevc"]

_RTNQ_EXT = ".rtnq"
_HEVC_EXT = ".hevc"


def compress_weight_layer(
    layer_name: str,
    weight_tensor: torch.Tensor,
    output_dir: str,
    *,
    mode: WeightCompressionMode = "rtn_lossless_hevc",
    qp: int = 3,
    hardware_accel: bool = True,
    frame_size: int = 1024,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    min_val = weight_tensor.min().item()
    max_val = weight_tensor.max().item()

    quantized, rtn_metadata = rtn_quantize_tensor(weight_tensor, min_val=min_val, max_val=max_val)
    original_size_bytes = weight_tensor.numel() * weight_tensor.element_size()

    extra_base = {
        "layer_name": layer_name,
        "min_val": min_val,
        "max_val": max_val,
        "compression_mode": mode,
    }

    if mode == "rtn_only":
        bin_path = os.path.join(output_dir, f"{layer_name}{_RTNQ_EXT}")
        quantized.cpu().numpy().tofile(bin_path)
        metadata = build_rtn_only_metadata(rtn_metadata, extra=extra_base)
        metadata_path = os.path.join(output_dir, f"{layer_name}_metadata.json")
        save_metadata(metadata, metadata_path)
        return {
            "layer_name": layer_name,
            "compression_mode": mode,
            "original_size_bytes": original_size_bytes,
            "compressed_size_bytes": os.path.getsize(bin_path),
            "metadata": metadata,
        }

    if mode == "rtn_lossless_hevc":
        lossless = True
    elif mode == "rtn_lossy_hevc":
        lossless = False
    else:
        raise ValueError(f"Unknown compression mode: {mode}")

    frames, frame_metadata = tensor_to_frames(quantized, frame_size=frame_size)

    # GPU-first (paper): NVENC when hardware_accel; no libx265 fallback unless encode fails
    # is disabled via allow_software_encoder_fallback on the encode path.
    if mode == "rtn_lossless_hevc":
        backend = "hevc_nvenc" if hardware_accel else "libx265"
    else:
        backend = "auto" if hardware_accel else "libx265"
    codec_job = CodecJob.square(
        frame_metadata["frame_size"],
        fps=1.0,
        intra_only=True,
        qp=qp,
        lossless=lossless,
        backend=backend,
    )

    bitstream_path = os.path.join(output_dir, f"{layer_name}{_HEVC_EXT}")
    encode_frames_to_bitstream(
        frames,
        bitstream_path,
        codec_job,
        allow_software_encoder_fallback=not hardware_accel,
    )

    metadata = build_pipeline_metadata(
        rtn_metadata,
        frame_metadata,
        codec_job,
        extra=extra_base,
    )
    metadata_path = os.path.join(output_dir, f"{layer_name}_metadata.json")
    save_metadata(metadata, metadata_path)

    return {
        "layer_name": layer_name,
        "compression_mode": mode,
        "original_size_bytes": original_size_bytes,
        "compressed_size_bytes": os.path.getsize(bitstream_path),
        "metadata": metadata,
    }


def _infer_legacy_compression_mode(metadata: dict, layer_name: str, input_dir: str) -> WeightCompressionMode:
    rt_path = os.path.join(input_dir, f"{layer_name}{_RTNQ_EXT}")
    hv_path = os.path.join(input_dir, f"{layer_name}{_HEVC_EXT}")
    if os.path.isfile(rt_path):
        return "rtn_only"
    if os.path.isfile(hv_path):
        cj = metadata.get("codec_job") if isinstance(metadata.get("codec_job"), dict) else {}
        lossless = bool(cj.get("lossless", metadata.get("lossless", True)))
        return "rtn_lossless_hevc" if lossless else "rtn_lossy_hevc"
    raise FileNotFoundError(
        f"No compressed payload for layer {layer_name}: missing {_RTNQ_EXT} or {_HEVC_EXT}"
    )


def decompress_weight_layer(
    layer_name: str,
    input_dir: str,
    *,
    hardware_accel: bool = True,
    hardware_decode: bool = True,
) -> torch.Tensor:
    # hardware_accel: API compatibility. hardware_decode: try hevc_cuvid before PNG (optional).
    metadata_path = os.path.join(input_dir, f"{layer_name}_metadata.json")
    metadata = load_metadata(metadata_path)

    mode: WeightCompressionMode | None = metadata.get("compression_mode")  # type: ignore[assignment]
    if mode is None:
        mode = _infer_legacy_compression_mode(metadata, layer_name, input_dir)

    if mode == "rtn_only":
        bin_path = os.path.join(input_dir, f"{layer_name}{_RTNQ_EXT}")
        shape = tuple(metadata["rtn"]["original_shape"])
        raw = np.fromfile(bin_path, dtype=np.uint8)
        expected = int(np.prod(shape))
        if raw.size != expected:
            raise ValueError(
                f"RTN-only size mismatch for {layer_name}: file has {raw.size} bytes, expected {expected}"
            )
        quantized = torch.from_numpy(raw.reshape(shape))
        weight_tensor = rtn_dequantize_tensor(quantized, metadata)
    else:
        bitstream_path = os.path.join(input_dir, f"{layer_name}{_HEVC_EXT}")
        job = codec_job_from_metadata(metadata)
        frames = decode_bitstream_to_frames(
            bitstream_path,
            job,
            force_software_decode=not hardware_decode,
            prefer_hardware_decode=hardware_decode,
        )
        quantized = frames_to_tensor(frames, metadata)
        weight_tensor = rtn_dequantize_tensor(quantized, metadata)

    dtype_str = metadata.get("original_dtype")
    if dtype_str is None and isinstance(metadata.get("rtn"), dict):
        dtype_str = metadata["rtn"].get("original_dtype")
    if dtype_str:
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
        }
        if dtype_str in dtype_map:
            weight_tensor = weight_tensor.to(dtype_map[dtype_str])

    return weight_tensor


def _load_state_dict_strict_false_warn(model: torch.nn.Module, state_dict: dict) -> None:
    """strict=False + explicit warning listing missing/unexpected keys (PyTorch 2.x API)."""
    inc = model.load_state_dict(state_dict, strict=False)
    missing = getattr(inc, "missing_keys", None)
    unexpected = getattr(inc, "unexpected_keys", None)
    if missing is None and unexpected is None:
        return
    if not missing and not unexpected:
        return
    print(
        "[weight_pipeline] Warning: load_state_dict(strict=False) completed with key mismatches.",
        flush=True,
    )
    if missing:
        head = missing[:32]
        tail = " ..." if len(missing) > 32 else ""
        print(f"  missing_keys ({len(missing)}): {head}{tail}", flush=True)
    if unexpected:
        head = unexpected[:32]
        tail = " ..." if len(unexpected) > 32 else ""
        print(f"  unexpected_keys ({len(unexpected)}): {head}{tail}", flush=True)


def _load_decompressed_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """Prefer strict load; fall back with explicit missing/unexpected reporting."""
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as err:
        print(
            f"[weight_pipeline] Warning: strict load_state_dict failed ({err}); "
            "retrying with strict=False.",
            flush=True,
        )
        _load_state_dict_strict_false_warn(model, state_dict)


def compress_model_weights(
    model: torch.nn.Module,
    output_dir: str,
    *,
    mode: WeightCompressionMode = "rtn_lossless_hevc",
    qp: int = 3,
    hardware_accel: bool = True,
    frame_size: int = 1024,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    stats = []

    for name, param in model.named_parameters():
        print(f"Compressing {name}...")
        safe_name = name.replace(".", SEPARATOR)
        stat = compress_weight_layer(
            safe_name,
            param.data.cpu(),
            output_dir,
            mode=mode,
            qp=qp,
            hardware_accel=hardware_accel,
            frame_size=frame_size,
        )
        stat["original_name"] = name
        stats.append(stat)

    total_original = sum(s["original_size_bytes"] for s in stats)
    total_compressed = sum(s["compressed_size_bytes"] for s in stats)

    summary = {
        "compression_mode": mode,
        "total_layers": len(stats),
        "total_original_bytes": total_original,
        "total_compressed_bytes": total_compressed,
        "compression_ratio": total_original / total_compressed if total_compressed > 0 else 0,
        "layers": stats,
        "hardware_accel": hardware_accel,
        "qp": qp,
        "frame_size": frame_size,
    }

    with open(os.path.join(output_dir, "compression_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCompression Summary:")
    print(f"  Mode: {mode}")
    print(f"  Layers: {summary['total_layers']}")
    print(f"  Original: {total_original / (1024**2):.2f} MB (element_size-accurate)")
    print(f"  Compressed: {total_compressed / (1024**2):.2f} MB")
    print(f"  Ratio: {summary['compression_ratio']:.2f}x")
    if mode == "rtn_lossless_hevc":
        print(
            "  Encoder: "
            + ("NVENC lossless" if hardware_accel else "libx265 lossless (CPU)")
        )
    else:
        print(f"  Hardware Acceleration (encode): {hardware_accel}")
    print(f"  QP (HEVC modes): {qp}")

    return summary


def decompress_model_weights(
    model: torch.nn.Module,
    input_dir: str,
    *,
    hardware_accel: bool = True,
    hardware_decode: bool = True,
) -> torch.nn.Module:
    summary_path = os.path.join(input_dir, "compression_summary.json")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    if hardware_decode:
        print(
            "[weight_pipeline] hardware_decode=True: using hevc_cuvid when possible. "
            "If perplexity explodes, reload without GPU decode (software PNG path).",
            flush=True,
        )

    state_dict = {}

    for layer_info in summary["layers"]:
        layer_name = layer_info["layer_name"]
        original_name = layer_info.get("original_name", layer_name.replace(SEPARATOR, "."))
        print(f"Decompressing {layer_name}...")
        weight = decompress_weight_layer(
            layer_name,
            input_dir,
            hardware_accel=hardware_accel,
            hardware_decode=hardware_decode,
        )
        state_dict[original_name] = weight

    _load_decompressed_state_dict(model, state_dict)
    return model
