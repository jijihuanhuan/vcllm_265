"""
JSON metadata for tensor codec pipeline: RTN stats, frame mapping, CodecJob.
"""
from __future__ import annotations

import json
import os
from typing import Any

from codec.codec_job import CodecJob

SCHEMA_VERSION = 1


def save_metadata(metadata: dict[str, Any], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_metadata(*metadata_list: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge (later keys overwrite). Prefer build_pipeline_metadata for new code."""
    merged: dict[str, Any] = {}
    for metadata in metadata_list:
        merged.update(metadata)
    return merged


def build_pipeline_metadata(
    rtn_metadata: dict[str, Any],
    frame_metadata: dict[str, Any],
    codec_job: CodecJob,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Structured, versioned snapshot for encode/decode round-trip.

    Includes nested sections `rtn`, `frame_mapping`, `codec_job`, plus flat keys
    compatible with rtn_dequantize_tensor / frames_to_tensor.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rtn": {
            "min_val": rtn_metadata["min_val"],
            "max_val": rtn_metadata["max_val"],
            "num_bits": rtn_metadata["num_bits"],
            "scale": rtn_metadata["scale"],
            "original_shape": rtn_metadata.get("original_shape"),
            "original_dtype": rtn_metadata.get("original_dtype"),
        },
        "frame_mapping": {
            "frame_size": frame_metadata["frame_size"],
            "width": frame_metadata.get("width", frame_metadata["frame_size"]),
            "height": frame_metadata.get("height", frame_metadata["frame_size"]),
            "num_frames": frame_metadata["num_frames"],
            "padding_len": frame_metadata["padding_len"],
            "original_shape": frame_metadata["original_shape"],
            "dtype": frame_metadata["dtype"],
            "content_frame_size": frame_metadata.get(
                "content_frame_size", frame_metadata["frame_size"]
            ),
            "logical_frame_size": frame_metadata.get(
                "logical_frame_size",
                frame_metadata.get("content_frame_size", frame_metadata["frame_size"]),
            ),
            "hardware_frame_size": frame_metadata.get(
                "hardware_frame_size", frame_metadata["frame_size"]
            ),
            "padded_to": frame_metadata.get("padded_to"),
        },
        "codec_job": codec_job.to_dict(),
    }
    if extra:
        payload.update(extra)
    payload.update(_flatten_for_decode_hooks(payload))
    return payload


def _flatten_for_decode_hooks(payload: dict[str, Any]) -> dict[str, Any]:
    """Flat keys expected by rtn_dequantize_tensor and frames_to_tensor."""
    rtn = payload["rtn"]
    fm = payload["frame_mapping"]
    flat = {
        "min_val": rtn["min_val"],
        "max_val": rtn["max_val"],
        "num_bits": rtn["num_bits"],
        "scale": rtn["scale"],
        "original_dtype": rtn.get("original_dtype"),
        "original_shape": fm["original_shape"],
        "padding_len": fm["padding_len"],
        "dtype": fm["dtype"],
        "frame_size": fm["frame_size"],
        "num_frames": fm["num_frames"],
        "content_frame_size": fm.get("content_frame_size", fm["frame_size"]),
        "hardware_frame_size": fm.get("hardware_frame_size", fm["frame_size"]),
    }
    if fm.get("padded_to") is not None:
        flat["padded_to"] = fm["padded_to"]
    return flat


def build_rtn_only_metadata(
    rtn_metadata: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Metadata for RTN-only storage (no video codec bitstream)."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "compression_mode": "rtn_only",
        "rtn": {
            "min_val": rtn_metadata["min_val"],
            "max_val": rtn_metadata["max_val"],
            "num_bits": rtn_metadata["num_bits"],
            "scale": rtn_metadata["scale"],
            "original_shape": rtn_metadata.get("original_shape"),
            "original_dtype": rtn_metadata.get("original_dtype"),
        },
        "frame_mapping": None,
        "codec_job": None,
    }
    if extra:
        payload.update(extra)
    payload.update(_flatten_rtn_only_decode(payload))
    return payload


def _flatten_rtn_only_decode(payload: dict[str, Any]) -> dict[str, Any]:
    """Flat keys for rtn_dequantize_tensor after loading raw uint8 tensor."""
    rtn = payload["rtn"]
    shape = rtn["original_shape"]
    return {
        "min_val": rtn["min_val"],
        "max_val": rtn["max_val"],
        "num_bits": rtn["num_bits"],
        "scale": rtn["scale"],
        "original_dtype": rtn.get("original_dtype"),
        "original_shape": shape,
        "padding_len": 0,
        "dtype": "torch.uint8",
        "frame_size": 0,
        "num_frames": 0,
    }


def codec_job_from_metadata(metadata: dict[str, Any]) -> CodecJob:
    """Restore CodecJob from saved metadata; supports legacy flat-only JSON."""
    if metadata.get("compression_mode") == "rtn_only":
        raise ValueError("RTN-only payloads do not contain a CodecJob")
    if metadata.get("schema_version") == SCHEMA_VERSION and metadata.get("codec_job"):
        return CodecJob.from_dict(metadata["codec_job"])

    fs = int(metadata.get("frame_size", 1024))
    return CodecJob(
        width=fs,
        height=fs,
        fps=1.0,
        intra_only=True,
        qp=int(metadata.get("qp", 0)),
        lossless=bool(metadata.get("lossless", True)),
        backend="auto",
    )
