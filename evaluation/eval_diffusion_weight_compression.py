"""
Diffusion (Stable Diffusion 1.5) UNet weight compression eval on the existing VcLLM
RTN + HEVC tensor pipeline.

This script generates aligned baseline/compressed image batches:
  outputs/baseline/img_0001.png ...
  outputs/compressed/img_0001.png ...
and writes a prompt manifest:
  outputs/prompts.csv

**Strict policy:** weight compression encode/decode uses **only** GPU hardware
(``hevc_nvenc`` / ``hevc_cuvid``). No CPU libx265 or PNG decode fallback is permitted
for this script; failures raise ``RuntimeError`` immediately.

Run from repository root:
  python evaluation/eval_diffusion_weight_compression.py --num-images 100
  python evaluation/eval_diffusion_weight_compression.py --num-images 100 --hevc-lossy --qp 0
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
from torch import Tensor

from codec.codec_job import CodecJob
from codec.frame_mapper import frames_to_tensor, tensor_to_frames
from codec.hevc_backend import (
    assert_gpu_hevc_hw_codecs_available,
    decode_bitstream_to_frames,
    encode_frames_to_bitstream,
)
from codec.metadata import build_pipeline_metadata, codec_job_from_metadata
from compression.rtn import rtn_dequantize_tensor, rtn_quantize_tensor

SEED = 42
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUTPUT_ROOT = os.path.join(_REPO_ROOT, "outputs")

# Fixed prompt set (COCO/DrawBench style) for reproducible batch evaluation.
PROMPTS = [
    "A cute golden retriever dog playing in the snow, high quality",
    "A red double-decker bus driving through rainy London streets at night",
    "A rustic kitchen table with fresh vegetables and warm morning sunlight",
    "A skateboarder jumping over stairs in an urban plaza, motion blur",
    "A macro photo of a sunflower covered with morning dew",
    "A snowy mountain landscape with a frozen lake and pine trees",
    "A portrait of an astronaut reading a book in a cozy cafe",
    "A bowl of ramen on a wooden table, shallow depth of field",
    "A futuristic city skyline at sunset with flying cars",
    "A watercolor painting of a lighthouse on a stormy coast",
    "A child flying a colorful kite in a large grassy field",
    "An old bookstore interior filled with warm lights and dust particles",
    "A close-up of a tabby cat looking out of a train window",
    "A blue bicycle parked beside a yellow wall with flowers",
    "A cinematic shot of a forest trail in dense fog",
    "A glass bottle floating in the ocean at golden hour",
    "A robot chef cooking pancakes in a modern kitchen",
    "A dragon made of clouds above a medieval castle",
    "A minimal Scandinavian living room with natural light",
    "An aerial view of winding desert roads and rock formations",
]


@dataclass(frozen=True)
class PromptRecord:
    image_name: str
    image_index: int
    prompt_id: int
    prompt: str
    seed: int


def _weight_matrix_view(w: Tensor) -> Tensor:
    """4D conv kernel -> 2D matrix (out, rest); 2D linear weight unchanged."""
    if w.dim() == 4:
        return w.reshape(w.shape[0], -1)
    if w.dim() == 2:
        return w
    raise ValueError(f"Expected Conv2d (4D) or Linear (2D) weight, got shape {tuple(w.shape)}")


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fwht_last_dim(x: Tensor) -> Tensor:
    """Orthonormal FWHT on the last dimension (last size must be a power of two)."""
    n = int(x.shape[-1])
    if n == 1:
        return x
    if not _is_pow2(n):
        raise ValueError(f"FWHT requires last-dim size power of 2, got {n}")
    prefix = x.shape[:-1]
    y = x.reshape(-1, n).float().clone()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.cat([a + b, a - b], dim=-1).reshape(-1, n)
        h *= 2
    y = y * (n**-0.5)
    return y.view(*prefix, n).to(dtype=x.dtype)


def _fwht_dim(x: Tensor, dim: int) -> Tensor:
    """FWHT along ``dim`` (that dimension's size must be a power of two)."""
    return _fwht_last_dim(x.movedim(dim, -1)).movedim(-1, dim)


def _apply_hadamard_conv4d(w: Tensor) -> tuple[Tensor, dict]:
    """Channel-only incoherence: FWHT on O (dim 0) and I (dim 1); Kh,Kw untouched."""
    o, i, kh, kw = w.shape
    y = w.float().clone()
    o_ok, i_ok = _is_pow2(o), _is_pow2(i)
    if o_ok:
        y = _fwht_dim(y, 0)
    if i_ok:
        y = _fwht_dim(y, 1)
    meta: dict = {
        "enabled": True,
        "kind": "conv4d",
        "o": int(o),
        "i": int(i),
        "kh": int(kh),
        "kw": int(kw),
        "o_fwht": o_ok,
        "i_fwht": i_ok,
    }
    return y, meta


def _invert_hadamard_conv4d(x: Tensor, meta: dict) -> Tensor:
    y = x.float().clone()
    if meta.get("i_fwht"):
        y = _fwht_dim(y, 1)
    if meta.get("o_fwht"):
        y = _fwht_dim(y, 0)
    return y.to(dtype=x.dtype)


def _apply_hadamard_linear2d(w: Tensor) -> tuple[Tensor, dict]:
    """Orthonormal FWHT on out-features then in-features (dim 0 then 1)."""
    o, i = w.shape
    y = w.float().clone()
    o_ok, i_ok = _is_pow2(o), _is_pow2(i)
    if o_ok:
        y = _fwht_dim(y, 0)
    if i_ok:
        y = _fwht_dim(y, 1)
    meta = {
        "enabled": True,
        "kind": "linear2d",
        "o": int(o),
        "i": int(i),
        "o_fwht": o_ok,
        "i_fwht": i_ok,
    }
    return y, meta


def _invert_hadamard_linear2d(x: Tensor, meta: dict) -> Tensor:
    y = x.float().clone()
    if meta.get("i_fwht"):
        y = _fwht_dim(y, 1)
    if meta.get("o_fwht"):
        y = _fwht_dim(y, 0)
    return y.to(dtype=x.dtype)


def _apply_hadamard_to_weight(w: Tensor, use: bool) -> tuple[Tensor, dict | None]:
    """Return (matrix_2d_or_same, hadamard_meta_for_json)."""
    if not use:
        return _weight_matrix_view(w), None
    if w.dim() == 4:
        t, meta = _apply_hadamard_conv4d(w.detach().cpu())
        m2 = t.reshape(t.shape[0], -1)
        return m2, meta
    if w.dim() == 2:
        t, meta = _apply_hadamard_linear2d(w.detach().cpu())
        return t, meta
    raise ValueError(f"use_hadamard: unsupported weight dim {w.dim()}")


def _invert_hadamard_matrix(mat: Tensor, meta: dict | None) -> Tensor:
    if meta is None or not meta.get("enabled"):
        return mat
    kind = meta.get("kind")
    if kind == "conv4d":
        o, i, kh, kw = meta["o"], meta["i"], meta["kh"], meta["kw"]
        t = mat.reshape(o, i, kh, kw).float()
        return _invert_hadamard_conv4d(t, meta).reshape(mat.shape).to(dtype=mat.dtype)
    if kind == "linear2d":
        return _invert_hadamard_linear2d(mat.float(), meta).to(dtype=mat.dtype)
    raise ValueError(f"Unknown hadamard kind {kind!r}")


ERR_NO_CUDA = (
    "本脚本需要 CUDA GPU：torch.cuda.is_available() 为 False。"
    "请先在同一环境中确认 `nvidia-smi` 与 `python -c \"import torch; print(torch.cuda.is_available())\"`。"
    "扩散推理与（非 debug）权重编解码依赖 GPU。"
    "禁止 CPU 软件编解码回退。"
)


def _flatten_pad_unpad_roundtrip(matrix: Tensor, frame_size: int) -> Tensor:
    """
    Control A: same path as ``tensor_to_frames`` → ``frames_to_tensor`` (1D pad + NVENC min HW pad).
    """
    m = matrix.detach().cpu()
    frames, meta = tensor_to_frames(m, frame_size=frame_size)
    out = frames_to_tensor(frames, meta)
    return out.to(dtype=m.dtype).to(device=matrix.device)


def _rtn_roundtrip_matrix(matrix: Tensor) -> Tensor:
    """Control B: asymmetric min–max 8-bit RTN quantize then dequantize (CPU)."""
    t = matrix.detach().cpu()
    mn, mx = t.min().item(), t.max().item()
    q, meta = rtn_quantize_tensor(t, min_val=mn, max_val=mx)
    return rtn_dequantize_tensor(q, meta).to(dtype=t.dtype)


def _compress_tensor(
    tensor: Tensor,
    *,
    qp: int = 3,
    frame_size: int = 1024,
    lossless: bool = True,
    hadamard_meta: dict | None = None,
    visual_optimization: bool = False,
) -> tuple[bytes, dict]:
    """
    In-memory RTN + HEVC (GPU NVENC only, no software encoder fallback).
    ``hadamard_meta`` (optional) is stored in sidecar metadata for inverse FWHT after decode.
    ``lossless=False`` uses NVENC constqp with ``qp`` (higher QP ≈ stronger compression / lower quality).
    When ``lossless=False`` and ``visual_optimization=False`` (default), NVENC disables spatial/temporal
    AQ (portable across ffmpeg 4.x; ``-tune psnr`` is not used on NVENC — unsupported on many builds).
    libx265 still uses ``-tune psnr`` and ``aq-mode=0:no-sao=1:no-deblock=1``. Set ``visual_optimization=True``
    to use encoder defaults for subjective video tuning (``--enable-visual-opts`` in the eval CLI).
    """
    tensor_cpu = tensor.detach().cpu()
    min_val = tensor_cpu.min().item()
    max_val = tensor_cpu.max().item()

    quantized, rtn_metadata = rtn_quantize_tensor(tensor_cpu, min_val=min_val, max_val=max_val)
    frames, frame_metadata = tensor_to_frames(quantized, frame_size=frame_size)

    codec_job = CodecJob.square(
        frame_metadata["frame_size"],
        fps=1.0,
        intra_only=True,
        qp=qp,
        lossless=lossless,
        backend="hevc_nvenc",
        visual_optimization=visual_optimization,
    )

    with tempfile.NamedTemporaryFile(suffix=".hevc", delete=False) as f:
        bitstream_path = f.name

    try:
        encode_frames_to_bitstream(
            frames,
            bitstream_path,
            codec_job,
            allow_software_encoder_fallback=False,
        )
        with open(bitstream_path, "rb") as bf:
            bitstream_data = bf.read()
    finally:
        if os.path.isfile(bitstream_path):
            os.remove(bitstream_path)

    mode = "rtn_lossless_hevc" if lossless else "rtn_lossy_hevc"
    extra: dict = {"compression_mode": mode}
    if hadamard_meta is not None:
        extra["hadamard"] = hadamard_meta
    metadata = build_pipeline_metadata(
        rtn_metadata,
        frame_metadata,
        codec_job,
        extra=extra,
    )
    return bitstream_data, metadata


def _decompress_tensor(bitstream_data: bytes, metadata: dict) -> Tensor:
    """Decode with hevc_cuvid only (no PNG/software decoder fallback)."""
    with tempfile.NamedTemporaryFile(suffix=".hevc", delete=False) as f:
        f.write(bitstream_data)
        bitstream_path = f.name
    try:
        job = codec_job_from_metadata(metadata)
        frames = decode_bitstream_to_frames(
            bitstream_path,
            job,
            force_software_decode=False,
            prefer_hardware_decode=True,
            allow_software_decoder_fallback=False,
        )
        quantized = frames_to_tensor(frames, metadata)
        weight_tensor = rtn_dequantize_tensor(quantized, metadata)
        had = metadata.get("hadamard")
        weight_tensor = _invert_hadamard_matrix(weight_tensor, had)
    finally:
        if os.path.isfile(bitstream_path):
            os.remove(bitstream_path)

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


def _unet_weight_bytes(unet: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in unet.parameters())


def _image_name(index: int, n_total: int) -> str:
    width = max(3, len(str(n_total)))
    return f"img_{index:0{width}d}.png"


def _build_prompt_records(num_images: int, seed: int) -> list[PromptRecord]:
    if num_images <= 0:
        raise ValueError("--num-images must be > 0")
    records: list[PromptRecord] = []
    for idx in range(1, num_images + 1):
        prompt_id = (idx - 1) % len(PROMPTS)
        records.append(
            PromptRecord(
                image_name=_image_name(idx, num_images),
                image_index=idx,
                prompt_id=prompt_id,
                prompt=PROMPTS[prompt_id],
                seed=seed + idx - 1,
            )
        )
    return records


def _save_prompt_manifest(path: str, records: list[PromptRecord]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_name", "image_index", "prompt_id", "seed", "prompt"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "image_name": r.image_name,
                    "image_index": r.image_index,
                    "prompt_id": r.prompt_id,
                    "seed": r.seed,
                    "prompt": r.prompt,
                }
            )


@torch.inference_mode()
def _generate_save(
    pipe,
    prompt: str,
    seed: int,
    out_path: str,
    num_inference_steps: int,
    height: int,
    width: int,
) -> None:
    device = pipe.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=gen,
    ).images[0]
    image.save(out_path)
    print(f"Saved {out_path}", flush=True)


def _generate_batch(
    pipe,
    records: list[PromptRecord],
    out_dir: str,
    *,
    num_inference_steps: int,
    height: int,
    width: int,
    label: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"[generate] {label}: writing {len(records)} images -> {out_dir}", flush=True)
    for i, record in enumerate(records, start=1):
        out_path = os.path.join(out_dir, record.image_name)
        print(
            f"[{label}] {i}/{len(records)} {record.image_name} | prompt_id={record.prompt_id} | seed={record.seed}",
            flush=True,
        )
        _generate_save(
            pipe,
            record.prompt,
            record.seed,
            out_path,
            num_inference_steps,
            height,
            width,
        )


def _module_in_subtree(root: nn.Module, target: nn.Module) -> bool:
    for m in root.modules():
        if m is target:
            return True
    return False


def _weight_module_in_up_blocks_only(unet: nn.Module, module: nn.Module) -> bool:
    """True iff ``module`` is a strict descendant of some ``unet.up_blocks`` child."""
    up_blocks = getattr(unet, "up_blocks", None)
    if up_blocks is None:
        return False
    for block in up_blocks:
        if _module_in_subtree(block, module):
            return True
    return False


def replace_unet_weights_with_compressed(
    unet: nn.Module,
    *,
    debug_mode: str | None,
    qp: int = 3,
    frame_size: int = 1024,
    hevc_lossless: bool = True,
    use_hadamard: bool = False,
    visual_optimization: bool = False,
) -> tuple[int, int, int, int]:
    """
    In-place replace Conv2d / Linear ``.weight`` per ``debug_mode``.

    Returns:
        (total_compressed_bytes, total_elems_touched, n_layers_mutated, n_hevc_layers)
    """
    total_bits = 0
    total_elems = 0
    n_mutated = 0
    n_hevc = 0

    for module in unet.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        w = module.weight
        if debug_mode == "partial_hevc" and not _weight_module_in_up_blocks_only(unet, module):
            continue

        orig_shape = w.shape
        matrix, had_meta = _apply_hadamard_to_weight(w.data, use_hadamard)
        n_mutated += 1
        total_elems += matrix.numel()

        if debug_mode == "reshape_only":
            restored_m = _flatten_pad_unpad_roundtrip(matrix, frame_size)
        elif debug_mode == "rtn_only":
            restored_m = _rtn_roundtrip_matrix(matrix)
        else:
            bitstream, meta = _compress_tensor(
                matrix,
                qp=qp,
                frame_size=frame_size,
                lossless=hevc_lossless,
                hadamard_meta=had_meta,
                visual_optimization=visual_optimization,
            )
            total_bits += len(bitstream) * 8
            n_hevc += 1
            restored = _decompress_tensor(bitstream, meta)
            if restored.shape != matrix.shape:
                raise RuntimeError(f"Shape mismatch after decode: {restored.shape} vs {matrix.shape}")
            restored_m = restored

        if use_hadamard and had_meta is not None and debug_mode in ("reshape_only", "rtn_only"):
            restored_m = _invert_hadamard_matrix(restored_m, had_meta)

        if restored_m.shape != matrix.shape:
            raise RuntimeError(f"Shape mismatch: {restored_m.shape} vs {matrix.shape}")
        restored_w = restored_m.reshape(orig_shape).to(device=w.device, dtype=w.dtype)
        w.data.copy_(restored_w)

    return total_bits // 8, total_elems, n_mutated, n_hevc


def main() -> None:
    parser = argparse.ArgumentParser(description="SD1.5 UNet weight compression + image fidelity (GPU HW codecs only)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for outputs/{baseline,compressed,prompts.csv}",
    )
    parser.add_argument("--num-images", type=int, default=1, help="Number of aligned images to generate")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--qp", type=int, default=3, help="HEVC QP when --hevc-lossy (ignored in lossless)")
    parser.add_argument(
        "--hevc-lossy",
        action="store_true",
        help="Use lossy NVENC (constqp + --qp) instead of lossless; for strength sweeps",
    )
    parser.add_argument(
        "--debug-mode",
        choices=["reshape_only", "rtn_only", "partial_hevc"],
        default=None,
        help="Control experiment: reshape_only | rtn_only | partial_hevc (up_blocks HEVC only)",
    )
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument(
        "--use-hadamard",
        action="store_true",
        help="Channel-only FWHT on Conv O/I dims (pow2) or on Linear rows/cols (pow2); compare BPE/quality vs baseline",
    )
    parser.add_argument(
        "--enable-visual-opts",
        action="store_true",
        help=(
            "Lossy HEVC only: set CodecJob.visual_optimization=True (encoder default AQ / subjective tuning). "
            "Default False uses tensor-oriented settings: NVENC disables spatial/temporal AQ; libx265 uses "
            "-tune psnr with aq-mode=0 / no SAO / no deblock for tensor data."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not call Hugging Face Hub: load SD from disk cache only (avoids ConnectTimeout if offline)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override model id or local snapshot path (default: runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument(
        "--no-offload-during-codec",
        action="store_true",
        help="Keep the full SD pipeline on GPU during RTN+HEVC (may OOM when ffmpeg hevc_cuvid creates a second CUDA context)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(ERR_NO_CUDA)

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: diffusers. Install with:\n  pip install diffusers accelerate"
        ) from e

    needs_hw_codec = args.debug_mode not in ("reshape_only", "rtn_only")
    if needs_hw_codec:
        assert_gpu_hevc_hw_codecs_available()

    out_root = os.path.abspath(args.output_dir)
    baseline_dir = os.path.join(out_root, "baseline")
    compressed_dir = os.path.join(out_root, "compressed")
    os.makedirs(out_root, exist_ok=True)
    records = _build_prompt_records(args.num_images, args.seed)
    manifest_path = os.path.join(out_root, "prompts.csv")
    _save_prompt_manifest(manifest_path, records)
    print(f"[manifest] Saved prompt manifest to {manifest_path}", flush=True)

    device = "cuda"
    dtype = torch.float16
    hevc_lossless = not args.hevc_lossy

    mode_line = (
        f"debug_mode={args.debug_mode!r}"
        if args.debug_mode
        else "full UNet RTN+HEVC"
    )
    lossy_line = f", HEVC lossless={hevc_lossless}, qp={args.qp}" if needs_hw_codec else ""
    if needs_hw_codec and args.enable_visual_opts:
        lossy_line += ", visual_optimization=True (encoder default visual tuning)"
    print(f"Device: {device}, dtype: {dtype} ({mode_line}{lossy_line})", flush=True)
    if args.debug_mode in ("reshape_only", "rtn_only"):
        print(
            "[info] Skipping GPU codec probe (no HEVC in reshape_only / rtn_only).",
            flush=True,
        )

    model_ref = args.model_id or MODEL_ID
    local_only = args.local_files_only or (
        os.environ.get("HF_HUB_OFFLINE", "").strip() == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip() == "1"
    )
    if local_only:
        print(
            "[info] local_files_only=True (CLI or HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE): "
            "no Hub metadata fetch; model must already be cached.",
            flush=True,
        )

    print(f"Loading {model_ref} ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=local_only,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    unet = pipe.unet
    unet_bytes = _unet_weight_bytes(unet)
    print(
        f"\n[UNet] Theoretical weight VRAM (all parameters): "
        f"{unet_bytes / (1024**2):.2f} MiB ({unet_bytes} bytes)",
        flush=True,
    )

    print(f"\n[1/2] FP16 baseline UNet — generating {len(records)} images ...", flush=True)
    _generate_batch(
        pipe,
        records,
        baseline_dir,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        label="baseline",
    )

    will_run_ffmpeg_codec = args.debug_mode not in ("reshape_only", "rtn_only")
    if will_run_ffmpeg_codec and not args.no_offload_during_codec:
        print(
            "\n[info] Moving full pipeline to CPU to free VRAM for ffmpeg NVENC/NVDEC "
            "(avoids CUDA_ERROR_OUT_OF_MEMORY in hevc_cuvid when SD already fills the GPU).",
            flush=True,
        )
        pipe.to("cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    ha = " + channel FWHT (pow2 O/I)" if args.use_hadamard else ""
    if args.debug_mode == "reshape_only":
        print(
            f"\n[compress] DEBUG reshape_only: flatten + pad + unpad (no RTN, no HEVC){ha} ...",
            flush=True,
        )
    elif args.debug_mode == "rtn_only":
        print(f"\n[compress] DEBUG rtn_only: 8-bit RTN round-trip (no HEVC){ha} ...", flush=True)
    elif args.debug_mode == "partial_hevc":
        print(
            f"\n[compress] DEBUG partial_hevc: RTN+HEVC on up_blocks weights only "
            f"(hevc_lossless={hevc_lossless}, qp={args.qp}){ha} ...",
            flush=True,
        )
    else:
        print(
            f"\n[compress] RTN + HEVC (hevc_nvenc / hevc_cuvid only) on all UNet Conv2d/Linear weights "
            f"(hevc_lossless={hevc_lossless}, qp={args.qp}){ha} ...",
            flush=True,
        )

    compressed_bytes, compressed_elems, n_layers, n_hevc = replace_unet_weights_with_compressed(
        unet,
        debug_mode=args.debug_mode,
        qp=args.qp,
        frame_size=args.frame_size,
        hevc_lossless=hevc_lossless,
        use_hadamard=args.use_hadamard,
        visual_optimization=args.enable_visual_opts,
    )

    if will_run_ffmpeg_codec and not args.no_offload_during_codec:
        print("\n[info] Moving pipeline back to CUDA for second inference pass ...", flush=True)
        pipe.to(device)

    avg_bpe = (compressed_bytes * 8) / compressed_elems if compressed_elems else 0.0
    print(
        f"\n[stats] Compressed bitstreams total: {compressed_bytes} bytes "
        f"({compressed_bytes / (1024**2):.3f} MiB)",
        flush=True,
    )
    print(f"[stats] Mutated Conv2d/Linear weight tensors: {n_layers}", flush=True)
    print(f"[stats] HEVC round-trip layers: {n_hevc}", flush=True)
    print(f"[stats] Total weight elements in mutated set: {compressed_elems}", flush=True)
    print(f"[stats] Average BPE (bits per element, HEVC path only): {avg_bpe:.4f}", flush=True)

    print(f"\n[2/2] Post-mutation UNet — generating {len(records)} aligned images ...", flush=True)
    _generate_batch(
        pipe,
        records,
        compressed_dir,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        label="compressed",
    )

    print(
        "\nDone. Outputs:\n"
        f"  baseline:   {baseline_dir}\n"
        f"  compressed: {compressed_dir}\n"
        f"  prompts:    {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
