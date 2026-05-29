"""
Phase 1 — capture baseline UNet feature maps across denoise steps and map to VcLLM frames.

Run from repository root::

  python evaluation/capture_diffusion_feature_maps.py \\
    --prompt "A cute golden retriever in the snow" \\
    --steps 30 \\
    --hook-site mid_block \\
    --map-to-frames \\
    --out-dir outputs/diffcodec_phase1

This script does **not** alter inference (no bypass). It only records ``F_t`` and
optional ``tensor_to_frames`` tiles for temporal redundancy analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from hooks.diffcodec import (
    DiffusionFeatureCapture,
    HookSite,
    attach_feature_hooks,
)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
SEED = 42

HOOK_SITE_CHOICES = {
    "mid_block": HookSite.MID_BLOCK,
    "up_blocks.0": HookSite.UP_BLOCK_0,
    "up_blocks.1": HookSite.UP_BLOCK_1,
    "down_blocks.2": HookSite.DOWN_BLOCK_2,
}


def _parse_sites(values: list[str]) -> tuple[HookSite, ...]:
    if not values:
        return (HookSite.MID_BLOCK,)
    out: list[HookSite] = []
    for v in values:
        if v not in HOOK_SITE_CHOICES:
            raise SystemExit(
                f"Unknown --hook-site {v!r}. Choices: {sorted(HOOK_SITE_CHOICES)}"
            )
        out.append(HOOK_SITE_CHOICES[v])
    return tuple(out)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiffCodec Phase 1: capture UNet feature maps + Tensor2Video mapping"
    )
    parser.add_argument("--prompt", type=str, default="A cute golden retriever in the snow")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--hook-site",
        action="append",
        dest="hook_sites",
        metavar="SITE",
        help=f"Repeatable. Sites: {', '.join(sorted(HOOK_SITE_CHOICES))}",
    )
    parser.add_argument("--map-to-frames", action="store_true")
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--channel-group-size", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default=os.path.join(_REPO_ROOT, "outputs", "diffcodec_phase1"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-id", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for SD inference in this eval script.")

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as e:
        raise SystemExit("pip install diffusers accelerate") from e

    sites = _parse_sites(args.hook_sites or [])
    os.makedirs(args.out_dir, exist_ok=True)

    model_ref = args.model_id or MODEL_ID
    local_only = args.local_files_only or (
        os.environ.get("HF_HUB_OFFLINE", "").strip() == "1"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        model_ref,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=local_only,
    ).to("cuda")

    capture = DiffusionFeatureCapture(
        sites=sites,
        map_to_frames=args.map_to_frames,
        frame_size=args.frame_size,
        channel_group_size=args.channel_group_size,
        store_device="cpu",
    )
    hook_bag = attach_feature_hooks(pipe.unet, capture)

    gen = torch.Generator(device=pipe.device)
    gen.manual_seed(args.seed)

    print(f"[capture] hook sites: {[s.value for s in sites]}", flush=True)
    pipe(
        args.prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        generator=gen,
        callback_on_step_end=capture.on_step_end,
    )
    hook_bag.remove()

    # Persist captures
    manifest: list[dict] = []
    for rec in capture.records:
        entry = {
            "timestep": rec.timestep,
            "step_index": rec.step_index,
            "site": rec.site,
            "shape": list(rec.feature.shape),
            "dtype": str(rec.feature.dtype),
        }
        base = f"step_{rec.step_index:03d}_{rec.site.replace('.', '_')}"
        feat_path = os.path.join(args.out_dir, f"{base}.pt")
        torch.save(rec.feature, feat_path)
        entry["feature_path"] = feat_path

        if rec.frames_meta is not None and rec.frames is not None:
            frames_path = os.path.join(args.out_dir, f"{base}_frames.npy")
            meta_path = os.path.join(args.out_dir, f"{base}_frames_meta.json")
            import numpy as np

            np.save(frames_path, rec.frames)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(rec.frames_meta.to_dict(), f, indent=2)
            entry["frames_path"] = frames_path
            entry["frames_meta_path"] = meta_path
            entry["num_frames"] = int(rec.frames.shape[0])

        manifest.append(entry)

    manifest_path = os.path.join(args.out_dir, "capture_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": args.prompt,
                "steps": args.steps,
                "hook_sites": [s.value for s in sites],
                "map_to_frames": args.map_to_frames,
                "records": manifest,
            },
            f,
            indent=2,
        )

    print(f"[done] {len(capture.records)} captures -> {args.out_dir}", flush=True)
    print(f"[done] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
