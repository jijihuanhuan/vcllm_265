from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.multimodal.clip_score import CLIPScore

try:
    import lpips  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: lpips. Install with:\n"
        "  pip install lpips"
    ) from e


@dataclass(frozen=True)
class PairItem:
    name: str
    baseline_path: Path
    compressed_path: Path
    prompt: str


def _list_png_files(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    return {p.name: p for p in sorted(folder.glob("*.png"))}


def _load_prompt_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt manifest not found: {path}")
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "prompt"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            name = row["image_name"].strip()
            prompt = row["prompt"].strip()
            if name:
                mapping[name] = prompt
    return mapping


def _collect_pairs(baseline_dir: Path, compressed_dir: Path, prompt_manifest: Path) -> list[PairItem]:
    baseline = _list_png_files(baseline_dir)
    compressed = _list_png_files(compressed_dir)
    prompts = _load_prompt_manifest(prompt_manifest)

    baseline_names = set(baseline.keys())
    compressed_names = set(compressed.keys())
    if baseline_names != compressed_names:
        only_base = sorted(baseline_names - compressed_names)
        only_comp = sorted(compressed_names - baseline_names)
        raise ValueError(
            "Baseline/compressed filenames are not aligned.\n"
            f"Only in baseline: {only_base[:5]}\n"
            f"Only in compressed: {only_comp[:5]}"
        )
    if not baseline_names:
        raise ValueError("No paired PNG files found.")

    pairs: list[PairItem] = []
    for name in sorted(baseline_names):
        if name not in prompts:
            raise ValueError(f"Missing prompt for {name} in {prompt_manifest}")
        pairs.append(
            PairItem(
                name=name,
                baseline_path=baseline[name],
                compressed_path=compressed[name],
                prompt=prompts[name],
            )
        )
    return pairs


def _load_image_uint8(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _draw_label(img: Image.Image, label: str, x: int, y: int) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle([(x, y), (x + 56, y + 28)], fill=(0, 0, 0))
    draw.text((x + 20, y + 8), label, fill=(255, 255, 255))


def export_blind_human_eval(
    pairs: list[PairItem],
    output_dir: Path,
    *,
    seed: int = 2026,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_pairs_dir = output_dir / "pairs"
    blind_pairs_dir.mkdir(parents=True, exist_ok=True)
    gt_csv = output_dir / "human_eval_ground_truth.csv"

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for idx, pair in enumerate(pairs, start=1):
        with Image.open(pair.baseline_path) as base_img:
            left_candidate = base_img.convert("RGB")
        with Image.open(pair.compressed_path) as comp_img:
            right_candidate = comp_img.convert("RGB")

        if left_candidate.size != right_candidate.size:
            raise ValueError(
                f"Image size mismatch for {pair.name}: "
                f"{left_candidate.size} vs {right_candidate.size}"
            )

        baseline_on_left = bool(rng.getrandbits(1))
        left_img = left_candidate if baseline_on_left else right_candidate
        right_img = right_candidate if baseline_on_left else left_candidate

        w, h = left_img.size
        combined = Image.new("RGB", (w * 2, h), color=(0, 0, 0))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (w, 0))
        _draw_label(combined, "A", 12, 12)
        _draw_label(combined, "B", w + 12, 12)

        pair_name = f"pair_{idx:04d}.png"
        combined.save(blind_pairs_dir / pair_name)
        rows.append(
            {
                "pair_image": pair_name,
                "source_image": pair.name,
                "prompt": pair.prompt,
                "A_source": "baseline" if baseline_on_left else "compressed",
                "B_source": "compressed" if baseline_on_left else "baseline",
            }
        )

    with gt_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_image", "source_image", "prompt", "A_source", "B_source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return gt_csv


@torch.inference_mode()
def compute_metrics(
    pairs: list[PairItem],
    *,
    device: torch.device,
    clip_model: str,
    fid_feature: int,
    skip_fid: bool = False,
    skip_clip: bool = False,
) -> dict[str, float]:
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    fid_metric: Optional[object] = None
    if not skip_fid:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ModuleNotFoundError as e:
            raise SystemExit(
                "FID dependency missing. Install with:\n"
                "  pip install torch-fidelity\n"
                "or run with --skip-fid to compute LPIPS/CLIP only."
            ) from e
        fid_metric = FrechetInceptionDistance(feature=fid_feature, normalize=False).to(device)
    clip_baseline: Optional[CLIPScore] = None
    clip_compressed: Optional[CLIPScore] = None
    if not skip_clip:
        try:
            clip_baseline = CLIPScore(model_name_or_path=clip_model).to(device)
            clip_compressed = CLIPScore(model_name_or_path=clip_model).to(device)
        except Exception as e:
            raise SystemExit(
                "CLIPScore model initialization failed. Common cause: transformers safety policy "
                "requires torch>=2.6 when loading legacy .bin weights.\n"
                "Options:\n"
                "  1) upgrade torch to >=2.6\n"
                "  2) try a safetensors-available CLIP model\n"
                "  3) run with --skip-clip"
            ) from e

    lpips_values: list[float] = []
    identical_pairs = 0

    for pair in pairs:
        baseline_u8 = _load_image_uint8(pair.baseline_path)
        compressed_u8 = _load_image_uint8(pair.compressed_path)
        if baseline_u8.shape != compressed_u8.shape:
            raise ValueError(
                f"Image shape mismatch for {pair.name}: "
                f"{tuple(baseline_u8.shape)} vs {tuple(compressed_u8.shape)}"
            )

        # Enforce exact zero when images are bitwise identical (e.g., qp=0 lossless).
        if torch.equal(baseline_u8, compressed_u8):
            lpips_values.append(0.0)
            identical_pairs += 1
        else:
            b_lp = (baseline_u8.to(device=device, dtype=torch.float32) / 255.0) * 2.0 - 1.0
            c_lp = (compressed_u8.to(device=device, dtype=torch.float32) / 255.0) * 2.0 - 1.0
            lp_val = lpips_model(b_lp.unsqueeze(0), c_lp.unsqueeze(0)).item()
            lpips_values.append(float(lp_val))

        b_dev = baseline_u8.unsqueeze(0).to(device=device)
        c_dev = compressed_u8.unsqueeze(0).to(device=device)
        if fid_metric is not None:
            fid_metric.update(b_dev, real=True)
            fid_metric.update(c_dev, real=False)

        if clip_baseline is not None and clip_compressed is not None:
            prompt_text = [pair.prompt]
            b_clip = baseline_u8.to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0
            c_clip = compressed_u8.to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0
            clip_baseline.update(b_clip, prompt_text)
            clip_compressed.update(c_clip, prompt_text)

    mean_lpips = float(sum(lpips_values) / len(lpips_values))
    fid_score = float(fid_metric.compute().item()) if fid_metric is not None else float("nan")
    if clip_baseline is not None and clip_compressed is not None:
        clip_baseline_mean = float(clip_baseline.compute().item())
        clip_compressed_mean = float(clip_compressed.compute().item())
        clip_drop = clip_baseline_mean - clip_compressed_mean
    else:
        clip_baseline_mean = float("nan")
        clip_compressed_mean = float("nan")
        clip_drop = float("nan")

    return {
        "num_pairs": float(len(pairs)),
        "identical_pairs": float(identical_pairs),
        "mean_lpips": mean_lpips,
        "fid_baseline_vs_compressed": fid_score,
        "clipscore_baseline_mean": clip_baseline_mean,
        "clipscore_compressed_mean": clip_compressed_mean,
        "clipscore_drop": clip_drop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative eval for diffusion weight compression outputs")
    parser.add_argument("--baseline-dir", type=str, default="outputs/baseline")
    parser.add_argument("--compressed-dir", type=str, default="outputs/compressed")
    parser.add_argument("--prompt-manifest", type=str, default="outputs/prompts.csv")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--fid-feature", type=int, default=2048, choices=[64, 192, 768, 2048])
    parser.add_argument("--skip-fid", action="store_true", help="Skip FID computation (no torch-fidelity needed)")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP score computation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--blind-output-dir", type=str, default="outputs/human_eval")
    parser.add_argument("--blind-seed", type=int, default=2026)
    parser.add_argument(
        "--skip-human-eval-export",
        action="store_true",
        help="Skip side-by-side A/B pair image export",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir).resolve()
    compressed_dir = Path(args.compressed_dir).resolve()
    prompt_manifest = Path(args.prompt_manifest).resolve()
    blind_output_dir = Path(args.blind_output_dir).resolve()
    device = torch.device(args.device)

    pairs = _collect_pairs(baseline_dir, compressed_dir, prompt_manifest)
    print(f"Found {len(pairs)} aligned image pairs.")

    metrics = compute_metrics(
        pairs,
        device=device,
        clip_model=args.clip_model,
        fid_feature=args.fid_feature,
        skip_fid=args.skip_fid,
        skip_clip=args.skip_clip,
    )
    print("\n[metrics]")
    for key, value in metrics.items():
        if key in {"num_pairs", "identical_pairs"}:
            print(f"{key}: {int(value)}")
        elif value != value:
            print(f"{key}: skipped")
        else:
            print(f"{key}: {value:.8f}")

    if not args.skip_human_eval_export:
        gt_csv = export_blind_human_eval(
            pairs,
            blind_output_dir,
            seed=args.blind_seed,
        )
        print(f"\n[human-eval] blind pairs exported to: {blind_output_dir / 'pairs'}")
        print(f"[human-eval] ground truth csv: {gt_csv}")


if __name__ == "__main__":
    main()
