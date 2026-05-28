#!/usr/bin/env python3
"""
LM Evaluation Harness integration for VcLLM compressed weights.

Loads a HuggingFace CausalLM in fp16 (baseline) or loads + ``decompress_model_weights``
(compressed checkpoint produced by ``run.py --mode compress_weights``), then runs
zero-shot tasks and writes JSON.

Example (from repo root)::

    # Baseline fp16
    python evaluation/lm_eval_integration.py \\
        --model EleutherAI/pythia-160m \\
        --output results/baseline.json

    # Decompressed VcLLM weights
    python evaluation/lm_eval_integration.py \\
        --model EleutherAI/pythia-160m \\
        --compressed-dir ./compressed_weights \\
        --output results/vcllm_decompressed.json

    # PIQA (or any task) with Phase-3 **3-bit RTN KV** baseline (no video codec)
    python evaluation/lm_eval_integration.py \\
        --model EleutherAI/pythia-160m \\
        --tasks piqa \\
        --kv-cache-mode rtn3 \\
        --output results/piqa_rtn3kv.json

    # PIQA with **VcLLM HEVC KV** (very slow; use --limit for smoke tests)
    python evaluation/lm_eval_integration.py \\
        --model EleutherAI/pythia-160m \\
        --tasks piqa \\
        --kv-cache-mode vcllm_hevc \\
        --kv-frame-size 1024 \\
        --output results/piqa_vcllm_kv.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo root on sys.path when running as ``python evaluation/lm_eval_integration.py``
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compression.weight_pipeline import decompress_model_weights  # noqa: E402
from evaluation.kv_injected_hflm import KVInjectedHFLM  # noqa: E402
from hooks.kv_cache_hook import KVCacheCompressionHook  # noqa: E402

from lm_eval.evaluator import simple_evaluate  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402
from lm_eval.utils import handle_non_serializable, make_table  # noqa: E402


DEFAULT_TASKS = [
    "piqa",
    "copa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "hellaswag",
    "rte",
    "openbookqa",
]


def _load_causal_lm(model_name: str, device_map_choice: str) -> tuple[torch.nn.Module, Any]:
    if torch.cuda.is_available() and device_map_choice == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
        )
    else:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=None,
        ).to(dev)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _metric_float(val: Any) -> float | None:
    """lm-eval may use the string ``'N/A'`` for stderr when ``bootstrap_iters=0``."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().upper() in ("N/A", "NA", ""):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_task_accuracy(results: dict[str, Any], task: str) -> dict[str, float | None]:
    """
    Pull primary accuracy fields from ``simple_evaluate`` output for paper-style tables.

    lm-eval stores metrics as ``acc,none`` / ``acc_norm,none`` with optional
    ``*_stderr,none`` when bootstrap_iters > 0.
    """
    row: dict[str, float | None] = {"acc": None, "acc_norm": None, "acc_stderr": None, "acc_norm_stderr": None}
    r = results.get("results", {}).get(task)
    if not r:
        return row
    if "acc,none" in r:
        row["acc"] = _metric_float(r["acc,none"])
    if "acc_stderr,none" in r:
        row["acc_stderr"] = _metric_float(r["acc_stderr,none"])
    if "acc_norm,none" in r:
        row["acc_norm"] = _metric_float(r["acc_norm,none"])
    if "acc_norm_stderr,none" in r:
        row["acc_norm_stderr"] = _metric_float(r["acc_norm_stderr,none"])
    return row


def _fmt4(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def print_paper_style_table(results: dict[str, Any], tasks: list[str], *, label: str) -> None:
    """Print a compact TSV-friendly block to stdout (paste into LaTeX / sheets)."""
    print(f"\n=== Paper-style summary ({label}) ===")
    print("task\tacc\tacc_stderr\tacc_norm\tacc_norm_stderr")
    for t in tasks:
        m = extract_task_accuracy(results, t)
        print(
            f"{t}\t{_fmt4(m['acc'])}\t{_fmt4(m['acc_stderr'])}\t{_fmt4(m['acc_norm'])}\t{_fmt4(m['acc_norm_stderr'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LM Eval Harness for VcLLM decompressed weights")
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path (architecture must match compression)")
    parser.add_argument(
        "--compressed-dir",
        type=str,
        default=None,
        help="Directory with compression_summary.json + layers (omit for fp16 baseline)",
    )
    parser.add_argument("--output", type=str, required=True, help="Write full results JSON here")
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated lm-eval task names",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=float, default=None, help="Per-task example limit (int or fraction); omit for full eval")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Global few-shot k (0 = zero-shot for supported tasks)")
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="Bootstrap iters for stderr (0 = fast, no stderr; use e.g. 100000 for paper-style CI)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="auto: HF device_map on GPU; none: single-GPU .to(cuda:0) or CPU",
    )
    parser.add_argument(
        "--no-hardware-decode",
        action="store_true",
        help="VcLLM: skip NVDEC on weight decompress (raises: decode is hevc_cuvid-only)",
    )
    parser.add_argument("--no-hardware-accel", action="store_true", help="VcLLM: passed to decompress (encode-side compat)")
    parser.add_argument("--log-samples", action="store_true", help="Include per-sample logs (very large JSON)")
    parser.add_argument(
        "--confirm-run-unsafe-code",
        action="store_true",
        help="Forward to lm-eval for tasks that need explicit confirmation",
    )
    parser.add_argument(
        "--kv-cache-mode",
        type=str,
        default="none",
        choices=["none", "rtn3", "vcllm_hevc"],
        help="Phase-3: inject fresh past_key_values per lm-eval forward (PIQA loglikelihood path). "
        "rtn3=3-bit min-max RTN KV round-trip; vcllm_hevc=RTN+HEVC (slow).",
    )
    parser.add_argument(
        "--rtn-kv-bits",
        type=int,
        default=3,
        help="Bit width for --kv-cache-mode rtn3 (default 3).",
    )
    parser.add_argument("--kv-qp", type=int, default=0, help="HEVC QP when using vcllm_hevc KV mode")
    parser.add_argument(
        "--kv-lossy",
        action="store_true",
        help="Use lossy HEVC for vcllm_hevc KV mode (default lossless)",
    )
    parser.add_argument(
        "--kv-frame-size",
        type=int,
        default=1024,
        help="tensor_to_frames frame_size for vcllm_hevc KV path",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    device_map_choice = args.device_map if torch.cuda.is_available() else "none"

    print(f"[lm_eval] Loading model {args.model!r} (fp16)...")
    model, tokenizer = _load_causal_lm(args.model, device_map_choice)

    run_label = "baseline_fp16"
    if args.compressed_dir:
        run_label = "vcllm_decompressed"
        print(f"[lm_eval] Decompressing VcLLM weights from {args.compressed_dir!r} ...")
        decompress_model_weights(
            model,
            args.compressed_dir,
            hardware_accel=not args.no_hardware_accel,
            hardware_decode=not args.no_hardware_decode,
        )

    if args.kv_cache_mode != "none":
        run_label = f"{run_label}_kv_{args.kv_cache_mode}"
        print(
            f"[lm_eval] KV cache injection: {args.kv_cache_mode!r} "
            f"(fresh cache per forward; vcllm_hevc is very slow on full tasks)",
            flush=True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_common = dict(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device,
        dtype="float16",
    )
    if args.kv_cache_mode == "none":
        lm = HFLM(**lm_common)
    else:
        kv_hook = None
        if args.kv_cache_mode == "vcllm_hevc":
            kv_hook = KVCacheCompressionHook(
                qp=args.kv_qp,
                lossless=not args.kv_lossy,
                frame_size=args.kv_frame_size,
                hardware_accel=not args.no_hardware_accel,
                hardware_decode=not args.no_hardware_decode,
                enabled=True,
            )
        lm = KVInjectedHFLM(
            **lm_common,
            vcllm_kv_cache_mode=args.kv_cache_mode,
            vcllm_kv_hook=kv_hook,
            rtn_kv_bits=args.rtn_kv_bits,
        )

    print(f"[lm_eval] Tasks ({len(tasks)}): {tasks}")
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=device,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=args.log_samples,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
    )
    if results is None:
        raise RuntimeError("simple_evaluate returned None (unexpected in single-process run)")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=handle_non_serializable)

    print(make_table(results))
    print_paper_style_table(results, tasks, label=run_label)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
