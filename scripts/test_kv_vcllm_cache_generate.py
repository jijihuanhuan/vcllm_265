#!/usr/bin/env python3
"""
Phase 3 smoke test: inject ``VcLLMCompressedCache`` into ``model.generate`` (pythia-160m).

Runs greedy generation for ``--max-new-tokens`` (default 256) and compares token ids to a
baseline DynamicCache run. Uses lossless HEVC + RTN by default for reproducibility.

Usage (from repo root)::

    python scripts/test_kv_vcllm_cache_generate.py
    python scripts/test_kv_vcllm_cache_generate.py --max-new-tokens 32 --no-hardware-accel
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.kv_cache_eval import run_generate_kv_eval
from hooks.kv_cache_hook import KVCacheCompressionHook


def main() -> None:
    p = argparse.ArgumentParser(description="VcLLMCompressedCache + pythia-160m generate test")
    p.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--qp", type=int, default=0)
    p.add_argument("--lossy", action="store_true", help="RTN + lossy HEVC (QP from --qp)")
    p.add_argument("--no-hardware-accel", action="store_true")
    p.add_argument("--no-hardware-decode", action="store_true")
    p.add_argument("--frame-size", type=int, default=1024)
    args = p.parse_args()

    load_kw: dict = {}
    if torch.cuda.is_available():
        load_kw["device_map"] = "auto"
        load_kw["dtype"] = torch.float16
    else:
        load_kw["dtype"] = torch.float32

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hook = KVCacheCompressionHook(
        qp=args.qp,
        lossless=not args.lossy,
        frame_size=args.frame_size,
        hardware_accel=not args.no_hardware_accel,
        hardware_decode=not args.no_hardware_decode,
        enabled=True,
    )

    prompt = (
        "The theory of relativity was developed by Albert Einstein. In plain terms, it states that "
        "space and time are linked. "
    )

    print(f"\nPrompt tokens: {tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]}")
    print(f"max_new_tokens: {args.max_new_tokens}\n")

    stats = run_generate_kv_eval(
        model,
        tokenizer,
        prompt,
        args.max_new_tokens,
        hook,
        seed=42,
    )

    print("=== Result ===")
    print(f"cache_class: {stats.get('cache_class')}")
    print(f"exact_match vs baseline: {stats['exact_match']}")
    print(f"token_match_rate: {stats['token_match_rate']:.6f}")
    print(f"dense KV bytes (post-gen, resident): {stats.get('dense_kv_bytes_after_generate')}")
    print(f"packed KV bytes (post-gen): {stats.get('compressed_kv_bytes_after_generate')}")

    if not stats["exact_match"] and not args.lossy:
        print(
            "\nWarning: mismatch under lossless mode may indicate a codec/NVDEC issue; "
            "inspect ffmpeg stderr from hevc_cuvid (decode is NVDEC-only)."
        )
        sys.exit(1)
    if not stats["exact_match"]:
        sys.exit(0)


if __name__ == "__main__":
    main()
