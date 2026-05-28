#!/usr/bin/env python3
"""
WikiText-2 (wikitext-2-raw-v1 / test) perplexity with **strict KV-cache autoregression**.

Does **not** use a single forward over the full window with ``labels=``. Instead, for each
sliding window we reset the cache and run **one token at a time**: ``forward`` with
``past_key_values``, take the last-step logits, apply CrossEntropy to the **next** token,
then feed the returned ``past_key_values`` into the next step — matching autoregressive
decoding and Phase-3 compressed-cache behaviour.

Modes (``--kv-cache-mode``):
  - ``none``: standard ``DynamicCache`` (FP16/BF16 KV tensors).
  - ``rtn3``: :func:`hooks.compressed_kv_cache.make_rtn3_kv_cache`.
  - ``vcllm_hevc``: :func:`hooks.compressed_kv_cache.make_compressed_kv_cache` + NVENC path.

Stride handling follows the usual WikiText sliding-window recipe (only the **last**
``stride`` next-token losses per window contribute when ``stride < window_len``), aligned
with masked-label bulk eval.

Examples::

    python evaluation/eval_wikitext_ppl.py --model EleutherAI/pythia-160m --kv-cache-mode none --limit-tokens 2048
    python evaluation/eval_wikitext_ppl.py --model EleutherAI/pythia-160m --kv-cache-mode rtn3 --limit-tokens 4096
    python evaluation/eval_wikitext_ppl.py --model EleutherAI/pythia-160m --kv-cache-mode vcllm_hevc --frame-size 1024 \\
        --output results/wikitext_kv_vcllm_hevc_full.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hooks.compressed_kv_cache import make_compressed_kv_cache, make_rtn3_kv_cache  # noqa: E402
from hooks.kv_cache_hook import KVCacheCompressionHook  # noqa: E402


def load_wikitext2_test_ids(tokenizer):
    """
    Load wikitext-2-raw-v1 test, join text, tokenize once, return ``input_ids`` [1, T] on CPU.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    parts = [str(t).strip() for t in ds["text"] if t is not None and str(t).strip()]
    full = "\n\n".join(parts)
    enc = tokenizer(full, return_tensors="pt", add_special_tokens=False, truncation=False)
    input_ids = enc["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


def _window_loss_span(window_len: int, stride: int) -> tuple[int, int]:
    """
    First / last **step index** ``t`` (input is token[t:t+1]) included in the loss,
    predicting token[t+1]. Matches ``labels[:, :-stride] = -100`` bulk masking when stride < L.
    """
    if window_len < 2:
        return 0, -1
    if stride >= window_len - 1:
        t_first, t_last = 0, window_len - 2
    else:
        t_first = window_len - 1 - stride
        t_last = window_len - 2
    return t_first, t_last


def make_past_for_mode(
    mode: str,
    config,
    hook: KVCacheCompressionHook | None,
    *,
    rtn_kv_bits: int = 3,
):
    if mode == "none":
        return DynamicCache(config=config)
    if mode == "rtn3":
        return make_rtn3_kv_cache(config, num_bits=rtn_kv_bits)
    if mode == "vcllm_hevc":
        if hook is None:
            raise ValueError("vcllm_hevc requires KVCacheCompressionHook")
        return make_compressed_kv_cache(config, hook)
    raise ValueError(f"Unknown --kv-cache-mode: {mode}")


@torch.inference_mode()
def perplexity_wikitext_kv_autoreg(
    model,
    input_ids: torch.Tensor,
    *,
    max_length: int,
    stride: int,
    kv_cache_mode: str,
    kv_hook: KVCacheCompressionHook | None,
    rtn_kv_bits: int,
    limit_tokens: int | None,
    log_every: int,
    device: torch.device,
):
    """
    Sum NLL over contributing next-token predictions; PPL = exp(mean NLL).
    """
    model.eval()
    ids = input_ids.to(device)
    seq_len = ids.shape[1]

    total_nll = 0.0
    n_predictions = 0
    windows_done = 0

    start = 0
    while start < seq_len - 1:
        end = min(start + max_length, seq_len)
        win = ids[:, start:end]
        L = win.shape[1]
        if L < 2:
            break

        t_first, t_last = _window_loss_span(L, stride)
        if t_first > t_last:
            start += stride
            continue

        past = make_past_for_mode(
            kv_cache_mode,
            model.config,
            kv_hook,
            rtn_kv_bits=rtn_kv_bits,
        )

        for t in range(t_first, t_last + 1):
            if limit_tokens is not None and n_predictions >= limit_tokens:
                break

            inp = win[:, t : t + 1]
            target = win[:, t + 1]

            out = model(inp, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

            loss = F.cross_entropy(logits.float(), target.long(), reduction="mean")
            total_nll += loss.item()
            n_predictions += 1

            if log_every > 0 and n_predictions % log_every == 0:
                avg = total_nll / n_predictions
                print(
                    f"[wikitext-ppl] predictions={n_predictions}  mean_loss={avg:.4f}  ppl={math.exp(avg):.2f}",
                    flush=True,
                )

        windows_done += 1
        if limit_tokens is not None and n_predictions >= limit_tokens:
            break

        start += stride
        if end >= seq_len:
            break

    if n_predictions == 0:
        return float("inf"), 0, windows_done

    mean_nll = total_nll / n_predictions
    return math.exp(mean_nll), n_predictions, windows_done


def main() -> None:
    p = argparse.ArgumentParser(description="WikiText-2 PPL with KV-cache token-by-token forward")
    p.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    p.add_argument(
        "--kv-cache-mode",
        type=str,
        choices=["none", "rtn3", "vcllm_hevc"],
        default="none",
        help="KV backend: FP16 DynamicCache | RTN 3-bit KV | VcLLM RTN+HEVC",
    )
    p.add_argument("--max-length", type=int, default=2048, help="Sliding window size (tokens)")
    p.add_argument("--stride", type=int, default=512, help="Sliding window stride (tokens)")
    p.add_argument(
        "--limit-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N contributing next-token losses (smoke test)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=256,
        help="Print running mean loss / PPL every N predictions (0 = only final)",
    )
    p.add_argument("--qp", type=int, default=0)
    p.add_argument("--kv-lossy", action="store_true")
    p.add_argument("--frame-size", type=int, default=1024)
    p.add_argument("--no-hardware-accel", action="store_true")
    p.add_argument("--no-hardware-decode", action="store_true")
    p.add_argument("--rtn-kv-bits", type=int, default=3)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional JSON path for metrics (perplexity, counts, hyperparameters)",
    )
    args = p.parse_args()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_kw: dict = {}
    if torch.cuda.is_available():
        load_kw["device_map"] = "auto"
        load_kw["dtype"] = dtype
    else:
        load_kw["dtype"] = dtype

    print(f"Loading model {args.model!r} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg_max = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", args.max_length
    )
    max_length = min(args.max_length, int(cfg_max))

    kv_hook = None
    if args.kv_cache_mode == "vcllm_hevc":
        kv_hook = KVCacheCompressionHook(
            qp=args.qp,
            lossless=not args.kv_lossy,
            frame_size=args.frame_size,
            hardware_accel=not args.no_hardware_accel,
            hardware_decode=not args.no_hardware_decode,
            enabled=True,
        )

    print("Loading WikiText-2 test (wikitext-2-raw-v1) ...", flush=True)
    input_ids = load_wikitext2_test_ids(tokenizer)
    print(f"Tokenized length: {input_ids.shape[1]} tokens", flush=True)

    device = next(model.parameters()).device
    print(
        f"KV mode={args.kv_cache_mode!r}  max_length={max_length}  stride={args.stride}  "
        f"limit_tokens={args.limit_tokens}",
        flush=True,
    )
    print("Computing perplexity (KV autoregressive, token-by-token) ...", flush=True)

    ppl, n_pred, n_win = perplexity_wikitext_kv_autoreg(
        model,
        input_ids,
        max_length=max_length,
        stride=args.stride,
        kv_cache_mode=args.kv_cache_mode,
        kv_hook=kv_hook,
        rtn_kv_bits=args.rtn_kv_bits,
        limit_tokens=args.limit_tokens,
        log_every=args.log_every,
        device=device,
    )

    if math.isinf(ppl):
        print("No predictions accumulated.", flush=True)
        sys.exit(1)

    mean_loss = math.log(ppl)
    print("", flush=True)
    print("=== WikiText-2 test (KV-cache autoreg) ===", flush=True)
    print(f"  contributing predictions : {n_pred}", flush=True)
    print(f"  sliding windows used      : {n_win}", flush=True)
    print(f"  cross-entropy (mean nll) : {mean_loss:.4f}", flush=True)
    print(f"  perplexity               : {ppl:.4f}", flush=True)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "kv_cache_mode": args.kv_cache_mode,
            "perplexity": ppl,
            "mean_nll": mean_loss,
            "contributing_predictions": n_pred,
            "sliding_windows": n_win,
            "max_length": max_length,
            "stride": args.stride,
            "limit_tokens": args.limit_tokens,
            "qp": args.qp,
            "kv_lossless": not args.kv_lossy,
            "frame_size": args.frame_size,
            "rtn_kv_bits": args.rtn_kv_bits,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics JSON: {out_path}", flush=True)


if __name__ == "__main__":
    main()
