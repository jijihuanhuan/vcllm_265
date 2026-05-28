"""KV cache compression evaluation (Phase 3): compare baseline vs HEVC-backed cache."""
from __future__ import annotations

import gc
import sys
from io import UnsupportedOperation

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hooks.compressed_kv_cache import (
    VcLLMCompressedCache,
    install_kv_compression_hooks,
    make_compressed_kv_cache,
    remove_kv_compression_hooks,
)
from hooks.kv_cache_hook import KVCacheCompressionHook


@torch.inference_mode()
def estimate_cache_payload_bytes(past_key_values) -> tuple[int, int]:
    """
    Return (approx_dense_kv_bytes, compressed_bitstream_bytes) for a DynamicCache
    using CompressedDynamicLayer. Dense counts resident tensors; compressed sums K/V bitstreams.
    """
    from hooks.compressed_kv_cache import ChunkedCompressedDynamicLayer

    dense_b = 0
    packed_b = 0
    if past_key_values is None:
        return 0, 0
    for layer in past_key_values.layers:
        if isinstance(layer, ChunkedCompressedDynamicLayer):
            for pk in layer.chunked_bitstreams_k:
                packed_b += len(pk[0])
            for pv in layer.chunked_bitstreams_v:
                packed_b += len(pv[0])
            if layer.uncompressed_buffer_k is not None and layer.uncompressed_buffer_k.numel():
                dense_b += layer.uncompressed_buffer_k.numel() * layer.uncompressed_buffer_k.element_size()
                dense_b += layer.uncompressed_buffer_v.numel() * layer.uncompressed_buffer_v.element_size()
            elif layer.keys is not None and layer.keys.numel():
                dense_b += layer.keys.numel() * layer.keys.element_size()
                dense_b += layer.values.numel() * layer.values.element_size()
        else:
            if getattr(layer, "keys", None) is not None and layer.keys.numel():
                dense_b += layer.keys.numel() * layer.keys.element_size()
                dense_b += layer.values.numel() * layer.values.element_size()
    return dense_b, packed_b


@torch.inference_mode()
def run_generate_kv_eval(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    kv_hook: KVCacheCompressionHook,
    seed: int = 42,
):
    """
    Greedy generation: baseline (default DynamicCache) vs **VcLLMCompressedCache** injected
    as ``past_key_values`` — no forward hooks required; compression runs inside ``Cache.update``.
    """
    from hooks.compressed_kv_cache import CompressedDynamicLayer

    device = next(model.parameters()).device
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    gen_kw = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}

    torch.manual_seed(seed)
    print(
        "[eval_kv_cache] Step 1/2: baseline greedy generate (standard DynamicCache) ...",
        flush=True,
    )
    out_base = model.generate(input_ids, attention_mask=attn_mask, use_cache=True, **gen_kw)
    ids_base = out_base[0].tolist()
    print(
        f"[eval_kv_cache] Step 1/2 done (output length {len(ids_base)} tokens).",
        flush=True,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.manual_seed(seed)
    n_layers = getattr(model.config, "num_hidden_layers", "?")
    print(
        "[eval_kv_cache] Step 2/2: VcLLMCompressedCache generate — each new token runs "
        f"RTN+HEVC on KV for all {n_layers} layers (very slow; minutes for small max_new_tokens is normal).",
        flush=True,
    )
    past = VcLLMCompressedCache(model.config, kv_hook)
    out_cmp = model.generate(
        input_ids,
        attention_mask=attn_mask,
        past_key_values=past,
        use_cache=True,
        **gen_kw,
    )
    print("[eval_kv_cache] Step 2/2 done.", flush=True)
    # One panel after generate (not per layer / per token): bitwidth & volume vs FP16 KV baseline.
    past.print_compression_summary(title="VcLLM KV — after greedy generate")

    ids_cmp = out_cmp[0].tolist()

    match = sum(1 for a, b in zip(ids_base, ids_cmp) if a == b)
    n = max(len(ids_base), len(ids_cmp))
    match_rate = match / n if n else 0.0

    dense_ref, packed_after = _footprint_from_cache(past)
    cs = past.compression_stats

    return {
        "prompt_tokens": input_ids.shape[1],
        "max_new_tokens": max_new_tokens,
        "baseline_ids": ids_base,
        "compressed_cache_ids": ids_cmp,
        "token_match_rate": match_rate,
        "exact_match": ids_base == ids_cmp,
        "compressed_layer_type": CompressedDynamicLayer.__name__,
        "cache_class": VcLLMCompressedCache.__name__,
        "dense_kv_bytes_after_generate": dense_ref,
        "compressed_kv_bytes_after_generate": packed_after,
        "kv_compress_calls": cs.compress_calls,
        "kv_total_elements": cs.total_elements,
        "kv_fp16_baseline_bytes": cs.total_fp16_bytes,
        "kv_total_compressed_bytes": cs.total_compressed_bytes,
        "kv_global_avg_bpe": cs.global_avg_bpe(),
    }


def _footprint_from_cache(past_key_values) -> tuple[int, int]:
    """Dense vs packed byte counts for a cache after generation (if layers still hold packed state)."""
    dense_b, packed_b = estimate_cache_payload_bytes(past_key_values)
    return dense_b, packed_b


@torch.inference_mode()
def run_generate_kv_eval_with_hooks(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    kv_hook: KVCacheCompressionHook,
    seed: int = 42,
):
    """
    Alternate path: forward pre/post hooks inject ``make_compressed_kv_cache`` when past is None.
    Useful when you cannot pass ``past_key_values`` explicitly.
    """
    from hooks.compressed_kv_cache import CompressedDynamicLayer

    device = next(model.parameters()).device
    torch.manual_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    gen_kw = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}

    torch.manual_seed(seed)
    out_base = model.generate(input_ids, attention_mask=attn_mask, use_cache=True, **gen_kw)
    ids_base = out_base[0].tolist()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    handles = install_kv_compression_hooks(model, kv_hook)
    torch.manual_seed(seed)
    try:
        out_cmp = model.generate(input_ids, attention_mask=attn_mask, use_cache=True, **gen_kw)
    finally:
        remove_kv_compression_hooks(handles)

    ids_cmp = out_cmp[0].tolist()
    match = sum(1 for a, b in zip(ids_base, ids_cmp) if a == b)
    n = max(len(ids_base), len(ids_cmp))

    return {
        "prompt_tokens": input_ids.shape[1],
        "max_new_tokens": max_new_tokens,
        "baseline_ids": ids_base,
        "compressed_cache_ids": ids_cmp,
        "token_match_rate": match / n if n else 0.0,
        "exact_match": ids_base == ids_cmp,
        "compressed_layer_type": CompressedDynamicLayer.__name__,
        "cache_class": "hook_injected",
    }


def evaluate_kv_cache_cli(
    model_name: str,
    *,
    prompt: str | None,
    max_new_tokens: int,
    qp: int,
    lossless: bool,
    frame_size: int,
    no_hardware_accel: bool,
    no_hardware_decode: bool,
    torch_dtype: str,
):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dt = dtype_map.get(torch_dtype, torch.float16)
    load_kw: dict = {}
    if torch.cuda.is_available():
        load_kw["device_map"] = "auto"
        load_kw["dtype"] = dt
    else:
        load_kw["dtype"] = torch.float32

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError, UnsupportedOperation):
            pass

    print(f"Loading {model_name} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hook = KVCacheCompressionHook(
        qp=qp,
        lossless=lossless,
        frame_size=frame_size,
        hardware_accel=not no_hardware_accel,
        hardware_decode=not no_hardware_decode,
        enabled=True,
    )

    text = prompt or (
        "The theory of relativity was developed by Albert Einstein. In plain terms, it states that "
    )

    print("\n=== KV cache compression (greedy generate, VcLLMCompressedCache injection) ===", flush=True)
    print(f"prompt (chars): {len(text)}, max_new_tokens: {max_new_tokens}", flush=True)
    stats = run_generate_kv_eval(model, tokenizer, text, max_new_tokens, hook)

    print(f"token_match_rate vs baseline cache: {stats['token_match_rate']:.4f}", flush=True)
    print(f"exact_match: {stats['exact_match']}", flush=True)
    if stats.get("compressed_kv_bytes_after_generate") is not None:
        print(
            f"after generate — dense KV bytes (resident): {stats['dense_kv_bytes_after_generate']}, "
            f"packed bitstreams: {stats['compressed_kv_bytes_after_generate']}",
            flush=True,
        )
    if not stats["exact_match"]:
        print(
            "Note: lossy codec / RTN drift may cause token divergence; prefer lossless for exact match.",
            flush=True,
        )

    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.inference_mode():
        out_dense = model(input_ids, use_cache=True)
        dense_ref, _ = estimate_cache_payload_bytes(out_dense.past_key_values)

    past = make_compressed_kv_cache(model.config, hook)
    with torch.inference_mode():
        out = model(input_ids, past_key_values=past, use_cache=True)
        pkv = out.past_key_values
        _, packed_b = estimate_cache_payload_bytes(pkv)
    past.print_compression_summary(title="VcLLM KV — after one forward (prefill)")

    print("\n=== KV footprint (same prompt, one full forward, use_cache=True) ===", flush=True)
    print(f"dense KV bytes (reference): {dense_ref}", flush=True)
    print(f"compressed KV bytes (K+V bitstreams): {packed_b}", flush=True)
    if packed_b > 0 and dense_ref > 0:
        print(f"nominal compression ratio (dense/compressed): {dense_ref / packed_b:.2f}x", flush=True)

    return stats
