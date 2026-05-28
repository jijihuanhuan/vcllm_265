"""
lm-eval ``HFLM`` subclass that injects a **fresh** ``past_key_values`` cache each forward.

Use for Phase-3 alignment: PIQA / MC tasks call ``_model_call`` with full sequences; causal
models still run ``Cache.update`` per layer, so :class:`~hooks.compressed_kv_cache.RTN3BitCache`
or :class:`~hooks.compressed_kv_cache.VcLLMCompressedCache` affects KV tensors inside that pass.

**Important:** ``vcllm_hevc`` runs RTN+HEVC on KV every layer update — extremely slow on full
evals; prefer ``rtn3`` for quick sweeps or ``--limit`` for smoke tests.
"""
from __future__ import annotations

from typing import Literal

import torch
import transformers
from lm_eval.models.huggingface import HFLM

from hooks.compressed_kv_cache import make_compressed_kv_cache, make_rtn3_kv_cache
from hooks.kv_cache_hook import KVCacheCompressionHook

KVCacheMode = Literal["none", "rtn3", "vcllm_hevc"]


class KVInjectedHFLM(HFLM):
    """Inject ``RTN3BitCache`` or ``VcLLMCompressedCache`` when ``past_key_values`` would be built."""

    def __init__(
        self,
        pretrained,
        *,
        vcllm_kv_cache_mode: KVCacheMode = "none",
        vcllm_kv_hook: KVCacheCompressionHook | None = None,
        rtn_kv_bits: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        self._vcllm_kv_cache_mode: KVCacheMode = vcllm_kv_cache_mode
        self._vcllm_kv_hook = vcllm_kv_hook
        self._rtn_kv_bits = int(rtn_kv_bits)

    def _fresh_past_key_values(self):
        if self._vcllm_kv_cache_mode == "none":
            return None
        cfg = self.model.config
        if self._vcllm_kv_cache_mode == "rtn3":
            return make_rtn3_kv_cache(cfg, num_bits=self._rtn_kv_bits)
        if self._vcllm_kv_cache_mode == "vcllm_hevc":
            if self._vcllm_kv_hook is None:
                raise ValueError("vcllm_kv_cache_mode=vcllm_hevc requires KVCacheCompressionHook (pass vcllm_kv_hook)")
            return make_compressed_kv_cache(cfg, self._vcllm_kv_hook)
        raise ValueError(f"Unknown vcllm_kv_cache_mode: {self._vcllm_kv_cache_mode!r}")

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits

            past = self._fresh_past_key_values()
            if past is not None and self.backend == "causal":
                return self.model(inps, use_cache=True, past_key_values=past).logits
            return self.model(inps).logits

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ) -> torch.Tensor:
        from lm_eval.models.utils_hf import stop_sequences_criteria

        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        if (temp := generation_kwargs.get("temperature")) == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and temp == 0.0:
            generation_kwargs.pop("temperature", None)
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        past = self._fresh_past_key_values()
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            gen_kw = dict(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )
            if past is not None:
                gen_kw["past_key_values"] = past
            return self.model.generate(**gen_kw)
