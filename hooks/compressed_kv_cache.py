"""
Phase 3: KV cache compression compatible with transformers DynamicCache / DynamicLayer.

**ChunkedVcLLMCache** stores KV incrementally: tail stays dense (FP16/BF16); every ``chunk_size``
tokens along sequence are RTN+HEVC-packed into bitstreams and appended to per-layer lists.
On each ``update``, materialized full K/V for attention are built by decompressing chunks +
concatenating the tail buffer **only as temporaries** — dense history is not retained.

**RTN3BitCache** (Phase-3 **codec-free** baseline): 3-bit asymmetric min–max RTN round-trip
on KV tensors after each ``DynamicLayer.update`` — no video encoder.

Hooks inject ``ChunkedVcLLMCache`` when ``past_key_values`` is None; chunked layers handle
compression inside ``update``, so post-hooks do not call ``compress_storage``.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.configuration_utils import PreTrainedConfig

from compression.rtn import rtn_minmax_roundtrip_tensor
from hooks.kv_cache_hook import KVCacheCompressionHook


class VcLLMCompressionStats:
    """
    Cumulative accounting for KV tensors passed through ``KVCacheCompressionHook._compress_tensor``.

    FP16 baseline is ``numel * 2`` bytes per compressed tensor. Global BPE is
    ``total_compressed_bytes * 8 / total_elements`` (bits per element).
    """

    __slots__ = ("total_elements", "total_compressed_bytes", "total_fp16_bytes", "compress_calls")

    def __init__(self) -> None:
        self.total_elements = 0
        self.total_compressed_bytes = 0
        self.total_fp16_bytes = 0
        self.compress_calls = 0

    def record(self, *, numel: int, bitstream_bytes: int) -> None:
        n = int(numel)
        b = int(bitstream_bytes)
        self.total_elements += n
        self.total_compressed_bytes += b
        self.total_fp16_bytes += n * 2
        self.compress_calls += 1

    def global_avg_bpe(self) -> float:
        if self.total_elements <= 0:
            return 0.0
        return (self.total_compressed_bytes * 8) / self.total_elements

    def reset(self) -> None:
        self.total_elements = 0
        self.total_compressed_bytes = 0
        self.total_fp16_bytes = 0
        self.compress_calls = 0


def get_inner_transformer(model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the decoder trunk (varies by architecture: Llama ``model``, GPT-NeoX ``gpt_neox``, GPT-2 ``transformer``)."""
    for attr in ("model", "gpt_neox", "transformer"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if isinstance(inner, torch.nn.Module):
                return inner
    raise ValueError(
        f"Unsupported architecture for KV hooks: {type(model).__name__}. "
        "Extend get_inner_transformer() with the correct submodule name."
    )


class ChunkedCompressedDynamicLayer(DynamicLayer):
    """
    Dual storage per layer:

    - ``chunked_bitstreams_k`` / ``chunked_bitstreams_v``: finalized HEVC bitstreams
      (each covers exactly ``chunk_size`` tokens on the sequence axis).
    - ``uncompressed_buffer_k`` / ``uncompressed_buffer_v``: dense tail (< chunk_size
      tokens) not yet frozen.

    ``update`` appends new K/V to the buffer, flushes full chunks to bitstreams, then returns
    **ephemeral** full tensors (chunk decompressions concatenated with buffer) for attention.
    """

    def __init__(self, hook: KVCacheCompressionHook, config=None, *, chunk_size: int = 64):
        super().__init__(config)
        self._hook = hook
        self._chunk_size = max(1, int(chunk_size))
        self.chunked_bitstreams_k: list[tuple[bytes, dict]] = []
        self.chunked_bitstreams_v: list[tuple[bytes, dict]] = []
        self.uncompressed_buffer_k: torch.Tensor | None = None
        self.uncompressed_buffer_v: torch.Tensor | None = None

    def _flush_buffer_chunks(self) -> None:
        """Move leading ``chunk_size`` tokens from buffers into compressed lists."""
        while (
            self.uncompressed_buffer_k is not None
            and self.uncompressed_buffer_k.shape[-2] >= self._chunk_size
        ):
            piece_k = self.uncompressed_buffer_k[:, :, : self._chunk_size, :]
            piece_v = self.uncompressed_buffer_v[:, :, : self._chunk_size, :]
            if self._hook.enabled:
                self.chunked_bitstreams_k.append(self._hook._compress_tensor(piece_k))
                self.chunked_bitstreams_v.append(self._hook._compress_tensor(piece_v))
            rest = self.uncompressed_buffer_k.shape[-2] - self._chunk_size
            if rest <= 0:
                self.uncompressed_buffer_k = None
                self.uncompressed_buffer_v = None
            else:
                self.uncompressed_buffer_k = self.uncompressed_buffer_k[:, :, self._chunk_size :, :]
                self.uncompressed_buffer_v = self.uncompressed_buffer_v[:, :, self._chunk_size :, :]

    def _materialize_kv_ephemeral(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress all chunks and concatenate with tail buffer (caller should not retain long-term)."""
        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        for pk in self.chunked_bitstreams_k:
            parts_k.append(self._hook._decompress_tensor(pk))
        for pv in self.chunked_bitstreams_v:
            parts_v.append(self._hook._decompress_tensor(pv))
        if self.uncompressed_buffer_k is not None:
            parts_k.append(self.uncompressed_buffer_k)
            parts_v.append(self.uncompressed_buffer_v)
        assert len(parts_k) == len(parts_v)
        if not parts_k:
            raise RuntimeError("ChunkedCompressedDynamicLayer: empty cache in materialize")
        keys = torch.cat(parts_k, dim=-2)
        values = torch.cat(parts_v, dim=-2)
        return keys, values

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        buf_len = (
            self.uncompressed_buffer_k.shape[-2] if self.uncompressed_buffer_k is not None else 0
        )
        return len(self.chunked_bitstreams_k) * self._chunk_size + buf_len

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.chunked_bitstreams_k = []
        self.chunked_bitstreams_v = []
        self.uncompressed_buffer_k = None
        self.uncompressed_buffer_v = None
        self.keys = None
        self.values = None
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._hook.enabled:
            return super().update(key_states, value_states, *args, **kwargs)

        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self.uncompressed_buffer_k is None:
            self.uncompressed_buffer_k = key_states
            self.uncompressed_buffer_v = value_states
        else:
            self.uncompressed_buffer_k = torch.cat([self.uncompressed_buffer_k, key_states], dim=-2)
            self.uncompressed_buffer_v = torch.cat([self.uncompressed_buffer_v, value_states], dim=-2)

        self._flush_buffer_chunks()

        keys, values = self._materialize_kv_ephemeral()
        # Do not retain full dense KV on the layer — only chunks + tail buffer.
        self.keys = None
        self.values = None
        return keys, values

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        if not self._hook.enabled:
            super().reorder_cache(beam_idx)
            return
        k, v = self._materialize_kv_ephemeral()
        self.chunked_bitstreams_k.clear()
        self.chunked_bitstreams_v.clear()
        self.uncompressed_buffer_k = k.index_select(0, beam_idx.to(k.device))
        self.uncompressed_buffer_v = v.index_select(0, beam_idx.to(v.device))
        self._flush_buffer_chunks()

    def crop(self, max_length: int) -> None:
        if self.get_seq_length() == 0:
            return
        if not self._hook.enabled:
            super().crop(max_length)
            return
        total = self.get_seq_length()
        if max_length < 0:
            max_length = total - abs(max_length)
        max_length = max(0, min(int(max_length), total))
        if total <= max_length:
            return
        k, v = self._materialize_kv_ephemeral()
        k = k[:, :, :max_length, :]
        v = v[:, :, :max_length, :]
        self.chunked_bitstreams_k.clear()
        self.chunked_bitstreams_v.clear()
        self.uncompressed_buffer_k = k
        self.uncompressed_buffer_v = v
        self._flush_buffer_chunks()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        if not self._hook.enabled:
            super().batch_repeat_interleave(repeats)
            return
        k, v = self._materialize_kv_ephemeral()
        self.chunked_bitstreams_k.clear()
        self.chunked_bitstreams_v.clear()
        self.uncompressed_buffer_k = k.repeat_interleave(repeats, dim=0)
        self.uncompressed_buffer_v = v.repeat_interleave(repeats, dim=0)
        self._flush_buffer_chunks()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        if not self._hook.enabled:
            super().batch_select_indices(indices)
            return
        k, v = self._materialize_kv_ephemeral()
        self.chunked_bitstreams_k.clear()
        self.chunked_bitstreams_v.clear()
        self.uncompressed_buffer_k = k[indices.to(k.device), ...]
        self.uncompressed_buffer_v = v[indices.to(v.device), ...]
        self._flush_buffer_chunks()

    def offload(self) -> None:
        if self.uncompressed_buffer_k is not None and self.uncompressed_buffer_k.device.type != "cpu":
            self.uncompressed_buffer_k = self.uncompressed_buffer_k.to("cpu", non_blocking=True)
            self.uncompressed_buffer_v = self.uncompressed_buffer_v.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        if self.uncompressed_buffer_k is None:
            return
        dev = getattr(self, "device", self.uncompressed_buffer_k.device)
        if self.uncompressed_buffer_k.device != dev:
            self.uncompressed_buffer_k = self.uncompressed_buffer_k.to(dev, non_blocking=True)
            self.uncompressed_buffer_v = self.uncompressed_buffer_v.to(dev, non_blocking=True)

    def reset(self) -> None:
        self.chunked_bitstreams_k.clear()
        self.chunked_bitstreams_v.clear()
        self.uncompressed_buffer_k = None
        self.uncompressed_buffer_v = None
        self.keys = None
        self.values = None


# Backward-compatible name for tests / docs
CompressedDynamicLayer = ChunkedCompressedDynamicLayer


class RTN3BitDynamicLayer(DynamicLayer):
    """
    After appending new keys/values, apply **min–max RTN** at ``num_bits`` and dequantize
    back to the layer dtype (fp16/bf16). No bitstream / codec.
    """

    def __init__(self, config=None, *, num_bits: int = 3, enabled: bool = True):
        super().__init__(config)
        self._num_bits = int(num_bits)
        self._enabled = bool(enabled)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(key_states, value_states, *args, **kwargs)
        if self._enabled and self.keys is not None and self.values is not None:
            self.keys = rtn_minmax_roundtrip_tensor(self.keys, self._num_bits)
            self.values = rtn_minmax_roundtrip_tensor(self.values, self._num_bits)
            keys, values = self.keys, self.values
        return keys, values


class RTN3BitCache(DynamicCache):
    """
    ``DynamicCache`` with per-layer :class:`RTN3BitDynamicLayer` (default **3-bit** RTN KV).
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        *,
        num_bits: int = 3,
        enabled: bool = True,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(config=config, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        self._num_bits = int(num_bits)
        self._enabled = bool(enabled)
        for i in range(len(self.layers)):
            self.layers[i] = RTN3BitDynamicLayer(config=None, num_bits=self._num_bits, enabled=self._enabled)


def make_rtn3_kv_cache(
    config: PreTrainedConfig,
    *,
    num_bits: int = 3,
    enabled: bool = True,
) -> RTN3BitCache:
    """Factory for the Phase-3 **3-bit RTN-only** KV baseline (no HEVC)."""
    return RTN3BitCache(config=config, num_bits=num_bits, enabled=enabled)


class ChunkedVcLLMCache(DynamicCache):
    """
    DynamicCache with per-layer :class:`ChunkedCompressedDynamicLayer`.

    Compression runs **inside** each layer's ``update`` (incremental chunks). No extra
    ``Cache.update`` post-step is required.

    **Telemetry:** ``hook.compression_stats`` records each ``_compress_tensor`` call (per chunk).
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        hook: KVCacheCompressionHook,
        *,
        chunk_size: int = 64,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(config=config, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        self._kv_hook = hook
        self._chunk_size = int(chunk_size)
        self._compression_stats = VcLLMCompressionStats()
        self._kv_hook.compression_stats = self._compression_stats
        for i in range(len(self.layers)):
            self.layers[i] = ChunkedCompressedDynamicLayer(
                hook, config=None, chunk_size=self._chunk_size
            )

    @property
    def compression_stats(self) -> VcLLMCompressionStats:
        return self._compression_stats

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def total_elements(self) -> int:
        return self._compression_stats.total_elements

    @property
    def total_compressed_bytes(self) -> int:
        return self._compression_stats.total_compressed_bytes

    @property
    def total_fp16_bytes(self) -> int:
        return self._compression_stats.total_fp16_bytes

    def reset_compression_stats(self) -> None:
        self._compression_stats.reset()

    def print_compression_summary(self, *, title: str = "VcLLM KV cache — compression telemetry") -> None:
        s = self._compression_stats
        if s.total_elements == 0:
            print(
                f"[{self.__class__.__name__}] No compression samples recorded.",
                flush=True,
            )
            return
        fp16_mb = s.total_fp16_bytes / (1024**2)
        cmp_mb = s.total_compressed_bytes / (1024**2)
        saved_mb = fp16_mb - cmp_mb
        bpe = s.global_avg_bpe()
        ratio = s.total_fp16_bytes / s.total_compressed_bytes if s.total_compressed_bytes else float("inf")

        def _bar() -> None:
            print("+" + "-" * 62 + "+", flush=True)

        _bar()
        print(f"| {title[:60]:<60} |", flush=True)
        _bar()
        print(f"|  chunk_size (tokens)                  : {self._chunk_size:<36} |", flush=True)
        print(f"|  Compress calls (K+V tensors)        : {s.compress_calls:<36} |", flush=True)
        print(f"|  Total elements (sum of numel)       : {s.total_elements:<36} |", flush=True)
        print(f"|  FP16 baseline volume                : {fp16_mb:>10.4f} MB{' ' * 23} |", flush=True)
        print(f"|  Compressed bitstream volume         : {cmp_mb:>10.4f} MB{' ' * 23} |", flush=True)
        print(f"|  Absolute savings (FP16 - packed)    : {saved_mb:>10.4f} MB{' ' * 23} |", flush=True)
        print(f"|  Mean global BPE (bits / element)    : {bpe:>10.4f}{' ' * 27} |", flush=True)
        print(f"|  Implied ratio (FP16 bytes / packed): {ratio:>10.4f}x{' ' * 25} |", flush=True)
        _bar()


VcLLMCompressedCache = ChunkedVcLLMCache


def make_compressed_kv_cache(
    config: PreTrainedConfig,
    hook: KVCacheCompressionHook,
    *,
    chunk_size: int = 64,
) -> ChunkedVcLLMCache:
    """Build :class:`ChunkedVcLLMCache` (incremental chunk compression, default ``chunk_size=64``)."""
    return ChunkedVcLLMCache(config=config, hook=hook, chunk_size=chunk_size)


def wrap_standard_dynamic_layers(
    cache: DynamicCache,
    hook: KVCacheCompressionHook,
    *,
    chunk_size: int = 64,
) -> None:
    """Replace plain DynamicLayer rows with :class:`ChunkedCompressedDynamicLayer`, migrating dense KV into chunks."""
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, ChunkedCompressedDynamicLayer):
            continue
        if type(layer) is not DynamicLayer:
            raise TypeError(
                f"KV compression expects DynamicLayer per cache row, got {type(layer).__name__}. "
                "Extend ChunkedCompressedDynamicLayer if needed."
            )
        new_layer = ChunkedCompressedDynamicLayer(hook, config=None, chunk_size=chunk_size)
        new_layer.dtype = getattr(layer, "dtype", None)
        new_layer.device = getattr(layer, "device", None)
        new_layer.is_initialized = layer.is_initialized
        if layer.is_initialized and layer.keys is not None and layer.keys.numel() > 0:
            new_layer.uncompressed_buffer_k = layer.keys
            new_layer.uncompressed_buffer_v = layer.values
            new_layer._flush_buffer_chunks()
        else:
            new_layer.keys = layer.keys
            new_layer.values = layer.values
        cache.layers[i] = new_layer


def install_kv_compression_hooks(model: torch.nn.Module, hook: KVCacheCompressionHook, *, chunk_size: int = 64):
    """
    Register forward pre/post hooks on the decoder trunk.

    - Pre: inject ``ChunkedVcLLMCache`` when ``past_key_values`` is None; wrap foreign caches.
    - Post: for non-chunked wrapped caches only, legacy compress (chunked layers compress in ``update``).
    """
    inner = get_inner_transformer(model)

    def _pre(module, args, kwargs):
        kwargs = dict(kwargs)
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = getattr(module.config, "use_cache", True)
        if use_cache is False:
            return args, kwargs

        pkv = kwargs.get("past_key_values")
        if pkv is None:
            kwargs["past_key_values"] = make_compressed_kv_cache(module.config, hook, chunk_size=chunk_size)
        else:
            wrap_standard_dynamic_layers(pkv, hook, chunk_size=chunk_size)

        # Chunked layers need no extra pre-step; dense DynamicLayer migration happens in wrap_standard_dynamic_layers.
        return args, kwargs

    def _post(module, args, kwargs, output):
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = getattr(module.config, "use_cache", True)
        if use_cache is False:
            return output
        pkv = getattr(output, "past_key_values", None)
        if pkv is None:
            return output
        if isinstance(pkv, ChunkedVcLLMCache):
            return output
        return output

    h1 = inner.register_forward_pre_hook(_pre, with_kwargs=True)
    h2 = inner.register_forward_hook(_post, with_kwargs=True)
    return (h1, h2)


def remove_kv_compression_hooks(handles: tuple) -> None:
    for h in handles:
        h.remove()
