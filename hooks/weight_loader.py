"""
Hooks for loading compressed weights.

NOTE (lazy / broken): ``load_weights_lazy`` previously registered backward hooks with the wrong
on-disk naming convention (``layer.weight`` -> ``layer_weight`` vs ``layer___DOT___weight`` in the
compressor) and mis-used gradients as a loading mechanism. Lazy loading remains **disabled** until
rewritten (e.g. forward hooks or explicit parameter wrappers aligned with ``compress_weight_layer``).
"""

from compression.weight_pipeline import decompress_model_weights


class WeightLoaderHook:
    def __init__(self, compressed_dir):
        self.compressed_dir = compressed_dir

    def load_weights_eager(self, model):
        return decompress_model_weights(model, self.compressed_dir)

    def load_weights_lazy(self, model):
        raise NotImplementedError(
            "load_weights_lazy is disabled: the previous implementation was incorrect "
            "(layer name mapping vs SEPARATOR '___DOT___', backward-hook misuse). "
            "Use load_weights_eager() / decompress_model_weights() instead."
        )

    def _create_lazy_hook(self, layer_name):
        # BROKEN / unused — kept only so any stale reference fails loudly.
        def hook(grad):
            raise RuntimeError("Lazy weight hook should never run; use eager load.")
        return hook


def load_model_with_compressed_weights(model, compressed_dir, mode="eager"):
    hook = WeightLoaderHook(compressed_dir)

    if mode == "eager":
        return hook.load_weights_eager(model)
    elif mode == "lazy":
        return hook.load_weights_lazy(model)
    else:
        raise ValueError(f"Unknown mode: {mode}")
