"""
RTN (round-to-nearest) uniform quantization.

Paper-style **phase-1** pipelines apply **incoherence processing** (blocked Hadamard +
Rademacher diagonal) *before* calling ``rtn_quantize_tensor``; see ``compression.transform``
and ``compress_weight_layer`` in ``weight_pipeline.py``.
"""
import torch
import numpy as np

def rtn_quantize_tensor(tensor, min_val=None, max_val=None, num_bits=8):
    if min_val is None:
        min_val = tensor.min().item()
    if max_val is None:
        max_val = tensor.max().item()

    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1e-6

    scale = range_val / (2**num_bits - 1)

    # Quantize: q = clamp( round( (x - min_val) / scale ), 0, 2^bits-1 )
    quantized = torch.clamp(torch.round((tensor - min_val) / scale), 0, 2**num_bits - 1)
    quantized = quantized.to(torch.uint8)

    metadata = {
        'min_val': min_val,
        'max_val': max_val,
        'num_bits': num_bits,
        'scale': scale,
        'original_shape': list(tensor.shape),
        'original_dtype': str(tensor.dtype)
    }

    return quantized, metadata

def rtn_minmax_roundtrip_tensor(tensor: torch.Tensor, num_bits: int = 3) -> torch.Tensor:
    """
    Asymmetric min–max RTN: quantize to ``num_bits`` then dequantize back to float.

    Used for Phase-3 **KV-only** baselines (e.g. 3-bit) without a video codec. Preserves
    input dtype (fp16/bf32) on output.
    """
    if tensor is None or tensor.numel() == 0:
        return tensor
    dt = tensor.dtype
    q, meta = rtn_quantize_tensor(tensor.detach().float(), num_bits=num_bits)
    out = rtn_dequantize_tensor(q, meta)
    return out.to(dtype=dt)


def rtn_dequantize_tensor(quantized, metadata):
    min_val = metadata['min_val']
    max_val = metadata['max_val']
    num_bits = metadata['num_bits']

    scale = metadata['scale']

    # Dequantize: x = q * scale + min_val
    dequantized = quantized.float() * scale + min_val

    if dequantized.isnan().any() or dequantized.isinf().any():
        print(f"WARNING: NaN or Inf detected in dequantized tensor!")
        print(f"  min_val={min_val}, max_val={max_val}, scale={scale}")
        dequantized = torch.clamp(dequantized, -1e10, 1e10)

    return dequantized