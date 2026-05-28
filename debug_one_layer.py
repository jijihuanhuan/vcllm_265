
import torch
import os
import numpy as np
from transformers import AutoModelForCausalLM
from compression.rtn import rtn_quantize_tensor, rtn_dequantize_tensor
from codec.codec_job import CodecJob
from codec.frame_mapper import tensor_to_frames, frames_to_tensor
from codec.hevc_backend import encode_frames_to_bitstream, decode_bitstream_to_frames

print("=" * 80)
print("Debugging: gpt_neox.final_layer_norm.weight")
print("=" * 80)

model_name = 'EleutherAI/pythia-160m'
model = AutoModelForCausalLM.from_pretrained(model_name)

target_name = None
target_tensor = None
for name, param in model.named_parameters():
    if name == 'gpt_neox.final_layer_norm.weight':
        target_name = name
        target_tensor = param.data.clone()
        break

print(f"Name: {target_name}")
print(f"Shape: {target_tensor.shape}")
print(f"Dtype: {target_tensor.dtype}")
print(f"Device: {target_tensor.device}")
print(f"Min: {target_tensor.min().item():.6f}")
print(f"Max: {target_tensor.max().item():.6f}")

print("\n--- Step 1: RTN Quantization Only ---")
min_val = target_tensor.min().item()
max_val = target_tensor.max().item()
q, meta = rtn_quantize_tensor(target_tensor.cpu(), min_val=min_val, max_val=max_val)
dq = rtn_dequantize_tensor(q, meta)
print(f"RTN only max error: {torch.abs(target_tensor.cpu() - dq).max().item():.10f}")

print("\n--- Step 2: RTN + Frame Mapper ---")
frames, fm_meta = tensor_to_frames(q)
back_q = frames_to_tensor(frames, fm_meta)
print(f"Frame mapper max error (uint8): {torch.abs(q - back_q).max().item():.10f}")
dq2 = rtn_dequantize_tensor(back_q, meta)
print(f"RTN+Frames max error: {torch.abs(target_tensor.cpu() - dq2).max().item():.10f}")

print("\n--- Step 3: Full Pipeline (RTN + Frames + HEVC ---")
import tempfile
with tempfile.NamedTemporaryFile(suffix='.hevc', delete=False) as f:
    hevc_path = f.name

_job = CodecJob.square(fm_meta["frame_size"], qp=0, lossless=True, intra_only=True, backend="libx265")
encode_frames_to_bitstream(frames, hevc_path, _job)
decoded_frames = decode_bitstream_to_frames(hevc_path, _job)
os.remove(hevc_path)
back_q2 = frames_to_tensor(decoded_frames, fm_meta)
print(f"HEVC round-trip max error (uint8): {torch.abs(q - back_q2).max().item():.10f}")
dq3 = rtn_dequantize_tensor(back_q2, meta)
print(f"Full pipeline max error: {torch.abs(target_tensor.cpu() - dq3).max().item():.10f}")

print("\n--- Comparison of values ---")
print(f"Original[:10]:   {target_tensor.cpu()[:10]}")
print(f"Full Pipe[:10]:  {dq3[:10]}")
