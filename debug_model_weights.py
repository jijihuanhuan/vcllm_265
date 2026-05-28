
import torch
import os
import json
from transformers import AutoModelForCausalLM
from compression.weight_pipeline import compress_model_weights, decompress_model_weights
from compression.weight_pipeline import compress_weight_layer, decompress_weight_layer

print("Loading model...")
model_name = 'EleutherAI/pythia-160m'
model = AutoModelForCausalLM.from_pretrained(model_name)

# Pick a small layer to debug
param_name = None
target_layer = None
for name, param in model.named_parameters():
    if 'input_layernorm' in name and 'weight' in name:
        param_name = name
        target_layer = param.data
        print(f"Testing layer: {name}, shape: {target_layer.shape}")
        break

# Step 1: Test with weight_pipeline functions
print("\n=== Testing weight_pipeline functions ===")
compressed_dir = '/tmp/debug_weights'
os.makedirs(compressed_dir, exist_ok=True)

safe_name = param_name.replace('.', '___DOT___')
stat = compress_weight_layer(
    safe_name, target_layer.cpu(), compressed_dir, mode="rtn_lossless_hevc", qp=0, hardware_accel=False
)
print(f"Compressed: original={stat['original_size_bytes']}, compressed={stat['compressed_size_bytes']}")

# Decompress
recovered = decompress_weight_layer(safe_name, compressed_dir, hardware_accel=False)
print(f"Recovered shape: {recovered.shape}, dtype: {recovered.dtype}")
print(f"Target dtype: {target_layer.dtype}")

# Check dtype
if recovered.dtype != target_layer.dtype:
    print(f"[WARNING] Mismatched dtypes! Converting to {target_layer.dtype}...")
    recovered = recovered.to(target_layer.dtype)

# Check
max_error = torch.abs(target_layer.cpu() - recovered).max().item()
all_close = torch.allclose(target_layer.cpu(), recovered, rtol=1e-2, atol=1e-2)
print(f"Max error: {max_error:.10f}")
print(f"All close: {all_close}")

# Cleanup
import shutil
if os.path.exists(compressed_dir):
    shutil.rmtree(compressed_dir)

print("\n=== Done ===")
