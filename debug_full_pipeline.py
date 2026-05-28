
import torch
import os
import json
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from compression.weight_pipeline import compress_model_weights, decompress_model_weights
from evaluation.perplexity import compute_perplexity

print("=" * 80)
print("1. Load Model & Compute Baseline")
print("=" * 80)
model_name = 'EleutherAI/pythia-160m'
test_text = """The quick brown fox jumps over the lazy dog. This is a test sentence to evaluate language model performance."""

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
baseline_ppl = compute_perplexity(model, tokenizer, test_text)
print(f"Baseline Perplexity: {baseline_ppl:.4f}")

print("\n" + "=" * 80)
print("2. Compress & Decompress")
print("=" * 80)

compressed_dir = '/tmp/debug_full_pipeline'
if os.path.exists(compressed_dir):
    shutil.rmtree(compressed_dir)

summary = compress_model_weights(
    model, compressed_dir, mode="rtn_lossless_hevc", qp=0, hardware_accel=False
)

# Load a fresh model for decompression
model_decompressed = AutoModelForCausalLM.from_pretrained(model_name)
decompress_model_weights(model_decompressed, compressed_dir, hardware_accel=False)

print("\n" + "=" * 80)
print("3. Compare Weights (Original vs Decompressed)")
print("=" * 80)

max_error_global = -1
mean_error_global = 0
param_count = 0
worst_layer = ""

with torch.no_grad():
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), model_decompressed.named_parameters()):
        assert name1 == name2, f"Name mismatch: {name1} vs {name2}"
        
        diff = torch.abs(param1 - param2)
        max_error = diff.max().item()
        mean_error = diff.mean().item()
        
        param_count += 1
        mean_error_global += mean_error
        
        if max_error > max_error_global:
            max_error_global = max_error
            worst_layer = name1
        
        print(f"{name1[:60]:60} | MaxErr={max_error:.8f} | MeanErr={mean_error:.8f} | dtype1={param1.dtype}, dtype2={param2.dtype}")

mean_error_global /= param_count

print("\n" + "-" * 80)
print(f"Global Max Error: {max_error_global:.8f} (Layer: {worst_layer})")
print(f"Global Mean Error: {mean_error_global:.8f}")
print("-" * 80)

print("\n" + "=" * 80)
print("4. Compute Decompressed Perplexity")
print("=" * 80)
decompressed_ppl = compute_perplexity(model_decompressed, tokenizer, test_text)
print(f"Baseline Perplexity:    {baseline_ppl:.4f}")
print(f"Decompressed Perplexity: {decompressed_ppl:.4f}")
print(f"Perplexity Increase:    {decompressed_ppl - baseline_ppl:.4f}")

print("\n" + "=" * 80)
print("Done!")
