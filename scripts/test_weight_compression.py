import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM
from compression.weight_pipeline import compress_model_weights, decompress_model_weights
from evaluation.perplexity import evaluate_perplexity_on_wikitext

def test_weight_compression():
    print("=" * 60)
    print("VcLLM Weight Compression Test")
    print("=" * 60)
    
    model_name = "EleutherAI/pythia-160m"
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Model loaded: {model.config.model_type}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    compressed_dir = "/tmp/test_compressed_weights"
    print(f"\nCompressing weights to: {compressed_dir}")
    summary = compress_model_weights(model, compressed_dir, mode="rtn_lossless_hevc", qp=0)
    
    print("\nDecompressing weights...")
    model_decompressed = AutoModelForCausalLM.from_pretrained(model_name)
    model_decompressed = decompress_model_weights(model_decompressed, compressed_dir)
    
    print("\nVerifying weights...")
    original_params = dict(model.named_parameters())
    decompressed_params = dict(model_decompressed.named_parameters())
    
    max_error = 0
    for name in original_params:
        original = original_params[name].data.cpu()
        decompressed = decompressed_params[name].data.cpu()
        error = torch.mean((original - decompressed) ** 2).item()
        max_error = max(max_error, error)
    
    print(f"Maximum reconstruction MSE: {max_error:.6f}")
    
    if max_error < 1e-5:
        print("\n✅ Weight compression/decompression successful!")
    else:
        print(f"\n❌ Weight reconstruction error too high: {max_error}")

if __name__ == "__main__":
    test_weight_compression()