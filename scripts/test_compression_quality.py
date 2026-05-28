import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from compression.weight_pipeline import compress_model_weights, decompress_model_weights
from evaluation.perplexity import compute_perplexity

def test_compression_quality(
    model_name, qp_values, test_text, mode: str = "rtn_lossy_hevc"
):
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, tokenizer, test_text)
    print(f"Baseline Perplexity: {baseline_ppl:.2f}")
    
    results = []
    
    for qp in qp_values:
        print(f"\n=== Testing QP={qp} (mode={mode}) ===")
        
        compressed_dir = f'/tmp/compression_test_qp{qp}'
        os.makedirs(compressed_dir, exist_ok=True)
        
        compress_model_weights(model, compressed_dir, mode=mode, qp=qp)
        
        model_decompressed = AutoModelForCausalLM.from_pretrained(model_name)
        decompress_model_weights(model_decompressed, compressed_dir)
        
        ppl = compute_perplexity(model_decompressed, tokenizer, test_text)
        print(f"Decompressed Perplexity: {ppl:.2f}")
        print(f"Perplexity Increase: {ppl - baseline_ppl:.2f}")
        
        summary_path = os.path.join(compressed_dir, 'compression_summary.json')
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        result = {
            'qp': qp,
            'mode': mode,
            'compression_ratio': summary['compression_ratio'],
            'original_size_mb': summary['total_original_bytes'] / (1024**2),
            'compressed_size_mb': summary['total_compressed_bytes'] / (1024**2),
            'baseline_ppl': baseline_ppl,
            'decompressed_ppl': ppl,
            'ppl_increase': ppl - baseline_ppl
        }
        results.append(result)
        
        os.system(f'rm -rf {compressed_dir}')
    
    print("\n=== Summary ===")
    print(f"{'QP':<4} {'Mode':<20} {'Ratio':<8} {'Original':<10} {'Compressed':<12} {'Baseline':<10} {'Decompressed':<14} {'Increase':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['qp']:<4} {str(r['mode']):<20} {r['compression_ratio']:<8.2f}x {r['original_size_mb']:<10.2f}MB {r['compressed_size_mb']:<12.2f}MB {r['baseline_ppl']:<10.2f} {r['decompressed_ppl']:<14.2f} {r['ppl_increase']:<8.2f}")

if __name__ == "__main__":
    test_text = """The quick brown fox jumps over the lazy dog. This is a test sentence to evaluate language model performance."""
    qp_values = [0, 10, 20, 30, 40]
    
    test_compression_quality('EleutherAI/pythia-160m', qp_values, test_text, mode="rtn_lossy_hevc")