import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="VcLLM: Video Coded LLM Compression Framework")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test_codec', 'compress_weights', 'eval_weight_codec', 'eval_kv_cache', 'help'],
        default='help',
        help='Run mode'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='EleutherAI/pythia-160m',
        help='Model name or path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory'
    )
    parser.add_argument(
        '--qp',
        type=int,
        default=0,
        help='HEVC QP (rtn_lossy_hevc; ignored for rtn_lossless_hevc / rtn_only)'
    )
    parser.add_argument(
        '--compress-mode',
        type=str,
        choices=['rtn_only', 'rtn_lossless_hevc', 'rtn_lossy_hevc'],
        default='rtn_lossy_hevc',
        help='rtn_lossy_hevc: NVENC+QP; rtn_lossless_hevc: NVENC lossless when GPU enabled',
    )
    parser.add_argument(
        '--frame-size',
        type=int,
        default=1024,
        help='Frame size for codec'
    )
    parser.add_argument(
        '--compressed-dir',
        type=str,
        default='./compressed_weights',
        help='Directory for compressed weights'
    )
    parser.add_argument(
        '--no-hardware-accel',
        action='store_true',
        help='CPU encode only (libx265). Default is NVENC; no libx265 fallback when GPU is used.',
    )
    parser.add_argument(
        '--no-hardware-decode',
        action='store_true',
        help='Disable GPU decode (hevc_cuvid); use software PNG decode for weights. Default: hardware decode.',
    )

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.mode == 'test_codec':
        from scripts.test_tensor_codec import test_tensor_codec
        test_tensor_codec()
    
    elif args.mode == 'compress_weights':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from compression.weight_pipeline import compress_model_weights
        
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(args.model)
        
        print(f"Compressing weights to: {args.compressed_dir}")
        compress_model_weights(
            model,
            args.compressed_dir,
            mode=args.compress_mode,
            qp=args.qp,
            hardware_accel=not args.no_hardware_accel,
            frame_size=args.frame_size,
        )
    
    elif args.mode == 'eval_weight_codec':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from compression.weight_pipeline import compress_model_weights, decompress_model_weights
        from evaluation.perplexity import evaluate_perplexity_on_wikitext
        
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        print("\n--- Baseline Perplexity ---")
        baseline_ppl = evaluate_perplexity_on_wikitext(model, tokenizer)
        
        print("\n--- Compressing Weights ---")
        compress_model_weights(
            model,
            args.compressed_dir,
            mode=args.compress_mode,
            qp=args.qp,
            hardware_accel=not args.no_hardware_accel,
            frame_size=args.frame_size,
        )
        
        print("\n--- Loading Compressed Weights ---")
        model_decompressed = AutoModelForCausalLM.from_pretrained(args.model)
        model_decompressed = decompress_model_weights(
            model_decompressed,
            args.compressed_dir,
            hardware_accel=not args.no_hardware_accel,
            hardware_decode=not args.no_hardware_decode,
        )
        
        print("\n--- Decompressed Perplexity ---")
        decompressed_ppl = evaluate_perplexity_on_wikitext(model_decompressed, tokenizer)
        
        print("\n=== Results ===")
        print(f"Baseline Perplexity: {baseline_ppl:.2f}")
        print(f"Decompressed Perplexity: {decompressed_ppl:.2f}")
        print(f"Perplexity Increase: {(decompressed_ppl - baseline_ppl):.2f}")
    
    elif args.mode == 'help':
        parser.print_help()
    
    else:
        print(f"Mode '{args.mode}' not yet implemented")
        parser.print_help()

if __name__ == "__main__":
    main()