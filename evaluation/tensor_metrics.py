import torch
import numpy as np

def compute_mse(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2).item()

def compute_mae(tensor1, tensor2):
    return torch.mean(torch.abs(tensor1 - tensor2)).item()

def compute_relative_error(tensor1, tensor2, eps=1e-8):
    denom = torch.max(torch.abs(tensor1), torch.abs(tensor2)) + eps
    return torch.mean(torch.abs(tensor1 - tensor2) / denom).item()

def compute_cosine_similarity(tensor1, tensor2):
    t1_flat = tensor1.flatten()
    t2_flat = tensor2.flatten()
    dot_product = torch.dot(t1_flat, t2_flat)
    norm1 = torch.norm(t1_flat)
    norm2 = torch.norm(t2_flat)
    return (dot_product / (norm1 * norm2)).item()

def compute_compression_ratio(original_size_bytes, compressed_size_bytes):
    return original_size_bytes / compressed_size_bytes

def compute_bits_per_value(original_size_bytes, compressed_size_bytes, num_elements):
    original_bits = original_size_bytes * 8
    compressed_bits = compressed_size_bytes * 8
    return compressed_bits / num_elements

def compute_all_metrics(original, reconstructed, original_size_bytes, compressed_size_bytes):
    metrics = {
        'mse': compute_mse(original, reconstructed),
        'mae': compute_mae(original, reconstructed),
        'relative_error': compute_relative_error(original, reconstructed),
        'cosine_similarity': compute_cosine_similarity(original, reconstructed),
        'compression_ratio': compute_compression_ratio(original_size_bytes, compressed_size_bytes),
        'bits_per_value': compute_bits_per_value(original_size_bytes, compressed_size_bytes, original.numel())
    }
    return metrics

def print_metrics(metrics):
    print("Tensor Reconstruction Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Relative Error: {metrics['relative_error']:.6f}")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
    print(f"  Bits per Value: {metrics['bits_per_value']:.2f} bpp")