import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.weight_pipeline import compress_model_weights, decompress_model_weights

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(50257, 768)
        self.layer1 = torch.nn.Linear(768, 768)
        self.layer2 = torch.nn.Linear(768, 768)
        self.layer3 = torch.nn.Linear(768, 50257)
        
        torch.nn.init.normal_(self.embedding.weight, std=0.02)
        torch.nn.init.normal_(self.layer1.weight, std=0.02)
        torch.nn.init.normal_(self.layer2.weight, std=0.02)
        torch.nn.init.normal_(self.layer3.weight, std=0.02)

def test_weight_compression():
    print("=" * 60)
    print("VcLLM Weight Compression Test (Local)")
    print("=" * 60)
    
    print("\nCreating mock model...")
    model = MockModel()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    compressed_dir = "/tmp/test_compressed_weights"
    print(f"\nCompressing weights to: {compressed_dir}")
    summary = compress_model_weights(model, compressed_dir, mode="rtn_lossless_hevc", qp=0)
    
    print("\nDecompressing weights...")
    model_decompressed = MockModel()
    model_decompressed = decompress_model_weights(model_decompressed, compressed_dir)
    
    print("\nVerifying weights...")
    max_error = 0
    for name in original_state:
        original = original_state[name].cpu()
        decompressed = dict(model_decompressed.named_parameters())[name].data.cpu()
        error = torch.mean((original - decompressed) ** 2).item()
        max_error = max(max_error, error)
        print(f"  {name}: MSE = {error:.6f}")
    
    print(f"\nMaximum reconstruction MSE: {max_error:.6f}")
    
    if max_error < 1e-3:
        print("\n✅ Weight compression/decompression successful!")
        print(f"  Reconstruction error within acceptable bounds for video codec compression")
    else:
        print(f"\n❌ Weight reconstruction error too high: {max_error}")

if __name__ == "__main__":
    test_weight_compression()