import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.rtn import rtn_quantize_tensor, rtn_dequantize_tensor
from codec.codec_job import CodecJob
from codec.frame_mapper import tensor_to_frames, frames_to_tensor
from codec.hevc_backend import encode_frames_to_bitstream, decode_bitstream_to_frames
from evaluation.tensor_metrics import compute_all_metrics, print_metrics

def test_tensor_codec():
    print("=" * 60)
    print("VcLLM Tensor Codec Pipeline Test")
    print("=" * 60)
    
    print("\n1. Creating test tensor...")
    tensor = torch.randn(1024, 1024)
    original_size_bytes = tensor.numel() * 4  
    print(f"   Shape: {tensor.shape}")
    print(f"   Original size: {original_size_bytes / (1024 * 1024):.2f} MB")
    
    print("\n2. RTN Quantization...")
    quantized, rtn_metadata = rtn_quantize_tensor(tensor, min_val=-3, max_val=3)
    print(f"   Quantized dtype: {quantized.dtype}")
    print(f"   Min: {rtn_metadata['min_val']}, Max: {rtn_metadata['max_val']}")
    
    print("\n3. Tensor to Frames...")
    frames, frame_metadata = tensor_to_frames(quantized)
    print(f"   Number of frames: {frame_metadata['num_frames']}")
    print(f"   Frame size: {frame_metadata['frame_size']}x{frame_metadata['frame_size']}")
    
    print("\n4. HEVC Encoding (NVENC only, no libx265 fallback)...")
    codec_job = CodecJob.square(
        frame_metadata["frame_size"],
        qp=0,
        lossless=True,
        intra_only=True,
        backend="hevc_nvenc",
    )
    encode_frames_to_bitstream(
        frames,
        '/tmp/test_tensor.hevc',
        codec_job,
        allow_software_encoder_fallback=False,
    )
    compressed_size_bytes = os.path.getsize('/tmp/test_tensor.hevc')
    print(f"   Compressed size: {compressed_size_bytes / 1024:.2f} KB")
    
    print("\n5. HEVC Decoding (NVDEC/hevc_cuvid only, no PNG fallback)...")
    decoded_frames = decode_bitstream_to_frames(
        '/tmp/test_tensor.hevc',
        codec_job,
        allow_software_decoder_fallback=False,
    )
    print(f"   Decoded frames: {len(decoded_frames)}")
    
    print("\n6. Frames to Tensor...")
    reconstructed_quantized = frames_to_tensor(decoded_frames, frame_metadata)
    
    print("\n7. RTN Dequantization...")
    reconstructed = rtn_dequantize_tensor(reconstructed_quantized, rtn_metadata)
    
    print("\n8. Computing Metrics...")
    metrics = compute_all_metrics(tensor, reconstructed, original_size_bytes, compressed_size_bytes)
    print_metrics(metrics)
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_tensor_codec()