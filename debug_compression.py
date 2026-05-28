
import torch
import numpy as np
from compression.rtn import rtn_quantize_tensor, rtn_dequantize_tensor
from codec.codec_job import CodecJob
from codec.frame_mapper import tensor_to_frames, frames_to_tensor
from codec.hevc_backend import encode_frames_to_bitstream, decode_bitstream_to_frames
import os

print("="*60)
print("Step 1: Testing RTN Quantization/Dequantization")
print("="*60)

# 创建一个简单的测试张量
test_tensor = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
print(f"Original: {test_tensor}")

# 量化
min_val = test_tensor.min().item()
max_val = test_tensor.max().item()
quantized, rtn_meta = rtn_quantize_tensor(test_tensor, min_val=min_val, max_val=max_val)
print(f"Quantized: {quantized}")

# 反量化
dequantized = rtn_dequantize_tensor(quantized, rtn_meta)
print(f"Dequantized: {dequantized}")
print(f"Error: {torch.abs(test_tensor - dequantized).max().item():.6f}")

print("\n" + "="*60)
print("Step 2: Testing Frame Mapper")
print("="*60)

test_tensor_2 = torch.randn(10, 10, dtype=torch.float32)
print(f"Original shape: {test_tensor_2.shape}")

# 转 frames
frames, frame_meta = tensor_to_frames(test_tensor_2)
print(f"Frames shape: {frames.shape}")

# 转回 tensor
recovered = frames_to_tensor(frames, frame_meta)
print(f"Recovered shape: {recovered.shape}")
print(f"Max error: {torch.abs(test_tensor_2 - recovered).max().item():.6f}")

print("\n" + "="*60)
print("Step 3: Testing RTN + Frame Mapper with uint8")
print("="*60)

test_tensor_3 = torch.tensor([[0, 25, 50], [75, 100, 125], [150, 175, 200]], dtype=torch.uint8)
print(f"Original:\n{test_tensor_3}")

frames, frame_meta = tensor_to_frames(test_tensor_3)
recovered = frames_to_tensor(frames, frame_meta)
print(f"Recovered:\n{recovered}")
print(f"All equal: {torch.all(test_tensor_3 == recovered)}")

print("\n" + "="*60)
print("Step 4: Full Pipeline (RTN -> Frames -> HEVC -> Frames -> RTN)")
print("="*60)

test_tensor_4 = torch.randn(3, 3, dtype=torch.float32) * 2
print(f"Original:\n{test_tensor_4}")

# 1. RTN Quant
min_val = test_tensor_4.min().item()
max_val = test_tensor_4.max().item()
quantized, rtn_meta = rtn_quantize_tensor(test_tensor_4, min_val=min_val, max_val=max_val)

# 2. To Frames
frames, frame_meta = tensor_to_frames(quantized)

# 3. HEVC Encode
bitstream_path = '/tmp/test_hevc.hevc'
_hevc_job = CodecJob.square(frame_meta["frame_size"], qp=0, lossless=True, intra_only=True, backend="libx265")
encode_frames_to_bitstream(frames, bitstream_path, _hevc_job)

# 4. HEVC Decode
decoded_frames = decode_bitstream_to_frames(bitstream_path, _hevc_job)

# 5. Back to Tensor
recovered_quant = frames_to_tensor(decoded_frames, frame_meta)

# 6. RTN Dequant
final_output = rtn_dequantize_tensor(recovered_quant, rtn_meta)

print(f"Final Output:\n{final_output}")
print(f"Max error: {torch.abs(test_tensor_4 - final_output).max().item():.6f}")
print(f"All close: {torch.allclose(test_tensor_4, final_output, rtol=1e-2, atol=1e-2)}")

# 清理
if os.path.exists(bitstream_path):
    os.remove(bitstream_path)
