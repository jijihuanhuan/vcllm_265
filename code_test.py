import torch
import numpy as np
from PIL import Image
import os

print("Step 1: Creating fake tensor...")
# 1. fake tensor (模拟 LLM weight)
tensor = torch.randn(1024, 1024)
print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
print(f"Tensor min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")

print("\nStep 2: RTN quantization...")
# 2. RTN quantization (论文步骤)
tensor_q = torch.clamp(tensor, -3, 3)
tensor_q = ((tensor_q + 3) / 6 * 255).to(torch.uint8)
print(f"Quantized tensor shape: {tensor_q.shape}, dtype: {tensor_q.dtype}")
print(f"Quantized tensor min: {tensor_q.min().item()}, max: {tensor_q.max().item()}")

frame = tensor_q.numpy()
print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

print("\nStep 3: Saving as image frame...")
# 3. 保存为 image frame
img = Image.fromarray(frame, mode='L')
img.save("frame.png")
if os.path.exists("frame.png"):
    print("frame.png saved successfully")
else:
    print("ERROR: frame.png not saved!")

print("\nStep 4: HEVC encode...")
# 4. HEVC encode (Intra-only)
encode_result = os.system(
    "ffmpeg -y -i frame.png "
    "-c:v hevc_nvenc -preset slow -g 1 -qp 0 "
    "tensor.hevc"
)
print(f"Encode exit code: {encode_result}")
if os.path.exists("tensor.hevc"):
    print("tensor.hevc saved successfully")
else:
    print("ERROR: tensor.hevc not saved!")

print("\nStep 5: Decode...")
# 5. decode
decode_result = os.system("ffmpeg -y -i tensor.hevc decoded.png")
print(f"Decode exit code: {decode_result}")
if os.path.exists("decoded.png"):
    print("decoded.png saved successfully")
else:
    print("ERROR: decoded.png not saved!")

print("\nStep 6: Recover tensor...")
decoded_img = Image.open("decoded.png").convert('L')
decoded = np.array(decoded_img)
if decoded is not None:
    print(f"Decoded image shape: {decoded.shape}, dtype: {decoded.dtype}")
else:
    print("ERROR: Failed to read decoded.png!")
    exit(1)

# 6. recover tensor
decoded = torch.tensor(decoded).float()
recovered = decoded / 255 * 6 - 3
print(f"Recovered tensor shape: {recovered.shape}, dtype: {recovered.dtype}")

print("\nStep 7: Calculate reconstruction error...")
error = torch.mean((tensor - recovered) ** 2).item()
print("Reconstruction Error:", error)