import torch
import numpy as np
from PIL import Image
import subprocess
import os

def test_frame_mapping():
    print("Testing frame mapping with large tensor...")
    
    tensor = torch.randn(50257, 768)
    print(f"Original tensor shape: {tensor.shape}")
    print(f"Original numel: {tensor.numel()}")
    
    frame_size = 1024
    flat = tensor.flatten().cpu().numpy()
    print(f"Flattened size: {len(flat)}")
    
    padding_len = (frame_size * frame_size - len(flat) % (frame_size * frame_size)) % (frame_size * frame_size)
    print(f"Padding length: {padding_len}")
    
    if padding_len > 0:
        flat = np.pad(flat, (0, padding_len), mode='constant')
    
    num_frames = len(flat) // (frame_size * frame_size)
    print(f"Number of frames: {num_frames}")
    
    frames = flat.reshape(num_frames, frame_size, frame_size)
    print(f"Frames shape: {frames.shape}")
    
    os.makedirs('/tmp/debug_frames', exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame.astype(np.uint8), mode='L')
        img.save(f'/tmp/debug_frames/frame_{i:04d}.png')
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '1',
        '-i', '/tmp/debug_frames/frame_%04d.png',
        '-c:v', 'libx265',
        '-preset', 'slow',
        '-g', '1',
        '-qp', '0',
        '-an',
        '/tmp/debug.hevc'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Encode return code: {result.returncode}")
    if result.returncode != 0:
        print(f"Encode error: {result.stderr}")
    
    os.makedirs('/tmp/debug_decoded', exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', '/tmp/debug.hevc',
        '-vsync', '0',
        '/tmp/debug_decoded/frame_%04d.png'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Decode return code: {result.returncode}")
    
    decoded_frames = []
    i = 0
    while True:
        frame_path = f'/tmp/debug_decoded/frame_{i:04d}.png'
        if os.path.exists(frame_path):
            img = Image.open(frame_path).convert('L')
            decoded_frames.append(np.array(img))
            os.remove(frame_path)
            i += 1
        else:
            break
    
    print(f"Decoded frames count: {len(decoded_frames)}")
    if decoded_frames:
        print(f"Decoded frame shape: {decoded_frames[0].shape}")
        print(f"Total decoded elements: {sum(f.size for f in decoded_frames)}")

if __name__ == "__main__":
    test_frame_mapping()