import torch
import torch.nn as nn
from compression.rtn import rtn_quantize_tensor, rtn_dequantize_tensor
from codec.codec_job import CodecJob
from codec.frame_mapper import tensor_to_frames, frames_to_tensor
from codec.hevc_backend import encode_frames_to_bitstream, decode_bitstream_to_frames

class KVCacheCompressionHook:
    def __init__(self, qp=0, lossless=True, enabled=True):
        self.qp = qp
        self.lossless = lossless
        self.enabled = enabled
        self.compress_cache = {}
    
    def compress_kv_cache(self, key_states, value_states):
        if not self.enabled:
            return key_states, value_states
        
        compressed_key = None
        compressed_value = None
        
        if key_states is not None:
            compressed_key = self._compress_tensor(key_states)
        
        if value_states is not None:
            compressed_value = self._compress_tensor(value_states)
        
        return compressed_key, compressed_value
    
    def decompress_kv_cache(self, compressed_key, compressed_value):
        if not self.enabled:
            return compressed_key, compressed_value
        
        key_states = None
        value_states = None
        
        if compressed_key is not None:
            key_states = self._decompress_tensor(compressed_key)
        
        if compressed_value is not None:
            value_states = self._decompress_tensor(compressed_value)
        
        return key_states, value_states
    
    def _compress_tensor(self, tensor):
        tensor_cpu = tensor.cpu()
        
        min_val = tensor_cpu.min().item()
        max_val = tensor_cpu.max().item()
        
        quantized, rtn_metadata = rtn_quantize_tensor(tensor_cpu, min_val=min_val, max_val=max_val)
        frames, frame_metadata = tensor_to_frames(quantized)

        codec_job = CodecJob.square(
            frame_metadata["frame_size"],
            qp=self.qp,
            lossless=self.lossless,
            intra_only=True,
            backend="auto",
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.hevc', delete=False) as f:
            bitstream_path = f.name

        encode_frames_to_bitstream(frames, bitstream_path, codec_job)
        
        with open(bitstream_path, 'rb') as f:
            bitstream_data = f.read()
        
        import os
        os.remove(bitstream_path)
        
        metadata = {
            'rtn_metadata': rtn_metadata,
            'frame_metadata': frame_metadata,
            'codec_job': codec_job.to_dict(),
            'original_device': str(tensor.device),
            'original_dtype': str(tensor.dtype)
        }
        
        return (bitstream_data, metadata)
    
    def _decompress_tensor(self, compressed_data):
        bitstream_data, metadata = compressed_data
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.hevc', delete=False) as f:
            f.write(bitstream_data)
            bitstream_path = f.name
        
        job = CodecJob.from_dict(metadata['codec_job'])
        frames = decode_bitstream_to_frames(bitstream_path, job)
        
        import os
        os.remove(bitstream_path)
        
        quantized = frames_to_tensor(frames, metadata['frame_metadata'])
        decompressed = rtn_dequantize_tensor(quantized, metadata['rtn_metadata'])
        
        device = torch.device(metadata['original_device'])
        decompressed = decompressed.to(device)
        
        return decompressed

def install_kv_cache_hook(model, qp=0, lossless=True):
    hook = KVCacheCompressionHook(qp=qp, lossless=lossless)
    
    def forward_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 3:
            hidden_states, past_key_values = output[0], output[1]
            
            if past_key_values is not None:
                new_past_key_values = []
                for layer_past in past_key_values:
                    key_states, value_states = layer_past
                    compressed_key, compressed_value = hook.compress_kv_cache(key_states, value_states)
                    new_past_key_values.append((compressed_key, compressed_value))
                
                return (hidden_states, tuple(new_past_key_values)) + output[2:]
        
        return output
    
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'decoder' in name.lower():
            module.register_forward_hook(forward_hook)
    
    return hook