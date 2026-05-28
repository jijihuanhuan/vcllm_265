
import torch

# 问题数据
test_tensor = torch.tensor([6.9180, 6.5430, 6.1680, 6.6680, 6.8281, 6.7031, 7.0234, 6.6172, 6.4766, 6.4570], dtype=torch.float16)
print("test_tensor:", test_tensor)

min_val = 4.484375  # 来自调试输出
max_val = 8.406250
num_bits = 8
print(f"\nmin_val: {min_val}")
print(f"max_val: {max_val}")

# --- 手动计算 RTN，找出问题 ---
print("\n--- Step by step calculation ---")
range_val = max_val - min_val
print(f"range_val: {range_val}")

scale = range_val / (2**num_bits - 1)
print(f"scale: {scale}")

zero_point = int(round(-min_val / scale))
print(f"zero_point: {zero_point}")

# 量化
x_q = torch.clamp(torch.round(test_tensor / scale + zero_point), 0, 2**num_bits - 1)
print(f"x_q: {x_q}")

# 反量化
x_dq = (x_q.float() - zero_point) * scale
print(f"x_dq (wrong): {x_dq}")
print(f"diff wrong: {torch.abs(test_tensor - x_dq)}")

# --- 正确的方法！---
print("\n--- Correct approach (symmetric/asymmetric but simpler) ---")
# 应该是：x_q = clamp(round( (x - min_val) / scale ), 0, 255)
# 然后 x_dq = x_q * scale + min_val
scale2 = range_val / 255
print(f"scale2: {scale2}")
x_q2 = torch.clamp(torch.round((test_tensor - min_val)/scale2), 0, 255).to(torch.uint8)
print(f"x_q2: {x_q2}")
x_dq2 = x_q2.float() * scale2 + min_val
print(f"x_dq2 (correct!): {x_dq2}")
print(f"diff correct: {torch.abs(test_tensor - x_dq2)}")
