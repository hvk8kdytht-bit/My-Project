"""直接调试 mask 生成"""
import sys
sys.path.insert(0, "H:/Program")

import numpy as np
from src.utils.mask_utils import generate_object_mask_from_pose_simple

# 模拟一个测试
K = np.array([
    [1066.778, 0, 312.987],
    [0, 1067.487, 241.311],
    [0, 0, 1]
])
img_shape = (480, 640)

# 测试用例：深度 1m，物体尺寸 10cm
R = np.eye(3)
t = np.array([0.0, 0.0, 1.0])  # 正前方 1m
size_3d = np.array([0.1, 0.1, 0.1])  # 10cm 立方体

mask = generate_object_mask_from_pose_simple(R, t, size_3d, K, img_shape)
print(f"正前方 1m, 10cm 立方体:")
print(f"  mask 像素数: {np.sum(mask)}")
print(f"  mask 中心附近: {mask[235:247, 307:319].astype(int)}")

# 测试用例2：深度 1m，偏左
t2 = np.array([-0.08, -0.06, 0.996])
mask2 = generate_object_mask_from_pose_simple(R, t2, size_3d, K, img_shape)
print(f"\n偏左 0.996m:")
print(f"  mask 像素数: {np.sum(mask2)}")

# 找中心
ys, xs = np.where(mask2)
if len(xs) > 0:
    print(f"  mask 范围 x: {xs.min()}-{xs.max()}, y: {ys.min()}-{ys.max()}")
    print(f"  中心约: ({xs.mean():.1f}, {ys.mean():.1f})")

# 手动计算
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
z = t2[2]
cx_pix = cx + t2[0] * fx / z
cy_pix = cy + t2[1] * fy / z
half_w = (size_3d[0] / 2) * fx / z
half_h = (size_3d[1] / 2) * fy / z
print(f"\n手动计算:")
print(f"  中心: ({cx_pix:.1f}, {cy_pix:.1f})")
print(f"  半宽/半高: ({half_w:.1f}, {half_h:.1f}) 像素")
print(f"  预期像素数: {3.14 * half_w * half_h:.0f}")
