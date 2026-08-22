"""
测试可视化工具（纯 numpy，不依赖 torch）
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from PIL import Image
import cv2
from src.utils.visualization import (
    visualize_bbox,
    visualize_pose_2d,
    create_pose_overlay,
)
from src.utils.metrics import project_points

print("=== 可视化工具测试 ===")
print()

# 生成测试图像
print("1. 生成测试图像...")
width, height = 640, 480
img = Image.new("RGB", (width, height), (200, 200, 200))

# 相机内参
K = np.array([
    [1066.778, 0, 312.9869],
    [0, 1067.487, 241.3109],
    [0, 0, 1],
])

# 物体位姿
angle = np.radians(30)
R = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1],
])
t = np.array([0.05, -0.03, 0.5])

# 立方体模型点
cube_size = 0.02
model_points = np.array([
    [-cube_size, -cube_size, -cube_size],
    [ cube_size, -cube_size, -cube_size],
    [ cube_size,  cube_size, -cube_size],
    [-cube_size,  cube_size, -cube_size],
    [-cube_size, -cube_size,  cube_size],
    [ cube_size, -cube_size,  cube_size],
    [ cube_size,  cube_size,  cube_size],
    [-cube_size,  cube_size,  cube_size],
])

# 测试1: 画BBox
print("2. 测试 visualize_bbox...")
bbox_2d = project_points(model_points, R, t, K)
x1, y1 = bbox_2d.min(axis=0)
x2, y2 = bbox_2d.max(axis=0)
img_bbox = visualize_bbox(
    img.copy(),
    np.array([x1, y1, x2 - x1, y2 - y1]),
    color=(255, 0, 0),
    label="obj_000001",
)
img_bbox.save("h:/Program/output/test_bbox.png")
print(f"   保存到 output/test_bbox.png")

# 测试2: 画2D位姿（3D点投影）
print("3. 测试 visualize_pose_2d...")
img_pose = visualize_pose_2d(
    img.copy(),
    model_points,
    R, t, K,
    color=(0, 255, 0),
    point_size=3,
)
img_pose.save("h:/Program/output/test_pose_2d.png")
print(f"   保存到 output/test_pose_2d.png")

# 测试3: Pose overlay（预测vs GT对比）
print("4. 测试 create_pose_overlay...")
# 生成预测位姿（带扰动）
t_pred = t + np.array([0.005, 0.003, 0])
pred_2d = project_points(model_points, R, t_pred, K)
gt_2d = project_points(model_points, R, t, K)

img_overlay = create_pose_overlay(
    np.array(img),
    pred_points_2d=pred_2d,
    gt_points_2d=gt_2d,
)
img_overlay.save("h:/Program/output/test_pose_overlay.png")
print(f"   保存到 output/test_pose_overlay.png")

# 测试4: 深度图可视化
print("5. 深度图可视化测试...")
depth = np.zeros((height, width), dtype=np.float32)
for y in range(height):
    for x in range(width):
        dx = (x - 320) / 100
        dy = (y - 240) / 100
        depth[y, x] = 0.5 + 0.1 * (dx**2 + dy**2)

depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite("h:/Program/output/test_depth.png", depth_colored)
print(f"   保存到 output/test_depth.png")

# 测试5: 综合可视化
print("6. 综合可视化测试 (bbox + pose points)...")
img_combo = img.copy()
img_combo = visualize_pose_2d(
    img_combo,
    model_points,
    R, t, K,
    color=(255, 0, 0),
    point_size=4,
)
img_combo = visualize_bbox(
    img_combo,
    np.array([x1, y1, x2 - x1, y2 - y1]),
    color=(0, 0, 255),
    label="cube (2cm)",
)
img_combo.save("h:/Program/output/test_combo.png")
print(f"   保存到 output/test_combo.png")

print()
print("✅ 所有可视化测试通过!")
print(f"   结果保存在 h:/Program/output/")
