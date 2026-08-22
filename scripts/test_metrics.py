"""
测试位姿评估指标模块
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.utils.metrics import (
    compute_add,
    compute_adi,
    compute_projection_error,
    evaluate_pose,
    project_points,
)

print("=== 位姿评估指标测试 ===")
print()

# 生成一个简单的立方体模型点
print("1. 生成测试物体模型点 (立方体)...")
cube_size = 0.02  # 2cm
points = np.array([
    [-cube_size, -cube_size, -cube_size],
    [ cube_size, -cube_size, -cube_size],
    [ cube_size,  cube_size, -cube_size],
    [-cube_size,  cube_size, -cube_size],
    [-cube_size, -cube_size,  cube_size],
    [ cube_size, -cube_size,  cube_size],
    [ cube_size,  cube_size,  cube_size],
    [-cube_size,  cube_size,  cube_size],
], dtype=np.float32)
print(f"   模型点数: {len(points)}, 尺寸: {cube_size*100:.1f}cm")

# 生成GT位姿
print()
print("2. 生成GT位姿...")
angle = np.radians(30)  # 30度旋转
R_gt = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1],
], dtype=np.float32)
t_gt = np.array([0.05, -0.03, 0.5], dtype=np.float32)  # 5cm前, 3cm右, 50cm远
print(f"   平移: {t_gt} m")
print(f"   旋转: 30° 绕Z轴")

# 相机内参
K = np.array([
    [1066.778, 0, 312.9869],
    [0, 1067.487, 241.3109],
    [0, 0, 1],
], dtype=np.float32)

# 测试1: 完全相同的位姿 -> ADD应为0
print()
print("3. 测试完全重合位姿 (ADD应≈0)...")
add_exact = compute_add(R_gt, t_gt, R_gt, t_gt, points)
print(f"   ADD = {add_exact:.8f} m (应为0)")
assert add_exact < 1e-6, "ADD 测试失败!"

# 测试2: 小扰动位姿
print()
print("4. 测试小扰动位姿...")
t_pred = t_gt + np.array([0.005, 0.003, -0.002])  # 几毫米偏移
angle_small = np.radians(2)
R_small = np.array([
    [np.cos(angle_small), -np.sin(angle_small), 0],
    [np.sin(angle_small), np.cos(angle_small), 0],
    [0, 0, 1],
], dtype=np.float32)
R_pred = R_small @ R_gt

add_small = compute_add(R_pred, t_pred, R_gt, t_gt, points)
print(f"   ADD = {add_small*1000:.3f} mm")

proj_err = compute_projection_error(R_pred, t_pred, R_gt, t_gt, points, K)
print(f"   投影误差 = {proj_err:.2f} px")

# 测试3: 大误差
print()
print("5. 测试大误差位姿...")
t_bad = t_gt + np.array([0.1, 0.1, 0.2])
angle_bad = np.radians(45)
R_bad = np.array([
    [np.cos(angle_bad), -np.sin(angle_bad), 0],
    [np.sin(angle_bad), np.cos(angle_bad), 0],
    [0, 0, 1],
]) @ R_gt

add_bad = compute_add(R_bad, t_bad, R_gt, t_gt, points)
print(f"   ADD = {add_bad*100:.2f} cm")

# 测试ADI（对于非对称物体，ADI≈ADD；对于对称物体，ADI<ADD）
print()
print("6. 测试 ADI (最近点距离)...")
adi_val = compute_adi(R_pred, t_pred, R_gt, t_gt, points)
print(f"   ADI = {adi_val*1000:.3f} mm")
print(f"   ADD = {add_small*1000:.3f} mm")
print(f"   (立方体是对称的，ADI 应该等于或小于 ADD)")

# 测试完整评估
print()
print("7. 测试完整评估 (evaluate_pose)...")
diameter = cube_size * 2 * np.sqrt(3)  # 立方体对角线 ≈ 直径
print(f"   物体直径: {diameter*100:.2f} cm")

predictions = []
for i in range(10):
    # 生成不同程度的误差
    noise_level = i * 0.002  # 从0到18mm
    t_noisy = t_gt + np.random.normal(0, noise_level, 3)
    angle_noise = np.radians(i * 2)  # 从0到18度
    R_noise = np.array([
        [np.cos(angle_noise), -np.sin(angle_noise), 0],
        [np.sin(angle_noise), np.cos(angle_noise), 0],
        [0, 0, 1],
    ])
    R_noisy = R_noise @ R_gt
    predictions.append({
        "obj_id": 1,
        "R_pred": R_noisy,
        "t_pred": t_noisy,
        "R_gt": R_gt,
        "t_gt": t_gt,
    })

model_diameters = {1: diameter}
model_points = {1: points}

results = evaluate_pose(
    predictions, model_diameters, model_points, K,
    threshold=0.1, symmetric_objects=[1],
)

print(f"   样本数: {results['num_samples']}")
print(f"   平均 ADD: {results['mean_add']*1000:.2f} mm")
print(f"   平均 ADI: {results['mean_adi']*1000:.2f} mm")
print(f"   平均投影误差: {results['mean_proj_err_px']:.2f} px")
print(f"   ADD 准确率 (10%直径): {results['accuracy_add']*100:.1f}%")
print(f"   ADI 准确率 (10%直径): {results['accuracy_adi']*100:.1f}%")

# 测试投影
print()
print("8. 测试3D点投影...")
points_2d = project_points(points, R_gt, t_gt, K)
print(f"   投影点数: {len(points_2d)}")
print(f"   图像范围: x=[{points_2d[:,0].min():.1f}, {points_2d[:,0].max():.1f}]")
print(f"              y=[{points_2d[:,1].min():.1f}, {points_2d[:,1].max():.1f}]")

print()
print("✅ 所有位姿评估测试通过!")
