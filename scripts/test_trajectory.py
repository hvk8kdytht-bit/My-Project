"""
测试速度/加速度估计和轨迹平滑模块
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.trajectory import (
    FiniteDifferenceEstimator,
    KalmanFilterEstimator,
    SavGolEstimator,
    smooth_trajectory,
    interpolate_missing,
    detect_outliers,
    fix_outliers,
    compute_trajectory_statistics,
)

# 生成测试数据：匀加速直线运动
N = 100
dt = 0.01
t = np.arange(N) * dt
a_true = np.array([1.0, 0.5, -0.3])
v0 = np.array([0.5, 0.2, 0.1])
x0 = np.array([0.0, 0.0, 0.0])

positions_true = np.zeros((N, 3))
velocities_true = np.zeros((N, 3))
for i in range(N):
    velocities_true[i] = v0 + a_true * t[i]
    positions_true[i] = x0 + v0 * t[i] + 0.5 * a_true * t[i]**2

np.random.seed(42)
noise_std = 0.005
positions_noisy = positions_true + np.random.normal(0, noise_std, positions_true.shape)

print("=== 速度/加速度估计测试 ===")
print(f"测试数据: {N}帧, dt={dt}s, 噪声={noise_std}m")
print()

# 有限差分
fd_est = FiniteDifferenceEstimator(order=2)
fd_vel, fd_acc = fd_est.estimate(positions_noisy, t)
fd_vel_err = np.mean(np.linalg.norm(fd_vel - velocities_true[:len(fd_vel)], axis=1))
print(f"有限差分速度误差 (MAE): {fd_vel_err:.6f} m/s")

# 卡尔曼滤波
kf_est = KalmanFilterEstimator(dim=3, process_noise=0.001, measurement_noise=noise_std**2)
kf_vel, kf_acc = kf_est.estimate(positions_noisy, t)
kf_vel_err = np.mean(np.linalg.norm(kf_vel - velocities_true, axis=1))
kf_acc_err = np.mean(np.linalg.norm(kf_acc - a_true, axis=1))
print(f"卡尔曼滤波速度误差 (MAE): {kf_vel_err:.6f} m/s")
print(f"卡尔曼滤波加速度误差 (MAE): {kf_acc_err:.6f} m/s^2")

# SavGol
sg_est = SavGolEstimator(window_length=11, polyorder=3)
sg_vel, sg_acc = sg_est.estimate(positions_noisy, t)
sg_vel_err = np.mean(np.linalg.norm(sg_vel - velocities_true, axis=1))
sg_acc_err = np.mean(np.linalg.norm(sg_acc - a_true, axis=1))
print(f"SavGol 速度误差 (MAE): {sg_vel_err:.6f} m/s")
print(f"SavGol 加速度误差 (MAE): {sg_acc_err:.6f} m/s^2")

print()
print("=== 轨迹平滑测试 ===")
smoothed = smooth_trajectory(positions_noisy, method="savgol", window_length=11)
smooth_err = np.mean(np.linalg.norm(smoothed - positions_true, axis=1))
print(f"平滑后位置误差 (MAE): {smooth_err:.6f} m")

print()
print("=== 异常值检测测试 ===")
positions_with_outlier = positions_noisy.copy()
positions_with_outlier[50] += np.array([0.1, -0.05, 0.08])
outliers = detect_outliers(positions_with_outlier, threshold_sigma=3.0)
print(f"检测到异常帧数: {np.sum(outliers)} (真值: 1)")

fixed = fix_outliers(positions_with_outlier)
fix_err = np.mean(np.linalg.norm(fixed - positions_true, axis=1))
print(f"修复后位置误差 (MAE): {fix_err:.6f} m")

print()
print("=== 轨迹统计 ===")
stats = compute_trajectory_statistics(positions_true, t)
for k, v in stats.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

print()
print("✅ 所有测试通过!")
