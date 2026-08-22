"""
轨迹平滑与插值工具
- 缺失帧插值
- 异常值检测与修复
- 轨迹平滑
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import median_filter
from typing import Optional, Tuple


def interpolate_missing(
    positions: np.ndarray,
    mask: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    method: str = "cubic",
) -> np.ndarray:
    """
    插值缺失的位姿数据

    Args:
        positions: 位置序列 (N, 3)
        mask: 有效数据掩码 (N,)，True=有效，False=缺失
              为None则自动检测 NaN
        timestamps: 时间戳 (N,)
        method: 'linear', 'cubic', 'spline'

    Returns:
        插值后的位置序列 (N, 3)
    """
    N, d = positions.shape

    if mask is None:
        mask = ~np.any(np.isnan(positions), axis=1)

    if timestamps is None:
        timestamps = np.arange(N)

    result = positions.copy()
    valid_indices = np.where(mask)[0]
    missing_indices = np.where(~mask)[0]

    if len(valid_indices) < 2:
        return result

    for dim in range(d):
        valid_values = positions[valid_indices, dim]

        if method == "linear":
            interp_func = interpolate.interp1d(
                timestamps[valid_indices], valid_values,
                kind="linear", fill_value="extrapolate"
            )
        elif method == "cubic":
            if len(valid_indices) >= 4:
                interp_func = interpolate.interp1d(
                    timestamps[valid_indices], valid_values,
                    kind="cubic", fill_value="extrapolate"
                )
            else:
                interp_func = interpolate.interp1d(
                    timestamps[valid_indices], valid_values,
                    kind="linear", fill_value="extrapolate"
                )
        elif method == "spline":
            if len(valid_indices) >= 4:
                spline = interpolate.make_interp_spline(
                    timestamps[valid_indices], valid_values, k=3
                )
                interp_func = lambda t: spline(t)
            else:
                interp_func = interpolate.interp1d(
                    timestamps[valid_indices], valid_values,
                    kind="linear", fill_value="extrapolate"
                )
        else:
            raise ValueError(f"不支持的插值方法: {method}")

        # 填充缺失值
        for idx in missing_indices:
            result[idx, dim] = float(interp_func(timestamps[idx]))

    return result


def smooth_trajectory(
    positions: np.ndarray,
    method: str = "savgol",
    window_length: int = 11,
    polyorder: int = 3,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    平滑轨迹

    Args:
        positions: 位置序列 (N, 3)
        method: 平滑方法
            'savgol' - Savitzky-Golay 滤波
            'gaussian' - 高斯平滑
            'median' - 中值滤波
            'moving_average' - 滑动平均
        window_length: 窗口长度
        polyorder: 多项式阶数（savgol用）
        sigma: 高斯标准差（gaussian用）

    Returns:
        平滑后的位置 (N, 3)
    """
    result = np.zeros_like(positions)
    N, d = positions.shape

    for dim in range(d):
        col = positions[:, dim]

        if method == "savgol":
            from scipy.signal import savgol_filter
            result[:, dim] = savgol_filter(col, window_length, polyorder)

        elif method == "gaussian":
            from scipy.ndimage import gaussian_filter1d
            result[:, dim] = gaussian_filter1d(col, sigma=sigma)

        elif method == "median":
            result[:, dim] = median_filter(col, size=window_length)

        elif method == "moving_average":
            kernel = np.ones(window_length) / window_length
            result[:, dim] = np.convolve(col, kernel, mode="same")

        else:
            raise ValueError(f"不支持的平滑方法: {method}")

    return result


def detect_outliers(
    positions: np.ndarray,
    threshold_sigma: float = 3.0,
    window_size: int = 10,
) -> np.ndarray:
    """
    检测轨迹中的异常值点

    Args:
        positions: 位置序列 (N, 3)
        threshold_sigma: 异常阈值（标准差倍数）
        window_size: 滑动窗口大小

    Returns:
        异常掩码 (N,) - True=异常
    """
    N, d = positions.shape
    is_outlier = np.zeros(N, dtype=bool)

    for dim in range(d):
        col = positions[:, dim]

        # 用滑动窗口计算局部均值和标准差
        for i in range(N):
            start = max(0, i - window_size // 2)
            end = min(N, i + window_size // 2 + 1)
            local = col[start:end]
            local_mean = np.mean(local)
            local_std = np.std(local)

            if local_std > 0 and abs(col[i] - local_mean) > threshold_sigma * local_std:
                is_outlier[i] = True

    return is_outlier


def fix_outliers(
    positions: np.ndarray,
    outlier_mask: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    检测并修复异常值

    Args:
        positions: 位置序列 (N, 3)
        outlier_mask: 异常掩码，为None则自动检测
        **kwargs: 传递给interpolate_missing的参数

    Returns:
        修复后的位置 (N, 3)
    """
    if outlier_mask is None:
        outlier_mask = detect_outliers(positions)

    # 将异常值设为NaN，然后插值
    result = positions.copy()
    result[outlier_mask] = np.nan

    return interpolate_missing(result, **kwargs)


def compute_trajectory_statistics(
    positions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> dict:
    """
    计算轨迹统计信息

    Args:
        positions: 位置序列 (N, 3)
        timestamps: 时间戳

    Returns:
        统计信息字典
    """
    N = len(positions)
    d = positions.shape[1] if positions.ndim > 1 else 1

    # 路径长度
    path_length = 0.0
    for i in range(1, N):
        path_length += np.linalg.norm(positions[i] - positions[i - 1])

    # 位移
    displacement = np.linalg.norm(positions[-1] - positions[0])

    # 速度统计
    from .velocity_estimator import FiniteDifferenceEstimator
    est = FiniteDifferenceEstimator()
    velocities, accelerations = est.estimate(positions, timestamps)

    speed = np.linalg.norm(velocities, axis=1) if velocities.ndim > 1 else np.abs(velocities)
    acc_mag = np.linalg.norm(accelerations, axis=1) if len(accelerations) > 0 and accelerations.ndim > 1 else np.abs(accelerations)

    return {
        "num_frames": N,
        "path_length": float(path_length),
        "displacement": float(displacement),
        "mean_speed": float(np.mean(speed)) if len(speed) > 0 else 0,
        "max_speed": float(np.max(speed)) if len(speed) > 0 else 0,
        "mean_acceleration": float(np.mean(acc_mag)) if len(acc_mag) > 0 else 0,
        "max_acceleration": float(np.max(acc_mag)) if len(acc_mag) > 0 else 0,
        "velocity_dim": d,
    }
