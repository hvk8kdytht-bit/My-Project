"""
速度与加速度估计器
从位姿时序估计线速度、角速度、线加速度、角加速度

支持的方法:
1. 有限差分 (FiniteDifference) - 最简单，噪声大
2. 卡尔曼滤波 (KalmanFilter) - 对噪声鲁棒
3. Savitzky-Golay 滤波 - 平滑求导

使用场景:
- 从 YCB-Video 视频序列的位姿估计结果计算速度/加速度
- 从 MuJoCo ground truth 验证估计精度
- 为安全抓取提供物体运动状态预测
"""

import numpy as np
from scipy import signal
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class VelocityEstimator(ABC):
    """速度估计器基类"""

    @abstractmethod
    def estimate(
        self,
        positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计速度和加速度

        Args:
            positions: 位置序列 (N, 3) 或四元数 (N, 4)
            timestamps: 时间戳 (N,)，为None则假设等间隔

        Returns:
            velocity: 速度 (N-1, 3) 或 (N-1, 3) 角速度
            acceleration: 加速度 (N-2, 3)
        """
        pass


class FiniteDifferenceEstimator(VelocityEstimator):
    """
    有限差分速度估计
    v[i] = (x[i+1] - x[i]) / dt
    a[i] = (v[i+1] - v[i]) / dt
    """

    def __init__(self, order: int = 1):
        """
        Args:
            order: 差分阶数 (1=前向差分, 2=中心差分)
        """
        self.order = order

    def estimate(
        self,
        positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(positions)
        if N < 2:
            raise ValueError("至少需要2个点计算速度")

        if timestamps is None:
            dt = 1.0
            dts = np.ones(N - 1)
        else:
            dts = np.diff(timestamps)
            dt = np.mean(dts)

        if self.order == 1:
            # 前向差分
            velocities = np.diff(positions, axis=0) / dts[:, np.newaxis]
        else:
            # 中心差分（端点用前向/后向）
            velocities = np.zeros_like(positions)
            velocities[1:-1] = (positions[2:] - positions[:-2]) / (dts[1:] + dts[:-1])[:, np.newaxis] * 2
            velocities[0] = (positions[1] - positions[0]) / dts[0]
            velocities[-1] = (positions[-1] - positions[-2]) / dts[-1]

        # 加速度
        if N < 3:
            accelerations = np.zeros((0, positions.shape[1]))
        else:
            if timestamps is None:
                acc_dts = dt
            else:
                acc_dts = (dts[1:] + dts[:-1]) / 2.0
            accelerations = np.diff(velocities[:len(dts)], axis=0) / acc_dts[:, np.newaxis]

        return velocities[:len(dts)], accelerations


class KalmanFilterEstimator(VelocityEstimator):
    """
    卡尔曼滤波速度/加速度估计
    状态: [位置, 速度, 加速度]
    观测: 位置

    对噪声较大的位姿估计结果效果更好。
    """

    def __init__(
        self,
        dim: int = 3,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_velocity_variance: float = 1.0,
        initial_acceleration_variance: float = 1.0,
    ):
        """
        Args:
            dim: 维度（通常3维平移）
            process_noise: 过程噪声方差
            measurement_noise: 测量噪声方差
            initial_velocity_variance: 初始速度不确定性
            initial_acceleration_variance: 初始加速度不确定性
        """
        self.dim = dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.init_vel_var = initial_velocity_variance
        self.init_acc_var = initial_acceleration_variance

    def estimate(
        self,
        positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(positions)
        d = self.dim

        if timestamps is None:
            dts = np.ones(N - 1)
        else:
            dts = np.diff(timestamps)

        # 状态向量: [pos_1...pos_d, vel_1...vel_d, acc_1...acc_d]
        state_size = 3 * d

        # 初始状态
        x = np.zeros(state_size)
        x[:d] = positions[0]

        # 初始协方差
        P = np.eye(state_size)
        P[:d, :d] *= self.measurement_noise
        P[d:2*d, d:2*d] *= self.init_vel_var
        P[2*d:, 2*d:] *= self.init_acc_var

        # 测量矩阵（只观测位置）
        H = np.zeros((d, state_size))
        H[:, :d] = np.eye(d)

        # 测量噪声
        R = np.eye(d) * self.measurement_noise

        velocities = np.zeros((N, d))
        accelerations = np.zeros((N, d))

        for i in range(N):
            if i > 0:
                dt = dts[i - 1]
                # 状态转移矩阵（常加速度模型）
                F = np.eye(state_size)
                F[:d, d:2*d] = np.eye(d) * dt
                F[:d, 2*d:] = np.eye(d) * 0.5 * dt**2
                F[d:2*d, 2*d:] = np.eye(d) * dt

                # 过程噪声
                Q = np.eye(state_size) * self.process_noise
                Q[:d, :d] *= dt**4 / 4
                Q[d:2*d, d:2*d] *= dt**2
                Q[2*d:, 2*d:] *= 1.0

                # 预测
                x = F @ x
                P = F @ P @ F.T + Q

            # 更新
            z = positions[i]
            y = z - H @ x  # 创新
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(state_size) - K @ H) @ P

            # 保存
            velocities[i] = x[d:2*d]
            accelerations[i] = x[2*d:]

        return velocities, accelerations


class SavGolEstimator(VelocityEstimator):
    """
    Savitzky-Golay 滤波速度估计
    先用多项式拟合平滑轨迹，再求导得到速度和加速度

    优点:
    - 有效去除噪声
    - 保留信号的形状和高度
    - 可以直接求各阶导数
    """

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 3,
    ):
        """
        Args:
            window_length: 窗口长度（奇数，越大越平滑）
            polyorder: 多项式阶数（通常2-4）
        """
        assert window_length % 2 == 1, "window_length 必须是奇数"
        assert polyorder < window_length, "polyorder 必须小于 window_length"
        self.window_length = window_length
        self.polyorder = polyorder

    def estimate(
        self,
        positions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(positions)
        d = positions.shape[1] if positions.ndim > 1 else 1

        if timestamps is None:
            dt = 1.0
        else:
            dt = np.mean(np.diff(timestamps))

        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)

        for i in range(d):
            pos_col = positions[:, i] if positions.ndim > 1 else positions

            # 平滑后的位置（0阶导数）
            smoothed = signal.savgol_filter(
                pos_col, self.window_length, self.polyorder, deriv=0, delta=dt
            )

            # 一阶导数 = 速度
            vel = signal.savgol_filter(
                pos_col, self.window_length, self.polyorder, deriv=1, delta=dt
            )

            # 二阶导数 = 加速度
            acc = signal.savgol_filter(
                pos_col, self.window_length, self.polyorder, deriv=2, delta=dt
            )

            if positions.ndim > 1:
                velocities[:, i] = vel
                accelerations[:, i] = acc
            else:
                velocities = vel
                accelerations = acc

        return velocities, accelerations


def estimate_angular_velocity(
    quaternions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    method: str = "finite_difference",
) -> np.ndarray:
    """
    从四元数序列估计角速度

    Args:
        quaternions: 四元数序列 (N, 4) - wxyz
        timestamps: 时间戳 (N,)
        method: 'finite_difference' 或 'savgol'

    Returns:
        角速度序列 (N-1, 3) - rad/s
    """
    N = len(quaternions)
    angular_velocities = []

    for i in range(1, N):
        q1 = quaternions[i - 1]
        q2 = quaternions[i]

        # 相对旋转: q_rel = q2 * q1^{-1}
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        # q1 的逆
        q1_inv = np.array([w1, -x1, -y1, -z1])

        # q2 * q1_inv
        q_rel = quaternion_multiply(q2, q1_inv)

        # 旋转角度
        angle = 2 * np.arccos(np.clip(q_rel[0], -1, 1))

        if abs(angle) < 1e-10:
            omega = np.zeros(3)
        else:
            # 旋转轴
            axis = q_rel[1:] / np.sin(angle / 2)
            # 角速度
            dt = 1.0 if timestamps is None else timestamps[i] - timestamps[i - 1]
            omega = axis * angle / dt

        angular_velocities.append(omega)

    return np.array(angular_velocities)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 q1 * q2 (wxyz)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])
