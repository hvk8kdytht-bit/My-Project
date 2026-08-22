"""
方案B: 光流法速度估计 Baseline
直接从连续两帧的RGB图像计算光流，
结合深度图和相机内参反投影到3D，得到物体速度估计。

适用场景: 无位姿模型时的纯视觉速度估计
输入: 连续两帧RGB + 深度图 + 物体掩码
输出: 物体线速度（米/秒）
"""
import numpy as np
import cv2
from typing import Dict, Optional, Tuple


class OpticalFlowVelocityEstimator:
    """
    光流法速度估计器

    方法:
    1. 对连续两帧在掩码区域内计算稠密光流（Farneback）
    2. 将像素光流通过深度和内参反投影到3D空间
    3. 取掩码内所有点的中位速度作为物体速度估计

    优点: 无需训练，纯几何方法
    缺点: 对光照变化敏感，低纹理区域不准，无法估计旋转
    """

    def __init__(
        self,
        method: str = "farneback",
        pyramid_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
    ):
        self.method = method
        self.pyramid_scale = pyramid_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations

    def estimate(
        self,
        rgb_prev: np.ndarray,
        rgb_curr: np.ndarray,
        depth_prev: np.ndarray,
        depth_curr: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        dt: float,
    ) -> Dict:
        """
        估计物体速度

        Args:
            rgb_prev: 前一帧 RGB (H,W,3) uint8
            rgb_curr: 当前帧 RGB (H,W,3) uint8
            depth_prev: 前一帧深度 (H,W) 米 float32
            depth_curr: 当前帧深度 (H,W) 米 float32
            mask: 物体掩码 (H,W) bool
            K: 相机内参 (3,3)
            dt: 两帧时间间隔（秒）

        Returns:
            dict: {
                'linear_velocity': (3,) - 相机系下的线速度估计 (m/s)
                'flow_median': (2,) - 像素光流中位数 (px)
                'confidence': float - 估计置信度 (0-1)
            }
        """
        if mask.sum() < 10:
            return {
                "linear_velocity": np.zeros(3, dtype=np.float32),
                "flow_median": np.zeros(2, dtype=np.float32),
                "confidence": 0.0,
            }

        gray_prev = cv2.cvtColor(rgb_prev, cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(rgb_curr, cv2.COLOR_RGB2GRAY)

        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            self.pyramid_scale, self.levels, self.winsize,
            self.iterations, 5, 1.2, 0,
        )  # (H, W, 2): dx, dy

        # 只取掩码区域内的光流
        masked_flow = flow[mask]  # (N, 2)
        masked_depth = depth_prev[mask]  # (N,)

        # 过滤无效深度
        valid = masked_depth > 0.01
        if valid.sum() < 10:
            return {
                "linear_velocity": np.zeros(3, dtype=np.float32),
                "flow_median": np.zeros(2, dtype=np.float32),
                "confidence": 0.0,
            }

        flow_valid = masked_flow[valid]
        depth_valid = masked_depth[valid]

        # 像素光流中位数
        flow_median = np.median(flow_valid, axis=0)

        # 反投影到3D: 用深度和内参将像素位移转为3D位移
        # 获取掩码区域的像素坐标
        ys, xs = np.where(mask)
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 前一帧的3D点
        X_prev = (xs - cx) * depth_valid / fx
        Y_prev = (ys - cy) * depth_valid / fy
        Z_prev = depth_valid

        # 当前帧的像素位置（光流偏移后）
        xs_curr = xs + flow_valid[:, 0]
        ys_curr = ys + flow_valid[:, 1]

        # 用当前帧深度反投影
        X_curr = (xs_curr - cx) * depth_valid / fx
        Y_curr = (ys_curr - cy) * depth_valid / fy
        Z_curr = depth_valid  # 近似: 用前一帧深度

        # 3D位移
        dX = X_curr - X_prev
        dY = Y_curr - Y_prev
        dZ = Z_curr - Z_prev

        # 中位数速度（鲁棒）
        vel_x = np.median(dX) / dt
        vel_y = np.median(dY) / dt
        vel_z = np.median(dZ) / dt

        # 置信度: 光流一致性（IQR越小置信度越高）
        flow_mag = np.linalg.norm(flow_valid, axis=1)
        iqr = np.percentile(flow_mag, 75) - np.percentile(flow_mag, 25)
        median_flow = np.median(flow_mag)
        confidence = float(np.clip(1.0 - iqr / max(median_flow * 2, 0.1), 0, 1))

        return {
            "linear_velocity": np.array([vel_x, vel_y, vel_z], dtype=np.float32),
            "flow_median": flow_median.astype(np.float32),
            "confidence": confidence,
        }


class LucasKanadeVelocityEstimator:
    """
    Lucas-Kanade 稀疏光流速度估计

    在掩码内检测角点，跟踪特征点运动，反投影到3D估计速度。
    比稠密光流更快，且对纹理少的区域更鲁棒。
    """

    def __init__(
        self,
        max_corners: int = 100,
        quality_level: float = 0.01,
        min_distance: int = 10,
        window_size: int = 21,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.window_size = window_size

    def estimate(
        self,
        rgb_prev: np.ndarray,
        rgb_curr: np.ndarray,
        depth_prev: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        dt: float,
    ) -> Dict:
        gray_prev = cv2.cvtColor(rgb_prev, cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(rgb_curr, cv2.COLOR_RGB2GRAY)

        # 在掩码内检测角点
        mask8u = (mask.astype(np.uint8) * 255)
        pts = cv2.goodFeaturesToTrack(
            gray_prev, self.max_corners,
            self.quality_level, self.min_distance,
            mask=mask8u,
        )

        if pts is None or len(pts) < 5:
            return {
                "linear_velocity": np.zeros(3, dtype=np.float32),
                "num_points": 0,
                "confidence": 0.0,
            }

        # LK 光流跟踪
        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, pts, None,
            winSize=(self.window_size, self.window_size),
            maxLevel=3,
        )

        good = status.ravel() == 1
        if good.sum() < 3:
            return {
                "linear_velocity": np.zeros(3, dtype=np.float32),
                "num_points": int(good.sum()),
                "confidence": 0.0,
            }

        pts_prev = pts[good].reshape(-1, 2)
        pts_curr = pts_next[good].reshape(-1, 2)

        # 反投影到3D
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        depths = np.array([depth_prev[int(p[1]), int(p[0])] for p in pts_prev])
        valid = depths > 0.01
        if valid.sum() < 3:
            return {
                "linear_velocity": np.zeros(3, dtype=np.float32),
                "num_points": int(valid.sum()),
                "confidence": 0.0,
            }

        pts_prev = pts_prev[valid]
        pts_curr = pts_curr[valid]
        depths = depths[valid]

        X_prev = (pts_prev[:, 0] - cx) * depths / fx
        Y_prev = (pts_prev[:, 1] - cy) * depths / fy
        Z_prev = depths

        X_curr = (pts_curr[:, 0] - cx) * depths / fx
        Y_curr = (pts_curr[:, 1] - cy) * depths / fy
        Z_curr = depths

        vel_x = np.median(X_curr - X_prev) / dt
        vel_y = np.median(Y_curr - Y_prev) / dt
        vel_z = np.median(Z_curr - Z_prev) / dt

        confidence = float(np.clip(valid.sum() / self.max_corners, 0, 1))

        return {
            "linear_velocity": np.array([vel_x, vel_y, vel_z], dtype=np.float32),
            "num_points": int(valid.sum()),
            "confidence": confidence,
        }
