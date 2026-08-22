"""
CoTracker 滑移检测器
通过跟踪物体和夹爪上的点，检测相对运动判断滑移
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class CoTrackerSlipDetector:
    """基于 CoTracker 的滑移检测
    
    同时跟踪物体和夹爪上的点，通过相对运动检测滑移
    
    Args:
        slip_threshold_mm: 滑移阈值 (mm)，超过则判定为滑移
        velocity_threshold: 相对速度阈值 (mm/s)
        min_slip_duration: 最小滑移持续帧数
        num_object_points: 物体上跟踪点数
        num_gripper_points: 夹爪上跟踪点数
    """
    
    def __init__(
        self,
        slip_threshold_mm: float = 1.0,
        velocity_threshold: float = 5.0,
        min_slip_duration: int = 3,
        num_object_points: int = 100,
        num_gripper_points: int = 50,
        fps: float = 30.0,
        device: str = "cuda",
    ):
        self.slip_threshold_mm = slip_threshold_mm
        self.velocity_threshold = velocity_threshold
        self.min_slip_duration = min_slip_duration
        self.num_object_points = num_object_points
        self.num_gripper_points = num_gripper_points
        self.fps = fps
        self.dt = 1.0 / fps
        self.device = device
        self._estimator = None
    
    def detect_slip_sequence(
        self,
        video_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
        object_mask: np.ndarray,
        gripper_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """检测视频序列中的滑移
        
        Args:
            video_frames: RGB 帧列表 (H, W, 3) uint8
            depth_frames: 深度图列表 (H, W) float32，米
            object_mask: 第一帧物体 mask (H, W) bool
            gripper_mask: 第一帧夹爪 mask (H, W) bool
            camera_intrinsics: 相机内参 (3, 3)
            
        Returns:
            包含滑移检测结果的字典
        """
        from src.velocity.cotracker_estimator import CoTrackerVelocityEstimator
        
        if self._estimator is None:
            self._estimator = CoTrackerVelocityEstimator(
                device=self.device,
                num_points=max(self.num_object_points, self.num_gripper_points),
                fps=self.fps,
            )
        
        T = len(video_frames)
        
        # 1. 跟踪物体上的点
        obj_result = self._estimator.estimate_velocity_sequence(
            video_frames, depth_frames, object_mask, camera_intrinsics
        )
        obj_traj_3d = obj_result["point_trajectories_3d"]  # (T, N_obj, 3)
        
        # 2. 跟踪夹爪上的点
        gripper_result = self._estimator.estimate_velocity_sequence(
            video_frames, depth_frames, gripper_mask, camera_intrinsics
        )
        gripper_traj_3d = gripper_result["point_trajectories_3d"]  # (T, N_grip, 3)
        
        # 3. 计算物体相对夹爪的位移
        relative_displacement = self._compute_relative_displacement(
            obj_traj_3d, gripper_traj_3d
        )  # (T, 3) 米
        
        # 4. 计算相对速度
        relative_velocity = self._compute_relative_velocity(relative_displacement)
        
        # 5. 滑移检测
        slip_detected = np.zeros(T, dtype=bool)
        slip_magnitude_mm = np.linalg.norm(relative_displacement * 1000, axis=1)
        rel_vel_magnitude_mms = np.linalg.norm(relative_velocity * 1000, axis=1)
        
        # 累积位移超过阈值 或 相对速度超过阈值
        for t in range(T):
            if (slip_magnitude_mm[t] > self.slip_threshold_mm or 
                rel_vel_magnitude_mms[t] > self.velocity_threshold):
                slip_detected[t] = True
        
        # 6. 形态学平滑：最小持续时长过滤
        slip_detected = self._filter_min_duration(slip_detected)
        
        return {
            "slip_detected": slip_detected,
            "relative_displacement_mm": relative_displacement * 1000,
            "relative_velocity_mms": relative_velocity * 1000,
            "slip_magnitude_mm": slip_magnitude_mm,
            "relative_velocity_magnitude_mms": rel_vel_magnitude_mms,
            "object_trajectories": obj_traj_3d,
            "gripper_trajectories": gripper_traj_3d,
        }
    
    def _compute_relative_displacement(
        self,
        obj_traj: np.ndarray,
        gripper_traj: np.ndarray,
    ) -> np.ndarray:
        """计算物体相对夹爪的位移
        
        用各自点的中位数代表整体，相减得到相对位移
        
        Args:
            obj_traj: (T, N_obj, 3) 物体点 3D 轨迹
            gripper_traj: (T, N_grip, 3) 夹爪点 3D 轨迹
            
        Returns:
            relative_disp: (T, 3) 相对位移（米）
        """
        T = obj_traj.shape[0]
        relative_disp = np.zeros((T, 3), dtype=np.float32)
        
        # 每帧取所有可见点的中位数作为代表
        for t in range(T):
            obj_pts = obj_traj[t]  # (N, 3)
            grip_pts = gripper_traj[t]
            
            obj_valid = obj_pts[~np.any(np.isnan(obj_pts), axis=1)]
            grip_valid = grip_pts[~np.any(np.isnan(grip_pts), axis=1)]
            
            if len(obj_valid) > 0 and len(grip_valid) > 0:
                obj_center = np.median(obj_valid, axis=0)
                grip_center = np.median(grip_valid, axis=0)
                relative_disp[t] = obj_center - grip_center
            else:
                relative_disp[t] = np.nan
        
        # 相对于第一帧
        if not np.any(np.isnan(relative_disp[0])):
            relative_disp = relative_disp - relative_disp[0]
        
        return relative_disp
    
    def _compute_relative_velocity(self, displacement: np.ndarray) -> np.ndarray:
        """从相对位移计算相对速度
        
        Args:
            displacement: (T, 3) 位移
            
        Returns:
            velocity: (T, 3) 速度 m/s
        """
        T = len(displacement)
        velocity = np.zeros_like(displacement)
        
        for t in range(1, T - 1):
            if not np.any(np.isnan(displacement[t-1:t+2])):
                velocity[t] = (displacement[t+1] - displacement[t-1]) / (2 * self.dt)
        
        if T > 1 and not np.any(np.isnan(displacement[:2])):
            velocity[0] = (displacement[1] - displacement[0]) / self.dt
        if T > 1 and not np.any(np.isnan(displacement[-2:])):
            velocity[-1] = (displacement[-1] - displacement[-2]) / self.dt
        
        return velocity
    
    def _filter_min_duration(self, slip_detected: np.ndarray) -> np.ndarray:
        """过滤持续时间太短的滑移检测"""
        T = len(slip_detected)
        filtered = slip_detected.copy()
        
        i = 0
        while i < T:
            if slip_detected[i]:
                j = i
                while j < T and slip_detected[j]:
                    j += 1
                duration = j - i
                if duration < self.min_slip_duration:
                    filtered[i:j] = False
                i = j
            else:
                i += 1
        
        return filtered


class CoTrackerContactDetector:
    """基于 CoTracker 的接触检测
    
    通过检测物体运动状态突变判断接触发生
    
    Args:
        velocity_drop_threshold: 速度下降比例阈值 (0-1)
        acceleration_threshold: 加速度阈值 m/s²
        min_contact_duration: 最小接触持续帧数
    """
    
    def __init__(
        self,
        velocity_drop_threshold: float = 0.5,
        acceleration_threshold: float = 2.0,
        min_contact_duration: int = 3,
        fps: float = 30.0,
        device: str = "cuda",
    ):
        self.velocity_drop_threshold = velocity_drop_threshold
        self.acceleration_threshold = acceleration_threshold
        self.min_contact_duration = min_contact_duration
        self.fps = fps
        self.device = device
    
    def detect_contact_sequence(
        self,
        video_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
        object_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """检测接触
        
        原理：夹爪接触物体时，物体自由运动被约束，速度会突然下降
        
        Args:
            video_frames: RGB 帧列表
            depth_frames: 深度图列表
            object_mask: 物体 mask
            camera_intrinsics: 相机内参
            
        Returns:
            接触检测结果
        """
        from src.velocity.cotracker_estimator import CoTrackerVelocityEstimator
        
        estimator = CoTrackerVelocityEstimator(
            device=self.device,
            fps=self.fps,
        )
        
        result = estimator.estimate_velocity_sequence(
            video_frames, depth_frames, object_mask, camera_intrinsics
        )
        
        velocity = result["velocity"]  # (T, 3)
        acceleration = result["acceleration"]  # (T, 3)
        
        vel_magnitude = np.linalg.norm(velocity, axis=1)
        acc_magnitude = np.linalg.norm(acceleration, axis=1)
        
        T = len(vel_magnitude)
        contact_detected = np.zeros(T, dtype=bool)
        
        # 找速度突然下降的时刻（接触后物体被约束）
        for t in range(1, T):
            if (vel_magnitude[t-1] > 0.01 and  # 之前在动
                vel_magnitude[t] < vel_magnitude[t-1] * (1 - self.velocity_drop_threshold)):
                # 速度骤降
                contact_detected[t:] = True
                break
        
        return {
            "contact_detected": contact_detected,
            "velocity_magnitude": vel_magnitude,
            "acceleration_magnitude": acc_magnitude,
            "contact_frame": np.argmax(contact_detected) if np.any(contact_detected) else -1,
        }
