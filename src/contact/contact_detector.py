"""
接触检测模块
从视觉观测或力觉信号检测夹爪与物体的接触状态

方法:
1. 阈值法 - 基于力/力矩阈值（有触觉传感器时）
2. 视觉法 - 基于物体位移/形变检测接触（纯视觉方案）
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List


class ContactDetector(ABC):
    """接触检测器基类"""

    @abstractmethod
    def detect(self, observation: dict) -> Tuple[bool, dict]:
        """
        检测接触状态

        Args:
            observation: 观测字典（可能包含力觉、视觉位姿、深度等）

        Returns:
            in_contact: 是否接触
            info: 详细信息字典
        """
        pass

    def reset(self):
        """重置检测器状态"""
        pass


class ThresholdContactDetector(ContactDetector):
    """
    基于力/力矩阈值的接触检测

    当夹爪力或力矩超过阈值时判定为接触。
    适用于有力/力矩传感器或触觉传感器的场景。
    """

    def __init__(
        self,
        force_threshold: float = 2.0,      # 力阈值 (N)
        torque_threshold: float = 0.1,      # 力矩阈值 (N·m)
        hysteresis: float = 0.2,            # 滞回系数 (0-1)
        debounce_frames: int = 3,           # 去抖帧数
    ):
        """
        Args:
            force_threshold: 接触力阈值
            torque_threshold: 接触力矩阈值
            hysteresis: 滞回系数，释放阈值 = threshold * (1 - hysteresis)
            debounce_frames: 需要连续多少帧才确认状态变化
        """
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.hysteresis = hysteresis
        self.debounce_frames = debounce_frames

        # 状态
        self._in_contact = False
        self._contact_counter = 0
        self._release_counter = 0

    def detect(self, observation: dict) -> Tuple[bool, dict]:
        force = observation.get("gripper_force", 0.0)
        torque = observation.get("gripper_torque", np.zeros(3))

        if isinstance(torque, np.ndarray):
            torque_mag = np.linalg.norm(torque)
        else:
            torque_mag = abs(torque)

        # 带滞回的阈值判断
        if self._in_contact:
            # 已接触状态，释放阈值更低
            release_thresh = self.force_threshold * (1 - self.hysteresis)
            release_torque_thresh = self.torque_threshold * (1 - self.hysteresis)
            would_release = force < release_thresh and torque_mag < release_torque_thresh

            if would_release:
                self._release_counter += 1
                self._contact_counter = 0
                if self._release_counter >= self.debounce_frames:
                    self._in_contact = False
            else:
                self._release_counter = 0
        else:
            # 未接触状态，接触阈值更高
            would_contact = force >= self.force_threshold or torque_mag >= self.torque_threshold
            if would_contact:
                self._contact_counter += 1
                self._release_counter = 0
                if self._contact_counter >= self.debounce_frames:
                    self._in_contact = True
            else:
                self._contact_counter = 0

        info = {
            "force": force,
            "torque_magnitude": torque_mag,
            "contact_confidence": min(1.0, force / self.force_threshold),
            "contact_frames": self._contact_counter,
            "release_frames": self._release_counter,
        }

        return self._in_contact, info

    def reset(self):
        self._in_contact = False
        self._contact_counter = 0
        self._release_counter = 0


class VisionContactDetector(ContactDetector):
    """
    纯视觉接触检测

    原理:
    - 夹爪闭合过程中，物体位姿/深度发生突变表明接触
    - 物体速度/加速度的突变是接触的标志
    - 深度图中接触区域的形变也可用于检测

    方法:
    1. 基于物体位姿突变 - 接触瞬间物体速度方向改变
    2. 基于深度图变化 - 接触区域深度梯度突变
    3. 基于夹爪闭合距离 - 夹爪间距不再减小 = 接触
    """

    def __init__(
        self,
        method: str = "pose_change",
        velocity_threshold: float = 0.05,      # 速度突变阈值 (m/s)
        acceleration_threshold: float = 5.0,   # 加速度阈值 (m/s^2)
        depth_change_threshold: float = 0.005, # 深度变化阈值 (m)
        min_history: int = 5,                   # 最少历史帧数
    ):
        """
        Args:
            method: 检测方法 ('pose_change', 'depth_change', 'combined')
            velocity_threshold: 速度突变阈值
            acceleration_threshold: 加速度阈值
            depth_change_threshold: 深度变化阈值
            min_history: 最少需要的历史帧数
        """
        self.method = method
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.depth_change_threshold = depth_change_threshold
        self.min_history = min_history

        # 历史记录
        self._pose_history: List[np.ndarray] = []
        self._depth_history: List[np.ndarray] = []
        self._gripper_width_history: List[float] = []
        self._in_contact = False

    def detect(self, observation: dict) -> Tuple[bool, dict]:
        info = {}

        # 记录历史
        if "object_pose" in observation:
            pose = observation["object_pose"]
            if isinstance(pose, np.ndarray) and pose.size >= 3:
                pos = pose[:3, 3] if pose.shape == (4, 4) else pose[:3]
                self._pose_history.append(pos.copy())

        if "depth" in observation and observation["depth"] is not None:
            self._depth_history.append(observation["depth"].copy())
            if len(self._depth_history) > 10:
                self._depth_history.pop(0)

        if "gripper_width" in observation:
            self._gripper_width_history.append(observation["gripper_width"])
            if len(self._gripper_width_history) > 20:
                self._gripper_width_history.pop(0)

        # 方法1: 基于位姿变化
        if self.method in ["pose_change", "combined"] and len(self._pose_history) >= self.min_history:
            positions = np.array(self._pose_history[-self.min_history:])
            velocities = np.diff(positions, axis=0)

            if len(velocities) >= 2:
                # 检测加速度突变
                accelerations = np.diff(velocities, axis=0)
                acc_magnitudes = np.linalg.norm(accelerations, axis=1)
                max_acc = np.max(acc_magnitudes)

                # 检测速度方向变化
                if len(velocities) >= 2:
                    v1 = velocities[-2]
                    v2 = velocities[-1]
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    if v1_norm > 0.001 and v2_norm > 0.001:
                        direction_change = 1 - np.dot(v1, v2) / (v1_norm * v2_norm)
                    else:
                        direction_change = 0.0

                    info["max_acceleration"] = max_acc
                    info["direction_change"] = direction_change

                    if max_acc > self.acceleration_threshold or direction_change > 0.5:
                        self._in_contact = True

        # 方法2: 基于夹爪闭合速度
        if len(self._gripper_width_history) >= 5:
            widths = np.array(self._gripper_width_history[-10:])
            width_velocity = np.diff(widths)
            if len(width_velocity) >= 3:
                # 夹爪闭合速度显著减小 = 接触
                recent_vel = np.mean(np.abs(width_velocity[-3:]))
                earlier_vel = np.mean(np.abs(width_velocity[:3])) if len(width_velocity) >= 6 else recent_vel

                if earlier_vel > 0.001 and recent_vel < earlier_vel * 0.3:
                    self._in_contact = True
                    info["gripper_deceleration"] = earlier_vel - recent_vel

        info["method"] = self.method
        info["history_length"] = len(self._pose_history)

        return self._in_contact, info

    def reset(self):
        self._pose_history.clear()
        self._depth_history.clear()
        self._gripper_width_history.clear()
        self._in_contact = False


def estimate_contact_quality(
    force: float,
    contact_area: float = 0.0,
    force_distribution: Optional[np.ndarray] = None,
) -> dict:
    """
    评估接触质量（稳定程度）

    Args:
        force: 接触力大小 (N)
        contact_area: 接触面积 (m^2)
        force_distribution: 力分布矩阵（触觉传感器）

    Returns:
        质量评估字典
    """
    # 压强
    pressure = force / contact_area if contact_area > 0 else float("inf")

    # 力分布均匀性
    uniformity = 1.0
    if force_distribution is not None and np.sum(force_distribution) > 0:
        normalized = force_distribution / np.sum(force_distribution)
        # 熵越大越均匀
        entropy = -np.sum(normalized[normalized > 0] * np.log(normalized[normalized > 0]))
        max_entropy = np.log(len(force_distribution.flatten()))
        uniformity = entropy / max_entropy if max_entropy > 0 else 1.0

    # 综合评分 (0-1)
    score = 0.0
    if force > 0:
        # 力在合理范围内
        force_score = min(1.0, force / 5.0)  # 5N为满分
        score = force_score * 0.5 + uniformity * 0.5

    return {
        "force": force,
        "pressure": pressure,
        "contact_area": contact_area,
        "force_uniformity": uniformity,
        "stability_score": score,
    }
