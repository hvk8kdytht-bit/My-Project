"""
滑移检测模块
检测物体与夹爪之间的滑移状态

方法:
1. 光流法 - 纯视觉，检测物体表面像素运动
2. 力觉法 - 基于切向力/力矩变化
3. 位姿差法 - 物体位姿与夹爪位姿的偏差
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from enum import Enum


class SlipState(Enum):
    STABLE = "stable"          # 稳定夹持
    INCIPIENT = "incipient"    # 初始滑移（即将滑移）
    SLIPPING = "slipping"      # 正在滑移
    UNKNOWN = "unknown"        # 未知


class SlipDetector(ABC):
    """滑移检测器基类"""

    @abstractmethod
    def detect(self, observation: dict) -> Tuple[SlipState, dict]:
        """
        检测滑移状态

        Args:
            observation: 观测字典

        Returns:
            slip_state: 滑移状态
            info: 详细信息
        """
        pass

    def reset(self):
        pass


class OpticalFlowSlipDetector(SlipDetector):
    """
    基于光流的纯视觉滑移检测

    原理:
    - 计算接触区域的光流
    - 如果光流方向与夹爪运动方向不一致，说明有滑移
    - 滑移速度 = 物体表面相对夹爪的运动速度

    适用于: 纯视觉方案，无触觉传感器
    """

    def __init__(
        self,
        slip_threshold: float = 2.0,           # 滑移速度阈值 (像素/帧)
        incipient_ratio: float = 0.5,          # 初始滑移阈值比例
        roi_margin: float = 0.2,               # ROI 边缘比例
        min_flow_points: int = 10,             # 最少光流点数
    ):
        """
        Args:
            slip_threshold: 判定滑移的速度阈值
            incipient_ratio: 初始滑移阈值 = slip_threshold * incipient_ratio
            roi_margin: 接触区域ROI边缘留白比例
            min_flow_points: 有效光流点的最小数量
        """
        self.slip_threshold = slip_threshold
        self.incipient_threshold = slip_threshold * incipient_ratio
        self.roi_margin = roi_margin
        self.min_flow_points = min_flow_points

        self._prev_image = None
        self._prev_roi = None

    def detect(self, observation: dict) -> Tuple[SlipState, dict]:
        image = observation.get("rgb")
        mask = observation.get("mask")
        bbox = observation.get("bbox")

        if image is None:
            return SlipState.UNKNOWN, {"error": "no_image"}

        # 转换为灰度
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        info = {}

        if self._prev_image is not None:
            try:
                import cv2

                # 定义ROI（物体周围区域）
                if bbox is not None and len(bbox) == 4:
                    x, y, w, h = bbox
                    # 扩大一点ROI
                    margin_x = int(w * self.roi_margin)
                    margin_y = int(h * self.roi_margin)
                    x1 = max(0, int(x - margin_x))
                    y1 = max(0, int(y - margin_y))
                    x2 = min(gray.shape[1], int(x + w + margin_x))
                    y2 = min(gray.shape[0], int(y + h + margin_y))

                    roi_prev = self._prev_image[y1:y2, x1:x2]
                    roi_curr = gray[y1:y2, x1:x2]

                    if roi_prev.size > 0 and roi_curr.size > 0:
                        # 计算光流
                        flow = cv2.calcOpticalFlowFarneback(
                            roi_prev, roi_curr, None,
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )

                        # 如果有mask，只考虑物体区域的光流
                        if mask is not None:
                            mask_resized = cv2.resize(
                                mask.astype(np.uint8),
                                (roi_prev.shape[1], roi_prev.shape[0])
                            ) > 0.5
                            if mask_resized.sum() > self.min_flow_points:
                                flow_masked = flow[mask_resized]
                                flow_magnitudes = np.linalg.norm(flow_masked, axis=1)
                            else:
                                flow_magnitudes = np.linalg.norm(flow, axis=2).flatten()
                        else:
                            flow_magnitudes = np.linalg.norm(flow, axis=2).flatten()

                        # 统计光流
                        if len(flow_magnitudes) > 0:
                            mean_flow = np.mean(flow_magnitudes)
                            max_flow = np.max(flow_magnitudes)
                            median_flow = np.median(flow_magnitudes)

                            info["mean_flow"] = float(mean_flow)
                            info["max_flow"] = float(max_flow)
                            info["median_flow"] = float(median_flow)

                            # 判断滑移状态
                            if median_flow > self.slip_threshold:
                                state = SlipState.SLIPPING
                            elif median_flow > self.incipient_threshold:
                                state = SlipState.INCIPIENT
                            else:
                                state = SlipState.STABLE
                        else:
                            state = SlipState.UNKNOWN
                    else:
                        state = SlipState.UNKNOWN
                else:
                    state = SlipState.UNKNOWN
                    info["error"] = "no_bbox"

            except ImportError:
                # OpenCV不可用，返回未知
                state = SlipState.UNKNOWN
                info["error"] = "opencv_not_available"
        else:
            state = SlipState.UNKNOWN
            info["error"] = "first_frame"

        # 更新历史
        self._prev_image = gray

        return state, info

    def reset(self):
        self._prev_image = None
        self._prev_roi = None


class ForceSlipDetector(SlipDetector):
    """
    基于力觉的滑移检测

    原理:
    - 切向力与法向力的比值 = 摩擦系数估计
    - 当摩擦系数接近静摩擦系数时，即将发生滑移
    - 当切向力快速变化时，可能正在滑移

    适用于: 有六维力传感器或触觉传感器
    """

    def __init__(
        self,
        static_friction_coeff: float = 0.8,   # 静摩擦系数估计
        incipient_ratio: float = 0.8,          # 初始滑移比例 (mu/mu_s)
        slip_ratio: float = 0.95,              # 滑移比例
        force_rate_threshold: float = 5.0,     # 力变化率阈值 (N/s)
        debounce_frames: int = 2,              # 去抖
    ):
        """
        Args:
            static_friction_coeff: 估计的静摩擦系数
            incipient_ratio: 初始滑移阈值 = static_friction_coeff * incipient_ratio
            slip_ratio: 滑移阈值 = static_friction_coeff * slip_ratio
            force_rate_threshold: 切向力变化率阈值
            debounce_frames: 去抖帧数
        """
        self.static_friction_coeff = static_friction_coeff
        self.incipient_threshold = static_friction_coeff * incipient_ratio
        self.slip_threshold = static_friction_coeff * slip_ratio
        self.force_rate_threshold = force_rate_threshold
        self.debounce_frames = debounce_frames

        self._prev_tangential_force = None
        self._slip_counter = 0
        self._stable_counter = 0
        self._last_state = SlipState.UNKNOWN

    def detect(self, observation: dict) -> Tuple[SlipState, dict]:
        normal_force = observation.get("normal_force", 0.0)
        tangential_force = observation.get("tangential_force", None)

        if tangential_force is None:
            # 从力向量分解
            force = observation.get("gripper_force_vec", np.zeros(3))
            normal_dir = observation.get("normal_direction", np.array([0, 0, 1]))

            normal_force = np.dot(force, normal_dir)
            tangential_vec = force - normal_force * normal_dir
            tangential_mag = np.linalg.norm(tangential_vec)
        elif isinstance(tangential_force, np.ndarray):
            tangential_mag = np.linalg.norm(tangential_force)
        else:
            tangential_mag = abs(tangential_force)

        info = {
            "normal_force": normal_force,
            "tangential_force": tangential_mag,
        }

        if normal_force <= 0.01:
            # 法向力太小，无法判断
            self._last_state = SlipState.UNKNOWN
            return SlipState.UNKNOWN, info

        # 摩擦系数估计
        friction_coeff = tangential_mag / normal_force
        info["friction_coefficient"] = friction_coeff
        info["friction_ratio"] = friction_coeff / self.static_friction_coeff

        # 切向力变化率
        if self._prev_tangential_force is not None:
            force_rate = tangential_mag - self._prev_tangential_force
            info["tangential_force_rate"] = force_rate
        else:
            force_rate = 0.0
            info["tangential_force_rate"] = 0.0

        self._prev_tangential_force = tangential_mag

        # 判断状态
        if friction_coeff >= self.slip_threshold or abs(force_rate) > self.force_rate_threshold:
            self._slip_counter += 1
            self._stable_counter = 0
            if self._slip_counter >= self.debounce_frames:
                state = SlipState.SLIPPING
            else:
                state = self._last_state
        elif friction_coeff >= self.incipient_threshold:
            state = SlipState.INCIPIENT
            self._slip_counter = 0
            self._stable_counter = 0
        else:
            self._stable_counter += 1
            self._slip_counter = 0
            if self._stable_counter >= self.debounce_frames:
                state = SlipState.STABLE
            else:
                state = self._last_state

        self._last_state = state
        return state, info

    def reset(self):
        self._prev_tangential_force = None
        self._slip_counter = 0
        self._stable_counter = 0
        self._last_state = SlipState.UNKNOWN


class PoseDifferenceSlipDetector(SlipDetector):
    """
    基于位姿差的滑移检测

    原理:
    - 比较物体位姿和夹爪位姿的相对变化
    - 如果物体相对夹爪发生了位移/旋转，说明有滑移

    适用于: 有物体位姿估计和夹爪位姿反馈
    """

    def __init__(
        self,
        translation_threshold: float = 0.001,  # 平移阈值 (m)
        rotation_threshold: float = 0.01,      # 旋转阈值 (rad)
        incipient_ratio: float = 0.3,          # 初始滑移比例
        min_history: int = 5,                  # 最少历史帧数
    ):
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.incipient_translation = translation_threshold * incipient_ratio
        self.incipient_rotation = rotation_threshold * incipient_ratio
        self.min_history = min_history

        self._relative_poses: List[np.ndarray] = []  # 物体相对夹爪的位姿历史

    def detect(self, observation: dict) -> Tuple[SlipState, dict]:
        obj_pose = observation.get("object_pose")
        gripper_pose = observation.get("gripper_pose")

        if obj_pose is None or gripper_pose is None:
            return SlipState.UNKNOWN, {"error": "missing_pose"}

        # 计算相对位姿
        # T_rel = T_gripper^{-1} @ T_object
        if isinstance(obj_pose, np.ndarray) and obj_pose.shape == (4, 4):
            gripper_inv = np.linalg.inv(gripper_pose)
            relative_pose = gripper_inv @ obj_pose
        else:
            # 只有平移，直接相减
            relative_pose = np.array(obj_pose) - np.array(gripper_pose)

        self._relative_poses.append(relative_pose.flatten())
        if len(self._relative_poses) > 20:
            self._relative_poses.pop(0)

        info = {"history_length": len(self._relative_poses)}

        if len(self._relative_poses) < self.min_history:
            return SlipState.UNKNOWN, info

        # 计算相对位姿变化
        poses = np.array(self._relative_poses)
        recent = poses[-1]
        baseline = np.mean(poses[-self.min_history:-1], axis=0) if len(poses) > self.min_history else poses[0]

        # 平移变化
        if recent.size >= 3:
            trans_change = np.linalg.norm(recent[:3] - baseline[:3])
            info["translation_change"] = trans_change
        else:
            trans_change = 0.0

        # 旋转变化（简化：如果有四元数或旋转矩阵）
        rot_change = 0.0
        if recent.size >= 7:
            # 假设有四元数 (wxyz)
            q1 = baseline[3:7]
            q2 = recent[3:7]
            if np.linalg.norm(q1) > 0 and np.linalg.norm(q2) > 0:
                q1 = q1 / np.linalg.norm(q1)
                q2 = q2 / np.linalg.norm(q2)
                dot = np.clip(np.dot(q1, q2), -1, 1)
                rot_change = 2 * np.arccos(dot)
                info["rotation_change"] = rot_change

        # 判断状态
        if trans_change > self.translation_threshold or rot_change > self.rotation_threshold:
            state = SlipState.SLIPPING
        elif trans_change > self.incipient_translation or rot_change > self.incipient_rotation:
            state = SlipState.INCIPIENT
        else:
            state = SlipState.STABLE

        return state, info

    def reset(self):
        self._relative_poses.clear()
