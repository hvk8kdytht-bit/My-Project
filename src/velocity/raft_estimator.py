"""
RAFT 深度学习光流速度估计器
基于 torchvision 内置的 RAFT (Recurrent All-Pairs Field Transforms)
比传统 Farneback 光流精度更高，尤其在大位移和低纹理区域
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
import torchvision.transforms.functional as F


class RaftOpticalFlowEstimator:
    """基于 RAFT 深度学习光流的速度估计器
    
    Args:
        model_size: 模型大小 ('small' 或 'large')
        device: 计算设备
        fps: 视频帧率
    """
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        fps: float = 30.0,
    ):
        self.model_size = model_size
        self.device = device
        self.fps = fps
        self.dt = 1.0 / fps
        self.model = None
        self.transform = None
        self._loaded = False
    
    def load_model(self):
        """加载 RAFT 模型"""
        if self._loaded:
            return
        
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._transform = weights.transforms()
        self._loaded = True
        print(f"RAFT small 模型已加载（设备: {self.device}）")
    
    def compute_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """计算两帧之间的光流
        
        Args:
            frame1: 第一帧 RGB (H, W, 3) uint8
            frame2: 第二帧 RGB (H, W, 3) uint8
            
        Returns:
            flow: 光流场 (H, W, 2)，单位像素（原始图像分辨率）
        """
        if not self._loaded:
            self.load_model()
        
        orig_h, orig_w = frame1.shape[:2]
        
        # 转 tensor
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0
        
        # 预处理（可能会 resize）
        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)
        
        img1_b, img2_b = self._transform(img1, img2)
        _, _, new_h, new_w = img1_b.shape
        
        # 推理
        with torch.no_grad():
            list_of_flows = self.model(img1_b, img2_b)
            predicted_flow = list_of_flows[-1]  # (1, 2, H_new, W_new)
        
        # 将光流 resize 回原始分辨率并缩放数值
        import torch.nn.functional as F
        flow_resized = F.interpolate(
            predicted_flow, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False
        )
        
        # 缩放光流值（图像缩放比例）
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
        flow_resized[:, 0] *= scale_x
        flow_resized[:, 1] *= scale_y
        
        # 转回 numpy (H, W, 2)
        flow = flow_resized[0].cpu().numpy().transpose(1, 2, 0)
        
        return flow
    
    def estimate_velocity_sequence(
        self,
        video_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
        object_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """估计视频序列中物体的速度
        
        Args:
            video_frames: RGB 帧列表 (H, W, 3) uint8
            depth_frames: 深度图列表 (H, W) float32，米
            object_mask: 物体 mask (H, W) bool
            camera_intrinsics: 相机内参 (3, 3)
            
        Returns:
            速度估计结果
        """
        if not self._loaded:
            self.load_model()
        
        T = len(video_frames)
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # 每帧的物体 3D 速度
        velocities_3d = np.zeros((T, 3), dtype=np.float32)
        
        # 逐帧计算光流
        for t in range(T - 1):
            flow = self.compute_flow(video_frames[t], video_frames[t+1])
            
            depth = depth_frames[t]
            H, W = depth.shape
            
            # 在物体 mask 内收集光流
            ys, xs = np.where(object_mask)
            if len(xs) == 0:
                continue
            
            # 计算每个点的 3D 位移
            displacements_3d = []
            
            for i in range(len(xs)):
                x, y = xs[i], ys[i]
                if depth[y, x] <= 0:
                    continue
                
                z = depth[y, x]
                dx_pix, dy_pix = flow[y, x]
                
                # 像素位移 -> 3D 位移（针孔相机近似）
                # X = (u - cx) * z / fx
                # dX = du * z / fx
                dX = dx_pix * z / fx
                dY = dy_pix * z / fy
                
                # Z 方向位移从深度图差分得到
                if t + 1 < len(depth_frames):
                    # 用邻域深度平均值更鲁棒
                    x2, y2 = int(x + dx_pix), int(y + dy_pix)
                    if 0 <= x2 < W and 0 <= y2 < H and depth_frames[t+1][y2, x2] > 0:
                        dZ = depth_frames[t+1][y2, x2] - z
                    else:
                        dZ = 0
                else:
                    dZ = 0
                
                displacements_3d.append([dX, dY, dZ])
            
            if len(displacements_3d) > 0:
                # 取中位数作为物体速度（更鲁棒）
                disp_median = np.median(displacements_3d, axis=0)
                velocities_3d[t] = disp_median / self.dt
        
        # 最后一帧用前一帧
        if T > 1:
            velocities_3d[-1] = velocities_3d[-2]
        
        # 计算加速度
        accelerations = np.zeros_like(velocities_3d)
        for t in range(1, T - 1):
            accelerations[t] = (velocities_3d[t+1] - velocities_3d[t-1]) / (2 * self.dt)
        
        return {
            "velocity": velocities_3d,
            "acceleration": accelerations,
            "method": f"raft_{self.model_size}",
        }


class RaftSlipDetector:
    """基于 RAFT 光流的滑移检测器
    
    通过比较物体区域和夹爪区域的光流差异检测滑移
    
    Args:
        slip_threshold_mm: 滑移阈值 mm
        velocity_threshold: 相对速度阈值 mm/s
        min_slip_duration: 最小持续帧数
    """
    
    def __init__(
        self,
        slip_threshold_mm: float = 1.0,
        velocity_threshold: float = 5.0,
        min_slip_duration: int = 3,
        model_size: str = "small",
        device: str = "cpu",
        fps: float = 30.0,
    ):
        self.slip_threshold_mm = slip_threshold_mm
        self.velocity_threshold = velocity_threshold
        self.min_slip_duration = min_slip_duration
        self.device = device
        self.fps = fps
        self.dt = 1.0 / fps
        
        self._flow_estimator = RaftOpticalFlowEstimator(
            model_size=model_size,
            device=device,
            fps=fps,
        )
    
    def detect_slip_sequence(
        self,
        video_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
        object_mask: np.ndarray,
        gripper_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """检测滑移
        
        Args:
            video_frames: RGB 帧列表
            depth_frames: 深度图列表
            object_mask: 物体 mask
            gripper_mask: 夹爪 mask
            camera_intrinsics: 相机内参
            
        Returns:
            滑移检测结果
        """
        T = len(video_frames)
        
        # 分别估计物体和夹爪的速度
        obj_result = self._flow_estimator.estimate_velocity_sequence(
            video_frames, depth_frames, object_mask, camera_intrinsics
        )
        grip_result = self._flow_estimator.estimate_velocity_sequence(
            video_frames, depth_frames, gripper_mask, camera_intrinsics
        )
        
        obj_vel = obj_result["velocity"]
        grip_vel = grip_result["velocity"]
        
        # 相对速度
        rel_vel = obj_vel - grip_vel
        rel_vel_mag = np.linalg.norm(rel_vel, axis=1) * 1000  # m/s -> mm/s
        
        # 累积相对位移
        cum_disp = np.cumsum(np.linalg.norm(rel_vel * self.dt, axis=1)) * 1000  # mm
        
        # 滑移检测
        slip_detected = np.zeros(T, dtype=bool)
        for t in range(T):
            if (rel_vel_mag[t] > self.velocity_threshold or 
                cum_disp[t] > self.slip_threshold_mm):
                slip_detected[t] = True
        
        # 最小持续时长过滤
        slip_detected = self._filter_min_duration(slip_detected)
        
        return {
            "slip_detected": slip_detected,
            "relative_velocity_mms": rel_vel_mag,
            "cumulative_displacement_mm": cum_disp,
            "object_velocity": obj_vel,
            "gripper_velocity": grip_vel,
        }
    
    def _filter_min_duration(self, slip_detected: np.ndarray) -> np.ndarray:
        T = len(slip_detected)
        filtered = slip_detected.copy()
        i = 0
        while i < T:
            if slip_detected[i]:
                j = i
                while j < T and slip_detected[j]:
                    j += 1
                if j - i < self.min_slip_duration:
                    filtered[i:j] = False
                i = j
            else:
                i += 1
        return filtered


class RaftContactDetector:
    """基于 RAFT 光流的接触检测器
    
    通过检测物体运动状态突变判断接触
    """
    
    def __init__(
        self,
        velocity_drop_threshold: float = 0.5,
        min_contact_duration: int = 3,
        model_size: str = "small",
        device: str = "cpu",
        fps: float = 30.0,
    ):
        self.velocity_drop_threshold = velocity_drop_threshold
        self.min_contact_duration = min_contact_duration
        self.device = device
        self.fps = fps
        
        self._flow_estimator = RaftOpticalFlowEstimator(
            model_size=model_size,
            device=device,
            fps=fps,
        )
    
    def detect_contact_sequence(
        self,
        video_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
        object_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """检测接触"""
        result = self._flow_estimator.estimate_velocity_sequence(
            video_frames, depth_frames, object_mask, camera_intrinsics
        )
        
        velocity = result["velocity"]
        vel_mag = np.linalg.norm(velocity, axis=1)
        
        T = len(vel_mag)
        contact_detected = np.zeros(T, dtype=bool)
        
        # 找速度骤降点
        for t in range(1, T):
            if (vel_mag[t-1] > 0.005 and 
                vel_mag[t] < vel_mag[t-1] * (1 - self.velocity_drop_threshold)):
                contact_detected[t:] = True
                break
        
        return {
            "contact_detected": contact_detected,
            "velocity_magnitude": vel_mag,
            "contact_frame": np.argmax(contact_detected) if np.any(contact_detected) else -1,
        }
