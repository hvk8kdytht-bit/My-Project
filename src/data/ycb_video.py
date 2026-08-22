"""
YCB-Video 数据集加载（BOP 格式）
支持 6D 位姿估计、目标检测、实例分割任务

BOP 格式目录结构:
    ycbv/
    ├── train_real/        # 真实训练图像（约80个场景）
    │   ├── 000001/
    │   │   ├── rgb/       # 彩色图像 .png
    │   │   ├── depth/     # 深度图像 .png
    │   │   ├── mask/      # 分割掩码
    │   │   ├── mask_visib/ # 可见部分掩码
    │   │   └── scene_gt.json  # 位姿标注
    │   └── ...
    ├── train_pbr/         # PBR合成训练图像（可选，较大）
    ├── test/              # 测试集
    ├── models/            # 3D模型 (.ply)
    └── camera.json        # 相机参数
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset


# YCB-Video 21个物体类别
YCB_OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
]

YCB_OBJECT_ID_TO_NAME = {i + 1: name for i, name in enumerate(YCB_OBJECTS)}
YCB_OBJECT_NAME_TO_ID = {name: i + 1 for i, name in enumerate(YCB_OBJECTS)}


class YCBVideoDataset(Dataset):
    """YCB-Video BOP 格式基础数据集 - 用于检测/分割"""

    def __init__(
        self,
        root_dir: str,
        split: str = "train_real",
        transform=None,
        load_depth: bool = True,
        load_mask: bool = True,
        scene_filter: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir: 数据集根目录 (e.g., H:/Program/datasets/ycb_video)
            split: 数据集划分 - 'train_real', 'train_pbr', 'test', 'val'
            transform: 图像变换
            load_depth: 是否加载深度图
            load_mask: 是否加载分割掩码
            scene_filter: 只加载指定场景（场景级划分，如 ["000048", "000049"]）
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.load_depth = load_depth
        self.load_mask = load_mask
        self.scene_filter = set(scene_filter) if scene_filter else None

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"数据集划分 {split} 不存在: {self.split_dir}\n"
                f"请确认数据已下载到 {root_dir}"
            )

        # 加载相机参数（根目录 camera_*.json，实际内参以每个场景的 scene_camera.json 为准）
        self.camera_params = None
        for cam_name in ["camera_cmu.json", "camera_uw.json", "camera.json"]:
            camera_path = self.root_dir / cam_name
            if camera_path.exists():
                with open(camera_path, "r") as f:
                    self.camera_params = json.load(f)
                break
        if self.camera_params is None:
            # base.zip 解压后相机文件位于 ycbv/ 子目录
            for cam_name in ["camera_cmu.json", "camera_uw.json"]:
                camera_path = self.root_dir / "ycbv" / cam_name
                if camera_path.exists():
                    with open(camera_path, "r") as f:
                        self.camera_params = json.load(f)
                    break

        # 收集所有样本
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Dict]:
        """收集所有图像样本的路径信息"""
        samples = []
        scene_dirs = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])

        for scene_dir in scene_dirs:
            scene_id = scene_dir.name
            if self.scene_filter and scene_id not in self.scene_filter:
                continue
            rgb_dir = scene_dir / "rgb"
            depth_dir = scene_dir / "depth"
            mask_dir = scene_dir / "mask_visib"

            if not rgb_dir.exists():
                continue

            # 加载场景GT标注
            gt_path = scene_dir / "scene_gt.json"
            gt_info_path = scene_dir / "scene_gt_info.json"
            scene_camera_path = scene_dir / "scene_camera.json"

            scene_gt = {}
            if gt_path.exists():
                with open(gt_path, "r") as f:
                    scene_gt = json.load(f)

            scene_gt_info = {}
            if gt_info_path.exists():
                with open(gt_info_path, "r") as f:
                    scene_gt_info = json.load(f)

            # BOP 标准: 逐帧相机内参 cam_K (3x3 行优先) + depth_scale
            scene_camera = {}
            if scene_camera_path.exists():
                with open(scene_camera_path, "r") as f:
                    scene_camera = json.load(f)

            # 遍历所有RGB图像
            rgb_files = sorted(rgb_dir.glob("*.png"))
            for rgb_path in rgb_files:
                img_id = rgb_path.stem  # e.g., "000001"
                # BOP 标注键为无补零整数字符串（"1"），图像文件名为补零（"000001"）
                ann_key = str(int(img_id))
                cam_info = scene_camera.get(ann_key, {})

                sample = {
                    "scene_id": scene_id,
                    "img_id": img_id,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_dir / f"{img_id}.png") if depth_dir.exists() else None,
                    "mask_dir": str(mask_dir) if mask_dir.exists() else None,
                    "annotations": scene_gt.get(ann_key, []),
                    "gt_info": scene_gt_info.get(ann_key, []),
                    "camera_intrinsics": cam_info.get("cam_K", None),
                    "depth_scale": cam_info.get("depth_scale", 1.0),
                }
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 加载RGB图像
        rgb = Image.open(sample["rgb_path"]).convert("RGB")

        result = {
            "rgb": rgb,
            "scene_id": sample["scene_id"],
            "img_id": sample["img_id"],
        }

        # 加载深度图
        if self.load_depth and sample["depth_path"] and os.path.exists(sample["depth_path"]):
            depth = np.array(Image.open(sample["depth_path"]))
            # BOP深度图通常是16位PNG，单位为毫米
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0  # 转为米
            result["depth"] = depth

        # 加载掩码和标注
        if self.load_mask and sample["mask_dir"]:
            result["annotations"] = sample["annotations"]
            result["gt_info"] = sample["gt_info"]

        # 应用变换
        if self.transform:
            result = self.transform(result)

        return result


class YCBVideoPoseDataset(YCBVideoDataset):
    """YCB-Video 6D 位姿估计数据集 - 输出 (RGB, 位姿, 内参)"""

    def __init__(
        self,
        root_dir: str,
        split: str = "train_real",
        transform=None,
        load_depth: bool = True,
        single_object: Optional[int] = None,
        scene_filter: Optional[List[str]] = None,
    ):
        """
        Args:
            single_object: 如果指定，则只返回该物体ID的样本(用于单物体位姿估计)
            scene_filter: 只加载指定场景（场景级划分）
        """
        super().__init__(root_dir, split, transform, load_depth, load_mask=True,
                         scene_filter=scene_filter)
        self.single_object = single_object
        self.pose_samples = self._collect_pose_samples()

    def _collect_pose_samples(self) -> List[Dict]:
        """按物体实例展开样本（每张图多个物体拆成多条）"""
        pose_samples = []

        for sample in self.samples:
            annotations = sample["annotations"]
            gt_info = sample["gt_info"]

            for obj_idx, ann in enumerate(annotations):
                obj_id = ann["obj_id"]

                # 单物体模式过滤
                if self.single_object is not None and obj_id != self.single_object:
                    continue

                # 过滤掉严重遮挡或不可见的物体
                if gt_info and obj_idx < len(gt_info):
                    visib_fract = gt_info[obj_idx].get("visib_fract", 0)
                    if visib_fract < 0.1:  # 可见度低于10%跳过
                        continue

                # 位姿: 旋转矩阵(3x3) + 平移向量(3,)
                # BOP 标准中 cam_R_m2c 无量纲、cam_t_m2c 单位为毫米，统一转为米
                R = np.array(ann["cam_R_m2c"]).reshape(3, 3)
                t = np.array(ann["cam_t_m2c"]).reshape(3) / 1000.0

                # 掩码路径
                mask_path = None
                if sample["mask_dir"]:
                    mask_filename = f"{sample['img_id']}_{obj_idx:06d}.png"
                    mask_path = os.path.join(sample["mask_dir"], mask_filename)

                pose_sample = {
                    "rgb_path": sample["rgb_path"],
                    "depth_path": sample["depth_path"],
                    "mask_path": mask_path,
                    "obj_id": obj_id,
                    "obj_name": YCB_OBJECT_ID_TO_NAME.get(obj_id, f"obj_{obj_id}"),
                    "rotation": R.astype(np.float32),
                    "translation": t.astype(np.float32),
                    "camera_intrinsics": sample.get("camera_intrinsics"),
                    "depth_scale": sample.get("depth_scale", 1.0),
                    "scene_id": sample["scene_id"],
                    "img_id": sample["img_id"],
                }
                pose_samples.append(pose_sample)

        return pose_samples

    def __len__(self) -> int:
        return len(self.pose_samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.pose_samples[idx]

        # 加载图像
        rgb = Image.open(sample["rgb_path"]).convert("RGB")

        result = {
            "rgb": rgb,
            "obj_id": sample["obj_id"],
            "obj_name": sample["obj_name"],
            "rotation": torch.from_numpy(sample["rotation"]),
            "translation": torch.from_numpy(sample["translation"]),
            "scene_id": sample["scene_id"],
            "img_id": sample["img_id"],
        }

        # 加载深度图（BOP 16位PNG，像素值 = depth_scale * 毫米）
        if self.load_depth and sample["depth_path"] and os.path.exists(sample["depth_path"]):
            depth = np.array(Image.open(sample["depth_path"]))
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * sample.get("depth_scale", 1.0) / 1000.0
            result["depth"] = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

        # 加载相机内参（BOP 逐帧 cam_K，行优先展平）
        if sample.get("camera_intrinsics") is not None:
            K = np.array(sample["camera_intrinsics"], dtype=np.float32).reshape(3, 3)
        elif self.camera_params:
            K = np.array([
                [self.camera_params["fx"], 0, self.camera_params["cx"]],
                [0, self.camera_params["fy"], self.camera_params["cy"]],
                [0, 0, 1],
            ], dtype=np.float32)
        else:
            K = np.array([[1066.778, 0, 312.9869],
                          [0, 1067.487, 241.3109],
                          [0, 0, 1]], dtype=np.float32)
        result["camera_intrinsics"] = torch.from_numpy(K)

        # 加载掩码
        if sample["mask_path"] and os.path.exists(sample["mask_path"]):
            mask = np.array(Image.open(sample["mask_path"]))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.float32)
            result["mask"] = torch.from_numpy(mask).unsqueeze(0)

        # 应用变换
        if self.transform:
            result = self.transform(result)

        return result


def compute_speed_and_acceleration(
    poses: List[Dict],
    timestamps: Optional[List[float]] = None,
) -> Dict:
    """
    从位姿序列计算速度和加速度（用于轨迹学习）

    Args:
        poses: 位姿列表，每个包含 rotation (3x3) 和 translation (3,)
        timestamps: 时间戳列表（秒），如果为None则假设等间隔1秒

    Returns:
        包含线速度、角速度、线加速度、角加速度的字典
    """
    n = len(poses)
    if n < 2:
        return {"linear_velocity": [], "angular_velocity": [],
                "linear_acceleration": [], "angular_acceleration": []}

    if timestamps is None:
        timestamps = list(range(n))

    linear_velocities = []
    angular_velocities = []

    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            dt = 1.0

        # 线速度
        t_prev = poses[i - 1]["translation"]
        t_curr = poses[i]["translation"]
        lin_vel = (t_curr - t_prev) / dt
        linear_velocities.append(lin_vel)

        # 角速度（从旋转矩阵差计算）
        R_prev = poses[i - 1]["rotation"]
        R_curr = poses[i]["rotation"]
        R_rel = R_curr @ R_prev.T
        # 从旋转矩阵提取轴角
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        if abs(angle) < 1e-6:
            ang_vel = np.zeros(3)
        else:
            s = 1 / (2 * np.sin(angle))
            rx = R_rel[2, 1] - R_rel[1, 2]
            ry = R_rel[0, 2] - R_rel[2, 0]
            rz = R_rel[1, 0] - R_rel[0, 1]
            axis = np.array([rx, ry, rz]) * s
            ang_vel = axis * angle / dt
        angular_velocities.append(ang_vel)

    # 加速度
    linear_accelerations = []
    angular_accelerations = []
    for i in range(1, len(linear_velocities)):
        dt = timestamps[i + 1] - timestamps[i]
        if dt <= 0:
            dt = 1.0
        lin_acc = (linear_velocities[i] - linear_velocities[i - 1]) / dt
        ang_acc = (angular_velocities[i] - angular_velocities[i - 1]) / dt
        linear_accelerations.append(lin_acc)
        angular_accelerations.append(ang_acc)

    return {
        "linear_velocity": np.array(linear_velocities),
        "angular_velocity": np.array(angular_velocities),
        "linear_acceleration": np.array(linear_accelerations),
        "angular_acceleration": np.array(angular_accelerations),
    }
