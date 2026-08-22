"""
DexYCB 数据集加载
用于人手抓取动作的 RGB-D 数据，包含物体 6D 位姿和手部姿态

目录结构:
    dexycb/
    ├── 20200709-subject-01/
    │   ├── <setup>/
    │   │   ├── <grip>/
    │   │   │   ├── <session>/
    │   │   │   │   ├── color/       # 彩色图像
    │   │   │   │   ├── depth/       # 深度图像
    │   │   │   │   ├── labels/      # 标注文件 (.npz)
    │   │   │   │   └── ...
    ├── calibration/
    └── models/
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset


class DexYCBDataset(Dataset):
    """DexYCB 数据集 - 抓取动作 RGB-D 序列"""

    def __init__(
        self,
        root_dir: str,
        subjects: Optional[List[str]] = None,
        split: str = "train",
        transform=None,
        load_depth: bool = True,
        load_pose: bool = True,
        load_hand: bool = False,
    ):
        """
        Args:
            root_dir: 数据集根目录
            subjects: 受试者列表，如 ['20200709-subject-01']，None表示全部
            split: 'train', 'val', 'test'
            transform: 图像变换
            load_depth: 是否加载深度图
            load_pose: 是否加载物体6D位姿
            load_hand: 是否加载手部姿态
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.load_depth = load_depth
        self.load_pose = load_pose
        self.load_hand = load_hand

        # 默认按受试者划分
        if subjects is None:
            subjects = [d.name for d in self.root_dir.iterdir()
                        if d.is_dir() and d.name.startswith("2020")]
        self.subjects = subjects

        # 加载物体模型信息
        self.models_dir = self.root_dir / "models"
        self.object_list = self._load_object_list()

        # 收集所有样本
        self.samples = self._collect_samples()

    def _load_object_list(self) -> List[str]:
        """加载物体列表（从YCB模型目录）"""
        if self.models_dir.exists():
            return sorted([d.name for d in self.models_dir.iterdir() if d.is_dir()])
        return []

    def _collect_samples(self) -> List[Dict]:
        """收集所有帧样本"""
        samples = []

        for subject_name in self.subjects:
            subject_dir = self.root_dir / subject_name
            if not subject_dir.exists():
                print(f"警告: 受试者目录不存在: {subject_dir}")
                continue

            # 遍历 setup -> grip -> session
            for setup_dir in subject_dir.iterdir():
                if not setup_dir.is_dir():
                    continue
                for grip_dir in setup_dir.iterdir():
                    if not grip_dir.is_dir():
                        continue
                    for session_dir in grip_dir.iterdir():
                        if not session_dir.is_dir():
                            continue

                        color_dir = session_dir / "color"
                        depth_dir = session_dir / "depth"
                        labels_dir = session_dir / "labels"

                        if not color_dir.exists():
                            continue

                        # 获取所有彩色图像
                        color_files = sorted(color_dir.glob("*.jpg"))
                        if not color_files:
                            color_files = sorted(color_dir.glob("*.png"))

                        for color_path in color_files:
                            frame_id = color_path.stem

                            depth_path = depth_dir / f"{frame_id}.png"
                            label_path = labels_dir / f"{frame_id}.npz"

                            sample = {
                                "subject": subject_name,
                                "setup": setup_dir.name,
                                "grip": grip_dir.name,
                                "session": session_dir.name,
                                "frame_id": frame_id,
                                "color_path": str(color_path),
                                "depth_path": str(depth_path) if depth_path.exists() else None,
                                "label_path": str(label_path) if label_path.exists() else None,
                            }
                            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 加载RGB图像
        rgb = Image.open(sample["color_path"]).convert("RGB")

        result = {
            "rgb": rgb,
            "subject": sample["subject"],
            "setup": sample["setup"],
            "grip": sample["grip"],
            "session": sample["session"],
            "frame_id": sample["frame_id"],
        }

        # 加载深度图
        if self.load_depth and sample["depth_path"]:
            depth = np.array(Image.open(sample["depth_path"]))
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            result["depth"] = depth

        # 加载标注（物体位姿 + 手部姿态）
        if self.load_pose and sample["label_path"]:
            try:
                label = np.load(sample["label_path"], allow_pickle=True)
                # 物体6D位姿
                if "pose_y" in label:
                    result["object_pose"] = label["pose_y"]  # (N, 3, 4)
                if "pose_m" in label:
                    result["hand_pose"] = label["pose_m"]  # MANO参数
                if "label" in label:
                    result["seg_mask"] = label["label"]  # 分割标签
            except Exception as e:
                print(f"加载标注失败: {sample['label_path']}, 错误: {e}")

        # 应用变换
        if self.transform:
            result = self.transform(result)

        return result


def get_dexycb_grasp_splits(root_dir: str, subject: str) -> Dict[str, List[str]]:
    """
    获取抓取类型划分（用于按抓取类型做微调）

    DexYCB中的抓取类型:
    - 10个抓取类别（按YCB物体划分）
    """
    subject_dir = Path(root_dir) / subject
    if not subject_dir.exists():
        return {}

    grip_types = set()
    for setup_dir in subject_dir.iterdir():
        if setup_dir.is_dir():
            for grip_dir in setup_dir.iterdir():
                if grip_dir.is_dir():
                    grip_types.add(grip_dir.name)

    return {"all_grips": sorted(grip_types)}
