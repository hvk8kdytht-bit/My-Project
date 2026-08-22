"""
RGBD1K 数据集加载
用于 RGB-D 目标跟踪预训练

目录结构:
    rgbd1k/
    ├── train/
    │   ├── <video_name>/
    │   │   ├── rgb/            # RGB图像
    │   │   ├── depth/          # 深度图像
    │   │   └── groundtruth.txt # 跟踪标注 (bbox)
    │   └── ...
    └── test/
        ├── <video_name>/
        │   ├── rgb/
        │   ├── depth/
        │   └── groundtruth.txt
        └── ...
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset


class RGBD1KDataset(Dataset):
    """RGBD1K 数据集 - RGB-D 目标跟踪"""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        load_depth: bool = True,
        sequence_length: int = 1,
        max_sequences: Optional[int] = None,
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train' 或 'test'
            transform: 图像变换
            load_depth: 是否加载深度图
            sequence_length: 返回连续帧的数量（1=单帧，>1=视频片段）
            max_sequences: 最多加载多少个视频序列（用于小规模测试）
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.load_depth = load_depth
        self.sequence_length = sequence_length

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"数据集划分 {split} 不存在: {self.split_dir}"
            )

        # 收集所有视频序列
        self.sequences = self._collect_sequences(max_sequences)

        # 构建帧索引（用于单帧模式）
        if sequence_length == 1:
            self.frame_index = self._build_frame_index()
        else:
            self.frame_index = self._build_sequence_index()

    def _collect_sequences(self, max_sequences: Optional[int]) -> List[Dict]:
        """收集所有视频序列信息"""
        sequences = []
        seq_dirs = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])

        if max_sequences is not None:
            seq_dirs = seq_dirs[:max_sequences]

        for seq_dir in seq_dirs:
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            gt_file = seq_dir / "groundtruth.txt"

            if not rgb_dir.exists():
                continue

            # 加载ground truth (bbox格式: x, y, w, h)
            gt_bboxes = {}
            if gt_file.exists():
                with open(gt_file, "r") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split(",")
                            if len(parts) >= 4:
                                try:
                                    frame_name = f"{i:08d}"
                                    x = float(parts[0])
                                    y = float(parts[1])
                                    w = float(parts[2])
                                    h = float(parts[3])
                                    gt_bboxes[frame_name] = [x, y, w, h]
                                except ValueError:
                                    pass

            # 收集RGB帧
            rgb_files = sorted(rgb_dir.glob("*.jpg"))
            if not rgb_files:
                rgb_files = sorted(rgb_dir.glob("*.png"))

            frame_names = [f.stem for f in rgb_files]

            sequences.append({
                "name": seq_dir.name,
                "rgb_dir": str(rgb_dir),
                "depth_dir": str(depth_dir) if depth_dir.exists() else None,
                "frame_names": frame_names,
                "gt_bboxes": gt_bboxes,
                "num_frames": len(frame_names),
            })

        return sequences

    def _build_frame_index(self) -> List[Dict]:
        """构建单帧索引"""
        index = []
        for seq_idx, seq in enumerate(self.sequences):
            for frame_idx, frame_name in enumerate(seq["frame_names"]):
                index.append({
                    "seq_idx": seq_idx,
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                })
        return index

    def _build_sequence_index(self) -> List[Dict]:
        """构建序列索引（用于多帧输入）"""
        index = []
        seq_len = self.sequence_length

        for seq_idx, seq in enumerate(self.sequences):
            num_frames = seq["num_frames"]
            for start_idx in range(0, num_frames - seq_len + 1, seq_len):
                index.append({
                    "seq_idx": seq_idx,
                    "start_frame": start_idx,
                    "end_frame": start_idx + seq_len,
                })
        return index

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> Dict:
        if self.sequence_length == 1:
            return self._get_single_frame(idx)
        else:
            return self._get_sequence(idx)

    def _get_single_frame(self, idx: int) -> Dict:
        """加载单帧数据"""
        entry = self.frame_index[idx]
        seq = self.sequences[entry["seq_idx"]]
        frame_name = entry["frame_name"]

        # RGB图像
        rgb_path = os.path.join(seq["rgb_dir"], f"{frame_name}.jpg")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(seq["rgb_dir"], f"{frame_name}.png")
        rgb = Image.open(rgb_path).convert("RGB")

        result = {
            "rgb": rgb,
            "sequence_name": seq["name"],
            "frame_name": frame_name,
            "frame_idx": entry["frame_idx"],
        }

        # 深度图
        if self.load_depth and seq["depth_dir"]:
            depth_path = os.path.join(seq["depth_dir"], f"{frame_name}.png")
            if os.path.exists(depth_path):
                depth = np.array(Image.open(depth_path))
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / 1000.0
                result["depth"] = depth

        # 跟踪bbox
        if frame_name in seq["gt_bboxes"]:
            result["bbox"] = np.array(seq["gt_bboxes"][frame_name], dtype=np.float32)

        # 应用变换
        if self.transform:
            result = self.transform(result)

        return result

    def _get_sequence(self, idx: int) -> Dict:
        """加载多帧序列"""
        entry = self.frame_index[idx]
        seq = self.sequences[entry["seq_idx"]]

        rgbs = []
        depths = []
        bboxes = []

        for i in range(entry["start_frame"], entry["end_frame"]):
            frame_name = seq["frame_names"][i]

            rgb_path = os.path.join(seq["rgb_dir"], f"{frame_name}.jpg")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(seq["rgb_dir"], f"{frame_name}.png")
            rgb = Image.open(rgb_path).convert("RGB")
            rgbs.append(rgb)

            if self.load_depth and seq["depth_dir"]:
                depth_path = os.path.join(seq["depth_dir"], f"{frame_name}.png")
                if os.path.exists(depth_path):
                    depth = np.array(Image.open(depth_path))
                    if depth.dtype == np.uint16:
                        depth = depth.astype(np.float32) / 1000.0
                    depths.append(depth)

            if frame_name in seq["gt_bboxes"]:
                bboxes.append(seq["gt_bboxes"][frame_name])

        result = {
            "rgb": rgbs,  # list of PIL Images
            "sequence_name": seq["name"],
            "start_frame": entry["start_frame"],
            "end_frame": entry["end_frame"],
            "num_frames": len(rgbs),
        }

        if depths:
            result["depth"] = depths
        if bboxes:
            result["bbox"] = np.array(bboxes, dtype=np.float32)

        if self.transform:
            result = self.transform(result)

        return result


def compute_track_statistics(dataset: RGBD1KDataset) -> Dict:
    """计算数据集统计信息"""
    num_sequences = len(dataset.sequences)
    total_frames = sum(s["num_frames"] for s in dataset.sequences)
    avg_length = total_frames / max(num_sequences, 1)

    labeled_frames = sum(len(s["gt_bboxes"]) for s in dataset.sequences)

    return {
        "num_sequences": num_sequences,
        "total_frames": total_frames,
        "average_sequence_length": avg_length,
        "labeled_frames": labeled_frames,
        "label_ratio": labeled_frames / max(total_frames, 1),
    }
