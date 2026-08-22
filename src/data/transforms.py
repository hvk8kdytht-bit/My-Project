"""
数据增强与预处理变换
用于训练和验证时的图像变换
"""

import numpy as np
from PIL import Image
from typing import Dict, List

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ToTensor:
    """将PIL图像和numpy数组转为Tensor"""
    def __call__(self, sample: Dict) -> Dict:
        if isinstance(sample.get("rgb"), Image.Image):
            sample["rgb"] = TF.to_tensor(sample["rgb"])  # (3, H, W), 0-1

        if "depth" in sample:
            depth = sample["depth"]
            if isinstance(depth, np.ndarray):
                if depth.ndim == 2:
                    depth = depth[np.newaxis, ...]
                sample["depth"] = torch.from_numpy(depth).float()

        if "mask" in sample:
            mask = sample["mask"]
            if isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    mask = mask[np.newaxis, ...]
                sample["mask"] = torch.from_numpy(mask).float()

        if "bbox" in sample and isinstance(sample["bbox"], np.ndarray):
            sample["bbox"] = torch.from_numpy(sample["bbox"]).float()

        return sample


class Resize:
    """调整图像大小"""
    def __init__(self, size=(480, 640)):
        self.size = size  # (H, W)

    def __call__(self, sample: Dict) -> Dict:
        h, w = self.size

        if isinstance(sample.get("rgb"), Image.Image):
            sample["rgb"] = TF.resize(sample["rgb"], (h, w))

        if "depth" in sample:
            depth = sample["depth"]
            if isinstance(depth, torch.Tensor):
                sample["depth"] = TF.resize(depth, (h, w), interpolation=T.InterpolationMode.NEAREST)
            elif isinstance(depth, np.ndarray):
                depth_pil = Image.fromarray(depth)
                depth_pil = depth_pil.resize((w, h), Image.NEAREST)
                sample["depth"] = np.array(depth_pil)

        if "mask" in sample:
            mask = sample["mask"]
            if isinstance(mask, torch.Tensor):
                sample["mask"] = TF.resize(mask, (h, w), interpolation=T.InterpolationMode.NEAREST)

        # 调整bbox
        if "bbox" in sample and "rgb" in sample:
            orig_h, orig_w = sample["rgb"].shape[-2:] if isinstance(sample["rgb"], torch.Tensor) else sample["rgb"].size[::-1]
            scale_x = w / orig_w
            scale_y = h / orig_h
            bbox = sample["bbox"].clone() if isinstance(sample["bbox"], torch.Tensor) else sample["bbox"].copy()
            bbox[..., 0] *= scale_x  # x
            bbox[..., 1] *= scale_y  # y
            bbox[..., 2] *= scale_x  # w
            bbox[..., 3] *= scale_y  # h
            sample["bbox"] = bbox

        return sample


class Normalize:
    """归一化图像"""
    def __init__(
        self,
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.229, 0.224, 0.225],
        depth_mean=None,
        depth_std=None,
    ):
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(self, sample: Dict) -> Dict:
        if "rgb" in sample and isinstance(sample["rgb"], torch.Tensor):
            sample["rgb"] = TF.normalize(sample["rgb"], self.rgb_mean, self.rgb_std)

        if "depth" in sample and isinstance(sample["depth"], torch.Tensor):
            if self.depth_mean is not None and self.depth_std is not None:
                sample["depth"] = (sample["depth"] - self.depth_mean) / self.depth_std
            else:
                # 简单归一化：除以最大深度（假设最大5米）
                sample["depth"] = sample["depth"] / 5.0

        return sample


class RandomHorizontalFlip:
    """随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if torch.rand(1).item() < self.p:
            if isinstance(sample.get("rgb"), (Image.Image, torch.Tensor)):
                sample["rgb"] = TF.hflip(sample["rgb"])

            if "depth" in sample and isinstance(sample["depth"], torch.Tensor):
                sample["depth"] = TF.hflip(sample["depth"])

            if "mask" in sample and isinstance(sample["mask"], torch.Tensor):
                sample["mask"] = TF.hflip(sample["mask"])

            # 翻转bbox的x坐标
            if "bbox" in sample:
                bbox = sample["bbox"]
                w = sample["rgb"].shape[-1] if isinstance(sample["rgb"], torch.Tensor) else sample["rgb"].width
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.clone()
                    bbox[..., 0] = w - bbox[..., 0] - bbox[..., 2]
                else:
                    bbox = bbox.copy()
                    bbox[..., 0] = w - bbox[..., 0] - bbox[..., 2]
                sample["bbox"] = bbox

        return sample


class ColorJitter:
    """颜色抖动（仅对RGB）"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample: Dict) -> Dict:
        if isinstance(sample.get("rgb"), Image.Image):
            sample["rgb"] = self.jitter(sample["rgb"])
        return sample


class RGBDConcat:
    """将RGB和Depth拼接为4通道输入"""
    def __call__(self, sample: Dict) -> Dict:
        if "rgb" in sample and "depth" in sample:
            rgb = sample["rgb"]
            depth = sample["depth"]
            if isinstance(rgb, torch.Tensor) and isinstance(depth, torch.Tensor):
                # 确保depth的尺寸和rgb一致
                if depth.shape[-2:] != rgb.shape[-2:]:
                    depth = TF.resize(depth, rgb.shape[-2:], interpolation=T.InterpolationMode.NEAREST)
                sample["rgbd"] = torch.cat([rgb, depth], dim=0)  # (4, H, W)
        return sample


def get_train_transforms(
    img_size=(480, 640),
    use_depth=True,
    use_color_jitter=True,
    use_flip=True,
    concat_rgbd=False,
):
    """获取训练时的数据变换流水线"""
    transforms = []

    if use_color_jitter:
        transforms.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))

    if use_flip:
        transforms.append(RandomHorizontalFlip(p=0.5))

    transforms.append(ToTensor())
    transforms.append(Resize(img_size))
    transforms.append(Normalize())

    if concat_rgbd and use_depth:
        transforms.append(RGBDConcat())

    return T.Compose(transforms)


def get_val_transforms(
    img_size=(480, 640),
    use_depth=True,
    concat_rgbd=False,
):
    """获取验证时的数据变换流水线"""
    transforms = [
        ToTensor(),
        Resize(img_size),
        Normalize(),
    ]

    if concat_rgbd and use_depth:
        transforms.append(RGBDConcat())

    return T.Compose(transforms)
