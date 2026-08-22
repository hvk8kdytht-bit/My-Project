"""
纯视觉 6D 位姿估计 baseline 模型
基于 CNN 直接回归物体 6D 位姿（旋转 + 平移）

Baseline 方法:
1. CNN backbone 提取特征
2. 旋转用四元数表示 (4维)
3. 平移用3D向量表示 (3维)
4. 损失函数: 旋转用L1 loss，平移用L1 loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PoseEstimatorCNN(nn.Module):
    """
    简单的CNN位姿估计器（轻量级，用于快速验证）
    输入: RGB图像 (3, H, W) 或 RGBD图像 (4, H, W)
    输出: 四元数 (4,) + 平移向量 (3,)
    """

    def __init__(
        self,
        input_channels: int = 3,
        img_h: int = 480,
        img_w: int = 640,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.img_h = img_h
        self.img_w = img_w

        # 简单的CNN特征提取
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 计算特征图大小
        feat_h = img_h // 128  # 经过5次下采样(2^5=32? 实际计算)
        feat_w = img_w // 128

        # 实际计算: 7x7 stride2 -> H/2, maxpool -> H/4
        # 5x5 stride2 -> H/8, maxpool -> H/16
        # 3x3 stride2 -> H/32
        # 3x3 stride2 -> H/64
        # 3x3 stride2 -> H/128
        # 对480x640: 480/128=3.75 -> 3, 640/128=5

        # 用自适应池化来处理
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 旋转回归头 (四元数)
        self.rotation_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

        # 平移回归头
        self.translation_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            dict with:
                rotation: 四元数 (B, 4)
                translation: 平移向量 (B, 3)
        """
        # 特征提取
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)  # (B, 512)

        # 位姿回归
        rotation = self.rotation_head(feat)
        # 归一化四元数
        rotation = F.normalize(rotation, p=2, dim=1)

        translation = self.translation_head(feat)

        return {
            "rotation": rotation,
            "translation": translation,
        }


class PoseEstimatorResNet(nn.Module):
    """
    基于 ResNet 的位姿估计器（更强的 baseline）
    使用预训练 ResNet18/34/50 作为 backbone
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        input_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_channels = input_channels

        # 加载预训练ResNet
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            feat_dim = 512
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            feat_dim = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"不支持的backbone: {backbone}")

        # 处理输入通道数（如果是RGBD，修改第一层）
        if input_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )
            # 用RGB通道的权重初始化前3个通道，深度通道随机初始化
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if input_channels > 3:
                    nn.init.kaiming_normal_(new_conv.weight[:, 3:])
            resnet.conv1 = new_conv

        # 提取backbone（去掉最后的fc层）
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = feat_dim

        # 旋转回归头
        self.rotation_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

        # 平移回归头
        self.translation_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            dict with rotation (B,4) and translation (B,3)
        """
        feat = self.backbone(x)
        feat = feat.flatten(1)  # (B, feat_dim)

        rotation = self.rotation_head(feat)
        rotation = F.normalize(rotation, p=2, dim=1)

        translation = self.translation_head(feat)

        return {
            "rotation": rotation,
            "translation": translation,
        }


class PoseLoss(nn.Module):
    """
    6D位姿估计损失函数
    - 旋转: 四元数 L1 损失 + 可选的测地线距离
    - 平移: L1 损失
    """

    def __init__(
        self,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        use_geodesic: bool = True,
    ):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.use_geodesic = use_geodesic

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: {'rotation': (B,4) quat, 'translation': (B,3)}
            target: {'rotation': (B,3,3) rot_mat or (B,4) quat,
                     'translation': (B,3)}

        Returns:
            dict with total_loss and individual losses
        """
        pred_rot = pred["rotation"]  # (B, 4) quaternion
        pred_trans = pred["translation"]  # (B, 3)

        target_rot = target["rotation"]
        target_trans = target["translation"]

        # 如果目标是旋转矩阵，转为四元数
        if target_rot.dim() == 3:
            target_rot = rotation_matrix_to_quaternion(target_rot)

        # 平移损失 (L1)
        trans_loss = F.l1_loss(pred_trans, target_trans)

        # 旋转损失
        if self.use_geodesic:
            # 测地线距离（四元数角度差）
            rot_loss = quaternion_geodesic_loss(pred_rot, target_rot)
        else:
            # L1 损失 (注意四元数的双覆盖问题: q 和 -q 表示同一旋转)
            rot_loss = quaternion_l1_loss(pred_rot, target_rot)

        total_loss = (
            self.rotation_weight * rot_loss
            + self.translation_weight * trans_loss
        )

        return {
            "total_loss": total_loss,
            "rotation_loss": rot_loss,
            "translation_loss": trans_loss,
        }


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为四元数 (w, x, y, z)
    Args:
        R: 旋转矩阵 (B, 3, 3)
    Returns:
        四元数 (B, 4) - (w, x, y, z)
    """
    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)

    for i in range(batch_size):
        m = R[i]
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        q[i] = torch.tensor([w, x, y, z], device=R.device, dtype=R.dtype)

    return q


def quaternion_l1_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """四元数 L1 损失（处理双覆盖 q = -q）"""
    loss1 = F.l1_loss(q1, q2)
    loss2 = F.l1_loss(q1, -q2)
    return torch.min(loss1, loss2)


def quaternion_geodesic_loss(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    四元数测地线距离损失
    角度差 = 2 * arccos(|<q1, q2>|)
    """
    # 点积的绝对值
    dot = torch.abs(torch.sum(q1 * q2, dim=1))
    dot = torch.clamp(dot, -1.0, 1.0)
    # 角度差（弧度）
    angle = 2.0 * torch.acos(dot)
    return angle.mean()
