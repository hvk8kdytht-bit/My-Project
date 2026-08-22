"""
简单的 RGB-D 目标跟踪 baseline
基于模板匹配 + 光流的简单跟踪器（作为下限 baseline）
以及基于轻量CNN的跟踪器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTracker(nn.Module):
    """
    简单的CNN跟踪器
    输入: 当前帧 + 模板帧 (或上一帧目标裁剪)
    输出: 目标bbox偏移 (dx, dy, dw, dh)
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # 特征提取（共享权重）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 2, hidden_dim),  # 模板 + 当前帧
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),  # dx, dy, dw, dh
        )

    def forward(
        self,
        current_frame: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_frame: 当前帧 (B, C, H, W)
            template: 目标模板 (B, C, H, W)

        Returns:
            bbox偏移 (B, 4) - [dx, dy, dw, dh]
        """
        feat_cur = self.feature_extractor(current_frame)
        feat_tmpl = self.feature_extractor(template)

        feat = torch.cat([feat_cur.flatten(1), feat_tmpl.flatten(1)], dim=1)
        bbox_delta = self.regressor(feat)

        return bbox_delta


class TrackingLoss(nn.Module):
    """跟踪损失（Smooth L1 + IoU）"""

    def __init__(self, iou_weight: float = 1.0):
        super().__init__()
        self.iou_weight = iou_weight

    def forward(
        self,
        pred_delta: torch.Tensor,
        target_delta: torch.Tensor,
        pred_bbox: torch.Tensor = None,
        target_bbox: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        smooth_l1 = F.smooth_l1_loss(pred_delta, target_delta)

        total_loss = smooth_l1

        if pred_bbox is not None and target_bbox is not None:
            iou = self._compute_iou(pred_bbox, target_bbox)
            iou_loss = 1.0 - iou.mean()
            total_loss = total_loss + self.iou_weight * iou_loss
            return {"total_loss": total_loss, "smooth_l1": smooth_l1, "iou_loss": iou_loss}

        return {"total_loss": total_loss, "smooth_l1": smooth_l1}

    def _compute_iou(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
        """计算IoU，bbox格式: (x, y, w, h)"""
        x1 = torch.max(bbox1[:, 0], bbox2[:, 0])
        y1 = torch.max(bbox1[:, 1], bbox2[:, 1])
        x2 = torch.min(bbox1[:, 0] + bbox1[:, 2], bbox2[:, 0] + bbox2[:, 2])
        y2 = torch.min(bbox1[:, 1] + bbox1[:, 3], bbox2[:, 1] + bbox2[:, 3])

        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = bbox1[:, 2] * bbox1[:, 3]
        area2 = bbox2[:, 2] * bbox2[:, 3]
        union = area1 + area2 - inter

        iou = inter / torch.clamp(union, min=1e-6)
        return iou
