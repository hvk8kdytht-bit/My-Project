"""
可视化工具
- 位姿估计结果2D投影可视化
- 跟踪bbox可视化
- 数据集样本可视化
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, List


def visualize_bbox(
    image: Image.Image,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
    label: Optional[str] = None,
) -> Image.Image:
    """
    在图像上绘制bbox

    Args:
        image: PIL Image
        bbox: [x, y, w, h]
        color: RGB颜色
        width: 线宽
        label: 标签文本

    Returns:
        标注后的图像
    """
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

    if label:
        try:
            font = ImageFont.load_default()
        except:
            font = None
        draw.text((x, y - 15), label, fill=color, font=font)

    return image


def visualize_pose_2d(
    image: Image.Image,
    model_points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    point_size: int = 1,
) -> Image.Image:
    """
    将3D模型点投影到2D并可视化

    Args:
        image: PIL Image
        model_points: 3D模型点 (N, 3)
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
        K: 相机内参 (3, 3)
        color: 点颜色
        point_size: 点大小

    Returns:
        标注后的图像
    """
    # 投影3D点到2D
    points_cam = (R @ model_points.T).T + t
    points_2d_h = (K @ points_cam.T).T
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]

    draw = ImageDraw.Draw(image)

    for pt in points_2d:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < image.width and 0 <= y < image.height:
            draw.ellipse(
                [x - point_size, y - point_size, x + point_size, y + point_size],
                fill=color,
            )

    return image


def create_pose_overlay(
    rgb: np.ndarray,
    depth: Optional[np.ndarray] = None,
    pred_points_2d: Optional[np.ndarray] = None,
    gt_points_2d: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    创建位姿估计对比图

    Args:
        rgb: RGB图像 (H, W, 3)
        depth: 深度图 (H, W) 可选
        pred_points_2d: 预测的2D点 (N, 2)
        gt_points_2d: GT的2D点 (N, 2)

    Returns:
        PIL Image
    """
    img = Image.fromarray(rgb).convert("RGB")

    if depth is not None:
        # 深度图转为伪彩色
        depth_normalized = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
        depth_colored = plt.cm.jet(depth_normalized)[:, :, :3]
        depth_img = Image.fromarray((depth_colored * 255).astype(np.uint8))

        # 拼接
        total_width = img.width + depth_img.width
        combined = Image.new("RGB", (total_width, img.height))
        combined.paste(img, (0, 0))
        combined.paste(depth_img, (img.width, 0))
        img = combined

    draw = ImageDraw.Draw(img)

    # 绘制GT点（绿色）
    if gt_points_2d is not None:
        for pt in gt_points_2d:
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 0))

    # 绘制预测点（红色）
    if pred_points_2d is not None:
        for pt in pred_points_2d:
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))

    return img


def visualize_dataset_sample(sample: dict, save_path: Optional[str] = None) -> Image.Image:
    """
    可视化数据集样本

    Args:
        sample: 数据集样本字典（包含 rgb, depth, mask, bbox 等）
        save_path: 保存路径（可选）

    Returns:
        PIL Image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = sample.get("rgb")
    # 延迟导入 torch
    try:
        import torch
        is_tensor = isinstance(rgb, torch.Tensor)
    except ImportError:
        is_tensor = False

    if is_tensor:
        rgb = rgb.permute(1, 2, 0).cpu().numpy()
        rgb = (rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    elif isinstance(rgb, Image.Image):
        rgb = np.array(rgb)

    depth = sample.get("depth")
    if is_tensor and depth is not None and hasattr(depth, 'cpu'):
        depth = depth.squeeze().cpu().numpy()

    mask = sample.get("mask")
    if is_tensor and mask is not None and hasattr(mask, 'cpu'):
        mask = mask.squeeze().cpu().numpy()

    bbox = sample.get("bbox")
    if is_tensor and bbox is not None and hasattr(bbox, 'cpu'):
        bbox = bbox.cpu().numpy()

    # 计算子图数量
    n_plots = 1
    if depth is not None:
        n_plots += 1
    if mask is not None:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    axes[idx].imshow(rgb)
    axes[idx].set_title("RGB")
    if bbox is not None and len(bbox) == 4:
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        axes[idx].add_patch(rect)
    idx += 1

    if depth is not None:
        im = axes[idx].imshow(depth, cmap="jet")
        axes[idx].set_title("Depth")
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        idx += 1

    if mask is not None:
        axes[idx].imshow(mask, cmap="gray")
        axes[idx].set_title("Mask")
        idx += 1

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return Image.open(save_path)

    # 转换为PIL Image返回
    fig.canvas.draw()
    img = Image.fromarray(np.array(fig.canvas.renderer.buffer_rgba()))
    plt.close()
    return img
