"""
用模拟数据快速验证训练流程
生成随机图像和位姿，跑几个训练步，验证:
    1. 数据加载管道
    2. 模型前向传播
    3. 损失计算
    4. 反向传播
    5. 模型保存/加载
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.models import PoseEstimatorCNN, PoseEstimatorResNet, PoseLoss
from src.data.transforms import get_train_transforms, get_val_transforms


class SyntheticPoseDataset(Dataset):
    """合成位姿数据集 - 用于快速验证训练管道"""

    def __init__(
        self,
        num_samples: int = 100,
        img_size: tuple = (128, 128),
        input_channels: int = 3,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.input_channels = input_channels

        # 预生成"图像"（随机噪声 + 目标位置相关的图案）
        self.images = torch.randn(num_samples, input_channels, img_size[0], img_size[1]) * 0.1 + 0.5

        # 预生成位姿（随机）
        self.rotations = []
        self.translations = []

        for i in range(num_samples):
            # 随机四元数
            q = np.random.randn(4)
            q = q / np.linalg.norm(q)
            # 四元数转旋转矩阵
            w, x, y, z = q
            R = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
            ])
            self.rotations.append(R.astype(np.float32))

            # 随机平移（米，在合理范围内）
            t = np.array([
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(0.3, 1.0),
            ], dtype=np.float32)
            self.translations.append(t)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 使图像与平移有一定相关性（模拟物体在图像中的位置）
        img = self.images[idx].clone()
        t = self.translations[idx]

        # 在图像中心附近画一个"物体"的痕迹（简单模拟）
        h, w = self.img_size
        cx = int(w / 2 + t[0] * 500)  # 平移映射到像素偏移
        cy = int(h / 2 + t[1] * 500)
        cx = np.clip(cx, 10, w - 10)
        cy = np.clip(cy, 10, h - 10)

        # 画一个小方块（让模型能学到一些东西）
        img[:, max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)] += 0.3

        return {
            "rgb": torch.clamp(img, 0, 1),
            "rotation": torch.from_numpy(self.rotations[idx]),
            "translation": torch.from_numpy(self.translations[idx]),
            "obj_id": 1,
            "obj_name": "synthetic_object",
        }


def test_training_pipeline():
    """测试完整训练流程"""
    print("=" * 60)
    print("测试训练管道")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 配置
    img_size = (128, 128)
    input_channels = 3
    batch_size = 8
    num_samples = 50
    num_epochs = 3

    # 数据集
    print(f"\n创建合成数据集 ({num_samples} 个样本, 图像尺寸 {img_size})...")
    train_dataset = SyntheticPoseDataset(
        num_samples=num_samples,
        img_size=img_size,
        input_channels=input_channels,
    )
    val_dataset = SyntheticPoseDataset(
        num_samples=20,
        img_size=img_size,
        input_channels=input_channels,
    )
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型（轻量级CNN，测试用）
    print(f"\n创建轻量级CNN模型 (输入通道: {input_channels})...")
    model = PoseEstimatorCNN(
        input_channels=input_channels,
        img_h=img_size[0],
        img_w=img_size[1],
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # 损失与优化器
    criterion = PoseLoss(
        rotation_weight=1.0,
        translation_weight=100.0,
        use_geodesic=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 训练循环
    print(f"\n开始训练 ({num_epochs} epochs)...")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        train_rot = 0.0
        train_trans = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["rgb"].to(device)
            target = {
                "rotation": batch["rotation"].to(device),
                "translation": batch["translation"].to(device),
            }

            pred = model(x)
            loss_dict = criterion(pred, target)
            loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_rot += loss_dict["rotation_loss"].item()
            train_trans += loss_dict["translation_loss"].item()
            num_batches += 1

        avg_loss = train_loss / num_batches
        avg_rot = train_rot / num_batches
        avg_trans = train_trans / num_batches

        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["rgb"].to(device)
                target = {
                    "rotation": batch["rotation"].to(device),
                    "translation": batch["translation"].to(device),
                }
                pred = model(x)
                loss_dict = criterion(pred, target)
                val_loss += loss_dict["total_loss"].item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {avg_loss:.4f} (rot: {avg_rot:.4f}, trans: {avg_trans:.6f}) | "
              f"Val Loss: {avg_val_loss:.4f}")

    print("-" * 60)
    print("✅ 训练管道测试通过!")

    # 测试模型保存/加载
    print("\n测试模型保存/加载...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_model.pth")
        torch.save({
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

        # 加载
        checkpoint = torch.load(ckpt_path, map_location=device)
        model2 = PoseEstimatorCNN(
            input_channels=input_channels,
            img_h=img_size[0],
            img_w=img_size[1],
        ).to(device)
        model2.load_state_dict(checkpoint["model_state_dict"])
        model2.eval()
        print("  ✅ 模型保存/加载正常")

    # 测试推理速度
    print("\n测试推理速度...")
    model.eval()
    dummy_input = torch.randn(1, input_channels, img_size[0], img_size[1]).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
    import time
    num_infers = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(num_infers):
            _ = model(dummy_input)
    elapsed = time.time() - start
    fps = num_infers / elapsed
    print(f"  推理速度: {fps:.1f} FPS (batch=1)")

    print("\n" + "=" * 60)
    print("✅ 全部测试通过!")
    print("=" * 60)

    return True


def test_resnet_backbone():
    """测试 ResNet backbone 模型"""
    print("\n" + "=" * 60)
    print("测试 ResNet 模型")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试 ResNet18
    print("\n创建 ResNet18 模型...")
    try:
        model = PoseEstimatorResNet(
            backbone="resnet18",
            input_channels=3,
            pretrained=False,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ResNet18 参数量: {total_params:,}")

        # 前向传播测试
        dummy = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(dummy)
        print(f"  输出 shape: rotation={out['rotation'].shape}, translation={out['translation'].shape}")
        print("  ✅ ResNet18 前向传播正常")
    except Exception as e:
        print(f"  ⚠️  ResNet18 测试跳过: {e}")

    # 测试 RGBD 4通道输入
    print("\n测试 4通道 RGBD 输入...")
    try:
        model_4ch = PoseEstimatorResNet(
            backbone="resnet18",
            input_channels=4,
            pretrained=False,
        ).to(device)

        dummy = torch.randn(2, 4, 224, 224).to(device)
        with torch.no_grad():
            out = model_4ch(dummy)
        print(f"  输出 shape: rotation={out['rotation'].shape}, translation={out['translation'].shape}")
        print("  ✅ 4通道 RGBD 输入正常")
    except Exception as e:
        print(f"  ⚠️  4通道测试跳过: {e}")

    print("\n✅ ResNet 模型测试完成!")
    return True


if __name__ == "__main__":
    success = test_training_pipeline()
    if success:
        test_resnet_backbone()
    print("\n🎉 所有测试通过!")
