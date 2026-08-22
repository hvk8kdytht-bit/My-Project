"""
6D 位姿估计训练脚本（纯视觉 baseline）

用法:
    python scripts/train_pose.py --data_path H:/Program/datasets/ycb_video --output_dir outputs/pose_baseline

数据流程:
    YCB-Video (BOP格式) -> DataLoader -> PoseEstimatorResNet -> PoseLoss -> 优化
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import YCBVideoPoseDataset, get_train_transforms, get_val_transforms
from src.models import PoseEstimatorResNet, PoseLoss


def parse_args():
    parser = argparse.ArgumentParser(description="纯视觉6D位姿估计训练")

    # 数据
    parser.add_argument("--data_path", type=str, required=True,
                        help="YCB-Video数据集根目录")
    parser.add_argument("--split_config", type=str, default=None,
                        help="场景级划分配置JSON（含train/val/test场景列表）")
    parser.add_argument("--data_split_dir", type=str, default="test",
                        help="数据所在子目录名（test_all解压后为test）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制训练样本数（调试/冒烟测试用）")
    parser.add_argument("--train_stride", type=int, default=1,
                        help="训练帧采样间隔（连续视频帧冗余，建议3-5）")
    parser.add_argument("--val_stride", type=int, default=1,
                        help="验证帧采样间隔")
    parser.add_argument("--img_size", type=int, nargs=2, default=[480, 640],
                        help="输入图像尺寸 [H, W]")
    parser.add_argument("--use_depth", action="store_true",
                        help="是否使用深度图（RGBD 4通道输入）")
    parser.add_argument("--single_object", type=int, default=None,
                        help="只训练单个物体（物体ID），None表示所有物体")

    # 模型
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="Backbone网络")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="使用预训练权重")

    # 训练
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # 损失权重
    parser.add_argument("--rot_weight", type=float, default=1.0,
                        help="旋转损失权重")
    parser.add_argument("--trans_weight", type=float, default=100.0,
                        help="平移损失权重（平移数值小，需要更大权重）")

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/pose_baseline")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 继续训练的路径")

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, log_interval):
    model.train()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # 输入
        if "rgbd" in batch:
            x = batch["rgbd"].to(device)
        elif "depth" in batch and batch["depth"] is not None:
            x = torch.cat([batch["rgb"], batch["depth"]], dim=1).to(device)
        else:
            x = batch["rgb"].to(device)

        # 目标
        target = {
            "rotation": batch["rotation"].to(device),
            "translation": batch["translation"].to(device),
        }

        # 前向传播
        pred = model(x)
        loss_dict = criterion(pred, target)
        loss = loss_dict["total_loss"]

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_rot_loss += loss_dict["rotation_loss"].item()
        total_trans_loss += loss_dict["translation_loss"].item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} "
                  f"(rot: {loss_dict['rotation_loss'].item():.4f}, "
                  f"trans: {loss_dict['translation_loss'].item():.4f})")

    avg_loss = total_loss / max(num_batches, 1)
    avg_rot = total_rot_loss / max(num_batches, 1)
    avg_trans = total_trans_loss / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "rotation_loss": avg_rot,
        "translation_loss": avg_trans,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_rot_loss = 0.0
    total_trans_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if "rgbd" in batch:
            x = batch["rgbd"].to(device)
        elif "depth" in batch and batch["depth"] is not None:
            x = torch.cat([batch["rgb"], batch["depth"]], dim=1).to(device)
        else:
            x = batch["rgb"].to(device)

        target = {
            "rotation": batch["rotation"].to(device),
            "translation": batch["translation"].to(device),
        }

        pred = model(x)
        loss_dict = criterion(pred, target)

        total_loss += loss_dict["total_loss"].item()
        total_rot_loss += loss_dict["rotation_loss"].item()
        total_trans_loss += loss_dict["translation_loss"].item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_rot = total_rot_loss / max(num_batches, 1)
    avg_trans = total_trans_loss / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "rotation_loss": avg_rot,
        "translation_loss": avg_trans,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # TensorBoard
    writer = SummaryWriter(output_dir / "logs")

    # 数据加载
    input_channels = 4 if args.use_depth else 3

    train_transform = get_train_transforms(
        img_size=tuple(args.img_size),
        use_depth=args.use_depth,
        concat_rgbd=args.use_depth,
    )
    val_transform = get_val_transforms(
        img_size=tuple(args.img_size),
        use_depth=args.use_depth,
        concat_rgbd=args.use_depth,
    )

    # 场景级划分
    split_scenes = {"train": None, "val": None}
    if args.split_config:
        with open(args.split_config) as f:
            cfg = json.load(f)
        split_scenes["train"] = cfg.get("train")
        split_scenes["val"] = cfg.get("val")
        print(f"使用场景划分配置: {args.split_config}")
        print(f"  训练场景: {split_scenes['train']}")
        print(f"  验证场景: {split_scenes['val']}")

    print("加载训练集...")
    train_dataset = YCBVideoPoseDataset(
        root_dir=args.data_path,
        split=args.data_split_dir,
        transform=train_transform,
        load_depth=args.use_depth,
        single_object=args.single_object,
        scene_filter=split_scenes["train"],
    )
    if args.train_stride > 1:
        train_dataset.pose_samples = train_dataset.pose_samples[::args.train_stride]
    if args.max_samples and len(train_dataset) > args.max_samples:
        train_dataset.pose_samples = train_dataset.pose_samples[:args.max_samples]
    print(f"  训练样本数: {len(train_dataset)}")

    # 验证集
    print("加载验证集...")
    val_dataset = YCBVideoPoseDataset(
        root_dir=args.data_path,
        split=args.data_split_dir,
        transform=val_transform,
        load_depth=args.use_depth,
        single_object=args.single_object,
        scene_filter=split_scenes["val"],
    )
    if args.val_stride > 1:
        val_dataset.pose_samples = val_dataset.pose_samples[::args.val_stride]
    print(f"  验证样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 模型
    print(f"创建模型: {args.backbone} (输入通道: {input_channels})")
    model = PoseEstimatorResNet(
        backbone=args.backbone,
        input_channels=input_channels,
        pretrained=args.pretrained,
    ).to(device)

    # 损失与优化器
    criterion = PoseLoss(
        rotation_weight=args.rot_weight,
        translation_weight=args.trans_weight,
        use_geodesic=True,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_val_loss = float("inf")
    start_epoch = 1
    
    # Resume 从 checkpoint 继续
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n从 checkpoint 恢复: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"  起始 epoch: {start_epoch}, best_val_loss: {best_val_loss:.4f}")
        else:
            print(f"  Warning: checkpoint 不存在 {resume_path}，从头开始训练")
    
    print(f"\n开始训练 (共 {args.epochs} epoch, 从 epoch {start_epoch} 开始)...")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.log_interval,
        )
        print(f"训练 Loss: {train_metrics['loss']:.4f} "
              f"(rot: {train_metrics['rotation_loss']:.4f}, "
              f"trans: {train_metrics['translation_loss']:.4f})")

        # 验证
        if epoch % args.val_interval == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"验证 Loss: {val_metrics['loss']:.4f} "
                  f"(rot: {val_metrics['rotation_loss']:.4f}, "
                  f"trans: {val_metrics['translation_loss']:.4f})")

            # 保存最佳模型
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": vars(args),
                }, ckpt_dir / "best.pth")
                print(f"  ✅ 保存最佳模型 (val_loss: {best_val_loss:.4f})")

            # TensorBoard
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # TensorBoard 训练日志
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # 定期保存
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
            }, ckpt_dir / f"epoch_{epoch}.pth")

        scheduler.step()

    # 保存最终模型
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
    }, ckpt_dir / "last.pth")

    print(f"\n{'='*50}")
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {ckpt_dir}")
    print(f"{'='*50}")

    writer.close()


if __name__ == "__main__":
    main()
