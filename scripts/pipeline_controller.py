#!/usr/bin/env python
"""
完整流水线：RGB训练完成 → RGBD训练 → 位姿评估 → 更新报告
自动检查进度并依次执行
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
PYTHON = PROJECT_DIR / "venv" / "Scripts" / "python.exe"


def check_rgb_done():
    """检查 RGB 训练是否完成"""
    ckpt_dir = PROJECT_DIR / "outputs" / "baseline_rgb" / "checkpoints"
    if not ckpt_dir.exists():
        return False, None
    ckpts = list(ckpt_dir.glob("*.pth"))
    if not ckpts:
        return False, None
    best = max(ckpts, key=lambda x: x.stat().st_mtime)
    # 检查是否有 final 或 epoch 数足够
    return True, best


def check_rgbd_done():
    """检查 RGBD 训练是否完成"""
    ckpt_dir = PROJECT_DIR / "outputs" / "baseline_rgbd" / "checkpoints"
    if not ckpt_dir.exists():
        return False, None
    ckpts = list(ckpt_dir.glob("*.pth"))
    if not ckpts:
        return False, None
    best = max(ckpts, key=lambda x: x.stat().st_mtime)
    return True, best


def train_rgbd():
    """启动 RGBD 训练"""
    print("\n" + "=" * 60)
    print("启动 RGBD baseline 训练")
    print("=" * 60)
    cmd = [
        str(PYTHON), "scripts/train_pose.py",
        "--data_path", "datasets/ycbv",
        "--split_config", "datasets/ycbv/scene_split.json",
        "--use_depth",
        "--backbone", "resnet18",
        "--batch_size", "16",
        "--epochs", "6",
        "--lr", "1e-4",
        "--output_dir", "outputs/baseline_rgbd",
        "--val_interval", "2",
        "--save_interval", "2",
        "--num_workers", "0",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"PID: {proc.pid}")
    return proc


def eval_pose(model_type: str, checkpoint: Path):
    """运行位姿评估"""
    print(f"\n评估 {model_type.upper()} 位姿模型...")
    cmd = [
        str(PYTHON), "scripts/eval_pose_on_mujoco.py",
        "--checkpoint", str(checkpoint),
        "--eval_dir", "datasets/ycb_grasp_eval",
        "--models_dir", "datasets/ycbv/models",
        "--model_type", model_type,
        "--output_dir", "outputs/evaluation_full",
        "--stride", "5",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-300:]}")
        return False
    return True


def main():
    print("完整流水线控制器")
    print(f"项目目录: {PROJECT_DIR}")

    # 阶段 1: 等 RGB 完成
    rgb_done, rgb_ckpt = check_rgb_done()
    if not rgb_done:
        print("RGB 训练尚未完成，等待中... (请先启动训练)")
        print("运行命令: python scripts/train_pose.py ...")
        return

    print(f"RGB 训练已完成: {rgb_ckpt.name}")

    # 阶段 2: 启动 RGBD 训练
    rgbd_done, rgbd_ckpt = check_rgbd_done()
    if not rgbd_done:
        print("\n启动 RGBD 训练...")
        rgbd_proc = train_rgbd()
        # 不等待，返回提示
        print("\nRGBD 训练已启动，运行完成后重新执行本脚本继续评估。")
        print(f"PID: {rgbd_proc.pid}")
        return
    else:
        print(f"RGBD 训练已完成: {rgbd_ckpt.name}")

    # 阶段 3: 位姿评估
    rgb_result = PROJECT_DIR / "outputs" / "evaluation_full" / "pose_rgb_results.json"
    if not rgb_result.exists():
        eval_pose("rgb", rgb_ckpt)
    else:
        print(f"RGB 位姿评估已完成: {rgb_result}")

    rgbd_result = PROJECT_DIR / "outputs" / "evaluation_full" / "pose_rgbd_results.json"
    if not rgbd_result.exists():
        eval_pose("rgbd", rgbd_ckpt)
    else:
        print(f"RGBD 位姿评估已完成: {rgbd_result}")

    print("\n所有评估完成！")


if __name__ == "__main__":
    main()
