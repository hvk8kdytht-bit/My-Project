"""
数据验证与探索脚本
检查数据集是否存在，并显示基本统计信息

用法:
    python scripts/verify_data.py --dataset ycb_video --path H:/Program/datasets/ycb_video
    python scripts/verify_data.py --dataset dexycb --path D:/datasets/dexycb
    python scripts/verify_data.py --dataset rgbd1k --path H:/Program/datasets/rgbd1k
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_ycb_video(path: str):
    """验证 YCB-Video BOP 格式数据集"""
    print("=" * 60)
    print(f"验证 YCB-Video 数据集: {path}")
    print("=" * 60)

    root = Path(path)

    if not root.exists():
        print(f"❌ 目录不存在: {path}")
        return False

    # 检查必要文件
    checks = {
        "camera.json": root / "camera.json",
        "models 目录": root / "models",
    }

    # 检查数据划分
    for split in ["train_real", "train_pbr", "test"]:
        split_dir = root / split
        if split_dir.exists():
            scene_count = len([d for d in split_dir.iterdir() if d.is_dir()])
            checks[f"{split} ({scene_count}个场景)"] = split_dir
        else:
            checks[f"{split} (缺失)"] = None

    all_ok = True
    for name, p in checks.items():
        if p and p.exists():
            # 计算大小
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  ✅ {name} ({size_mb:.1f} MB)")
            else:
                # 估算目录大小
                try:
                    total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    size_gb = total_size / (1024 ** 3)
                    print(f"  ✅ {name} ({size_gb:.2f} GB)")
                except:
                    print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} - 未找到")
            all_ok = False

    # 统计图像数量
    for split in ["train_real", "test"]:
        split_dir = root / split
        if split_dir.exists():
            total_images = 0
            for scene_dir in split_dir.iterdir():
                if scene_dir.is_dir():
                    rgb_dir = scene_dir / "rgb"
                    if rgb_dir.exists():
                        total_images += len(list(rgb_dir.glob("*.png")))
            print(f"  📊 {split}: {total_images} 张图像")

    print()
    if all_ok:
        print("✅ YCB-Video 数据集验证通过")
    else:
        print("⚠️  YCB-Video 数据集部分缺失")

    return all_ok


def verify_dexycb(path: str):
    """验证 DexYCB 数据集"""
    print("=" * 60)
    print(f"验证 DexYCB 数据集: {path}")
    print("=" * 60)

    root = Path(path)

    if not root.exists():
        print(f"❌ 目录不存在: {path}")
        return False

    checks = {
        "models 目录": root / "models",
        "calibration 目录": root / "calibration",
    }

    # 检查受试者目录
    subject_dirs = sorted([d for d in root.iterdir()
                           if d.is_dir() and d.name.startswith("2020")])
    checks[f"受试者 ({len(subject_dirs)}个)"] = subject_dirs[0] if subject_dirs else None

    all_ok = True
    for name, p in checks.items():
        if p and (p.exists() if isinstance(p, Path) else True):
            if isinstance(p, Path) and p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  ✅ {name} ({size_mb:.1f} MB)")
            elif isinstance(p, Path) and p.is_dir():
                print(f"  ✅ {name}")
            else:
                print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} - 未找到")
            all_ok = False

    # 统计受试者信息
    total_frames = 0
    for subj_dir in subject_dirs:
        subj_frames = 0
        for setup_dir in subj_dir.glob("*"):
            if setup_dir.is_dir():
                for grip_dir in setup_dir.glob("*"):
                    if grip_dir.is_dir():
                        for sess_dir in grip_dir.glob("*"):
                            if sess_dir.is_dir():
                                color_dir = sess_dir / "color"
                                if color_dir.exists():
                                    subj_frames += len(list(color_dir.glob("*.jpg")))
        total_frames += subj_frames
        print(f"  📊 {subj_dir.name}: {subj_frames} 帧")

    if len(subject_dirs) > 1:
        print(f"  📊 总计: {len(subject_dirs)} 个受试者, {total_frames} 帧")

    print()
    if all_ok:
        print("✅ DexYCB 数据集验证通过")
    else:
        print("⚠️  DexYCB 数据集部分缺失")

    return all_ok


def verify_rgbd1k(path: str):
    """验证 RGBD1K 数据集"""
    print("=" * 60)
    print(f"验证 RGBD1K 数据集: {path}")
    print("=" * 60)

    root = Path(path)

    if not root.exists():
        print(f"❌ 目录不存在: {path}")
        return False

    all_ok = True

    for split in ["train", "test"]:
        split_dir = root / split
        if split_dir.exists():
            seq_count = len([d for d in split_dir.iterdir() if d.is_dir()])
            total_frames = 0
            total_gt = 0

            for seq_dir in split_dir.iterdir():
                if seq_dir.is_dir():
                    rgb_dir = seq_dir / "rgb"
                    if rgb_dir.exists():
                        total_frames += len(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
                    gt_file = seq_dir / "groundtruth.txt"
                    if gt_file.exists():
                        with open(gt_file, "r") as f:
                            lines = [l for l in f.readlines() if l.strip() and not l.startswith("#")]
                            total_gt += len(lines)

            print(f"  ✅ {split}: {seq_count} 个序列, {total_frames} 帧, {total_gt} 个标注")
        else:
            print(f"  ❌ {split} - 未找到")
            all_ok = False

    print()
    if all_ok:
        print("✅ RGBD1K 数据集验证通过")
    else:
        print("⚠️  RGBD1K 数据集部分缺失")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="数据集验证与探索")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ycb_video", "dexycb", "rgbd1k", "all"],
                        help="数据集名称")
    parser.add_argument("--path", type=str, required=True,
                        help="数据集根目录路径")
    args = parser.parse_args()

    if args.dataset == "ycb_video":
        verify_ycb_video(args.path)
    elif args.dataset == "dexycb":
        verify_dexycb(args.path)
    elif args.dataset == "rgbd1k":
        verify_rgbd1k(args.path)
    elif args.dataset == "all":
        verify_ycb_video(os.path.join(args.path, "ycb_video"))
        print()
        verify_dexycb(os.path.join(args.path, "dexycb"))
        print()
        verify_rgbd1k(os.path.join(args.path, "rgbd1k"))


if __name__ == "__main__":
    main()
