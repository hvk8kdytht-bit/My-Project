"""
6D 位姿估计评估脚本
计算 ADD / ADI / 投影误差 等指标

用法:
    python scripts/eval_pose.py --checkpoint outputs/pose_baseline/checkpoints/best.pth --data_path H:/Program/datasets/ycb_video
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import YCBVideoPoseDataset, get_val_transforms
from src.models import PoseEstimatorResNet
from src.utils.metrics import evaluate_pose, project_points


def parse_args():
    parser = argparse.ArgumentParser(description="6D位姿估计评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data_path", type=str, required=True, help="数据集根目录")
    parser.add_argument("--split", type=str, default="test", help="评估数据集划分")
    parser.add_argument("--img_size", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    input_channels = 4 if config.get("use_depth", False) else 3
    model = PoseEstimatorResNet(
        backbone=config.get("backbone", "resnet18"),
        input_channels=input_channels,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 数据集
    transform = get_val_transforms(
        img_size=tuple(args.img_size),
        use_depth=args.use_depth,
        concat_rgbd=args.use_depth,
    )

    dataset = YCBVideoPoseDataset(
        root_dir=args.data_path,
        split=args.split,
        transform=transform,
        load_depth=args.use_depth,
    )
    print(f"评估集大小: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 推理收集预测结果
    predictions = []
    print("开始推理...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if args.use_depth:
                x = batch["rgbd"].to(device)
            else:
                x = batch["rgb"].to(device)

            pred = model(x)

            # 将四元数转为旋转矩阵
            pred_quat = pred["rotation"].cpu().numpy()
            pred_trans = pred["translation"].cpu().numpy()

            gt_rot = batch["rotation"].numpy()  # (B, 3, 3)
            gt_trans = batch["translation"].numpy()
            obj_ids = batch["obj_id"].numpy()

            for j in range(len(pred_quat)):
                # 四元数转旋转矩阵
                w, x_q, y, z = pred_quat[j]
                R_pred = np.array([
                    [1 - 2*y*y - 2*z*z, 2*x_q*y - 2*w*z, 2*x_q*z + 2*w*y],
                    [2*x_q*y + 2*w*z, 1 - 2*x_q*x_q - 2*z*z, 2*y*z - 2*w*x_q],
                    [2*x_q*z - 2*w*y, 2*y*z + 2*w*x_q, 1 - 2*x_q*x_q - 2*y*y],
                ])

                predictions.append({
                    "obj_id": int(obj_ids[j]),
                    "R_pred": R_pred,
                    "t_pred": pred_trans[j],
                    "R_gt": gt_rot[j],
                    "t_gt": gt_trans[j],
                })

            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(dataloader)} batches")

    print(f"完成推理，共 {len(predictions)} 个样本")

    # 注意：完整评估需要3D模型点和直径信息
    # 这里先输出基本统计
    print("\n" + "=" * 50)
    print("初步评估结果（需加载3D模型以计算ADD/ADI）")
    print("=" * 50)

    # 计算平移误差
    trans_errors = []
    rot_errors = []
    for pred in predictions:
        trans_err = np.linalg.norm(pred["t_pred"] - pred["t_gt"])
        trans_errors.append(trans_err)

        # 旋转误差（测地线距离）
        R_rel = pred["R_pred"] @ pred["R_gt"].T
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    print(f"平移误差 (米):")
    print(f"  均值:   {trans_errors.mean():.4f}")
    print(f"  中位数: {np.median(trans_errors):.4f}")
    print(f"  RMSE:   {np.sqrt(np.mean(trans_errors**2)):.4f}")

    print(f"\n旋转误差 (度):")
    print(f"  均值:   {rot_errors.mean():.2f}")
    print(f"  中位数: {np.median(rot_errors):.2f}")

    # 5cm/5度 准确率
    accurate = (trans_errors < 0.05) & (rot_errors < 5.0)
    print(f"\n5cm/5° 准确率: {accurate.mean()*100:.1f}%")

    print(f"\n提示: 加载物体3D模型后可计算 ADD/ADI 等标准指标")


if __name__ == "__main__":
    main()
