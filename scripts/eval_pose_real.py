"""
真实数据位姿评估脚本
在 YCB-Video 真实测试集上评估位姿模型，计算 ADD / ADI / 投影误差 / 5cm5°准确率

用法:
    python scripts/eval_pose_real.py --checkpoint outputs/baseline_rgb/checkpoints/best.pth
    python scripts/eval_pose_real.py --checkpoint outputs/baseline_rgbd/checkpoints/best.pth --use_depth
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ycb_video import YCBVideoPoseDataset
from src.data.transforms import get_val_transforms
from src.models.pose_estimator import PoseEstimatorResNet
from src.utils.metrics import (
    compute_add,
    compute_adi,
    compute_projection_error,
    project_points,
)


# YCB-Video 21 个物体 ID -> 名称映射
YCB_OBJECTS = {
    1: "001_master_chef_can",
    2: "002_cracker_box",
    3: "003_sugar_box",
    4: "004_tomato_soup_can",
    5: "005_mustard_bottle",
    6: "006_tuna_fish_can",
    7: "007_pudding_box",
    8: "008_gelatin_box",
    9: "009_potted_meat_can",
    10: "010_banana",
    11: "011_pitcher_base",
    12: "012_bleach_cleanser",
    13: "013_bowl",
    14: "014_mug",
    15: "015_power_drill",
    16: "016_wood_block",
    17: "017_scissors",
    18: "018_large_marker",
    19: "019_large_clamp",
    20: "020_extra_large_clamp",
    21: "021_foam_brick",
}

# 对称物体（用 ADI 评估）
SYMMETRIC_OBJECTS = [1, 4, 6, 13, 16, 18, 19, 20]


def load_model_points(models_dir: str, obj_id: int, num_points: int = 3000) -> np.ndarray:
    """加载物体3D模型点云"""
    obj_name = YCB_OBJECTS.get(obj_id, f"obj_{obj_id:06d}")
    ply_path = Path(models_dir) / f"obj_{obj_id:06d}.ply"

    if not ply_path.exists():
        # 试试 models_eval
        ply_path2 = Path(models_dir).parent / "models_eval" / f"obj_{obj_id:06d}.ply"
        if ply_path2.exists():
            ply_path = ply_path2
        else:
            return None

    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        if len(mesh.vertices) == 0:
            return None
        # 采样点
        pcd = mesh.sample_points_poisson_disk(num_points)
        points = np.asarray(pcd.points).astype(np.float32)
        return points
    except ImportError:
        # 没有 open3d，从 ply 手动读顶点
        try:
            with open(ply_path) as f:
                lines = f.readlines()
            # 找 vertex 数量
            n_verts = 0
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                if line.strip() == "end_header":
                    header_end = i + 1
                    break
            # 读顶点
            points = []
            for i in range(header_end, header_end + n_verts):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            points = np.array(points, dtype=np.float32)
            # 下采样到 num_points
            if len(points) > num_points:
                idx = np.random.choice(len(points), num_points, replace=False)
                points = points[idx]
            return points / 1000.0  # mm -> m
        except Exception:
            return None


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) 转旋转矩阵"""
    w, x, y, z = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


def rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """旋转误差（度）"""
    R_rel = R_pred @ R_gt.T
    trace = np.trace(R_rel)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    return np.degrees(angle)


def parse_args():
    parser = argparse.ArgumentParser(description="真实数据位姿评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="datasets/ycbv")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="3D模型目录，默认 data_path/models")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split_config", type=str, default="datasets/ycbv/scene_split.json")
    parser.add_argument("--split_name", type=str, default="test",
                        help="使用 split_config 中的哪个划分 (val/test)")
    parser.add_argument("--img_size", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径 (JSON)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型目录
    models_dir = args.models_dir or str(Path(args.data_path) / "models")

    # 场景筛选
    scene_filter = None
    if args.split_config and Path(args.split_config).exists():
        with open(args.split_config) as f:
            cfg = json.load(f)
        scene_filter = cfg.get(args.split_name)
        print(f"使用划分: {args.split_name} ({len(scene_filter) if scene_filter else '全部'} 个场景)")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    input_channels = 4 if config.get("use_depth", args.use_depth) else 3
    model = PoseEstimatorResNet(
        backbone=config.get("backbone", "resnet18"),
        input_channels=input_channels,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Backbone: {config.get('backbone', 'resnet18')}")
    print(f"  输入通道: {input_channels}")

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
        scene_filter=scene_filter,
    )
    print(f"评估集大小: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 推理收集
    predictions = []
    print("\n开始推理...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if args.use_depth:
                x = batch["rgbd"].to(device)
            else:
                x = batch["rgb"].to(device)

            pred = model(x)
            pred_quat = pred["rotation"].cpu().numpy()
            pred_trans = pred["translation"].cpu().numpy()

            gt_rot = batch["rotation"].numpy()
            gt_trans = batch["translation"].numpy()
            obj_ids = batch["obj_id"].numpy()
            K = batch["camera_intrinsics"].numpy()

            for j in range(len(pred_quat)):
                R_pred = quat_to_rotmat(pred_quat[j])
                predictions.append({
                    "obj_id": int(obj_ids[j]),
                    "R_pred": R_pred,
                    "t_pred": pred_trans[j],
                    "R_gt": gt_rot[j],
                    "t_gt": gt_trans[j],
                    "K": K[j],
                })

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(dataloader)} batches")

    print(f"完成推理，共 {len(predictions)} 个样本")

    # 计算各指标
    print("\n" + "=" * 60)
    print("评估结果（真实数据 YCB-Video）")
    print("=" * 60)

    # 预加载模型点云
    model_points = {}
    obj_ids_in_data = set(p["obj_id"] for p in predictions)
    for obj_id in obj_ids_in_data:
        pts = load_model_points(models_dir, obj_id)
        if pts is not None:
            model_points[obj_id] = pts

    print(f"成功加载 {len(model_points)}/{len(obj_ids_in_data)} 个物体的3D模型")

    # 计算每个样本的指标
    add_values = []
    adi_values = []
    proj_errors = []
    trans_errors = []
    rot_errors = []
    per_obj = {}

    for pred in predictions:
        obj_id = pred["obj_id"]
        R_pred, t_pred = pred["R_pred"], pred["t_pred"]
        R_gt, t_gt = pred["R_gt"], pred["t_gt"]
        K = pred["K"]

        # 平移误差
        t_err = np.linalg.norm(t_pred - t_gt)
        trans_errors.append(t_err)

        # 旋转误差
        r_err = rotation_error(R_pred, R_gt)
        rot_errors.append(r_err)

        # ADD / ADI
        if obj_id in model_points:
            pts = model_points[obj_id]
            add_val = compute_add(R_pred, t_pred, R_gt, t_gt, pts)
            adi_val = compute_adi(R_pred, t_pred, R_gt, t_gt, pts)
            add_values.append(add_val)
            adi_values.append(adi_val)

            # 投影误差
            proj_err = compute_projection_error(R_pred, t_pred, R_gt, t_gt, pts, K)
            proj_errors.append(proj_err)

        # 分物体统计
        if obj_id not in per_obj:
            per_obj[obj_id] = {
                "count": 0,
                "trans_err_sum": 0.0,
                "rot_err_sum": 0.0,
                "add_sum": 0.0,
                "add_count": 0,
                "accurate_5cm5deg": 0,
            }
        per_obj[obj_id]["count"] += 1
        per_obj[obj_id]["trans_err_sum"] += t_err
        per_obj[obj_id]["rot_err_sum"] += r_err
        if obj_id in model_points:
            per_obj[obj_id]["add_sum"] += add_val
            per_obj[obj_id]["add_count"] += 1
        if t_err < 0.05 and r_err < 5.0:
            per_obj[obj_id]["accurate_5cm5deg"] += 1

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    add_values = np.array(add_values) if add_values else np.array([0])
    adi_values = np.array(adi_values) if adi_values else np.array([0])
    proj_errors = np.array(proj_errors) if proj_errors else np.array([0])

    # 总体结果
    print(f"\n总体指标:")
    print(f"  样本数: {len(predictions)}")
    print(f"  平移误差 (cm): 均值 {trans_errors.mean()*100:.2f}  中位数 {np.median(trans_errors)*100:.2f}  RMSE {np.sqrt(np.mean(trans_errors**2))*100:.2f}")
    print(f"  旋转误差 (°):  均值 {rot_errors.mean():.2f}  中位数 {np.median(rot_errors):.2f}")
    print(f"  5cm/5° 准确率: {( (trans_errors < 0.05) & (rot_errors < 5.0) ).mean()*100:.1f}%")

    if len(add_values) > 0:
        print(f"  ADD (cm): 均值 {add_values.mean()*100:.2f}  中位数 {np.median(add_values)*100:.2f}")
        print(f"  ADI (cm): 均值 {adi_values.mean()*100:.2f}  中位数 {np.median(adi_values)*100:.2f}")
    if len(proj_errors) > 0:
        print(f"  投影误差 (px): 均值 {proj_errors.mean():.2f}  中位数 {np.median(proj_errors):.2f}")

    # 分物体结果
    print(f"\n分物体结果:")
    print(f"  {'物体':<25s} {'样本':>5s} {'平移(cm)':>10s} {'旋转(°)':>8s} {'5cm5°':>8s} {'ADD(cm)':>9s}")
    print(f"  {'-'*75}")
    for obj_id in sorted(per_obj.keys()):
        info = per_obj[obj_id]
        name = YCB_OBJECTS.get(obj_id, f"obj_{obj_id}")
        t_avg = info["trans_err_sum"] / info["count"] * 100
        r_avg = info["rot_err_sum"] / info["count"]
        acc = info["accurate_5cm5deg"] / info["count"] * 100
        add_avg = (info["add_sum"] / info["add_count"] * 100) if info["add_count"] > 0 else -1
        add_str = f"{add_avg:.2f}" if add_avg >= 0 else "N/A"
        print(f"  {name:<25s} {info['count']:>5d} {t_avg:>10.2f} {r_avg:>8.2f} {acc:>7.1f}% {add_str:>9s}")

    # 保存结果
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "checkpoint": args.checkpoint,
            "dataset": "YCB-Video (real)",
            "split": args.split_name,
            "num_samples": len(predictions),
            "use_depth": args.use_depth,
            "metrics": {
                "translation_mean_cm": float(trans_errors.mean() * 100),
                "translation_median_cm": float(np.median(trans_errors) * 100),
                "translation_rmse_cm": float(np.sqrt(np.mean(trans_errors**2)) * 100),
                "rotation_mean_deg": float(rot_errors.mean()),
                "rotation_median_deg": float(np.median(rot_errors)),
                "accuracy_5cm_5deg": float(((trans_errors < 0.05) & (rot_errors < 5.0)).mean()),
            },
            "per_object": {},
        }
        if len(add_values) > 0:
            result["metrics"]["add_mean_cm"] = float(add_values.mean() * 100)
            result["metrics"]["add_median_cm"] = float(np.median(add_values) * 100)
            result["metrics"]["adi_mean_cm"] = float(adi_values.mean() * 100)
            result["metrics"]["projection_error_mean_px"] = float(proj_errors.mean())
        for obj_id in per_obj:
            info = per_obj[obj_id]
            result["per_object"][str(obj_id)] = {
                "name": YCB_OBJECTS.get(obj_id, f"obj_{obj_id}"),
                "count": info["count"],
                "translation_mean_cm": float(info["trans_err_sum"] / info["count"] * 100),
                "rotation_mean_deg": float(info["rot_err_sum"] / info["count"]),
                "accuracy_5cm_5deg": float(info["accurate_5cm5deg"] / info["count"]),
            }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
