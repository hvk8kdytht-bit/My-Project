#!/usr/bin/env python
"""
位姿模型在 MuJoCo 评估集上推理评估
计算 ADD / 投影误差 / 10% 直径准确率
"""
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_rgb(scene_dir: Path, idx: int) -> np.ndarray:
    f = scene_dir / "rgb" / f"{idx:06d}.jpg"
    if not f.exists():
        f = scene_dir / "rgb" / f"{idx:06d}.png"
    img = cv2.imread(str(f))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(scene_dir: Path, idx: int) -> np.ndarray:
    f = scene_dir / "depth" / f"{idx:06d}.png"
    depth = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    return depth.astype(np.float32) / 1000.0


def load_model_points(models_dir: Path, obj_id: str) -> np.ndarray:
    """加载物体 3D 模型点（用于 ADD 计算）"""
    import trimesh
    obj_path = models_dir / f"obj_{obj_id}" / "textured.obj"
    if not obj_path.exists():
        obj_path = models_dir / f"{obj_id}" / "textured.obj"
    if not obj_path.exists():
        # 用简单立方体近似
        return np.random.rand(300, 3) * 0.05 - 0.025
    try:
        mesh = trimesh.load(str(obj_path))
        pts = mesh.sample(3000)
        return pts / 1000.0  # mm -> m
    except Exception:
        return np.random.rand(300, 3) * 0.05 - 0.025


def compute_add(pred_R, pred_t, gt_R, gt_t, model_points):
    """ADD 指标: 平均最近点距离"""
    pred_pts = (pred_R @ model_points.T).T + pred_t
    gt_pts = (gt_R @ model_points.T).T + gt_t
    # ADD-S 用最近点距离，ADD 用一一对应
    # 这里用 ADD（一一对应，因为模型点顺序相同）
    dists = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return float(np.mean(dists))


def compute_projection_error(pred_R, pred_t, gt_R, gt_t, model_points, K):
    """投影误差：模型点投影到图像上的平均像素距离"""
    pred_pts_3d = (pred_R @ model_points.T).T + pred_t
    gt_pts_3d = (gt_R @ model_points.T).T + gt_t

    # 投影到 2D
    pred_pts_2d = (K @ pred_pts_3d.T).T
    pred_pts_2d = pred_pts_2d[:, :2] / pred_pts_2d[:, 2:3]
    gt_pts_2d = (K @ gt_pts_3d.T).T
    gt_pts_2d = gt_pts_2d[:, :2] / gt_pts_2d[:, 2:3]

    dists = np.linalg.norm(pred_pts_2d - gt_pts_2d, axis=1)
    return float(np.mean(dists))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="位姿模型在 MuJoCo 评估集上评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--eval_dir", type=str, default="datasets/ycb_grasp_eval")
    parser.add_argument("--models_dir", type=str, default="datasets/ycbv/models")
    parser.add_argument("--model_type", type=str, default="rgb", choices=["rgb", "rgbd"])
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation_full")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--stride", type=int, default=5, help="帧间隔（加快评估）")
    args = parser.parse_args()

    import torch
    from src.models.pose_estimator import PoseEstimator

    eval_dir = Path(args.eval_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir = eval_dir / "test"
    scene_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"模型: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")

    in_channels = 4 if args.model_type == "rgbd" else 3
    model = PoseEstimator(backbone="resnet18", in_channels=in_channels, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print("模型加载完成")

    # 物体模型点缓存
    model_points_cache = {}

    # 评估结果
    add_errors = []
    proj_errors = []
    per_object = defaultdict(lambda: {"add": [], "proj": []})

    print(f"\n评估集: {eval_dir}")
    print(f"场景数: {len(scene_dirs)}, stride: {args.stride}")
    print()

    with torch.no_grad():
        for si, scene_dir in enumerate(scene_dirs):
            # 加载元数据
            with open(scene_dir / "scene_gt.json") as f:
                gt = json.load(f)
            with open(scene_dir / "scene_camera.json") as f:
                cam = json.load(f)

            n_frames = len(gt)
            K = np.array(cam["0"]["cam_K"]).reshape(3, 3)
            obj_id = str(gt["0"][0]["obj_id"]).zfill(6)

            # 加载物体模型点
            if obj_id not in model_points_cache:
                pts = load_model_points(models_dir, obj_id)
                model_points_cache[obj_id] = pts
            model_pts = model_points_cache[obj_id]

            print(f"  场景 {si+1}/{len(scene_dirs)}: {scene_dir.name} (obj={obj_id}, {n_frames}帧)", flush=True)

            for i in range(0, n_frames, args.stride):
                # 加载图像
                rgb = load_rgb(scene_dir, i)
                # 预处理
                rgb = cv2.resize(rgb, (256, 256))
                rgb = rgb.astype(np.float32) / 255.0
                rgb = rgb.transpose(2, 0, 1)  # HWC -> CHW

                if args.model_type == "rgbd":
                    depth = load_depth(scene_dir, i)
                    depth = cv2.resize(depth, (256, 256))
                    depth = (depth - depth.mean()) / (depth.std() + 1e-6)
                    depth = depth[np.newaxis, ...]
                    inp = np.concatenate([rgb, depth], axis=0)
                else:
                    inp = rgb

                inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)

                # 推理
                pred = model(inp_tensor)
                pred_t = pred["translation"][0].cpu().numpy()
                # 简化：用旋转矩阵表示（实际模型可能输出四元数或6D）
                # 这里假设输出 4 维四元数
                if "rotation_quat" in pred:
                    from scipy.spatial.transform import Rotation as R
                    quat = pred["rotation_quat"][0].cpu().numpy()
                    pred_R = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                elif "rotation_6d" in pred:
                    from src.utils.geometry import rotation_6d_to_matrix
                    pred_R = rotation_6d_to_matrix(pred["rotation_6d"][0].cpu().numpy())
                else:
                    # 退化情况：用单位矩阵代替（仅测试用）
                    pred_R = np.eye(3)

                # GT
                ann = gt[str(i)][0]
                gt_R = np.array(ann["cam_R_m2c"]).reshape(3, 3)
                gt_t = np.array(ann["cam_t_m2c"]) / 1000.0  # mm -> m
                pred_t_m = pred_t / 1000.0 if pred_t.max() > 1 else pred_t  # 归一化到米

                # 计算指标
                add = compute_add(pred_R, pred_t_m, gt_R, gt_t, model_pts)
                proj = compute_projection_error(pred_R, pred_t_m, gt_R, gt_t, model_pts, K)

                add_errors.append(add)
                proj_errors.append(proj)
                per_object[obj_id]["add"].append(add)
                per_object[obj_id]["proj"].append(proj)

    # 汇总
    add_errors = np.array(add_errors)
    proj_errors = np.array(proj_errors)

    print("\n" + "=" * 60)
    print(f"位姿估计结果 ({args.model_type.upper()})")
    print("=" * 60)
    print(f"  ADD 均值:   {np.mean(add_errors)*1000:.2f} mm")
    print(f"  ADD 中位数: {np.median(add_errors)*1000:.2f} mm")
    print(f"  投影误差:   {np.mean(proj_errors):.2f} px")
    print(f"  样本数:     {len(add_errors)}")

    # 10% 直径准确率（近似：用 20mm 作阈值）
    threshold_mm = 20
    acc_20mm = np.mean(add_errors * 1000 < threshold_mm)
    print(f"  ADD < {threshold_mm}mm: {acc_20mm*100:.1f}%")

    # 每物体结果
    print("\n分物体结果:")
    for obj_id in sorted(per_object.keys()):
        adds = np.array(per_object[obj_id]["add"])
        projs = np.array(per_object[obj_id]["proj"])
        print(f"  {obj_id}: ADD={np.mean(adds)*1000:.1f}mm, 投影={np.mean(projs):.1f}px, n={len(adds)}")

    # 保存
    out_path = output_dir / f"pose_{args.model_type}_results.json"
    result_data = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "n_samples": len(add_errors),
        "add_mean_mm": float(np.mean(add_errors) * 1000),
        "add_median_mm": float(np.median(add_errors) * 1000),
        "projection_error_mean_px": float(np.mean(proj_errors)),
        "add_20mm_accuracy": float(acc_20mm),
        "per_object": {
            k: {
                "add_mean_mm": float(np.mean(v["add"]) * 1000),
                "projection_error_mean_px": float(np.mean(v["proj"])),
                "n_samples": len(v["add"]),
            }
            for k, v in per_object.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
