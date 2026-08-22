#!/usr/bin/env python
"""
光流法速度估计评估
在 MuJoCo 评估集上运行稠密光流和稀疏光流速度估计，与 GT 比较。
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
    return depth.astype(np.float32) / 1000.0  # mm -> m


def load_gray(scene_dir: Path, idx: int) -> np.ndarray:
    f = scene_dir / "rgb" / f"{idx:06d}.jpg"
    if not f.exists():
        f = scene_dir / "rgb" / f"{idx:06d}.png"
    return cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)


def estimate_farneback(gray_prev, gray_curr, depth_prev, mask, K, dt):
    """稠密光流 (Farneback) 速度估计"""
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # 只在掩码区域内计算
    valid = mask > 0
    if valid.sum() < 10:
        return np.zeros(3)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    h, w = depth_prev.shape
    ys, xs = np.mgrid[0:h, 0:w]

    # 反投影到 3D (prev frame)
    Z = depth_prev
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    # 光流 -> 3D 速度 (近似: 假设深度不变)
    vx_flow = flow[..., 0]
    vy_flow = flow[..., 1]

    # 像素速度 -> 3D 速度 (X_dot = Z_dot/fx * u + Z/fx * u_dot ... 简化)
    # 简化: vx_3d ≈ vx_flow * Z / fx, vy_3d ≈ vy_flow * Z / fy
    vx_3d = vx_flow * Z / fx / dt
    vy_3d = vy_flow * Z / fy / dt

    # Z 方向速度从深度变化估计
    depth_diff = np.zeros_like(depth_prev)
    depth_diff[1:-1, 1:-1] = depth_prev[1:-1, 1:-1] - depth_prev[1:-1, 1:-1]  # 近似
    vz_3d = np.zeros_like(depth_prev)

    # 取掩码内中位数
    mask_flat = valid.flatten()
    vx_med = np.median(vx_3d.flatten()[mask_flat])
    vy_med = np.median(vy_3d.flatten()[mask_flat])
    vz_med = 0.0  # 深度变化估计不准，暂时置零

    return np.array([vx_med, vy_med, vz_med])


def estimate_lucas_kanade(gray_prev, gray_curr, depth_prev, mask, K, dt):
    """稀疏光流 (Lucas-Kanade) 速度估计"""
    # 在前一帧掩码内找角点
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return np.zeros(3)

    # 采样特征点
    p0 = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    # 降采样避免过多点
    if len(p0) > 500:
        idx = np.random.choice(len(p0), 500, replace=False)
        p0 = p0[idx]
    p0 = p0.reshape(-1, 1, 2)

    # LK 光流
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, p0, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    good = st.flatten() == 1
    if good.sum() < 5:
        return np.zeros(3)

    p0_good = p0[good].reshape(-1, 2)
    p1_good = p1[good].reshape(-1, 2)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 每个点的 3D 速度
    vx_list = []
    vy_list = []
    for i in range(len(p0_good)):
        x0, y0 = p0_good[i]
        x1, y1 = p1_good[i]
        xi, yi = int(x0), int(y0)
        if 0 <= yi < depth_prev.shape[0] and 0 <= xi < depth_prev.shape[1]:
            Z = depth_prev[yi, xi]
            if Z > 0:
                vx = (x1 - x0) * Z / fx / dt
                vy = (y1 - y0) * Z / fy / dt
                vx_list.append(vx)
                vy_list.append(vy)

    if len(vx_list) < 5:
        return np.zeros(3)

    vx_med = np.median(vx_list)
    vy_med = np.median(vy_list)
    return np.array([vx_med, vy_med, 0.0])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="datasets/ycb_grasp_eval")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation_full")
    parser.add_argument("--stride", type=int, default=10, help="帧间隔（加快速度）")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir = eval_dir / "test"
    scene_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    dt = 0.002 * args.stride  # 实际时间间隔

    print(f"光流法速度估计评估")
    print(f"评估集: {eval_dir}")
    print(f"场景数: {len(scene_dirs)}, stride: {args.stride}, dt: {dt:.4f}s")
    print()

    results = defaultdict(list)

    for si, scene_dir in enumerate(scene_dirs):
        # 加载 GT
        with open(scene_dir / "scene_gt.json") as f:
            gt = json.load(f)
        with open(scene_dir / "scene_velocity.json") as f:
            vel_gt = json.load(f)
        with open(scene_dir / "scene_camera.json") as f:
            cam = json.load(f)

        n_frames = len(gt)
        K = np.array(cam["0"]["cam_K"]).reshape(3, 3)

        # 生成近似掩码（用 bbox 近似）
        # 实际评估集中没存 mask，用物体中心附近区域近似
        # 这里简单处理：取图像中心一块区域作为物体掩码（物体始终在视野中央）
        h, w = 480, 640
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        cv2.circle(mask, (cx, cy), 80, 255, -1)

        print(f"  场景 {si+1}/{len(scene_dirs)}: {scene_dir.name} ({n_frames} 帧)", flush=True)

        prev_gray = None
        prev_depth = None

        for i in range(0, n_frames - args.stride, args.stride):
            curr_gray = load_gray(scene_dir, i)
            curr_depth = load_depth(scene_dir, i)

            if prev_gray is not None:
                # 稠密光流
                vel_fb = estimate_farneback(prev_gray, curr_gray, prev_depth, mask, K, dt)
                # 稀疏光流
                vel_lk = estimate_lucas_kanade(prev_gray, curr_gray, prev_depth, mask, K, dt)

                # GT 速度
                key = str(i)
                if key in vel_gt:
                    gt_vel = np.array(vel_gt[key]["linear_velocity_m_s"])
                    err_fb = np.linalg.norm(vel_fb - gt_vel)
                    err_lk = np.linalg.norm(vel_lk - gt_vel)
                    results["farneback"].append(float(err_fb))
                    results["lucas_kanade"].append(float(err_lk))

            prev_gray = curr_gray
            prev_depth = curr_depth

    # 汇总
    print()
    print("=" * 60)
    print("光流法速度估计结果")
    print("=" * 60)
    for name in ["farneback", "lucas_kanade"]:
        errs = np.array(results[name])
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        mae = float(np.mean(errs))
        print(f"  {name:<15} RMSE={rmse:.4f} m/s   MAE={mae:.4f} m/s   (n={len(errs)})")

    # 保存
    out_path = output_dir / "optical_flow_results.json"
    out_data = {}
    for name in results:
        errs = np.array(results[name])
        out_data[name] = {
            "linear_velocity_rmse_m_s": float(np.sqrt(np.mean(errs ** 2))),
            "linear_velocity_mae_m_s": float(np.mean(errs)),
            "n_samples": len(errs),
        }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
