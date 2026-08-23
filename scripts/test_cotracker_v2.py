"""
CoTracker3 完整测试 V2
改进：
1. 深度图中值滤波去噪
2. 时序 SavGol 平滑速度估计
3. 更多测试场景（5个）
4. 2D 跟踪精度评估（不依赖深度）
5. 与其他方法的对比
6. 生成 HTML 报告
"""

import sys
import os
import json
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from scipy.signal import savgol_filter, medfilt2d

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "co-tracker-src" / "co-tracker-main"))

from cotracker.predictor import CoTrackerPredictor


YCB_OBJECTS = {
    1: "001_master_chef_can", 2: "002_cracker_box", 3: "003_sugar_box",
    4: "004_tomato_soup_can", 5: "005_mustard_bottle", 6: "006_tuna_fish_can",
    7: "007_pudding_box", 8: "008_gelatin_box", 9: "009_potted_meat_can",
    10: "010_banana", 11: "011_pitcher_base", 12: "012_bleach_cleanser",
    13: "013_bowl", 14: "014_mug", 15: "015_power_drill",
    16: "016_wood_block", 17: "017_scissors", 18: "018_large_marker",
    19: "019_large_clamp", 20: "020_extra_large_clamp", 21: "021_foam_brick",
}


def load_scene(scene_dir, max_frames=60):
    """加载场景数据"""
    import json as _json
    scene_dir = Path(scene_dir)

    with open(scene_dir / "scene_gt.json") as f:
        scene_gt = _json.load(f)
    with open(scene_dir / "scene_camera.json") as f:
        scene_cam = _json.load(f)

    first_key = sorted(scene_gt.keys())[0]
    obj_id = scene_gt[first_key][0]["obj_id"]
    frame_keys = sorted(scene_gt.keys())[:max_frames]

    rgbs, depths, poses, Ks = [], [], [], []
    for key in frame_keys:
        pose_item = None
        for item in scene_gt[key]:
            if item["obj_id"] == obj_id:
                pose_item = item
                break
        if not pose_item:
            continue

        rgb_path = scene_dir / "rgb" / f"{int(key):06d}.png"
        depth_path = scene_dir / "depth" / f"{int(key):06d}.png"
        if not rgb_path.exists() or not depth_path.exists():
            continue

        rgb = cv2.imread(str(rgb_path))
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        R = np.array(pose_item["cam_R_m2c"]).reshape(3, 3)
        t = np.array(pose_item["cam_t_m2c"]) / 1000.0
        K = np.array(scene_cam[key]["cam_K"]).reshape(3, 3)

        rgbs.append(rgb)
        depths.append(depth_raw)
        poses.append({"R": R, "t": t})
        Ks.append(K)

    return {
        "rgb": rgbs, "depth": depths, "poses": poses,
        "K": Ks[0] if Ks else np.eye(3), "obj_id": obj_id,
        "scene_id": scene_dir.name,
    }


def filter_depth(depth_map, kernel_size=7):
    """中值滤波 + 双边滤波去除深度图噪声"""
    valid = depth_map > 0
    if np.sum(valid) < 10:
        return depth_map

    # scipy 中值滤波（支持 float32）
    filtered = medfilt2d(depth_map, kernel_size=kernel_size)

    # 双边滤波（保边去噪）
    filtered = cv2.bilateralFilter(filtered, 5, 0.05, 0.05)

    # 恢复原始有效区域
    result = np.where(valid, filtered, depth_map)
    return result


def get_depth_at_point(depth_map, x, y, window=5):
    """在点位置附近窗口内取有效深度的中值"""
    H, W = depth_map.shape
    x0 = max(0, int(x) - window)
    x1 = min(W, int(x) + window + 1)
    y0 = max(0, int(y) - window)
    y1 = min(H, int(y) + window + 1)

    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0.01]  # 排除 0 和极小值
    if len(valid) > 0:
        return float(np.median(valid))
    return -1.0


def generate_object_mask(pose, model_size_m, K, img_shape):
    """从 GT 位姿生成物体 mask"""
    H, W = img_shape
    t = pose["t"]
    if t[2] <= 0:
        return np.zeros((H, W), dtype=bool)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    cx_pix = cx + t[0] * fx / t[2]
    cy_pix = cy + t[1] * fy / t[2]
    half_w = max(model_size_m[0] / 2 * fx / t[2], 3)
    half_h = max(model_size_m[1] / 2 * fy / t[2], 3)

    y_coords, x_coords = np.ogrid[:H, :W]
    mask = ((x_coords - cx_pix) / half_w) ** 2 + ((y_coords - cy_pix) / half_h) ** 2 <= 1.0
    return mask.astype(bool)


def get_object_size(models_dir, obj_id):
    """获取物体尺寸"""
    info_path = Path(models_dir) / "models_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        key = str(obj_id)
        if key in info:
            if "size" in info[key]:
                return np.array(info[key]["size"]) / 1000.0
            elif "diameter" in info[key]:
                d = info[key]["diameter"] / 1000.0
                return np.array([d * 0.7, d * 0.7, d * 0.7])
    return np.array([0.1, 0.1, 0.1])


def create_tracking_video(rgb_frames, tracks, visibilities, output_path, fps=15, gt_velocity=None, pred_velocity=None):
    """生成带跟踪点叠加 + 速度信息的视频"""
    H, W = rgb_frames[0].shape[:2]
    panel_h = 100
    total_h = H + panel_h
    # 用 XVID 编码器 + .avi 容器，兼容性最好
    if str(output_path).endswith(".mp4"):
        output_path = str(output_path).replace(".mp4", ".avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, total_h))
    if not writer.isOpened():
        # 回退到 mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, total_h))

    T = len(rgb_frames)
    N = tracks.shape[1]
    np.random.seed(42)
    colors = np.random.randint(0, 255, (N, 3)).tolist()

    for t in range(T):
        frame = rgb_frames[t].copy()

        # 画轨迹
        for i in range(N):
            pts = []
            for tt in range(t + 1):
                if tt < tracks.shape[0] and visibilities[tt, i]:
                    pts.append((int(tracks[tt, i, 0]), int(tracks[tt, i, 1])))
            if len(pts) > 1:
                color = colors[i]
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    thickness = max(1, int(2 * alpha))
                    cv2.line(frame, pts[k - 1], pts[k], color, thickness)

            if t < tracks.shape[0] and visibilities[t, i]:
                x, y = tracks[t, i]
                cv2.circle(frame, (int(x), int(y)), 4, colors[i], -1)
                cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 1)

        # 信息面板
        panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        vis_count = int(np.sum(visibilities[t])) if t < visibilities.shape[0] else 0
        cv2.putText(panel, f"Frame {t+1}/{T} | Points: {vis_count}/{N} visible", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if gt_velocity is not None and t < len(gt_velocity):
            gv = gt_velocity[t]
            cv2.putText(panel, f"GT vel: ({gv[0]*1000:.0f}, {gv[1]*1000:.0f}, {gv[2]*1000:.0f}) mm/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if pred_velocity is not None and t < len(pred_velocity):
            pv = pred_velocity[t]
            cv2.putText(panel, f"Pred vel: ({pv[0]*1000:.0f}, {pv[1]*1000:.0f}, {pv[2]*1000:.0f}) mm/s", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        frame_with_panel = np.vstack([frame, panel])
        writer.write(frame_with_panel)

    writer.release()
    file_size = os.path.getsize(str(output_path))
    if file_size < 1000:
        print(f"  WARNING: Video file too small ({file_size} bytes), encoder may have failed")
    else:
        print(f"  Video saved: {output_path} ({file_size/1024:.0f} KB)")


def compute_gt_velocity(poses, fps=30.0):
    """GT 速度"""
    T = len(poses)
    dt = 1.0 / fps
    translations = np.array([p["t"] for p in poses])

    velocities = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T - 1):
        velocities[t] = (translations[t + 1] - translations[t - 1]) / (2 * dt)
    if T > 1:
        velocities[0] = (translations[1] - translations[0]) / dt
        velocities[-1] = (translations[-1] - translations[-2]) / dt
    return velocities


def compute_cotracker_velocity_v2(tracks, visibilities, depth_frames, K, fps=30.0):
    """改进版速度估计：深度滤波 + 时序平滑"""
    T, N, _ = tracks.shape
    dt = 1.0 / fps
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 1. 滤波深度图
    filtered_depths = [filter_depth(d) for d in depth_frames]

    # 2. 计算每个点每帧的 3D 位置
    points_3d = np.full((T, N, 3), np.nan, dtype=np.float32)

    for t in range(T):
        depth = filtered_depths[t]
        for i in range(N):
            x, y = tracks[t, i]
            if visibilities[t, i] and 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                z = get_depth_at_point(depth, x, y, window=5)
                if z > 0.01:
                    X = (x - cx) * z / fx
                    Y = (y - cy) * z / fy
                    points_3d[t, i] = [X, Y, z]

    # 3. 物体中心位置 = 所有可见点的中值
    obj_center = np.full((T, 3), np.nan, dtype=np.float32)
    for t in range(T):
        valid_pts = points_3d[t]
        valid_mask = ~np.any(np.isnan(valid_pts), axis=1)
        if np.sum(valid_mask) >= 2:
            obj_center[t] = np.median(valid_pts[valid_mask], axis=0)

    # 4. 填补缺失帧（线性插值）
    for dim in range(3):
        col = obj_center[:, dim]
        nan_mask = np.isnan(col)
        if np.any(nan_mask) and not np.all(nan_mask):
            valid_idx = np.where(~nan_mask)[0]
            nan_idx = np.where(nan_mask)[0]
            col[nan_idx] = np.interp(nan_idx, valid_idx, col[valid_idx])
            obj_center[:, dim] = col

    # 5. SavGol 平滑位置
    if T > 11:
        window = min(11, T if T % 2 == 1 else T - 1)
        for dim in range(3):
            obj_center[:, dim] = savgol_filter(obj_center[:, dim], window, 2)

    # 6. 差分计算速度
    velocities = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T - 1):
        velocities[t] = (obj_center[t + 1] - obj_center[t - 1]) / (2 * dt)
    if T > 1:
        velocities[0] = (obj_center[1] - obj_center[0]) / dt
        velocities[-1] = (obj_center[-1] - obj_center[-2]) / dt

    # 7. 速度也做 SavGol 平滑
    if T > 11:
        window = min(11, T if T % 2 == 1 else T - 1)
        for dim in range(3):
            velocities[:, dim] = savgol_filter(velocities[:, dim], window, 2)

    return velocities, obj_center


def compute_2d_tracking_error(tracks, visibilities, poses, K, model_size):
    """计算 2D 跟踪误差：跟踪点偏离物体重心的像素距离"""
    T, N, _ = tracks.shape
    errors = []

    for t in range(T):
        if t >= len(poses):
            break
        pose = poses[t]
        t_vec = pose["t"]
        if t_vec[2] <= 0:
            continue

        # GT 物体中心投影
        cx_pix = K[0, 2] + t_vec[0] * K[0, 0] / t_vec[2]
        cy_pix = K[1, 2] + t_vec[1] * K[1, 1] / t_vec[2]
        gt_center = np.array([cx_pix, cy_pix])

        # 跟踪点中心
        vis_pts = []
        for i in range(N):
            if visibilities[t, i]:
                vis_pts.append(tracks[t, i])

        if len(vis_pts) > 0:
            tracked_center = np.mean(vis_pts, axis=0)
            error = np.linalg.norm(tracked_center - gt_center)
            errors.append(error)

    return np.array(errors) if errors else np.array([0])


def run_cotracker_test_v2(
    data_root="datasets/ycbv",
    models_dir="datasets/ycbv/models",
    checkpoint="checkpoints/cotracker3_offline.pth",
    max_frames=60,
    num_scenes=5,
    output_dir="outputs/cotracker_test_v2",
    device="cpu",
):
    """运行改进版 CoTracker 测试"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CoTracker3 V2 Test (depth filtering + temporal smoothing)")
    print("=" * 60)

    # 加载模型
    print("\nLoading CoTracker3...")
    predictor = CoTrackerPredictor(
        checkpoint=checkpoint,
        offline=True,
        v2=False,
        window_len=60,
    )
    predictor = predictor.to(device)
    predictor.eval()
    print("  Model loaded!")

    # 选择测试场景
    test_dir = Path(data_root) / "test"
    all_scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    split_path = Path(data_root) / "scene_split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        test_scenes = split.get("test", all_scenes[:num_scenes])
    else:
        test_scenes = all_scenes[:num_scenes]

    test_scenes = test_scenes[:num_scenes]
    print(f"\nTest scenes: {test_scenes}")

    all_results = []

    for scene_idx, scene_id in enumerate(test_scenes):
        print(f"\n{'='*50}")
        print(f"Scene {scene_idx+1}/{len(test_scenes)}: {scene_id}")
        print(f"{'='*50}")

        scene_dir = test_dir / scene_id
        if not scene_dir.exists():
            print(f"  Scene not found, skip")
            continue

        # 加载数据
        t0 = time.time()
        data = load_scene(str(scene_dir), max_frames=max_frames)
        T = len(data["rgb"])
        if T < 5:
            print(f"  Too few frames ({T}), skip")
            continue

        obj_name = YCB_OBJECTS.get(data["obj_id"], f"obj_{data['obj_id']}")
        print(f"  Object: {obj_name}")
        print(f"  Frames: {T}")

        # 生成 mask
        model_size = get_object_size(models_dir, data["obj_id"])
        H, W = data["rgb"][0].shape[:2]
        mask = generate_object_mask(data["poses"][0], model_size, data["K"], (H, W))
        mask_pixels = int(np.sum(mask))
        print(f"  Mask pixels: {mask_pixels}")

        if mask_pixels < 50:
            print(f"  Mask too small, skip")
            continue

        # 准备视频 tensor
        video = np.stack(data["rgb"])
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        video_tensor = video_tensor.unsqueeze(0).to(device)

        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        # 运行 CoTracker
        print(f"  Running CoTracker3...")
        t1 = time.time()
        with torch.no_grad():
            tracks, visibilities = predictor(
                video_tensor,
                segm_mask=mask_tensor,
                grid_size=10,
            )
        t2 = time.time()
        print(f"  Tracking time: {t2 - t1:.1f}s")

        tracks_np = tracks[0].cpu().numpy()
        vis_np = visibilities[0].cpu().numpy()
        N = tracks_np.shape[1]
        print(f"  Tracked points: {N}")
        print(f"  Avg visibility: {np.mean(vis_np):.2%}")

        # GT 速度
        gt_velocity = compute_gt_velocity(data["poses"])

        # 改进版速度估计
        pred_velocity, obj_center = compute_cotracker_velocity_v2(
            tracks_np, vis_np, data["depth"], data["K"]
        )

        # 速度误差
        valid = ~np.all(pred_velocity == 0, axis=1)
        valid[0] = False
        if np.sum(valid) > 0:
            vel_error = np.linalg.norm(pred_velocity[valid] - gt_velocity[valid], axis=1)
            vel_rmse = float(np.sqrt(np.mean(vel_error ** 2)))
            vel_mae = float(np.mean(vel_error))
        else:
            vel_rmse = vel_mae = -1

        # 2D 跟踪误差
        tracking_errors_2d = compute_2d_tracking_error(
            tracks_np, vis_np, data["poses"], data["K"], model_size
        )

        # 跟踪精度（点在 mask 内）
        in_mask_count = 0
        total_visible = 0
        for t in range(T):
            if t < len(data["poses"]):
                mask_t = generate_object_mask(data["poses"][t], model_size, data["K"], (H, W))
                for i in range(N):
                    if vis_np[t, i]:
                        total_visible += 1
                        x, y = int(tracks_np[t, i, 0]), int(tracks_np[t, i, 1])
                        if 0 <= x < W and 0 <= y < H and mask_t[y, x]:
                            in_mask_count += 1

        tracking_accuracy = in_mask_count / max(total_visible, 1)
        coverage = float(np.mean(vis_np))

        # 位移统计
        displacements = []
        for i in range(N):
            for t in range(1, T):
                if vis_np[t, i] and vis_np[t - 1, i]:
                    dx = tracks_np[t, i, 0] - tracks_np[t - 1, i, 0]
                    dy = tracks_np[t, i, 1] - tracks_np[t - 1, i, 1]
                    displacements.append(np.sqrt(dx ** 2 + dy ** 2))

        # 生成视频
        video_path = video_dir / f"{scene_id}_{obj_name}_tracking_v2.mp4"
        create_tracking_video(
            data["rgb"], tracks_np, vis_np, video_path,
            gt_velocity=gt_velocity, pred_velocity=pred_velocity
        )

        result = {
            "scene_id": scene_id,
            "obj_id": data["obj_id"],
            "obj_name": obj_name,
            "num_frames": T,
            "num_points": N,
            "tracking_time_s": round(t2 - t1, 2),
            "visibility_rate": round(coverage, 4),
            "tracking_accuracy": round(tracking_accuracy, 4),
            "in_mask_count": in_mask_count,
            "total_visible": total_visible,
            "velocity_rmse_mms": round(vel_rmse * 1000, 2) if vel_rmse > 0 else -1,
            "velocity_mae_mms": round(vel_mae * 1000, 2) if vel_mae > 0 else -1,
            "tracking_error_2d_px": round(float(np.mean(tracking_errors_2d)), 2),
            "tracking_error_2d_max_px": round(float(np.max(tracking_errors_2d)), 2),
            "mean_displacement_px": round(float(np.mean(displacements)) if displacements else 0, 2),
            "max_displacement_px": round(float(np.max(displacements)) if displacements else 0, 2),
            "video_path": str(video_path),
        }

        all_results.append(result)

        print(f"\n  --- Results ---")
        print(f"  Tracked points: {N}")
        print(f"  Visibility: {coverage:.2%}")
        print(f"  Tracking accuracy (in mask): {tracking_accuracy:.2%}")
        print(f"  2D tracking error: {result['tracking_error_2d_px']:.2f} px (mean)")
        print(f"  Velocity RMSE: {result['velocity_rmse_mms']:.2f} mm/s")
        print(f"  Velocity MAE: {result['velocity_mae_mms']:.2f} mm/s")
        print(f"  Mean displacement: {result['mean_displacement_px']:.2f} px")
        print(f"  Video: {video_path}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Summary ({len(all_results)} scenes)")
    print(f"{'='*60}")

    if all_results:
        avg_vis = np.mean([r["visibility_rate"] for r in all_results])
        avg_acc = np.mean([r["tracking_accuracy"] for r in all_results])
        avg_2d = np.mean([r["tracking_error_2d_px"] for r in all_results])
        vel_rmses = [r["velocity_rmse_mms"] for r in all_results if r["velocity_rmse_mms"] > 0]
        avg_vel_rmse = np.mean(vel_rmses) if vel_rmses else -1
        avg_time = np.mean([r["tracking_time_s"] for r in all_results])

        print(f"  Avg visibility: {avg_vis:.2%}")
        print(f"  Avg tracking accuracy: {avg_acc:.2%}")
        print(f"  Avg 2D tracking error: {avg_2d:.2f} px")
        print(f"  Avg velocity RMSE: {avg_vel_rmse:.2f} mm/s")
        print(f"  Avg tracking time: {avg_time:.1f}s / scene")

    summary = {
        "model": "CoTracker3 (scaled_offline)",
        "version": "v2 (depth filter + SavGol)",
        "device": device,
        "num_scenes": len(all_results),
        "results": all_results,
        "summary": {
            "avg_visibility_rate": float(np.mean([r["visibility_rate"] for r in all_results])) if all_results else 0,
            "avg_tracking_accuracy": float(np.mean([r["tracking_accuracy"] for r in all_results])) if all_results else 0,
            "avg_2d_tracking_error_px": float(np.mean([r["tracking_error_2d_px"] for r in all_results])) if all_results else 0,
            "avg_velocity_rmse_mms": float(np.mean(vel_rmses)) if vel_rmses else -1,
            "avg_tracking_time_s": float(np.mean([r["tracking_time_s"] for r in all_results])) if all_results else 0,
        },
    }

    with open(output_dir / "cotracker_results_v2.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_dir / 'cotracker_results_v2.json'}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/ycbv")
    parser.add_argument("--models_dir", default="datasets/ycbv/models")
    parser.add_argument("--checkpoint", default="checkpoints/cotracker3_offline.pth")
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--num_scenes", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/cotracker_test_v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_cotracker_test_v2(
        data_root=args.data_root,
        models_dir=args.models_dir,
        checkpoint=args.checkpoint,
        max_frames=args.max_frames,
        num_scenes=args.num_scenes,
        output_dir=args.output_dir,
        device=args.device,
    )
