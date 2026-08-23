"""
CoTracker3 动态场景测试
从 YCB-Video 中选取物体运动最大的 60 帧片段进行测试
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
}


def find_most_dynamic_segment(scene_dir, obj_id, segment_len=60):
    """在场景中找到物体运动最大的 segment_len 帧"""
    with open(scene_dir / "scene_gt.json") as f:
        gt = json.load(f)

    keys = sorted(gt.keys())
    translations = []
    for k in keys:
        for item in gt[k]:
            if item["obj_id"] == obj_id:
                translations.append((k, item["cam_t_m2c"]))
                break

    if len(translations) < segment_len:
        return translations

    trans = np.array([t[1] for t in translations]) / 1000.0

    # 滑动窗口找运动最大的片段
    best_start = 0
    best_disp = 0
    for i in range(len(trans) - segment_len):
        disp = np.linalg.norm(trans[i + segment_len - 1] - trans[i])
        if disp > best_disp:
            best_disp = disp
            best_start = i

    return translations[best_start:best_start + segment_len], best_disp * 1000


def load_dynamic_segment(scene_dir, segment, max_frames=60):
    """加载动态片段"""
    with open(scene_dir / "scene_camera.json") as f:
        scene_cam = json.load(f)

    rgbs, depths, poses, Ks = [], [], [], []
    obj_id = None

    for key, t_raw in segment[:max_frames]:
        key_str = str(key)
        if key_str not in scene_cam:
            continue

        rgb_path = scene_dir / "rgb" / f"{int(key):06d}.png"
        depth_path = scene_dir / "depth" / f"{int(key):06d}.png"
        if not rgb_path.exists() or not depth_path.exists():
            continue

        with open(scene_dir / "scene_gt.json") as f:
            gt = json.load(f)
        pose_item = None
        for item in gt[key_str]:
            if item["obj_id"] == segment[0][1] if False else True:
                # 找对应物体的位姿
                pass
        # 重新读取
        for item in gt[key_str]:
            pose_item = item
            break

        obj_id = pose_item["obj_id"]
        rgb = cv2.imread(str(rgb_path))
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        R = np.array(pose_item["cam_R_m2c"]).reshape(3, 3)
        t = np.array(pose_item["cam_t_m2c"]) / 1000.0
        K = np.array(scene_cam[key_str]["cam_K"]).reshape(3, 3)

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
    valid = depth_map > 0
    if np.sum(valid) < 10:
        return depth_map
    filtered = medfilt2d(depth_map, kernel_size=kernel_size)
    filtered = cv2.bilateralFilter(filtered, 5, 0.05, 0.05)
    result = np.where(valid, filtered, depth_map)
    return result


def get_depth_at_point(depth_map, x, y, window=5):
    H, W = depth_map.shape
    x0 = max(0, int(x) - window)
    x1 = min(W, int(x) + window + 1)
    y0 = max(0, int(y) - window)
    y1 = min(H, int(y) + window + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[patch > 0.01]
    if len(valid) > 0:
        return float(np.median(valid))
    return -1.0


def generate_object_mask(pose, model_size_m, K, img_shape):
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


def create_tracking_video(rgb_frames, tracks, visibilities, output_path, fps=15, gt_velocity=None, pred_velocity=None, scene_name="", obj_name=""):
    H, W = rgb_frames[0].shape[:2]
    panel_h = 120
    total_h = H + panel_h

    if str(output_path).endswith(".avi"):
        mp4_path = str(output_path).replace(".avi", ".mp4")
    else:
        mp4_path = str(output_path).replace(".mp4", ".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (W, total_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        mp4_path = mp4_path.replace(".mp4", ".avi")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (W, total_h))

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
        title = f"{scene_name} | {obj_name} | Frame {t+1}/{T} | Points: {vis_count}/{N}"
        cv2.putText(panel, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if gt_velocity is not None and t < len(gt_velocity):
            gv = gt_velocity[t]
            speed = np.linalg.norm(gv) * 1000
            cv2.putText(panel, f"GT vel: ({gv[0]*1000:.0f}, {gv[1]*1000:.0f}, {gv[2]*1000:.0f}) mm/s  |speed={speed:.0f} mm/s", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if pred_velocity is not None and t < len(pred_velocity):
            pv = pred_velocity[t]
            pspeed = np.linalg.norm(pv) * 1000
            cv2.putText(panel, f"Pred:   ({pv[0]*1000:.0f}, {pv[1]*1000:.0f}, {pv[2]*1000:.0f}) mm/s  |speed={pspeed:.0f} mm/s", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # 误差
        if gt_velocity is not None and pred_velocity is not None and t < len(gt_velocity) and t < len(pred_velocity):
            err = np.linalg.norm(gt_velocity[t] - pred_velocity[t]) * 1000
            cv2.putText(panel, f"Error: {err:.0f} mm/s", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)

        frame_with_panel = np.vstack([frame, panel])
        writer.write(frame_with_panel)

    writer.release()
    file_size = os.path.getsize(mp4_path)
    if file_size < 1000:
        print(f"  WARNING: Video too small ({file_size} bytes)")
    else:
        print(f"  Video saved: {mp4_path} ({file_size/1024:.0f} KB)")
    return mp4_path


def compute_gt_velocity(poses, fps=30.0):
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
    T, N, _ = tracks.shape
    dt = 1.0 / fps
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    filtered_depths = [filter_depth(d) for d in depth_frames]

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

    obj_center = np.full((T, 3), np.nan, dtype=np.float32)
    for t in range(T):
        valid_pts = points_3d[t]
        valid_mask = ~np.any(np.isnan(valid_pts), axis=1)
        if np.sum(valid_mask) >= 2:
            obj_center[t] = np.median(valid_pts[valid_mask], axis=0)

    for dim in range(3):
        col = obj_center[:, dim]
        nan_mask = np.isnan(col)
        if np.any(nan_mask) and not np.all(nan_mask):
            valid_idx = np.where(~nan_mask)[0]
            nan_idx = np.where(nan_mask)[0]
            col[nan_idx] = np.interp(nan_idx, valid_idx, col[valid_idx])
            obj_center[:, dim] = col

    if T > 11:
        window = min(11, T if T % 2 == 1 else T - 1)
        for dim in range(3):
            obj_center[:, dim] = savgol_filter(obj_center[:, dim], window, 2)

    velocities = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T - 1):
        velocities[t] = (obj_center[t + 1] - obj_center[t - 1]) / (2 * dt)
    if T > 1:
        velocities[0] = (obj_center[1] - obj_center[0]) / dt
        velocities[-1] = (obj_center[-1] - obj_center[-2]) / dt

    if T > 11:
        window = min(11, T if T % 2 == 1 else T - 1)
        for dim in range(3):
            velocities[:, dim] = savgol_filter(velocities[:, dim], window, 2)

    return velocities, obj_center


def run_dynamic_test(
    data_root="datasets/ycbv",
    models_dir="datasets/ycbv/models",
    checkpoint="checkpoints/cotracker3_offline.pth",
    max_frames=60,
    output_dir="outputs/cotracker_dynamic",
    device="cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CoTracker3 Dynamic Scene Test")
    print("=" * 60)

    # 加载模型
    print("\nLoading CoTracker3...")
    predictor = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
    predictor = predictor.to(device)
    predictor.eval()
    print("  Model loaded!")

    # 所有场景
    test_dir = Path(data_root) / "test"
    all_scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    # 找每个场景中运动最大的片段
    print("\nScanning scenes for most dynamic segments...")
    scene_segments = []

    for scene in all_scenes:
        scene_dir = test_dir / scene
        gt_path = scene_dir / "scene_gt.json"
        if not gt_path.exists():
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        keys = sorted(gt.keys())
        if len(keys) < max_frames:
            continue

        obj_id = gt[keys[0]][0]["obj_id"]

        # 获取所有帧的位姿
        translations = []
        for k in keys:
            for item in gt[k]:
                if item["obj_id"] == obj_id:
                    translations.append((k, item["cam_t_m2c"]))
                    break

        if len(translations) < max_frames:
            continue

        trans = np.array([t[1] for t in translations]) / 1000.0

        # 滑动窗口找运动最大片段
        best_start = 0
        best_disp = 0
        for i in range(len(trans) - max_frames):
            disp = np.linalg.norm(trans[i + max_frames - 1] - trans[i])
            if disp > best_disp:
                best_disp = disp
                best_start = i

        segment = translations[best_start:best_start + max_frames]
        avg_speed = np.mean([np.linalg.norm(trans[i+1] - trans[i]) for i in range(best_start, best_start + max_frames - 1)])

        scene_segments.append({
            "scene": scene,
            "obj_id": obj_id,
            "start_frame": best_start,
            "displacement_mm": best_disp * 1000,
            "avg_speed_mms": avg_speed * 1000,
            "segment": segment,
        })

    # 按运动量排序，取前5个
    scene_segments.sort(key=lambda x: x["displacement_mm"], reverse=True)
    top_scenes = scene_segments[:5]

    print(f"\nTop {len(top_scenes)} most dynamic scenes:")
    for i, s in enumerate(top_scenes):
        obj_name = YCB_OBJECTS.get(s["obj_id"], f"obj_{s['obj_id']}")
        print(f"  {i+1}. Scene {s['scene']} - {obj_name} - Disp: {s['displacement_mm']:.1f}mm, Avg speed: {s['avg_speed_mms']:.1f}mm/s")

    all_results = []

    for idx, seg_info in enumerate(top_scenes):
        scene = seg_info["scene"]
        obj_id = seg_info["obj_id"]
        obj_name = YCB_OBJECTS.get(obj_id, f"obj_{obj_id}")

        print(f"\n{'='*50}")
        print(f"Test {idx+1}/{len(top_scenes)}: Scene {scene} - {obj_name}")
        print(f"  Segment: frames {seg_info['start_frame']}-{seg_info['start_frame']+max_frames}")
        print(f"  Total displacement: {seg_info['displacement_mm']:.1f}mm")
        print(f"  Avg speed: {seg_info['avg_speed_mms']:.1f}mm/s")
        print(f"{'='*50}")

        scene_dir = test_dir / scene

        # 加载片段
        with open(scene_dir / "scene_camera.json") as f:
            scene_cam = json.load(f)
        with open(scene_dir / "scene_gt.json") as f:
            gt = json.load(f)

        rgbs, depths, poses = [], [], []
        for key, t_raw in seg_info["segment"]:
            key_str = str(key)
            rgb_path = scene_dir / "rgb" / f"{int(key):06d}.png"
            depth_path = scene_dir / "depth" / f"{int(key):06d}.png"
            if not rgb_path.exists() or not depth_path.exists():
                continue

            pose_item = None
            for item in gt[key_str]:
                if item["obj_id"] == obj_id:
                    pose_item = item
                    break
            if not pose_item:
                continue

            rgb = cv2.imread(str(rgb_path))
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            R = np.array(pose_item["cam_R_m2c"]).reshape(3, 3)
            t = np.array(pose_item["cam_t_m2c"]) / 1000.0
            K = np.array(scene_cam[key_str]["cam_K"]).reshape(3, 3)

            rgbs.append(rgb)
            depths.append(depth_raw)
            poses.append({"R": R, "t": t})

        T = len(rgbs)
        if T < 10:
            print(f"  Too few frames ({T}), skip")
            continue

        K = np.array(scene_cam[str(seg_info["segment"][0][0])]["cam_K"]).reshape(3, 3)
        print(f"  Loaded {T} frames")

        # 生成 mask
        model_size = get_object_size(models_dir, obj_id)
        H, W = rgbs[0].shape[:2]
        mask = generate_object_mask(poses[0], model_size, K, (H, W))
        mask_pixels = int(np.sum(mask))
        print(f"  Mask pixels: {mask_pixels}")

        if mask_pixels < 50:
            print(f"  Mask too small, skip")
            continue

        # 运行 CoTracker
        video = np.stack(rgbs)
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        video_tensor = video_tensor.unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        print(f"  Running CoTracker3...")
        t1 = time.time()
        with torch.no_grad():
            tracks, visibilities = predictor(video_tensor, segm_mask=mask_tensor, grid_size=10)
        t2 = time.time()
        print(f"  Tracking time: {t2 - t1:.1f}s")

        tracks_np = tracks[0].cpu().numpy()
        vis_np = visibilities[0].cpu().numpy()
        N = tracks_np.shape[1]
        print(f"  Tracked points: {N}")
        print(f"  Avg visibility: {np.mean(vis_np):.2%}")

        # GT 速度
        gt_velocity = compute_gt_velocity(poses)

        # CoTracker 速度
        pred_velocity, obj_center = compute_cotracker_velocity_v2(tracks_np, vis_np, depths, K)

        # 速度误差
        valid = ~np.all(pred_velocity == 0, axis=1)
        valid[0] = False
        if np.sum(valid) > 0:
            vel_error = np.linalg.norm(pred_velocity[valid] - gt_velocity[valid], axis=1)
            vel_rmse = float(np.sqrt(np.mean(vel_error ** 2)))
            vel_mae = float(np.mean(vel_error))
        else:
            vel_rmse = vel_mae = -1

        # 跟踪精度
        in_mask_count = 0
        total_visible = 0
        for t in range(T):
            mask_t = generate_object_mask(poses[t], model_size, K, (H, W))
            for i in range(N):
                if vis_np[t, i]:
                    total_visible += 1
                    x, y = int(tracks_np[t, i, 0]), int(tracks_np[t, i, 1])
                    if 0 <= x < W and 0 <= y < H and mask_t[y, x]:
                        in_mask_count += 1

        tracking_accuracy = in_mask_count / max(total_visible, 1)

        # 位移统计
        displacements = []
        for i in range(N):
            for t in range(1, T):
                if vis_np[t, i] and vis_np[t - 1, i]:
                    dx = tracks_np[t, i, 0] - tracks_np[t - 1, i, 0]
                    dy = tracks_np[t, i, 1] - tracks_np[t - 1, i, 1]
                    displacements.append(np.sqrt(dx ** 2 + dy ** 2))

        # 生成视频
        video_path = video_dir / f"{scene}_{obj_name}_dynamic.mp4"
        actual_video = create_tracking_video(
            rgbs, tracks_np, vis_np, video_path,
            gt_velocity=gt_velocity, pred_velocity=pred_velocity,
            scene_name=f"Scene {scene}", obj_name=obj_name
        )

        result = {
            "scene_id": scene,
            "obj_id": obj_id,
            "obj_name": obj_name,
            "segment_start": seg_info["start_frame"],
            "num_frames": T,
            "num_points": N,
            "tracking_time_s": round(t2 - t1, 2),
            "visibility_rate": round(float(np.mean(vis_np)), 4),
            "tracking_accuracy": round(tracking_accuracy, 4),
            "velocity_rmse_mms": round(vel_rmse * 1000, 2) if vel_rmse > 0 else -1,
            "velocity_mae_mms": round(vel_mae * 1000, 2) if vel_mae > 0 else -1,
            "total_displacement_mm": round(seg_info["displacement_mm"], 1),
            "avg_gt_speed_mms": round(seg_info["avg_speed_mms"], 1),
            "mean_pixel_displacement_px": round(float(np.mean(displacements)) if displacements else 0, 2),
            "max_pixel_displacement_px": round(float(np.max(displacements)) if displacements else 0, 2),
            "video_path": actual_video,
        }

        all_results.append(result)

        print(f"\n  --- Results ---")
        print(f"  Tracked points: {N}")
        print(f"  Visibility: {np.mean(vis_np):.2%}")
        print(f"  Tracking accuracy: {tracking_accuracy:.2%}")
        print(f"  Velocity RMSE: {result['velocity_rmse_mms']:.2f} mm/s")
        print(f"  Velocity MAE: {result['velocity_mae_mms']:.2f} mm/s")
        print(f"  Pixel displacement: mean={result['mean_pixel_displacement_px']:.1f}px, max={result['max_pixel_displacement_px']:.1f}px")
        print(f"  Video: {actual_video}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Summary ({len(all_results)} scenes)")
    print(f"{'='*60}")

    if all_results:
        avg_vis = np.mean([r["visibility_rate"] for r in all_results])
        avg_acc = np.mean([r["tracking_accuracy"] for r in all_results])
        vel_rmses = [r["velocity_rmse_mms"] for r in all_results if r["velocity_rmse_mms"] > 0]
        avg_vel_rmse = np.mean(vel_rmses) if vel_rmses else -1
        avg_time = np.mean([r["tracking_time_s"] for r in all_results])
        avg_disp = np.mean([r["total_displacement_mm"] for r in all_results])

        print(f"  Avg total displacement: {avg_disp:.1f} mm")
        print(f"  Avg visibility: {avg_vis:.2%}")
        print(f"  Avg tracking accuracy: {avg_acc:.2%}")
        print(f"  Avg velocity RMSE: {avg_vel_rmse:.2f} mm/s")
        print(f"  Avg tracking time: {avg_time:.1f}s / scene")

    summary = {
        "model": "CoTracker3 (scaled_offline)",
        "version": "dynamic_v2 (depth filter + SavGol + dynamic segments)",
        "device": device,
        "num_scenes": len(all_results),
        "results": all_results,
        "summary": {
            "avg_visibility_rate": float(np.mean([r["visibility_rate"] for r in all_results])) if all_results else 0,
            "avg_tracking_accuracy": float(np.mean([r["tracking_accuracy"] for r in all_results])) if all_results else 0,
            "avg_velocity_rmse_mms": float(np.mean(vel_rmses)) if vel_rmses else -1,
            "avg_total_displacement_mm": float(np.mean([r["total_displacement_mm"] for r in all_results])) if all_results else 0,
            "avg_tracking_time_s": float(np.mean([r["tracking_time_s"] for r in all_results])) if all_results else 0,
        },
    }

    with open(output_dir / "cotracker_dynamic_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_dir / 'cotracker_dynamic_results.json'}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/ycbv")
    parser.add_argument("--models_dir", default="datasets/ycbv/models")
    parser.add_argument("--checkpoint", default="checkpoints/cotracker3_offline.pth")
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--output_dir", default="outputs/cotracker_dynamic")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dynamic_test(
        data_root=args.data_root,
        models_dir=args.models_dir,
        checkpoint=args.checkpoint,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        device=args.device,
    )
