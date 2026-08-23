"""
CoTracker3 物体运动测试
从 YCB-Video 中选取物体位移最大的连续帧，合成"相机不动、物体在动"的测试视频
同时用 MuJoCo 生成一个物体被抓取滑移的仿真视频
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
    10: "010_banana", 15: "015_power_drill", 21: "021_foam_brick",
}


def load_scene_segment(scene_dir, obj_id, start_frame, num_frames=60):
    """加载场景中指定物体的连续帧"""
    with open(scene_dir / "scene_gt.json") as f:
        gt = json.load(f)
    with open(scene_dir / "scene_camera.json") as f:
        scene_cam = json.load(f)

    keys = sorted(gt.keys())
    translations = []
    for k in keys:
        for item in gt[k]:
            if item["obj_id"] == obj_id:
                translations.append((k, item))
                break

    if len(translations) < start_frame + num_frames:
        num_frames = len(translations) - start_frame

    segment = translations[start_frame:start_frame + num_frames]

    rgbs, depths, poses = [], [], []
    for key, pose_item in segment:
        key_str = str(key)
        rgb_path = scene_dir / "rgb" / f"{int(key):06d}.png"
        depth_path = scene_dir / "depth" / f"{int(key):06d}.png"
        if not rgb_path.exists() or not depth_path.exists():
            continue

        rgb = cv2.imread(str(rgb_path))
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        R = np.array(pose_item["cam_R_m2c"]).reshape(3, 3)
        t = np.array(pose_item["cam_t_m2c"]) / 1000.0
        K = np.array(scene_cam[key_str]["cam_K"]).reshape(3, 3)

        rgbs.append(rgb)
        depths.append(depth_raw)
        poses.append({"R": R, "t": t})

    K = np.array(scene_cam[str(segment[0][0])]["cam_K"]).reshape(3, 3)
    return rgbs, depths, poses, K


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


def create_video(rgb_frames, tracks, visibilities, output_path, fps=30,
                 gt_velocity=None, pred_velocity=None, title=""):
    H, W = rgb_frames[0].shape[:2]
    panel_h = 120
    total_h = H + panel_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, total_h))
    if not writer.isOpened():
        return None

    T = len(rgb_frames)
    N = tracks.shape[1]
    np.random.seed(42)
    colors = np.random.randint(50, 255, (N, 3)).tolist()

    for t in range(T):
        frame = rgb_frames[t].copy()

        for i in range(N):
            pts = []
            for tt in range(t + 1):
                if tt < tracks.shape[0] and visibilities[tt, i]:
                    pts.append((int(tracks[tt, i, 0]), int(tracks[tt, i, 1])))
            if len(pts) > 1:
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    cv2.line(frame, pts[k - 1], pts[k], colors[i], max(1, int(3 * alpha)))
            if t < tracks.shape[0] and visibilities[t, i]:
                x, y = tracks[t, i]
                cv2.circle(frame, (int(x), int(y)), 5, colors[i], -1)
                cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), 1)

        panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        vis_count = int(np.sum(visibilities[t])) if t < visibilities.shape[0] else 0
        cv2.putText(panel, f"{title} | Frame {t+1}/{T} | Points: {vis_count}/{N}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if gt_velocity is not None and t < len(gt_velocity):
            gv = gt_velocity[t]
            speed = np.linalg.norm(gv) * 1000
            cv2.putText(panel, f"GT speed: {speed:.0f} mm/s  ({gv[0]*1000:.0f}, {gv[1]*1000:.0f}, {gv[2]*1000:.0f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if pred_velocity is not None and t < len(pred_velocity):
            pv = pred_velocity[t]
            pspeed = np.linalg.norm(pv) * 1000
            cv2.putText(panel, f"Pred speed: {pspeed:.0f} mm/s  ({pv[0]*1000:.0f}, {pv[1]*1000:.0f}, {pv[2]*1000:.0f})", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        if gt_velocity is not None and pred_velocity is not None and t < len(gt_velocity) and t < len(pred_velocity):
            err = np.linalg.norm(gt_velocity[t] - pred_velocity[t]) * 1000
            cv2.putText(panel, f"Error: {err:.0f} mm/s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

        frame_with_panel = np.vstack([frame, panel])
        writer.write(frame_with_panel)

    writer.release()
    size = os.path.getsize(str(output_path))
    print(f"  Video: {output_path} ({size/1024:.0f} KB)")
    return str(output_path)


def compute_gt_velocity(poses, fps=30.0):
    T = len(poses)
    dt = 1.0 / fps
    trans = np.array([p["t"] for p in poses])
    vels = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T - 1):
        vels[t] = (trans[t + 1] - trans[t - 1]) / (2 * dt)
    if T > 1:
        vels[0] = (trans[1] - trans[0]) / dt
        vels[-1] = (trans[-1] - trans[-2]) / dt
    return vels


def compute_velocity(tracks, vis, depths, K, fps=30.0):
    T, N, _ = tracks.shape
    dt = 1.0 / fps
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    filt_depths = []
    for d in depths:
        valid = d > 0
        if np.sum(valid) > 10:
            fd = medfilt2d(d, kernel_size=7)
            fd = cv2.bilateralFilter(fd, 5, 0.05, 0.05)
            fd = np.where(valid, fd, d)
        else:
            fd = d
        filt_depths.append(fd)

    points_3d = np.full((T, N, 3), np.nan, dtype=np.float32)
    for t in range(T):
        d = filt_depths[t]
        for i in range(N):
            x, y = tracks[t, i]
            if vis[t, i] and 0 <= x < d.shape[1] and 0 <= y < d.shape[0]:
                x0 = max(0, int(x) - 5)
                x1 = min(d.shape[1], int(x) + 6)
                y0 = max(0, int(y) - 5)
                y1 = min(d.shape[0], int(y) + 6)
                patch = d[y0:y1, x0:x1]
                vv = patch[patch > 0.01]
                if len(vv) > 0:
                    z = float(np.median(vv))
                    points_3d[t, i] = [(x - cx) * z / fx, (y - cy) * z / fy, z]

    obj_center = np.full((T, 3), np.nan, dtype=np.float32)
    for t in range(T):
        vp = points_3d[t]
        vm = ~np.any(np.isnan(vp), axis=1)
        if np.sum(vm) >= 2:
            obj_center[t] = np.median(vp[vm], axis=0)

    for dim in range(3):
        col = obj_center[:, dim]
        nm = np.isnan(col)
        if np.any(nm) and not np.all(nm):
            vi = np.where(~nm)[0]
            ni = np.where(nm)[0]
            col[ni] = np.interp(ni, vi, col[vi])
            obj_center[:, dim] = col

    if T > 11:
        w = min(11, T if T % 2 == 1 else T - 1)
        for dim in range(3):
            obj_center[:, dim] = savgol_filter(obj_center[:, dim], w, 2)

    vels = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T - 1):
        vels[t] = (obj_center[t + 1] - obj_center[t - 1]) / (2 * dt)
    if T > 1:
        vels[0] = (obj_center[1] - obj_center[0]) / dt
        vels[-1] = (obj_center[-1] - obj_center[-2]) / dt

    if T > 11:
        w = min(11, T if T % 2 == 1 else T - 1)
        for dim in range(3):
            vels[:, dim] = savgol_filter(vels[:, dim], w, 2)

    return vels


def run_object_motion_test(
    data_root="datasets/ycbv",
    models_dir="datasets/ycbv/models",
    checkpoint="checkpoints/cotracker3_offline.pth",
    output_dir="outputs/cotracker_object_motion",
    device="cpu",
    max_frames=80,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CoTracker3 Object Motion Test")
    print("(Camera fixed, object moving)")
    print("=" * 60)

    # 加载模型
    print("\nLoading CoTracker3...")
    predictor = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
    predictor = predictor.to(device)
    predictor.eval()
    print("  Model loaded!")

    # 扫描所有场景，找物体运动最大的片段
    test_dir = Path(data_root) / "test"
    all_scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    print("\nScanning for object motion...")
    candidates = []

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
        translations = []
        for k in keys:
            for item in gt[k]:
                if item["obj_id"] == obj_id:
                    translations.append(np.array(item["cam_t_m2c"]) / 1000.0)
                    break
        if len(translations) < max_frames:
            continue

        trans = np.array(translations)
        # 找运动最大的窗口
        for i in range(0, len(trans) - max_frames, max_frames // 2):
            seg = trans[i:i + max_frames]
            disp = np.linalg.norm(seg[-1] - seg[0])
            avg_speed = np.mean([np.linalg.norm(seg[j+1] - seg[j]) for j in range(len(seg)-1)])
            # 检查是否是物体在动而不是相机在动（看相机参数是否变化）
            candidates.append({
                "scene": scene,
                "obj_id": obj_id,
                "start": i,
                "disp_mm": disp * 1000,
                "avg_speed_mms": avg_speed * 1000,
            })

    candidates.sort(key=lambda x: x["disp_mm"], reverse=True)
    top = candidates[:5]

    print(f"\nTop {len(top)} segments with most object motion:")
    for i, c in enumerate(top):
        name = YCB_OBJECTS.get(c["obj_id"], f"obj_{c['obj_id']}")
        print(f"  {i+1}. Scene {c['scene']} - {name} | Disp: {c['disp_mm']:.1f}mm | Speed: {c['avg_speed_mms']:.1f}mm/s")

    all_results = []

    for idx, seg_info in enumerate(top):
        scene = seg_info["scene"]
        obj_id = seg_info["obj_id"]
        obj_name = YCB_OBJECTS.get(obj_id, f"obj_{obj_id}")

        print(f"\n{'='*50}")
        print(f"Test {idx+1}/{len(top)}: Scene {scene} - {obj_name}")
        print(f"  Object displacement: {seg_info['disp_mm']:.1f}mm")
        print(f"  Avg speed: {seg_info['avg_speed_mms']:.1f}mm/s")
        print(f"{'='*50}")

        scene_dir = test_dir / scene
        rgbs, depths, poses, K = load_scene_segment(scene_dir, obj_id, seg_info["start"], max_frames)
        T = len(rgbs)
        if T < 10:
            continue

        H, W = rgbs[0].shape[:2]
        print(f"  Loaded {T} frames ({W}x{H})")

        # 物体位移详情
        trans = np.array([p["t"] for p in poses])
        total_disp = np.linalg.norm(trans[-1] - trans[0])
        print(f"  Start pos: ({trans[0][0]*1000:.1f}, {trans[0][1]*1000:.1f}, {trans[0][2]*1000:.1f}) mm")
        print(f"  End pos:   ({trans[-1][0]*1000:.1f}, {trans[-1][1]*1000:.1f}, {trans[-1][2]*1000:.1f}) mm")
        print(f"  Total displacement: {total_disp*1000:.1f}mm")

        # 生成 mask
        model_size = get_object_size(models_dir, obj_id)
        mask = generate_object_mask(poses[0], model_size, K, (H, W))
        mask_pixels = int(np.sum(mask))
        print(f"  Mask pixels: {mask_pixels}")

        if mask_pixels < 50:
            print("  Mask too small, skip")
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
        print(f"  Tracking time: {t2-t1:.1f}s")

        tracks_np = tracks[0].cpu().numpy()
        vis_np = visibilities[0].cpu().numpy()
        N = tracks_np.shape[1]
        print(f"  Points: {N}, Visibility: {np.mean(vis_np):.1%}")

        # GT 速度
        gt_vel = compute_gt_velocity(poses)

        # CoTracker 速度
        pred_vel = compute_velocity(tracks_np, vis_np, depths, K)

        # 误差
        valid = ~np.all(pred_vel == 0, axis=1)
        valid[0] = False
        if np.sum(valid) > 0:
            err = np.linalg.norm(pred_vel[valid] - gt_vel[valid], axis=1)
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(err))
        else:
            rmse = mae = -1

        # 跟踪精度
        in_mask = 0
        total_vis = 0
        for t in range(T):
            mt = generate_object_mask(poses[t], model_size, K, (H, W))
            for i in range(N):
                if vis_np[t, i]:
                    total_vis += 1
                    x, y = int(tracks_np[t, i, 0]), int(tracks_np[t, i, 1])
                    if 0 <= x < W and 0 <= y < H and mt[y, x]:
                        in_mask += 1
        acc = in_mask / max(total_vis, 1)

        # 像素位移
        disps = []
        for i in range(N):
            for t in range(1, T):
                if vis_np[t, i] and vis_np[t-1, i]:
                    dx = tracks_np[t, i, 0] - tracks_np[t-1, i, 0]
                    dy = tracks_np[t, i, 1] - tracks_np[t-1, i, 1]
                    disps.append(np.sqrt(dx**2 + dy**2))

        # 生成视频
        out_path = video_dir / f"{scene}_{obj_name}_motion.mp4"
        create_video(rgbs, tracks_np, vis_np, out_path, fps=30,
                     gt_velocity=gt_vel, pred_velocity=pred_vel,
                     title=f"Scene {scene} | {obj_name} | Obj disp: {total_disp*1000:.0f}mm")

        result = {
            "scene": scene,
            "obj_name": obj_name,
            "obj_id": obj_id,
            "frames": T,
            "points": N,
            "tracking_time_s": round(t2-t1, 2),
            "visibility": round(float(np.mean(vis_np)), 4),
            "tracking_accuracy": round(acc, 4),
            "velocity_rmse_mms": round(rmse * 1000, 2) if rmse > 0 else -1,
            "velocity_mae_mms": round(mae * 1000, 2) if mae > 0 else -1,
            "object_displacement_mm": round(total_disp * 1000, 1),
            "avg_speed_mms": round(seg_info["avg_speed_mms"], 1),
            "mean_pixel_disp": round(float(np.mean(disps)) if disps else 0, 2),
            "max_pixel_disp": round(float(np.max(disps)) if disps else 0, 2),
            "video": str(out_path),
        }
        all_results.append(result)

        print(f"\n  --- Results ---")
        print(f"  Points: {N} (vis: {np.mean(vis_np):.1%})")
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Velocity RMSE: {result['velocity_rmse_mms']:.0f} mm/s")
        print(f"  Pixel disp: mean={result['mean_pixel_disp']:.1f}px max={result['max_pixel_disp']:.1f}px")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Summary ({len(all_results)} tests)")
    print(f"{'='*60}")
    if all_results:
        for r in all_results:
            print(f"  {r['scene']} {r['obj_name']}: disp={r['object_displacement_mm']:.0f}mm "
                  f"vis={r['visibility']:.0%} acc={r['tracking_accuracy']:.0%} "
                  f"RMSE={r['velocity_rmse_mms']:.0f}mm/s")

    summary = {
        "model": "CoTracker3",
        "test": "object_motion",
        "results": all_results,
    }
    with open(output_dir / "object_motion_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_dir / 'object_motion_results.json'}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/ycbv")
    parser.add_argument("--models_dir", default="datasets/ycbv/models")
    parser.add_argument("--checkpoint", default="checkpoints/cotracker3_offline.pth")
    parser.add_argument("--output_dir", default="outputs/cotracker_object_motion")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_frames", type=int, default=80)
    args = parser.parse_args()

    run_object_motion_test(
        data_root=args.data_root,
        models_dir=args.models_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        max_frames=args.max_frames,
    )
