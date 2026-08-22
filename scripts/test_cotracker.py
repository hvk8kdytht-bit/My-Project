"""
CoTracker 完整测试脚本
在 YCB-Video 真实数据上测试 CoTracker3 的表现

输出:
1. 跟踪视频回放（带点轨迹叠加的 MP4）
2. 量化指标（跟踪精度、速度估计误差、覆盖率）
3. 可视化图表（轨迹图、误差分布图）
4. HTML 报告
"""

import sys
import os
import json
import time
import numpy as np
import cv2
import torch
from pathlib import Path

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


def load_scene(scene_dir, max_frames=50):
    """加载场景的 RGB、深度、GT 位姿"""
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
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        R = np.array(pose_item["cam_R_m2c"]).reshape(3, 3)
        t = np.array(pose_item["cam_t_m2c"]) / 1000.0
        K = np.array(scene_cam[key]["cam_K"]).reshape(3, 3)

        rgbs.append(rgb)
        depths.append(depth)
        poses.append({"R": R, "t": t})
        Ks.append(K)

    return {
        "rgb": rgbs, "depth": depths, "poses": poses,
        "K": Ks[0] if Ks else np.eye(3), "obj_id": obj_id,
        "scene_id": scene_dir.name,
    }


def generate_object_mask(pose, model_size_m, K, img_shape):
    """从 GT 位姿生成物体 mask（椭圆近似）"""
    H, W = img_shape
    t = pose["t"]
    if t[2] <= 0:
        return np.zeros((H, W), dtype=bool)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    cx_pix = cx + t[0] * fx / t[2]
    cy_pix = cy + t[1] * fy / t[2]
    half_w = max(model_size_m[0] / 2 * fx / t[2], 2)
    half_h = max(model_size_m[1] / 2 * fy / t[2], 2)

    y_coords, x_coords = np.ogrid[:H, :W]
    mask = ((x_coords - cx_pix) / half_w) ** 2 + ((y_coords - cy_pix) / half_h) ** 2 <= 1.0
    return mask.astype(bool)


def get_object_size(models_dir, obj_id):
    """从 models_info.json 获取物体尺寸"""
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


def create_tracking_video(rgb_frames, tracks, visibilities, output_path, fps=15):
    """生成带跟踪点叠加的视频"""
    H, W = rgb_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    T = len(rgb_frames)
    N = tracks.shape[1] if len(tracks.shape) == 3 else tracks.shape[0]

    # 颜色：按点索引分配不同颜色
    np.random.seed(42)
    colors = np.random.randint(0, 255, (N, 3)).tolist()

    for t in range(T):
        frame = rgb_frames[t].copy()

        # 画轨迹线（从开始到当前帧）
        for i in range(min(N, tracks.shape[1] if len(tracks.shape) == 3 else tracks.shape[0])):
            pts = []
            for tt in range(t + 1):
                if tt < tracks.shape[0]:
                    x, y = tracks[tt, i]
                    if visibilities[tt, i]:
                        pts.append((int(x), int(y)))
            if len(pts) > 1:
                pts_array = np.array(pts)
                color = colors[i]
                # 画轨迹线（渐变透明度）
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    thickness = max(1, int(2 * alpha))
                    cv2.line(frame, pts[k - 1], pts[k], color, thickness)

            # 画当前点
            if t < tracks.shape[0]:
                x, y = tracks[t, i]
                if visibilities[t, i]:
                    cv2.circle(frame, (int(x), int(y)), 4, colors[i], -1)
                    cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 1)

        # 添加帧号和统计信息
        vis_count = np.sum(visibilities[t]) if t < visibilities.shape[0] else 0
        info_text = f"Frame {t+1}/{T} | Visible: {vis_count}/{N}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(frame)

    writer.release()
    print(f"  视频已保存: {output_path}")


def compute_gt_velocity(poses, fps=30.0):
    """从 GT 位姿计算物体速度"""
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


def compute_cotracker_velocity(tracks, visibilities, depth_frames, K, fps=30.0):
    """从 CoTracker 跟踪点轨迹 + 深度图计算 3D 速度"""
    T, N, _ = tracks.shape
    dt = 1.0 / fps
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 每个点每帧的 3D 位置
    points_3d = np.full((T, N, 3), np.nan, dtype=np.float32)

    for t in range(T):
        depth = depth_frames[t]
        H, W = depth.shape
        for i in range(N):
            x, y = tracks[t, i]
            xi, yi = int(round(x)), int(round(y))
            if visibilities[t, i] and 0 <= xi < W and 0 <= yi < H and depth[yi, xi] > 0:
                z = depth[yi, xi]
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                points_3d[t, i] = [X, Y, z]

    # 物体中心速度（所有可见点的中位数）
    velocities = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T):
        if t > 0:
            pts_curr = points_3d[t]
            pts_prev = points_3d[t - 1]
            valid = ~np.any(np.isnan(pts_curr), axis=1) & ~np.any(np.isnan(pts_prev), axis=1)
            if np.sum(valid) > 0:
                disp = pts_curr[valid] - pts_prev[valid]
                velocities[t] = np.median(disp, axis=0) / dt

    return velocities


def run_cotracker_test(
    data_root="datasets/ycbv",
    models_dir="datasets/ycbv/models",
    checkpoint="checkpoints/cotracker3_offline.pth",
    max_frames=50,
    num_scenes=3,
    output_dir="outputs/cotracker_test",
    device="cpu",
):
    """运行完整 CoTracker 测试"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    # 加载模型
    print("=" * 60)
    print("CoTracker3 完整测试")
    print("=" * 60)

    print(f"\n加载 CoTracker3 模型...")
    predictor = CoTrackerPredictor(
        checkpoint=checkpoint,
        offline=True,
        v2=False,
        window_len=60,
    )
    predictor = predictor.to(device)
    predictor.eval()
    print(f"  模型加载成功")

    # 选择测试场景
    test_dir = Path(data_root) / "test"
    all_scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    # 读取场景划分
    split_path = Path(data_root) / "scene_split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        test_scenes = split.get("test", all_scenes[:num_scenes])
    else:
        test_scenes = all_scenes[:num_scenes]

    test_scenes = test_scenes[:num_scenes]
    print(f"\n测试场景: {test_scenes}")

    all_results = []

    for scene_idx, scene_id in enumerate(test_scenes):
        print(f"\n{'='*50}")
        print(f"场景 {scene_idx+1}/{len(test_scenes)}: {scene_id}")
        print(f"{'='*50}")

        scene_dir = test_dir / scene_id
        if not scene_dir.exists():
            print(f"  场景不存在，跳过")
            continue

        # 加载数据
        t0 = time.time()
        data = load_scene(str(scene_dir), max_frames=max_frames)
        T = len(data["rgb"])
        if T < 5:
            print(f"  帧数太少 ({T})，跳过")
            continue

        obj_name = YCB_OBJECTS.get(data["obj_id"], f"obj_{data['obj_id']}")
        print(f"  物体: {obj_name}")
        print(f"  帧数: {T}")

        # 生成物体 mask
        model_size = get_object_size(models_dir, data["obj_id"])
        H, W = data["rgb"][0].shape[:2]
        mask = generate_object_mask(data["poses"][0], model_size, data["K"], (H, W))
        mask_pixels = np.sum(mask)
        print(f"  Mask 像素: {mask_pixels}")

        if mask_pixels < 50:
            print(f"  Mask 太小，跳过")
            continue

        # 准备视频 tensor
        video = np.stack(data["rgb"])  # (T, H, W, 3)
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        video_tensor = video_tensor.unsqueeze(0)  # (1, T, 3, H, W)
        video_tensor = video_tensor.to(device)

        # mask tensor
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask_tensor = mask_tensor.to(device)

        # 运行 CoTracker
        print(f"  运行 CoTracker3 跟踪...")
        t1 = time.time()
        with torch.no_grad():
            tracks, visibilities = predictor(
                video_tensor,
                segm_mask=mask_tensor,
                grid_size=10,
            )
        t2 = time.time()
        print(f"  跟踪耗时: {t2 - t1:.1f}s")

        # 转回 numpy
        tracks_np = tracks[0].cpu().numpy()  # (T, N, 2)
        vis_np = visibilities[0].cpu().numpy()  # (T, N)
        N = tracks_np.shape[1]
        print(f"  跟踪点数: {N}")
        print(f"  平均可见率: {np.mean(vis_np):.2%}")

        # 生成跟踪视频
        video_path = video_dir / f"{scene_id}_{obj_name}_tracking.mp4"
        create_tracking_video(data["rgb"], tracks_np, vis_np, video_path)

        # 计算 GT 速度
        gt_velocity = compute_gt_velocity(data["poses"])

        # 计算 CoTracker 估计速度
        pred_velocity = compute_cotracker_velocity(
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

        # 跟踪精度：点是否还在物体 mask 内
        in_mask_count = 0
        total_visible = 0
        for t in range(T):
            # 用当前帧的 GT 位姿生成 mask
            if t < len(data["poses"]):
                mask_t = generate_object_mask(data["poses"][t], model_size, data["K"], (H, W))
                for i in range(N):
                    if vis_np[t, i]:
                        total_visible += 1
                        x, y = int(tracks_np[t, i, 0]), int(tracks_np[t, i, 1])
                        if 0 <= x < W and 0 <= y < H and mask_t[y, x]:
                            in_mask_count += 1

        tracking_accuracy = in_mask_count / max(total_visible, 1)

        # 覆盖率
        coverage = float(np.mean(vis_np))

        # 点位移统计
        displacements = []
        for i in range(N):
            for t in range(1, T):
                if vis_np[t, i] and vis_np[t - 1, i]:
                    dx = tracks_np[t, i, 0] - tracks_np[t - 1, i, 0]
                    dy = tracks_np[t, i, 1] - tracks_np[t - 1, i, 1]
                    displacements.append(np.sqrt(dx ** 2 + dy ** 2))

        result = {
            "scene_id": scene_id,
            "obj_id": data["obj_id"],
            "obj_name": obj_name,
            "num_frames": T,
            "num_points": N,
            "tracking_time_s": round(t2 - t1, 2),
            "visibility_rate": round(coverage, 4),
            "tracking_accuracy": round(tracking_accuracy, 4),
            "in_mask_ratio": round(in_mask_count / max(total_visible, 1), 4),
            "total_visible": total_visible,
            "in_mask_count": in_mask_count,
            "velocity_rmse_mms": round(vel_rmse * 1000, 2) if vel_rmse > 0 else -1,
            "velocity_mae_mms": round(vel_mae * 1000, 2) if vel_mae > 0 else -1,
            "mean_displacement_px": round(float(np.mean(displacements)) if displacements else 0, 2),
            "max_displacement_px": round(float(np.max(displacements)) if displacements else 0, 2),
            "video_path": str(video_path),
        }

        all_results.append(result)

        print(f"\n  --- 结果 ---")
        print(f"  跟踪点数: {N}")
        print(f"  可见率: {coverage:.2%}")
        print(f"  跟踪精度（点在物体内）: {tracking_accuracy:.2%}")
        print(f"  速度 RMSE: {vel_rmse*1000:.2f} mm/s")
        print(f"  速度 MAE: {vel_mae*1000:.2f} mm/s")
        print(f"  平均位移: {result['mean_displacement_px']:.2f} px")
        print(f"  视频已保存: {video_path}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"汇总 ({len(all_results)} 个场景)")
    print(f"{'='*60}")

    if all_results:
        avg_vis = np.mean([r["visibility_rate"] for r in all_results])
        avg_acc = np.mean([r["tracking_accuracy"] for r in all_results])
        vel_rmses = [r["velocity_rmse_mms"] for r in all_results if r["velocity_rmse_mms"] > 0]
        avg_vel_rmse = np.mean(vel_rmses) if vel_rmses else -1
        avg_time = np.mean([r["tracking_time_s"] for r in all_results])

        print(f"  平均可见率: {avg_vis:.2%}")
        print(f"  平均跟踪精度: {avg_acc:.2%}")
        print(f"  平均速度 RMSE: {avg_vel_rmse:.2f} mm/s")
        print(f"  平均跟踪耗时: {avg_time:.1f}s / 场景")

    # 保存 JSON 结果
    summary = {
        "model": "CoTracker3 (scaled_offline)",
        "device": device,
        "num_scenes": len(all_results),
        "results": all_results,
        "summary": {
            "avg_visibility_rate": float(np.mean([r["visibility_rate"] for r in all_results])) if all_results else 0,
            "avg_tracking_accuracy": float(np.mean([r["tracking_accuracy"] for r in all_results])) if all_results else 0,
            "avg_velocity_rmse_mms": float(np.mean(vel_rmses)) if vel_rmses else -1,
            "avg_tracking_time_s": float(np.mean([r["tracking_time_s"] for r in all_results])) if all_results else 0,
        },
    }

    with open(output_dir / "cotracker_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_dir / 'cotracker_results.json'}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/ycbv")
    parser.add_argument("--models_dir", default="datasets/ycbv/models")
    parser.add_argument("--checkpoint", default="checkpoints/cotracker3_offline.pth")
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--num_scenes", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs/cotracker_test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_cotracker_test(
        data_root=args.data_root,
        models_dir=args.models_dir,
        checkpoint=args.checkpoint,
        max_frames=args.max_frames,
        num_scenes=args.num_scenes,
        output_dir=args.output_dir,
        device=args.device,
    )
