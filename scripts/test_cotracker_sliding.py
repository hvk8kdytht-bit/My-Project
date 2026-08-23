"""
CoTracker3 测试：MuJoCo 仿真的物体滑移视频
相机固定，物体在桌面缓慢滑动
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


def load_simulation_frames(rgb_dir, max_frames=150):
    """加载仿真帧"""
    rgb_dir = Path(rgb_dir)
    files = sorted(rgb_dir.glob("*.png"))
    frames = []
    for f in files[:max_frames]:
        img = cv2.imread(str(f))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def create_tracking_video(rgb_frames, tracks, visibilities, output_path, fps=30,
                          gt_data=None, title=""):
    """生成跟踪视频"""
    H, W = rgb_frames[0].shape[:2]
    panel_h = 140
    total_h = H + panel_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, total_h))

    T = len(rgb_frames)
    N = tracks.shape[1]
    np.random.seed(42)
    colors = np.random.randint(50, 255, (N, 3)).tolist()

    for t in range(T):
        frame = cv2.cvtColor(rgb_frames[t], cv2.COLOR_RGB2BGR)

        # 画轨迹
        for i in range(N):
            pts = []
            for tt in range(min(t + 1, tracks.shape[0])):
                if visibilities[tt, i]:
                    pts.append((int(tracks[tt, i, 0]), int(tracks[tt, i, 1])))
            if len(pts) > 1:
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    cv2.line(frame, pts[k - 1], pts[k], colors[i], max(1, int(3 * alpha)))
            if t < tracks.shape[0] and visibilities[t, i]:
                x, y = tracks[t, i]
                cv2.circle(frame, (int(x), int(y)), 4, colors[i], -1)
                cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 1)

        # 信息面板
        panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        vis_count = int(np.sum(visibilities[t])) if t < visibilities.shape[0] else 0
        cv2.putText(panel, f"{title} | Frame {t+1}/{T} | Points: {vis_count}/{N}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 像素位移
        if t > 0:
            total_disp = 0
            count = 0
            for i in range(N):
                if visibilities[t, i] and visibilities[t - 1, i]:
                    dx = tracks[t, i, 0] - tracks[t - 1, i, 0]
                    dy = tracks[t, i, 1] - tracks[t - 1, i, 1]
                    total_disp += np.sqrt(dx**2 + dy**2)
                    count += 1
            avg_disp = total_disp / max(count, 1)
            cv2.putText(panel, f"Pixel displacement: {avg_disp:.1f} px/frame", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # GT 物体位置
        if gt_data and t < len(gt_data):
            gt = gt_data[t]
            for j, obj in enumerate(["box", "ball", "cyl"]):
                pos = gt[obj]["pos"]
                y_pos = 75 + j * 20
                color = [(0, 0, 255), (255, 0, 0), (0, 255, 0)][j]
                cv2.putText(panel, f"{obj}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        frame_with_panel = np.vstack([frame, panel])
        writer.write(frame_with_panel)

    writer.release()
    print(f"  Video: {output_path} ({os.path.getsize(str(output_path))/1024:.0f} KB)")


def run_test(
    sim_dir="outputs/sliding_simulation",
    checkpoint="checkpoints/cotracker3_offline.pth",
    output_dir="outputs/cotracker_sliding_test",
    device="cpu",
    grid_size=15,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CoTracker3 Sliding Object Test")
    print("(Fixed camera, objects sliding on table)")
    print("=" * 60)

    # 加载模型
    print("\nLoading CoTracker3...")
    predictor = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
    predictor = predictor.to(device)
    predictor.eval()
    print("  Model loaded!")

    # 加载仿真数据
    rgb_dir = Path(sim_dir) / "rgb"
    gt_path = Path(sim_dir) / "sliding_gt.json"

    frames = load_simulation_frames(rgb_dir)
    T = len(frames)
    H, W = frames[0].shape[:2]
    print(f"\nLoaded {T} frames ({W}x{H})")

    with open(gt_path) as f:
        gt_all = json.load(f)
    gt_data = gt_all["frames"]

    # GT 位移统计
    print("\nGT object motion:")
    for obj in ["box", "ball", "cyl"]:
        p0 = np.array(gt_data[0][obj]["pos"])
        p_end = np.array(gt_data[-1][obj]["pos"])
        disp = np.linalg.norm(p_end - p0) * 1000
        print(f"  {obj}: {disp:.0f}mm total displacement")

    # 运行 CoTracker（网格点跟踪）
    video = np.stack(frames)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    video_tensor = video_tensor.unsqueeze(0).to(device)

    print(f"\nRunning CoTracker3 (grid_size={grid_size})...")
    t1 = time.time()
    with torch.no_grad():
        tracks, visibilities = predictor(video_tensor, grid_size=grid_size)
    t2 = time.time()
    print(f"  Tracking time: {t2 - t1:.1f}s")

    tracks_np = tracks[0].cpu().numpy()
    vis_np = visibilities[0].cpu().numpy()
    N = tracks_np.shape[1]
    print(f"  Points: {N}")
    print(f"  Visibility: {np.mean(vis_np):.1%}")

    # 像素位移统计
    point_disps = []
    for i in range(N):
        total_d = 0
        for t in range(1, T):
            if vis_np[t, i] and vis_np[t - 1, i]:
                dx = tracks_np[t, i, 0] - tracks_np[t - 1, i, 0]
                dy = tracks_np[t, i, 1] - tracks_np[t - 1, i, 1]
                total_d += np.sqrt(dx**2 + dy**2)
        point_disps.append(total_d)

    point_disps = np.array(point_disps)
    moving = np.sum(point_disps > 3)
    static = N - moving

    print(f"\n  Moving points: {moving}/{N} ({moving/N:.0%})")
    print(f"  Static points: {static}/{N}")
    print(f"  Mean total displacement: {np.mean(point_disps):.1f} px")
    print(f"  Max total displacement: {np.max(point_disps):.1f} px")

    # 找运动最大的点
    top_moving = np.argsort(point_disps)[-5:][::-1]
    print(f"\n  Top 5 moving points:")
    for idx in top_moving:
        print(f"    Point {idx}: {point_disps[idx]:.1f}px total, vis={np.mean(vis_np[:, idx]):.0%}")

    # 生成视频
    out_path = video_dir / "sliding_tracking.mp4"
    create_tracking_video(frames, tracks_np, vis_np, out_path, fps=30,
                          gt_data=gt_data, title="Sliding Objects | Fixed Camera")

    # 保存结果
    result = {
        "model": "CoTracker3",
        "test": "sliding_simulation",
        "frames": T,
        "resolution": f"{W}x{H}",
        "grid_size": grid_size,
        "num_points": N,
        "tracking_time_s": round(t2 - t1, 2),
        "visibility": round(float(np.mean(vis_np)), 4),
        "moving_points": int(moving),
        "static_points": int(static),
        "mean_pixel_disp": round(float(np.mean(point_disps)), 2),
        "max_pixel_disp": round(float(np.max(point_disps)), 2),
        "gt_object_displacements_mm": {
            "box": round(np.linalg.norm(np.array(gt_data[-1]["box"]["pos"]) - np.array(gt_data[0]["box"]["pos"])) * 1000, 1),
            "ball": round(np.linalg.norm(np.array(gt_data[-1]["ball"]["pos"]) - np.array(gt_data[0]["ball"]["pos"])) * 1000, 1),
            "cyl": round(np.linalg.norm(np.array(gt_data[-1]["cyl"]["pos"]) - np.array(gt_data[0]["cyl"]["pos"])) * 1000, 1),
        },
        "video": str(out_path),
    }

    with open(output_dir / "sliding_results.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_dir / 'sliding_results.json'}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", default="outputs/sliding_simulation")
    parser.add_argument("--checkpoint", default="checkpoints/cotracker3_offline.pth")
    parser.add_argument("--output_dir", default="outputs/cotracker_sliding_test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid_size", type=int, default=15)
    args = parser.parse_args()

    run_test(
        sim_dir=args.sim_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        grid_size=args.grid_size,
    )
