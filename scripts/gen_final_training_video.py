"""
生成obj1_rgb最终CoTracker训练视频
- 左右分屏：原始视频 + 跟踪可视化
- 底部面板：位移/速度曲线
- 带详细运动分析标注
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


def run_cotracker(video_path, checkpoint, device="cpu", grid_size=20, max_frames=300):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    T = len(frames)
    H, W = frames[0].shape[:2]
    print(f"  Loaded {T} frames ({W}x{H})")

    video = np.stack(frames)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    video_tensor = video_tensor.unsqueeze(0).to(device)

    print(f"  Running CoTracker3 (grid_size={grid_size})...")
    t1 = time.time()
    with torch.no_grad():
        tracks, visibilities = predictor(video_tensor, grid_size=grid_size)
    t2 = time.time()
    print(f"  Tracking time: {t2-t1:.1f}s")

    return frames, tracks[0].cpu().numpy(), visibilities[0].cpu().numpy()


def compute_motion(tracks, vis, T):
    N = tracks.shape[1]
    disps_per_frame = np.zeros((T, N))
    total_disps = np.zeros(N)
    speeds = np.zeros(N)

    for i in range(N):
        for t in range(1, T):
            if vis[t, i] and vis[t-1, i]:
                dx = tracks[t, i, 0] - tracks[t-1, i, 0]
                dy = tracks[t, i, 1] - tracks[t-1, i, 1]
                d = np.sqrt(dx**2 + dy**2)
                disps_per_frame[t, i] = d
                total_disps[i] += d
                if d > speeds[i]:
                    speeds[i] = d

    return disps_per_frame, total_disps, speeds


def create_final_video(frames, tracks, vis, disps_per_frame, total_disps, speeds,
                        output_path, fps=30):
    T = len(frames)
    N = tracks.shape[1]
    H, W = frames[0].shape[:2]

    # Layout: left (original) | right (tracking) | bottom panel
    panel_h = 140
    total_w = W * 2  # left + right side by side
    total_h = H + panel_h

    np.random.seed(42)
    colors = np.random.randint(60, 255, (N, 3)).tolist()

    # Precompute average displacement per frame
    avg_disp_per_frame = np.array([disps_per_frame[t].mean() if disps_per_frame[t].max() > 0 else 0
                                    for t in range(T)])
    max_avg = max(avg_disp_per_frame.max(), 1)

    # Precompute cumulative displacement
    cum_disp = np.zeros(T)
    for t in range(1, T):
        cum_disp[t] = cum_disp[t-1] + avg_disp_per_frame[t]
    max_cum = max(cum_disp.max(), 1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))

    moving_pts = int(np.sum(total_disps > 5))
    vis_rate = float(np.mean(vis))

    for t in range(T):
        frame_bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)

        # Left: original with frame number
        left = frame_bgr.copy()
        cv2.putText(left, f"Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(left, f"Frame {t+1}/{T}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Right: tracking visualization
        right = frame_bgr.copy()
        for i in range(N):
            pts = []
            for tt in range(min(t + 1, tracks.shape[0])):
                if vis[tt, i]:
                    pts.append((int(tracks[tt, i, 0]), int(tracks[tt, i, 1])))
            if len(pts) > 1:
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    cv2.line(right, pts[k-1], pts[k], colors[i], max(1, int(2 * alpha)))
            if t < tracks.shape[0] and vis[t, i]:
                x, y = tracks[t, i]
                cv2.circle(right, (int(x), int(y)), 3, colors[i], -1)

        cv2.putText(right, f"CoTracker3 Tracking ({N} pts)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        vis_count = int(np.sum(vis[t])) if t < vis.shape[0] else 0
        cv2.putText(right, f"Visible: {vis_count}/{N}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Combine left and right
        combined = np.hstack([left, right])

        # Bottom panel
        panel = np.zeros((panel_h, total_w, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        # Title bar
        cv2.putText(panel, f"obj1_rgb | CoTracker3 | Moving: {moving_pts}/{N} | "
                    f"Vis: {vis_rate:.0%} | Max Disp: {total_disps.max():.0f}px | "
                    f"Max Speed: {speeds.max():.1f}px/f",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Displacement curve (left half of panel)
        chart_x = 10
        chart_y = 35
        chart_w = (total_w // 2) - 20
        chart_h = panel_h - 45

        # Draw axis
        cv2.rectangle(panel, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h),
                       (50, 50, 50), 1)
        cv2.putText(panel, f"Displacement/frame (px)", (chart_x, chart_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if t > 0:
            for ft in range(1, t + 1):
                x1 = chart_x + int((ft - 1) / T * chart_w)
                x2 = chart_x + int(ft / T * chart_w)
                val = min(avg_disp_per_frame[ft] / max_avg, 1.0)
                y1 = chart_y + chart_h - int(val * chart_h)
                y2 = chart_y + chart_h
                cv2.line(panel, (x1, y2), (x2, y1), (0, 200, 255), 2)

        # Cumulative displacement curve (right half of panel)
        chart2_x = total_w // 2 + 10
        chart2_w = (total_w // 2) - 20

        cv2.rectangle(panel, (chart2_x, chart_y), (chart2_x + chart2_w, chart_y + chart_h),
                       (50, 50, 50), 1)
        cv2.putText(panel, f"Cumulative Disp (px)", (chart2_x, chart_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if t > 0:
            for ft in range(1, t + 1):
                x1 = chart2_x + int((ft - 1) / T * chart2_w)
                x2 = chart2_x + int(ft / T * chart2_w)
                val = min(cum_disp[ft] / max_cum, 1.0)
                y1 = chart_y + chart_h - int(val * chart_h)
                y2 = chart_y + chart_h
                cv2.line(panel, (x1, y2), (x2, y1), (0, 255, 100), 2)

        # Current values
        cv2.putText(panel, f"Current: {avg_disp_per_frame[t]:.1f} px/f",
                    (chart_x + 5, chart_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        cv2.putText(panel, f"Total: {cum_disp[t]:.0f} px",
                    (chart2_x + 5, chart_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)

        # Combine
        final = np.vstack([combined, panel])
        writer.write(final)

    writer.release()
    print(f"  Final video: {output_path} ({os.path.getsize(str(output_path))/1024:.0f} KB)")


if __name__ == "__main__":
    video_path = "outputs/slipping_videos/obj1_rgb.mp4"
    checkpoint = "checkpoints/cotracker3_offline.pth"
    output_dir = Path("outputs/cotracker_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    output_path = output_dir / "videos" / "obj1_rgb_final_tracking.mp4"

    device = "cpu"
    grid_size = 20

    print("=" * 60)
    print("Final Training Video: obj1_rgb")
    print("=" * 60)

    # Initialize model
    print("\nLoading CoTracker3...")
    predictor = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
    predictor = predictor.to(device)
    predictor.eval()
    print("  Model loaded!")

    # Run tracking
    print(f"\nProcessing: {video_path}")
    frames, tracks, vis = run_cotracker(video_path, checkpoint, device, grid_size)

    # Compute motion
    T = len(frames)
    N = tracks.shape[1]
    disps_per_frame, total_disps, speeds = compute_motion(tracks, vis, T)

    moving = int(np.sum(total_disps > 5))
    print(f"\n  Points: {N}")
    print(f"  Visibility: {np.mean(vis):.1%}")
    print(f"  Moving: {moving}/{N}")
    print(f"  Max displacement: {total_disps.max():.1f}px")
    print(f"  Max speed: {speeds.max():.1f}px/frame")
    print(f"  Mean displacement: {total_disps.mean():.1f}px")

    # Top 10 moving points
    top_idx = np.argsort(total_disps)[-10:][::-1]
    print(f"\n  Top 10 moving points:")
    for idx in top_idx:
        print(f"    Point {idx}: {total_disps[idx]:.1f}px, speed={speeds[idx]:.1f}px/f, vis={np.mean(vis[:,idx]):.0%}")

    # Generate final video
    print(f"\nGenerating final video...")
    create_final_video(frames, tracks, vis, disps_per_frame, total_disps, speeds,
                        output_path, fps=30)

    # Save results
    results = {
        "video": "obj1_rgb",
        "type": "RGB",
        "frames": T,
        "resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
        "grid_size": grid_size,
        "points": N,
        "visibility": round(float(np.mean(vis)), 4),
        "moving_points": moving,
        "max_pixel_disp": round(float(total_disps.max()), 2),
        "mean_pixel_disp": round(float(total_disps.mean()), 2),
        "max_speed_px_per_frame": round(float(speeds.max()), 2),
        "mean_speed_px_per_frame": round(float(speeds.mean()), 2),
        "top_10_points": [
            {"point": int(idx), "disp_px": round(float(total_disps[idx]), 2),
             "speed_px_f": round(float(speeds[idx]), 2),
             "vis": round(float(np.mean(vis[:, idx])), 4)}
            for idx in top_idx
        ],
        "video_output": str(output_path),
    }
    with open(output_dir / "obj1_rgb_final_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults: {output_dir / 'obj1_rgb_final_results.json'}")
    print("\nDone!")
