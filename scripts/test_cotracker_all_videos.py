"""
CoTracker3 全视频测试：从8个滑移视频中找出最适合CoTracker测试的视频
测试4个RGB + 4个RGBD视频，输出综合对比结果
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


def create_tracking_video(rgb_frames, tracks, visibilities, output_path, fps=30,
                          title="", panel_info=None):
    H, W = rgb_frames[0].shape[:2]
    panel_h = 100
    total_h = H + panel_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, total_h))

    T = len(rgb_frames)
    N = tracks.shape[1]
    np.random.seed(42)
    colors = np.random.randint(50, 255, (N, 3)).tolist()

    for t in range(T):
        frame = rgb_frames[t].copy()

        for i in range(N):
            pts = []
            for tt in range(min(t + 1, tracks.shape[0])):
                if visibilities[tt, i]:
                    pts.append((int(tracks[tt, i, 0]), int(tracks[tt, i, 1])))
            if len(pts) > 1:
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    cv2.line(frame, pts[k - 1], pts[k], colors[i], max(1, int(2 * alpha)))
            if t < tracks.shape[0] and visibilities[t, i]:
                x, y = tracks[t, i]
                cv2.circle(frame, (int(x), int(y)), 3, colors[i], -1)

        panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        vis_count = int(np.sum(visibilities[t])) if t < visibilities.shape[0] else 0
        cv2.putText(panel, f"{title} | Frame {t+1}/{T} | Points: {vis_count}/{N}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
            cv2.putText(panel, f"Avg pixel disp: {avg_disp:.1f} px/frame",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        if panel_info:
            cv2.putText(panel, panel_info, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

        frame_with_panel = np.vstack([frame, panel])
        writer.write(frame_with_panel)

    writer.release()
    print(f"  Video: {output_path} ({os.path.getsize(str(output_path))/1024:.0f} KB)")


def analyze_motion(tracks, vis, T):
    N = tracks.shape[1]
    point_disps = []
    point_speeds = []
    for i in range(N):
        total_d = 0
        max_speed = 0
        for t in range(1, T):
            if vis[t, i] and vis[t-1, i]:
                dx = tracks[t, i, 0] - tracks[t-1, i, 0]
                dy = tracks[t, i, 1] - tracks[t-1, i, 1]
                d = np.sqrt(dx**2 + dy**2)
                total_d += d
                max_speed = max(max_speed, d)
        point_disps.append(total_d)
        point_speeds.append(max_speed)
    return np.array(point_disps), np.array(point_speeds)


def run_test(
    video_dir="outputs/slipping_videos",
    checkpoint="checkpoints/cotracker3_offline.pth",
    output_dir="outputs/cotracker_all_test",
    device="cpu",
    grid_size=20,
    max_frames=300,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir_out = output_dir / "videos"
    video_dir_out.mkdir(exist_ok=True)

    print("=" * 60)
    print("CoTracker3 All Videos Test (RGB + RGBD)")
    print("=" * 60)

    print("\nLoading CoTracker3...")
    predictor = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
    predictor = predictor.to(device)
    predictor.eval()
    print("  Model loaded!")

    all_videos = [
        ("obj1_rgb", "RGB"), ("obj1_rgbd", "RGBD"),
        ("obj2_rgb", "RGB"), ("obj2_rgbd", "RGBD"),
        ("obj3_rgb", "RGB"), ("obj3_rgbd", "RGBD"),
        ("obj4_rgb", "RGB"), ("obj4_rgbd", "RGBD"),
    ]

    all_results = []

    for vid_idx, (vid_name, vid_type) in enumerate(all_videos):
        video_path = Path(video_dir) / f"{vid_name}.mp4"
        if not video_path.exists():
            print(f"\n[SKIP] {vid_name}.mp4 not found")
            continue

        print(f"\n{'='*50}")
        print(f"Video {vid_idx+1}/{len(all_videos)}: {vid_name} ({vid_type})")
        print(f"{'='*50}")

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or len(frames) >= max_frames:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        T = len(frames)
        if T == 0:
            print("  No frames loaded, skipping")
            continue
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

        tracks_np = tracks[0].cpu().numpy()
        vis_np = visibilities[0].cpu().numpy()
        N = tracks_np.shape[1]

        disps, speeds = analyze_motion(tracks_np, vis_np, T)
        moving = int(np.sum(disps > 5))
        static = N - moving
        vis_rate = float(np.mean(vis_np))

        print(f"  Points: {N} (vis: {vis_rate:.1%})")
        print(f"  Moving: {moving}/{N} ({moving/N:.0%})")
        print(f"  Max disp: {np.max(disps):.1f}px, Max speed: {np.max(speeds):.1f}px/frame")
        print(f"  Mean disp: {np.mean(disps):.1f}px, Median disp: {np.median(disps):.1f}px")

        top_idx = np.argsort(disps)[-5:][::-1]
        print(f"  Top 5 moving points:")
        for idx in top_idx:
            print(f"    Point {idx}: {disps[idx]:.1f}px total, vis={np.mean(vis_np[:,idx]):.0%}")

        out_path = video_dir_out / f"{vid_name}_tracking.mp4"
        info = f"Moving: {moving}/{N} | Max disp: {np.max(disps):.0f}px | Type: {vid_type}"
        create_tracking_video(
            [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames],
            tracks_np, vis_np, out_path, fps=30,
            title=f"{vid_name} ({vid_type})",
            panel_info=info,
        )

        result = {
            "video": vid_name,
            "type": vid_type,
            "frames": T,
            "resolution": f"{W}x{H}",
            "points": N,
            "tracking_time_s": round(t2-t1, 2),
            "visibility": round(vis_rate, 4),
            "moving_points": moving,
            "static_points": static,
            "max_pixel_disp": round(float(np.max(disps)), 2),
            "mean_pixel_disp": round(float(np.mean(disps)), 2),
            "median_pixel_disp": round(float(np.median(disps)), 2),
            "max_speed_px_per_frame": round(float(np.max(speeds)), 2),
            "mean_speed_px_per_frame": round(float(np.mean(speeds)), 2),
            "video_output": str(out_path),
        }
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"Summary ({len(all_results)} videos)")
    print(f"{'='*60}")
    print(f"{'Video':<15} {'Type':<5} {'Pts':<5} {'Vis%':<6} {'Move':<5} {'MaxDisp':<8} {'MeanDisp':<9}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['video']:<15} {r['type']:<5} {r['points']:<5} "
              f"{r['visibility']:.0%}   {r['moving_points']:<5} "
              f"{r['max_pixel_disp']:<8.1f} {r['mean_pixel_disp']:<9.1f}")

    best_rgb = max([r for r in all_results if r["type"] == "RGB"],
                   key=lambda x: x["max_pixel_disp"], default=None)
    best_rgbd = max([r for r in all_results if r["type"] == "RGBD"],
                    key=lambda x: x["max_pixel_disp"], default=None)

    print(f"\nBest RGB:  {best_rgb['video']} (max_disp={best_rgb['max_pixel_disp']:.0f}px)")
    print(f"Best RGBD: {best_rgbd['video']} (max_disp={best_rgbd['max_pixel_disp']:.0f}px)")

    summary = {
        "model": "CoTracker3",
        "test": "all_videos_comparison",
        "grid_size": grid_size,
        "results": all_results,
        "best_rgb": best_rgb,
        "best_rgbd": best_rgbd,
    }
    with open(output_dir / "all_test_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_dir / 'all_test_results.json'}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="outputs/slipping_videos")
    parser.add_argument("--checkpoint", default="checkpoints/cotracker3_offline.pth")
    parser.add_argument("--output_dir", default="outputs/cotracker_all_test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid_size", type=int, default=20)
    args = parser.parse_args()

    run_test(
        video_dir=args.video_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        grid_size=args.grid_size,
    )
