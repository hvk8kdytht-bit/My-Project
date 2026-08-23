"""
CoTracker3 外部视频测试
用 CoTracker 官方演示视频 (apple.mp4) 和其他运动视频测试点跟踪效果
"""
import sys
import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "co-tracker-src" / "co-tracker-main"))

from cotracker.predictor import CoTrackerPredictor


def load_video(video_path, max_frames=120):
    """加载视频帧"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) >= max_frames:
            break
    cap.release()

    print(f"  Loaded {len(frames)} frames, {fps:.1f} fps, {frames[0].shape[1]}x{frames[0].shape[0]}")
    return frames, fps


def create_tracking_video_rgb(rgb_frames, tracks, visibilities, output_path, fps=30):
    """生成跟踪视频（纯RGB，不需要深度）"""
    H, W = rgb_frames[0].shape[:2]
    panel_h = 80
    total_h = H + panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, total_h))
    if not writer.isOpened():
        print("  VideoWriter failed to open!")
        return

    T = len(rgb_frames)
    N = tracks.shape[1]
    np.random.seed(42)
    colors = np.random.randint(50, 255, (N, 3)).tolist()

    for t in range(T):
        frame = cv2.cvtColor(rgb_frames[t], cv2.COLOR_RGB2BGR)

        # 画轨迹线
        for i in range(N):
            pts = []
            for tt in range(t + 1):
                if tt < tracks.shape[0] and visibilities[tt, i]:
                    pts.append((int(tracks[tt, i, 0]), int(tracks[tt, i, 1])))
            if len(pts) > 1:
                color = colors[i]
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, pts[k - 1], pts[k], color, thickness)

            if t < tracks.shape[0] and visibilities[t, i]:
                x, y = tracks[t, i]
                cv2.circle(frame, (int(x), int(y)), 5, colors[i], -1)
                cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), 1)

        # 信息面板
        panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        vis_count = int(np.sum(visibilities[t])) if t < visibilities.shape[0] else 0
        cv2.putText(panel, f"Frame {t+1}/{T} | Points: {vis_count}/{N} visible | CoTracker3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 像素位移统计
        if t > 0:
            total_disp = 0
            count = 0
            for i in range(N):
                if visibilities[t, i] and visibilities[t-1, i]:
                    dx = tracks[t, i, 0] - tracks[t-1, i, 0]
                    dy = tracks[t, i, 1] - tracks[t-1, i, 1]
                    total_disp += np.sqrt(dx**2 + dy**2)
                    count += 1
            avg_disp = total_disp / max(count, 1)
            cv2.putText(panel, f"Avg pixel displacement: {avg_disp:.1f} px", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        frame_with_panel = np.vstack([frame, panel])
        writer.write(frame_with_panel)

    writer.release()
    file_size = os.path.getsize(str(output_path))
    print(f"  Video saved: {output_path} ({file_size/1024:.0f} KB)")


def run_external_test(
    video_dir="datasets/external_videos",
    checkpoint="checkpoints/cotracker3_offline.pth",
    output_dir="outputs/cotracker_external",
    device="cpu",
    grid_size=20,
    max_frames=120,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_out = output_dir / "videos"
    video_out.mkdir(exist_ok=True)

    print("=" * 60)
    print("CoTracker3 External Video Test")
    print("=" * 60)

    # 加载模型
    print("\nLoading CoTracker3...")
    predictor = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
    predictor = predictor.to(device)
    predictor.eval()
    print("  Model loaded!")

    # 找所有视频文件
    video_dir = Path(video_dir)
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
    print(f"\nFound {len(videos)} videos: {[v.name for v in videos]}")

    all_results = []

    for vid_idx, video_path in enumerate(videos):
        print(f"\n{'='*50}")
        print(f"Video {vid_idx+1}/{len(videos)}: {video_path.name}")
        print(f"{'='*50}")

        # 加载视频
        frames, fps = load_video(video_path, max_frames=max_frames)
        if frames is None or len(frames) < 5:
            continue

        T = len(frames)
        H, W = frames[0].shape[:2]

        # 准备 tensor
        video = np.stack(frames)
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        video_tensor = video_tensor.unsqueeze(0).to(device)

        # 方式1: 网格点跟踪
        print(f"  Running CoTracker3 (grid_size={grid_size})...")
        t1 = time.time()
        with torch.no_grad():
            tracks, visibilities = predictor(video_tensor, grid_size=grid_size)
        t2 = time.time()
        print(f"  Tracking time: {t2 - t1:.1f}s")

        tracks_np = tracks[0].cpu().numpy()
        vis_np = visibilities[0].cpu().numpy()
        N = tracks_np.shape[1]
        print(f"  Tracked points: {N}")
        print(f"  Avg visibility: {np.mean(vis_np):.2%}")

        # 计算像素位移统计
        displacements = []
        for i in range(N):
            for t in range(1, T):
                if vis_np[t, i] and vis_np[t-1, i]:
                    dx = tracks_np[t, i, 0] - tracks_np[t-1, i, 0]
                    dy = tracks_np[t, i, 1] - tracks_np[t-1, i, 1]
                    displacements.append(np.sqrt(dx**2 + dy**2))

        # 每个点的总位移
        point_displacements = []
        for i in range(N):
            total_d = 0
            for t in range(1, T):
                if vis_np[t, i] and vis_np[t-1, i]:
                    dx = tracks_np[t, i, 0] - tracks_np[t-1, i, 0]
                    dy = tracks_np[t, i, 1] - tracks_np[t-1, i, 1]
                    total_d += np.sqrt(dx**2 + dy**2)
            point_displacements.append(total_d)

        moving_points = np.sum(np.array(point_displacements) > 5)
        static_points = N - moving_points

        # 生成视频
        out_path = video_out / f"{video_path.stem}_cotracker.mp4"
        create_tracking_video_rgb(frames, tracks_np, vis_np, out_path, fps=fps)

        result = {
            "video": video_path.name,
            "frames": T,
            "fps": round(fps, 1),
            "resolution": f"{W}x{H}",
            "num_points": N,
            "tracking_time_s": round(t2 - t1, 2),
            "visibility_rate": round(float(np.mean(vis_np)), 4),
            "moving_points": int(moving_points),
            "static_points": int(static_points),
            "mean_pixel_disp_per_frame": round(float(np.mean(displacements)) if displacements else 0, 2),
            "max_pixel_disp_per_frame": round(float(np.max(displacements)) if displacements else 0, 2),
            "mean_total_pixel_disp": round(float(np.mean(point_displacements)), 2),
            "max_total_pixel_disp": round(float(np.max(point_displacements)), 2),
            "video_output": str(out_path),
        }

        all_results.append(result)

        print(f"\n  --- Results ---")
        print(f"  Tracked points: {N} (moving: {moving_points}, static: {static_points})")
        print(f"  Visibility: {np.mean(vis_np):.2%}")
        print(f"  Mean displacement/frame: {result['mean_pixel_disp_per_frame']:.2f} px")
        print(f"  Max displacement/frame: {result['max_pixel_disp_per_frame']:.2f} px")
        print(f"  Mean total displacement: {result['mean_total_pixel_disp']:.2f} px")
        print(f"  Max total displacement: {result['max_total_pixel_disp']:.2f} px")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Summary ({len(all_results)} videos)")
    print(f"{'='*60}")
    if all_results:
        for r in all_results:
            print(f"  {r['video']}: {r['num_points']} pts, vis={r['visibility_rate']:.0%}, "
                  f"moving={r['moving_points']}, max_disp={r['max_total_pixel_disp']:.0f}px")

    import json
    summary = {
        "model": "CoTracker3 (scaled_offline)",
        "test": "external_videos",
        "num_videos": len(all_results),
        "results": all_results,
    }
    with open(output_dir / "external_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_dir / 'external_results.json'}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="datasets/external_videos")
    parser.add_argument("--checkpoint", default="checkpoints/cotracker3_offline.pth")
    parser.add_argument("--output_dir", default="outputs/cotracker_external")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=120)
    args = parser.parse_args()

    run_external_test(
        video_dir=args.video_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        grid_size=args.grid_size,
        max_frames=args.max_frames,
    )
