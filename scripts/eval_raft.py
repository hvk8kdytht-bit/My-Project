"""RAFT 深度学习光流在 YCB-Video 上的评估"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.velocity.raft_estimator import RaftOpticalFlowEstimator
from scripts.eval_cotracker import (
    load_scene_data, compute_gt_velocity_from_poses,
)
from src.utils.mask_utils import get_object_model_size, generate_object_mask_from_pose_simple


def main():
    data_root = Path("datasets/ycbv")
    test_dir = data_root / "test"
    all_scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    test_scenes = all_scenes[-2:]  # 最后2个场景做测试

    print(f"评估场景: {test_scenes}")

    estimator = RaftOpticalFlowEstimator(model_size="small", device="cpu", fps=30.0)
    estimator.load_model()

    all_errors = []
    for scene_id in test_scenes:
        scene_dir = test_dir / scene_id
        print(f"\n处理场景 {scene_id}...")

        data = load_scene_data(str(scene_dir), max_frames=30)
        if len(data["rgb"]) < 10:
            print("  帧数太少，跳过")
            continue

        print(f"  帧数: {len(data['rgb'])}, 物体: {data['obj_id']}")

        size_3d, diameter = get_object_model_size(str(data_root / "models"), data["obj_id"])
        first_pose = data["poses"][0]
        H, W = data["rgb"][0].shape[:2]
        object_mask = generate_object_mask_from_pose_simple(
            first_pose["R"], first_pose["t"], size_3d, data["camera_k"], (H, W)
        )

        mask_pixels = np.sum(object_mask)
        print(f"  mask 像素数: {mask_pixels}")
        if mask_pixels < 100:
            print("  mask 太小，跳过")
            continue

        gt_velocity = compute_gt_velocity_from_poses(data["poses"])

        result = estimator.estimate_velocity_sequence(
            data["rgb"], data["depth"], object_mask, data["camera_k"]
        )

        pred_velocity = result["velocity"]

        valid = ~np.any(np.isnan(pred_velocity), axis=1) & ~np.any(np.isnan(gt_velocity), axis=1)
        valid[0] = False  # 第一帧跳过

        if np.sum(valid) > 0:
            error = np.linalg.norm(pred_velocity[valid] - gt_velocity[valid], axis=1)
            all_errors.extend(error.tolist())

            rmse = np.sqrt(np.mean(error**2))
            mae = np.mean(error)
            gt_speed = np.linalg.norm(gt_velocity[valid], axis=1)
            print(f"  速度 RMSE: {rmse*1000:.2f} mm/s")
            print(f"  速度 MAE: {mae*1000:.2f} mm/s")
            print(f"  GT 速度范围: {np.min(gt_speed)*1000:.2f} - {np.max(gt_speed)*1000:.2f} mm/s")

    if all_errors:
        all_errors = np.array(all_errors)
        result = {
            "method": "raft_deep_optical_flow",
            "num_samples": len(all_errors),
            "velocity_rmse_m_s": float(np.sqrt(np.mean(all_errors**2))),
            "velocity_mae_m_s": float(np.mean(all_errors)),
            "velocity_median_m_s": float(np.median(all_errors)),
        }
        print(f"\n=== RAFT 深度学习光流评估结果 ===")
        print(f"  样本数: {len(all_errors)}")
        print(f"  RMSE: {result['velocity_rmse_m_s']*1000:.2f} mm/s")
        print(f"  MAE: {result['velocity_mae_m_s']*1000:.2f} mm/s")
        print(f"  Median: {result['velocity_median_m_s']*1000:.2f} mm/s")

        output_dir = Path("outputs/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "raft_velocity_result.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到 {output_dir / 'raft_velocity_result.json'}")
    else:
        print("没有有效结果")


if __name__ == "__main__":
    main()
