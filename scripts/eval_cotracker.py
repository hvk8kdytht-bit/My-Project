"""
CoTracker 在 YCB-Video 真实数据集上的评估
用真实视频 + GT 位姿验证 CoTracker 速度估计精度
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# 项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_scene_data(scene_dir: str, max_frames: int = 100) -> Dict:
    """加载一个场景的数据
    
    Args:
        scene_dir: 场景目录路径
        max_frames: 最大帧数
        
    Returns:
        包含 rgb, depth, poses, camera 的字典
    """
    import cv2
    
    scene_dir = Path(scene_dir)
    
    # 加载 GT 位姿
    with open(scene_dir / "scene_gt.json") as f:
        scene_gt = json.load(f)
    
    # 加载相机参数
    with open(scene_dir / "scene_camera.json") as f:
        scene_cam = json.load(f)
    
    # 获取第一帧的物体列表
    first_key = sorted(scene_gt.keys())[0]
    obj_ids = [item["obj_id"] for item in scene_gt[first_key]]
    
    # 只取第一个物体，限制帧数
    obj_id = obj_ids[0]
    frame_keys = sorted(scene_gt.keys())[:max_frames]
    
    rgb_frames = []
    depth_frames = []
    poses = []
    camera_ks = []
    
    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"
    
    for key in frame_keys:
        # 找到对应 obj_id 的位姿
        pose_item = None
        for item in scene_gt[key]:
            if item["obj_id"] == obj_id:
                pose_item = item
                break
        if pose_item is None:
            continue
        
        # RGB
        rgb_path = rgb_dir / f"{int(key):06d}.png"
        if not rgb_path.exists():
            continue
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Depth
        depth_path = depth_dir / f"{int(key):06d}.png"
        if not depth_path.exists():
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth_m = depth.astype(np.float32) / 1000.0  # mm -> m
        
        # 位姿
        R = np.array(pose_item["cam_R_m2c"]).reshape(3, 3)
        t = np.array(pose_item["cam_t_m2c"]) / 1000.0  # mm -> m
        
        # 相机内参
        cam_info = scene_cam[key]
        K = np.array(cam_info["cam_K"]).reshape(3, 3)
        
        rgb_frames.append(rgb)
        depth_frames.append(depth_m)
        poses.append({"R": R, "t": t})
        camera_ks.append(K)
    
    return {
        "rgb": rgb_frames,
        "depth": depth_frames,
        "poses": poses,
        "camera_k": camera_ks[0] if camera_ks else np.eye(3),
        "obj_id": obj_id,
        "scene_id": scene_dir.name,
    }


def compute_gt_velocity_from_poses(poses: List[Dict], fps: float = 30.0) -> np.ndarray:
    """从 GT 位姿序列计算物体中心速度
    
    Args:
        poses: 位姿列表，每项含 R 和 t
        fps: 帧率
        
    Returns:
        velocity: (T, 3) 速度 m/s
    """
    T = len(poses)
    dt = 1.0 / fps
    velocity = np.zeros((T, 3), dtype=np.float32)
    
    # 物体中心在相机坐标系中的位置就是 t
    translations = np.array([p["t"] for p in poses])  # (T, 3)
    
    # 中心差分
    for t in range(1, T - 1):
        velocity[t] = (translations[t+1] - translations[t-1]) / (2 * dt)
    
    if T > 1:
        velocity[0] = (translations[1] - translations[0]) / dt
        velocity[-1] = (translations[-1] - translations[-2]) / dt
    
    return velocity


def generate_object_mask_from_pose(
    pose: Dict,
    model_points: np.ndarray,
    K: np.ndarray,
    img_shape: Tuple[int, int],
) -> np.ndarray:
    """根据位姿和物体模型点投影生成 mask
    
    Args:
        pose: 位姿字典 (R, t)
        model_points: 物体 3D 模型点 (N, 3) 米
        K: 相机内参 (3, 3)
        img_shape: (H, W)
        
    Returns:
        mask: (H, W) bool
    """
    H, W = img_shape
    R = pose["R"]
    t = pose["t"]
    
    # 投影 3D 点到 2D
    points_cam = (R @ model_points.T).T + t  # (N, 3)
    points_2d = (K @ points_cam.T).T  # (N, 3)
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # (N, 2)
    
    # 生成 mask（用凸包近似）
    mask = np.zeros((H, W), dtype=bool)
    valid = points_cam[:, 2] > 0.01  # 深度为正
    pts = points_2d[valid].astype(np.int32)
    
    if len(pts) > 3:
        import cv2
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, True)
    
    return mask


def load_object_model(models_dir: str, obj_id: int) -> np.ndarray:
    """加载物体模型点（从 ply 文件采样）"""
    import cv2
    
    ply_path = Path(models_dir) / f"obj_{obj_id:06d}.ply"
    if not ply_path.exists():
        return np.array([[0, 0, 0]])
    
    # 简单读取 ply 顶点
    try:
        points = []
        with open(ply_path, "r") as f:
            reading_vertices = False
            vertex_count = 0
            for line in f:
                line = line.strip()
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line == "end_header":
                    reading_vertices = True
                    continue
                elif reading_vertex and len(points) < vertex_count:
                    parts = line.split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        if len(points) > 500:
            # 降采样到 500 个点
            indices = np.linspace(0, len(points)-1, 500, dtype=int)
            points = [points[i] for i in indices]
        
        return np.array(points) / 1000.0  # mm -> m
    except:
        return np.array([[0, 0, 0]])


def evaluate_cotracker_on_ycbv(
    data_root: str,
    models_dir: str,
    scene_ids: List[str] = None,
    max_frames_per_scene: int = 100,
    device: str = "cpu",
) -> Dict:
    """在 YCB-Video 上评估 CoTracker 速度估计
    
    Args:
        data_root: YCB-Video 数据根目录
        models_dir: 物体模型目录
        scene_ids: 要评估的场景 ID 列表
        max_frames_per_scene: 每个场景最大帧数
        device: 计算设备
        
    Returns:
        评估结果字典
    """
    data_root = Path(data_root) / "test"
    
    if scene_ids is None:
        scene_ids = sorted([d.name for d in data_root.iterdir() if d.is_dir()])[:3]
    
    print(f"评估 {len(scene_ids)} 个场景，每场景最多 {max_frames_per_scene} 帧")
    print(f"设备: {device}")
    
    all_errors = []
    
    for scene_id in scene_ids:
        scene_dir = data_root / scene_id
        if not scene_dir.exists():
            print(f"  跳过 {scene_id}（不存在）")
            continue
        
        print(f"\n处理场景 {scene_id}...")
        
        try:
            # 加载场景数据
            data = load_scene_data(str(scene_dir), max_frames=max_frames_per_scene)
            
            if len(data["rgb"]) < 10:
                print(f"  帧数太少（{len(data['rgb'])}），跳过")
                continue
            
            print(f"  帧数: {len(data['rgb'])}, 物体: {data['obj_id']}")
            
            # 加载物体模型
            model_points = load_object_model(models_dir, data["obj_id"])
            
            # 用 GT 位姿生成第一帧 mask
            first_pose = data["poses"][0]
            H, W = data["rgb"][0].shape[:2]
            object_mask = generate_object_mask_from_pose(
                first_pose, model_points, data["camera_k"], (H, W)
            )
            
            if not np.any(object_mask):
                print(f"  mask 为空，跳过")
                continue
            
            # GT 速度
            gt_velocity = compute_gt_velocity_from_poses(data["poses"])
            
            # CoTracker 估计速度
            try:
                from src.velocity.cotracker_estimator import CoTrackerVelocityEstimator
                
                estimator = CoTrackerVelocityEstimator(
                    device=device,
                    num_points=50,
                    fps=30.0,
                )
                estimator.load_model()
                
                result = estimator.estimate_velocity_sequence(
                    data["rgb"],
                    data["depth"],
                    object_mask,
                    data["camera_k"],
                )
                
                pred_velocity = result["velocity"]
                
                # 计算误差
                valid = ~np.any(np.isnan(pred_velocity), axis=1)
                if np.sum(valid) > 0:
                    error = np.linalg.norm(
                        pred_velocity[valid] - gt_velocity[valid], axis=1
                    )
                    all_errors.extend(error.tolist())
                    
                    rmse = np.sqrt(np.mean(error**2))
                    mae = np.mean(error)
                    print(f"  速度 RMSE: {rmse*1000:.2f} mm/s")
                    print(f"  速度 MAE: {mae*1000:.2f} mm/s")
                else:
                    print(f"  无有效预测")
                
            except Exception as e:
                print(f"  CoTracker 评估失败: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"  加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    if all_errors:
        all_errors = np.array(all_errors)
        result = {
            "method": "cotracker",
            "num_samples": len(all_errors),
            "velocity_rmse_m_s": float(np.sqrt(np.mean(all_errors**2))),
            "velocity_mae_m_s": float(np.mean(all_errors)),
            "velocity_median_m_s": float(np.median(all_errors)),
        }
    else:
        result = {
            "method": "cotracker",
            "num_samples": 0,
            "error": "no valid results",
        }
    
    print(f"\n{'='*50}")
    print(f"CoTracker 速度估计评估结果:")
    print(f"  样本数: {result.get('num_samples', 0)}")
    if "velocity_rmse_m_s" in result:
        print(f"  RMSE: {result['velocity_rmse_m_s']*1000:.2f} mm/s")
        print(f"  MAE: {result['velocity_mae_m_s']*1000:.2f} mm/s")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/ycbv", help="YCB-Video 数据根目录")
    parser.add_argument("--models_dir", default="datasets/ycbv/models", help="物体模型目录")
    parser.add_argument("--max_frames", type=int, default=50, help="每场景最大帧数")
    parser.add_argument("--num_scenes", type=int, default=3, help="评估场景数")
    parser.add_argument("--device", default="cpu", help="计算设备")
    args = parser.parse_args()
    
    # 找测试场景
    test_dir = Path(args.data_root) / "test"
    all_scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    # 取测试集场景
    from src.data.ycb_video import get_scene_split
    try:
        split = get_scene_split(args.data_root)
        scene_ids = split.get("test", all_scenes[:args.num_scenes])
    except:
        scene_ids = all_scenes[:args.num_scenes]
    
    scene_ids = scene_ids[:args.num_scenes]
    
    result = evaluate_cotracker_on_ycbv(
        data_root=args.data_root,
        models_dir=args.models_dir,
        scene_ids=scene_ids,
        max_frames_per_scene=args.max_frames,
        device=args.device,
    )
    
    # 保存结果
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "cotracker_velocity_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 {output_dir / 'cotracker_velocity_result.json'}")
