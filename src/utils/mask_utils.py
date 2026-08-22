"""生成物体 mask 的工具函数"""

import numpy as np
from typing import Tuple
from pathlib import Path
import json


def get_object_model_size(models_dir: str, obj_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """获取物体模型的尺寸和直径
    
    Args:
        models_dir: 模型目录
        obj_id: 物体 ID
        
    Returns:
        size: (3,) 物体尺寸 (x, y, z) 米
        diameter: 物体直径 米
    """
    models_dir = Path(models_dir)
    
    # 先试试 models_info.json
    info_path = models_dir / "models_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        obj_key = str(obj_id)
        if obj_key in info:
            info_item = info[obj_key]
            if "size" in info_item:
                size = np.array(info_item["size"]) / 1000.0  # mm->m
            elif "diameter" in info_item:
                # 没有 size 时用直径近似立方体
                d = info_item["diameter"] / 1000.0
                size = np.array([d * 0.7, d * 0.7, d * 0.7])
            else:
                size = np.array([0.1, 0.1, 0.1])
            diameter = info_item.get("diameter", 0.15) / 1000.0
            return size, diameter
    
    # 从 ply 读顶点
    ply_path = models_dir / f"obj_{obj_id:06d}.ply"
    if ply_path.exists():
        try:
            points = []
            with open(ply_path, "r") as f:
                reading = False
                for line in f:
                    line = line.strip()
                    if line.startswith("element vertex"):
                        n_verts = int(line.split()[-1])
                    elif line == "end_header":
                        reading = True
                        continue
                    elif reading and len(points) < n_verts:
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            except:
                                pass
            
            if len(points) > 10:
                pts = np.array(points) / 1000.0  # mm -> m
                mins = np.min(pts, axis=0)
                maxs = np.max(pts, axis=0)
                size = maxs - mins
                diameter = np.max(size)
                return size, diameter
        except:
            pass
    
    # 默认值
    return np.array([0.1, 0.1, 0.1]), 0.15


def generate_object_mask_from_pose_simple(
    pose_R: np.ndarray,
    pose_t: np.ndarray,
    size_3d: np.ndarray,
    K: np.ndarray,
    img_shape: Tuple[int, int],
) -> np.ndarray:
    """从位姿和物体尺寸生成椭圆 mask（近似）
    
    Args:
        pose_R: 旋转矩阵 (3, 3)
        pose_t: 平移向量 (3,) 米
        size_3d: 物体 3D 尺寸 (3,) 米
        K: 相机内参 (3, 3)
        img_shape: (H, W)
        
    Returns:
        mask: (H, W) bool
    """
    H, W = img_shape
    
    # 物体中心投影
    center_3d = pose_t  # 相机坐标系下的物体中心
    if center_3d[2] <= 0:
        return np.zeros((H, W), dtype=bool)
    
    # 投影到像素
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    cx_pix = cx + center_3d[0] * fx / center_3d[2]
    cy_pix = cy + center_3d[1] * fy / center_3d[2]
    
    # 近似像素尺寸（用最大的两个维度）
    z = center_3d[2]
    half_w_pix = (size_3d[0] / 2) * fx / z
    half_h_pix = (size_3d[1] / 2) * fy / z
    
    # 生成椭圆 mask
    y_coords, x_coords = np.ogrid[:H, :W]
    mask = ((x_coords - cx_pix) / max(half_w_pix, 1))**2 + \
           ((y_coords - cy_pix) / max(half_h_pix, 1))**2 <= 1.0
    
    return mask.astype(bool)


def generate_gripper_mask_simple(
    K: np.ndarray,
    img_shape: Tuple[int, int],
    gripper_width_m: float = 0.08,
    gripper_height_m: float = 0.02,
    depth_m: float = 0.5,
) -> np.ndarray:
    """生成简单的夹爪 mask（图像底部中间区域）
    
    用于无标注时的近似
    """
    H, W = img_shape
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    half_w_pix = (gripper_width_m / 2) * fx / depth_m
    half_h_pix = (gripper_height_m / 2) * fy / depth_m
    
    # 夹爪在图像底部
    gripper_y = H - int(half_h_pix * 2)
    gripper_x = cx
    
    y_coords, x_coords = np.ogrid[:H, :W]
    mask = ((x_coords - gripper_x) / max(half_w_pix, 1))**2 + \
           ((y_coords - gripper_y) / max(half_h_pix, 1))**2 <= 1.0
    
    return mask.astype(bool)
