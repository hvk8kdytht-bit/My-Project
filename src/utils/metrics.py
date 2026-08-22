"""
6D 位姿评估指标
- ADD (Average Distance)：平均点距离
- ADI (Average Distance Index)：最邻近点平均距离
- 投影误差：2D 重投影误差
- 准确率：ADD < 0.1 * diameter 比例
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_add(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    model_points: np.ndarray,
) -> float:
    """
    计算 ADD (Average Distance of model points)

    ADD = (1/n) * sum(||(R_pred * X + t_pred) - (R_gt * X + t_gt)||)

    Args:
        R_pred: 预测旋转矩阵 (3, 3)
        t_pred: 预测平移向量 (3,)
        R_gt: GT旋转矩阵 (3, 3)
        t_gt: GT平移向量 (3,)
        model_points: 物体模型3D点 (N, 3)

    Returns:
        ADD 值（米）
    """
    pred_points = (R_pred @ model_points.T).T + t_pred
    gt_points = (R_gt @ model_points.T).T + t_gt

    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return float(np.mean(distances))


def compute_adi(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    model_points: np.ndarray,
) -> float:
    """
    计算 ADI (Average Distance Index) - 用于对称物体

    对每个预测点，找最近的GT点（而不是一一对应）

    Args:
        R_pred: 预测旋转矩阵 (3, 3)
        t_pred: 预测平移向量 (3,)
        R_gt: GT旋转矩阵 (3, 3)
        t_gt: GT平移向量 (3,)
        model_points: 物体模型3D点 (N, 3)

    Returns:
        ADI 值（米）
    """
    from scipy.spatial.distance import cdist

    pred_points = (R_pred @ model_points.T).T + t_pred
    gt_points = (R_gt @ model_points.T).T + t_gt

    # 计算所有点对的距离 (N_pred, N_gt)
    distances = cdist(pred_points, gt_points)
    # 每个预测点找最近的GT点
    min_distances = np.min(distances, axis=1)
    return float(np.mean(min_distances))


def compute_projection_error(
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    model_points: np.ndarray,
    K: np.ndarray,
) -> float:
    """
    计算 2D 重投影误差（像素）

    Args:
        R_pred: 预测旋转矩阵 (3, 3)
        t_pred: 预测平移向量 (3,)
        R_gt: GT旋转矩阵 (3, 3)
        t_gt: GT平移向量 (3,)
        model_points: 物体模型3D点 (N, 3)
        K: 相机内参 (3, 3)

    Returns:
        平均2D投影误差（像素）
    """
    # 投影预测点
    pred_2d = project_points(model_points, R_pred, t_pred, K)
    # 投影GT点
    gt_2d = project_points(model_points, R_gt, t_gt, K)

    errors = np.linalg.norm(pred_2d - gt_2d, axis=1)
    return float(np.mean(errors))


def project_points(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """将3D点投影到2D图像平面"""
    # 变换到相机坐标系
    points_cam = (R @ points_3d.T).T + t  # (N, 3)
    # 投影
    points_2d_h = (K @ points_cam.T).T  # (N, 3)
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]  # (N, 2)
    return points_2d


def evaluate_pose(
    predictions: List[Dict],
    model_diameters: Dict[int, float],
    model_points: Dict[int, np.ndarray],
    K: np.ndarray,
    threshold: float = 0.1,
    symmetric_objects: List[int] = None,
) -> Dict:
    """
    评估位姿估计结果

    Args:
        predictions: 预测列表，每个包含 obj_id, R_pred, t_pred, R_gt, t_gt
        model_diameters: 每个物体的直径 {obj_id: diameter_in_meters}
        model_points: 每个物体的3D模型点 {obj_id: (N,3)}
        K: 相机内参 (3, 3)
        threshold: ADD/ADI 准确率阈值（直径的比例）
        symmetric_objects: 对称物体ID列表（使用ADI而不是ADD）

    Returns:
        评估指标字典
    """
    if symmetric_objects is None:
        symmetric_objects = []

    add_values = []
    adi_values = []
    proj_errors = []
    per_object_results = {}

    for pred in predictions:
        obj_id = pred["obj_id"]
        R_pred = pred["R_pred"]
        t_pred = pred["t_pred"]
        R_gt = pred["R_gt"]
        t_gt = pred["t_gt"]

        points = model_points.get(obj_id)
        if points is None:
            continue

        # 计算 ADD
        add = compute_add(R_pred, t_pred, R_gt, t_gt, points)
        add_values.append(add)

        # 计算 ADI
        adi = compute_adi(R_pred, t_pred, R_gt, t_gt, points)
        adi_values.append(adi)

        # 计算投影误差
        proj_err = compute_projection_error(R_pred, t_pred, R_gt, t_gt, points, K)
        proj_errors.append(proj_err)

        # 逐物体统计
        if obj_id not in per_object_results:
            per_object_results[obj_id] = {"add": [], "adi": [], "proj_err": [], "count": 0}
        per_object_results[obj_id]["add"].append(add)
        per_object_results[obj_id]["adi"].append(adi)
        per_object_results[obj_id]["proj_err"].append(proj_err)
        per_object_results[obj_id]["count"] += 1

    # 计算整体准确率
    if not add_values:
        return {"accuracy_add": 0.0, "accuracy_adi": 0.0, "mean_add": 0.0,
                "mean_proj_err": 0.0, "num_samples": 0}

    add_values = np.array(add_values)
    adi_values = np.array(adi_values)
    proj_errors = np.array(proj_errors)

    # 计算每个样本是否准确
    add_accurate = np.zeros(len(predictions), dtype=bool)
    adi_accurate = np.zeros(len(predictions), dtype=bool)

    for i, pred in enumerate(predictions):
        obj_id = pred["obj_id"]
        diameter = model_diameters.get(obj_id, 0.1)  # 默认10cm
        add_threshold = threshold * diameter

        if obj_id in symmetric_objects:
            adi_accurate[i] = adi_values[i] < add_threshold
        else:
            add_accurate[i] = add_values[i] < add_threshold

    results = {
        "num_samples": len(predictions),
        "mean_add": float(np.mean(add_values)),
        "median_add": float(np.median(add_values)),
        "mean_adi": float(np.mean(adi_values)),
        "median_adi": float(np.median(adi_values)),
        "mean_proj_err_px": float(np.mean(proj_errors)),
        "median_proj_err_px": float(np.median(proj_errors)),
        "accuracy_add": float(np.mean(add_accurate)),
        "accuracy_adi": float(np.mean(adi_accurate)),
        "per_object": {},
    }

    # 逐物体统计
    for obj_id, obj_data in per_object_results.items():
        diameter = model_diameters.get(obj_id, 0.1)
        add_threshold = threshold * diameter

        obj_add = np.array(obj_data["add"])
        obj_adi = np.array(obj_data["adi"])

        results["per_object"][obj_id] = {
            "count": obj_data["count"],
            "mean_add": float(np.mean(obj_add)),
            "mean_adi": float(np.mean(obj_adi)),
            "mean_proj_err_px": float(np.mean(obj_data["proj_err"])),
            "accuracy_add": float(np.mean(obj_add < add_threshold)),
            "accuracy_adi": float(np.mean(obj_adi < add_threshold)),
            "diameter": diameter,
        }

    return results
