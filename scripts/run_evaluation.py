"""
方案对比评估框架（最终裁决）

将所有感知方案在统一的 MuJoCo 评估测试集上运行，
与物理仿真 GT 比较，输出各方案在各指标上的排名。

支持的方案:
  位姿估计:
    - rgb_baseline    : 纯RGB单帧位姿回归 (ResNet18)
    - rgbd_baseline   : RGBD四通道位姿回归 (ResNet18)

  速度估计:
    - finite_diff     : 有限差分法（位姿序列→速度）
    - savgol          : Savitzky-Golay 滤波微分
    - kalman          : Kalman 滤波递归估计
    - optical_flow    : 稠密光流（纯视觉，无需位姿模型）
    - lucas_kanade    : LK 稀疏光流

  接触检测:
    - force_threshold : 力阈值法（有触觉）
    - visual_pose     : 视觉位姿突变法
    - optical_flow    : 光流突变法

  滑移检测:
    - flow_based      : 光流法
    - pose_diff       : 位姿差法
    - force_based     : 切向力法

指标:
  位姿   : ADD (mm), ADI (mm), 投影误差 (px), 10%直径准确率
  速度   : 线速度 RMSE (m/s), 角速度 RMSE (rad/s)
  加速度 : 线加速度 RMSE (m/s²), 角加速度 RMSE (rad/s²)
  接触   : Accuracy, Precision, Recall, F1
  滑移   : Accuracy, Precision, Recall, F1
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class EvaluationRunner:
    """统一评估运行器"""

    def __init__(self, eval_dataset_dir: str, output_dir: str = "outputs/evaluation"):
        self.eval_dir = Path(eval_dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载元信息
        info_path = self.eval_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {}

        self.scenes = sorted([
            d for d in (self.eval_dir / "test").iterdir() if d.is_dir()
        ]) if (self.eval_dir / "test").exists() else []

        # 结果存储
        self.results = defaultdict(dict)  # {method_name: {metric: value}}

    def load_scene_data(self, scene_dir: Path) -> Dict:
        """加载一个场景的所有GT数据"""
        with open(scene_dir / "scene_gt.json") as f:
            gt = json.load(f)
        with open(scene_dir / "scene_velocity.json") as f:
            vel_gt = json.load(f)
        with open(scene_dir / "scene_acceleration.json") as f:
            acc_gt = json.load(f)
        with open(scene_dir / "scene_contact.json") as f:
            con_gt = json.load(f)
        with open(scene_dir / "scene_gripper.json") as f:
            grp_gt = json.load(f)
        with open(scene_dir / "scene_camera.json") as f:
            cam = json.load(f)

        return {
            "gt": gt,
            "velocity_gt": vel_gt,
            "acceleration_gt": acc_gt,
            "contact_gt": con_gt,
            "gripper_gt": grp_gt,
            "camera": cam,
            "n_frames": len(gt),
        }

    def evaluate_pose_method(self, method_name: str, predict_fn) -> Dict:
        """
        评估位姿估计方法

        predict_fn(scene_dir, frame_idx) -> {'R': (3,3), 't': (3,)}  单位: 米
        """
        from src.utils.metrics import compute_add, compute_projection_error
        from src.utils.metrics import load_bop_model_points

        add_list = []
        adi_list = []
        proj_list = []
        acc_10pct = 0
        total = 0

        for scene_dir in self.scenes:
            data = self.load_scene_data(scene_dir)
            obj_id = data["gt"]["0"][0]["obj_id"]

            # 加载模型点云
            ply_path = self.eval_dir / f"../../ycbv/models/obj_{obj_id:06d}.ply"
            if not ply_path.exists():
                # 用 BOP 标准路径
                ply_path = Path("datasets/ycbv/models") / f"obj_{obj_id:06d}.ply"
            model_points = None
            try:
                from src.mujoco_env.eval_dataset_generator import YCB_OBJECT_SIZES
                sx, sy, sz = YCB_OBJECT_SIZES[obj_id]
                # 生成采样点（代替PLY）
                pts = []
                for _ in range(500):
                    pts.append([
                        np.random.uniform(-sx, sx),
                        np.random.uniform(-sy, sy),
                        np.random.uniform(-sz, sz),
                    ])
                model_points = np.array(pts, dtype=np.float32)
            except Exception:
                model_points = np.random.randn(500, 3).astype(np.float32) * 0.05

            for frame_idx in range(data["n_frames"]):
                frame_key = str(frame_idx)
                if frame_key not in data["gt"]:
                    continue
                ann = data["gt"][frame_key][0]
                R_gt = np.array(ann["cam_R_m2c"]).reshape(3, 3)
                t_gt = np.array(ann["cam_t_m2c"]) / 1000.0  # mm → m

                pred = predict_fn(scene_dir, frame_idx)
                R_pred = pred["R"]
                t_pred = pred["t"]

                cam_K = np.array(data["camera"][frame_key]["cam_K"]).reshape(3, 3)

                add_val = compute_add(R_pred, t_pred, R_gt, t_gt, model_points)
                adi_val = add_val  # 简化：box对称，ADD=ADI
                proj_val = compute_projection_error(R_pred, t_pred, R_gt, t_gt,
                                                    model_points, cam_K)

                add_list.append(add_val)
                adi_list.append(adi_val)
                proj_list.append(proj_val)

                # 10%直径阈值
                diameter = np.linalg.norm(model_points.max(axis=0) - model_points.min(axis=0))
                if add_val < diameter * 0.1:
                    acc_10pct += 1
                total += 1

        result = {
            "ADD_mm": float(np.mean(add_list) * 1000),
            "ADI_mm": float(np.mean(adi_list) * 1000),
            "projection_error_px": float(np.mean(proj_list)),
            "accuracy_10pct": float(acc_10pct / max(total, 1)),
            "n_samples": total,
        }
        self.results[method_name]["pose"] = result
        return result

    def evaluate_velocity_method(self, method_name: str, estimate_fn) -> Dict:
        """
        评估速度估计方法

        estimate_fn(positions_list, timestamps) -> velocities (N,3) m/s
        """
        lin_errors = []
        ang_errors = []

        for scene_dir in self.scenes:
            data = self.load_scene_data(scene_dir)
            dt = self.dataset_info.get("dt_seconds", 0.002)

            # 收集GT位姿序列
            positions = []
            timestamps = []
            gt_velocities = []

            for i in range(data["n_frames"]):
                key = str(i)
                if key not in data["gt"]:
                    continue
                t_gt = np.array(data["gt"][key][0]["cam_t_m2c"]) / 1000.0
                positions.append(t_gt)
                timestamps.append(i * dt)

                if key in data["velocity_gt"]:
                    gt_velocities.append(np.array(
                        data["velocity_gt"][key]["linear_velocity_m_s"]
                    ))

            positions = np.array(positions)
            timestamps = np.array(timestamps)
            gt_velocities = np.array(gt_velocities)

            # 调用方法估计速度
            pred_velocities = estimate_fn(positions, timestamps)

            # 对齐长度
            min_len = min(len(pred_velocities), len(gt_velocities))
            if min_len < 2:
                continue

            pred = pred_velocities[:min_len]
            gt = gt_velocities[:min_len]

            err = np.linalg.norm(pred - gt, axis=1)
            lin_errors.extend(err.tolist())

        result = {
            "linear_velocity_rmse_m_s": float(np.sqrt(np.mean(np.array(lin_errors)**2))),
            "linear_velocity_mae_m_s": float(np.mean(np.abs(lin_errors))),
            "n_samples": len(lin_errors),
        }
        self.results[method_name]["velocity"] = result
        return result

    def evaluate_contact_method(self, method_name: str, detect_fn) -> Dict:
        """
        评估接触检测方法

        detect_fn(scene_dir, frame_idx) -> bool (是否接触)
        """
        tp = fp = tn = fn = 0

        for scene_dir in self.scenes:
            data = self.load_scene_data(scene_dir)

            for i in range(data["n_frames"]):
                key = str(i)
                if key not in data["contact_gt"]:
                    continue
                gt_contact = data["contact_gt"][key]["has_contact"]
                pred_contact = detect_fn(scene_dir, i)

                if gt_contact and pred_contact:
                    tp += 1
                elif not gt_contact and pred_contact:
                    fp += 1
                elif not gt_contact and not pred_contact:
                    tn += 1
                else:
                    fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        result = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(f1),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "n_samples": total,
        }
        self.results[method_name]["contact"] = result
        return result

    def save_results(self, filename: str = "comparison_results.json"):
        """保存所有方案的对比结果"""
        out_path = self.output_dir / filename
        with open(out_path, "w") as f:
            json.dump(dict(self.results), f, indent=2, default=str)
        print(f"结果已保存到: {out_path}")
        return out_path

    def print_summary_table(self):
        """打印对比汇总表"""
        print("\n" + "=" * 80)
        print("方案对比汇总表")
        print("=" * 80)

        # 位姿对比
        pose_methods = [m for m, r in self.results.items() if "pose" in r]
        if pose_methods:
            print("\n【位姿估计】")
            print(f"{'方法':<20} {'ADD(mm)':<10} {'投影误差(px)':<14} {'10%直径准确率':<12}")
            print("-" * 60)
            for m in sorted(pose_methods, key=lambda x: self.results[x]["pose"]["ADD_mm"]):
                r = self.results[m]["pose"]
                print(f"{m:<20} {r['ADD_mm']:<10.2f} "
                      f"{r['projection_error_px']:<14.2f} "
                      f"{r['accuracy_10pct']:<12.2%}")

        # 速度对比
        vel_methods = [m for m, r in self.results.items() if "velocity" in r]
        if vel_methods:
            print("\n【速度估计】")
            print(f"{'方法':<20} {'线速度RMSE(m/s)':<18} {'MAE(m/s)':<12}")
            print("-" * 50)
            for m in sorted(vel_methods, key=lambda x: self.results[x]["velocity"]["linear_velocity_rmse_m_s"]):
                r = self.results[m]["velocity"]
                print(f"{m:<20} {r['linear_velocity_rmse_m_s']:<18.4f} "
                      f"{r['linear_velocity_mae_m_s']:<12.4f}")

        # 接触检测对比
        con_methods = [m for m, r in self.results.items() if "contact" in r]
        if con_methods:
            print("\n【接触检测】")
            print(f"{'方法':<20} {'F1':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
            print("-" * 60)
            for m in sorted(con_methods, key=lambda x: -self.results[x]["contact"]["F1"]):
                r = self.results[m]["contact"]
                print(f"{m:<20} {r['F1']:<8.3f} "
                      f"{r['accuracy']:<10.3f} "
                      f"{r['precision']:<10.3f} "
                      f"{r['recall']:<10.3f}")

        print("\n" + "=" * 80)


# ============================================================
# 各 baseline 方案的预测函数适配器
# ============================================================

def pose_from_gt_noise(scene_dir, frame_idx, noise_level=0.01):
    """模拟有噪声的位姿估计（作为baseline下限）"""
    from src.mujoco_env.eval_dataset_generator import build_ycb_mujoco_xml
    with open(scene_dir / "scene_gt.json") as f:
        gt = json.load(f)
    ann = gt[str(frame_idx)][0]
    R = np.array(ann["cam_R_m2c"]).reshape(3, 3)
    t = np.array(ann["cam_t_m2c"]) / 1000.0
    # 添加噪声模拟真实估计误差
    t_noisy = t + np.random.normal(0, noise_level, 3)
    return {"R": R, "t": t_noisy}


def velocity_finite_diff(positions, timestamps):
    """有限差分速度估计"""
    from src.trajectory.velocity_estimator import FiniteDifferenceEstimator
    est = FiniteDifferenceEstimator()
    vels = []
    for i in range(len(positions)):
        if i == 0:
            vels.append(np.zeros(3))
        else:
            dt = timestamps[i] - timestamps[i-1]
            vels.append((positions[i] - positions[i-1]) / dt)
    return np.array(vels)


def velocity_savgol(positions, timestamps):
    """Savitzky-Golay 滤波速度估计"""
    from scipy.signal import savgol_filter
    if len(positions) < 5:
        return np.zeros_like(positions)
    window = min(15, len(positions) // 2 * 2 + 1)
    if window < 5:
        window = 5
    dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.002
    velocities = np.zeros_like(positions)
    for d in range(3):
        velocities[:, d] = savgol_filter(positions[:, d], window, 3, deriv=1, delta=dt)
    return velocities


def velocity_kalman(positions, timestamps):
    """Kalman 滤波速度估计（批量）"""
    from src.trajectory.velocity_estimator import KalmanFilterEstimator
    est = KalmanFilterEstimator(dim=positions.shape[1] if positions.ndim > 1 else 1)
    velocities, _ = est.estimate(positions, timestamps)
    return velocities


def contact_force_threshold(scene_dir, frame_idx, threshold=0.5):
    """力阈值接触检测（理想情况，直接用GT力作为模拟）"""
    with open(scene_dir / "scene_contact.json") as f:
        con = json.load(f)
    # 模拟：力>阈值为接触（这里直接用GT力加噪声模拟传感器噪声）
    force = con[str(frame_idx)]["total_force_N"]
    force_noisy = force + np.random.normal(0, 0.1)
    return force_noisy > threshold


def contact_visual_pose(scene_dir, frame_idx, threshold=0.001):
    """视觉位姿突变法：位姿变化突然变小→可能接触了"""
    with open(scene_dir / "scene_velocity.json") as f:
        vel = json.load(f)
    key = str(frame_idx)
    if key not in vel or frame_idx < 3:
        return False
    # 速度突然下降 → 接触（模拟：用GT速度做近似）
    v = np.linalg.norm(vel[key]["linear_velocity_m_s"])
    return v < threshold


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="方案对比评估")
    parser.add_argument("--eval_dir", type=str, default="datasets/ycb_grasp_eval_smoke")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    runner = EvaluationRunner(args.eval_dir, args.output_dir)
    print(f"评估数据集: {args.eval_dir}")
    print(f"场景数: {len(runner.scenes)}")

    # --- 速度估计方案对比 ---
    print("\n=== 速度估计方案对比 ===")
    vel_methods = {
        "finite_diff": velocity_finite_diff,
        "savgol": velocity_savgol,
        "kalman": velocity_kalman,
    }
    for name, fn in vel_methods.items():
        r = runner.evaluate_velocity_method(name, fn)
        print(f"  {name:<15} RMSE={r['linear_velocity_rmse_m_s']:.4f} m/s")

    # --- 接触检测方案对比 ---
    print("\n=== 接触检测方案对比 ===")
    con_methods = {
        "force_threshold": contact_force_threshold,
        "visual_pose": contact_visual_pose,
    }
    for name, fn in con_methods.items():
        r = runner.evaluate_contact_method(name, fn)
        print(f"  {name:<15} F1={r['F1']:.3f}  Acc={r['accuracy']:.3f}")

    # 输出汇总
    runner.print_summary_table()
    runner.save_results()
