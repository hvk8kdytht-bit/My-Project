#!/usr/bin/env python
"""
全方案对比评估（完整版）
在 MuJoCo 评估测试集上运行所有 baseline 方案，
与物理仿真 GT 比较，输出各方案排名。
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FullEvaluation:
    def __init__(self, eval_dir: str, output_dir: str = "outputs/evaluation_full"):
        self.eval_dir = Path(eval_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        info_path = self.eval_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {"dt_seconds": 0.002}

        self.test_dir = self.eval_dir / "test"
        self.scene_dirs = sorted([d for d in self.test_dir.iterdir() if d.is_dir()])
        self.dt = self.dataset_info.get("dt_seconds", 0.002)
        self.results = defaultdict(dict)

        print(f"评估集: {eval_dir}")
        print(f"场景数: {len(self.scene_dirs)}, dt: {self.dt}s")

    def load_scene(self, scene_dir: Path) -> Dict:
        with open(scene_dir / "scene_gt.json") as f:
            gt = json.load(f)
        with open(scene_dir / "scene_velocity.json") as f:
            vel_gt = json.load(f)
        with open(scene_dir / "scene_contact.json") as f:
            con_gt = json.load(f)

        n = len(gt)
        positions = np.zeros((n, 3), dtype=np.float32)
        gt_velocities = np.zeros((n, 3), dtype=np.float32)
        gt_contacts = np.zeros(n, dtype=bool)
        gt_forces = np.zeros(n, dtype=np.float32)
        gt_slips = np.zeros(n, dtype=bool)

        for i in range(n):
            key = str(i)
            ann = gt[key][0]
            positions[i] = np.array(ann["cam_t_m2c"]) / 1000.0
            if key in vel_gt:
                gt_velocities[i] = np.array(vel_gt[key]["linear_velocity_m_s"])
            if key in con_gt:
                gt_contacts[i] = con_gt[key]["has_contact"]
                gt_forces[i] = con_gt[key]["total_force_N"]
                gt_slips[i] = con_gt[key]["is_slipping"]

        return {
            "n_frames": n,
            "positions": positions,
            "gt_velocities": gt_velocities,
            "gt_contacts": gt_contacts,
            "gt_forces": gt_forces,
            "gt_slips": gt_slips,
            "obj_id": gt["0"][0]["obj_id"],
        }

    # ---- 速度估计 ----
    def eval_velocity(self):
        print("\n" + "=" * 70)
        print("速度估计方案评估")
        print("=" * 70)

        methods = {
            "finite_diff": self._vel_finite_diff,
            "savgol": self._vel_savgol,
            "kalman": self._vel_kalman,
        }

        for name, method_fn in methods.items():
            all_errors = []
            for scene_dir in self.scene_dirs:
                data = self.load_scene(scene_dir)
                timestamps = np.arange(data["n_frames"]) * self.dt
                pred = method_fn(data["positions"], timestamps)
                min_len = min(len(pred), data["n_frames"])
                s, e = 5, min_len - 5
                if e <= s:
                    continue
                err = np.linalg.norm(pred[s:e] - data["gt_velocities"][s:e], axis=1)
                all_errors.extend(err.tolist())

            all_errors = np.array(all_errors)
            rmse = float(np.sqrt(np.mean(all_errors ** 2)))
            mae = float(np.mean(all_errors))
            self.results[name]["velocity"] = {
                "linear_velocity_rmse_m_s": rmse,
                "linear_velocity_mae_m_s": mae,
                "n_samples": len(all_errors),
            }
            print(f"  {name:<15} RMSE={rmse:.4f} m/s   MAE={mae:.4f} m/s   (n={len(all_errors)})")

        ranked = sorted(methods.keys(), key=lambda k: self.results[k]["velocity"]["linear_velocity_rmse_m_s"])
        print(f"\n  排名: {' → '.join(ranked)}")

    @staticmethod
    def _vel_finite_diff(positions, timestamps):
        n = len(positions)
        vels = np.zeros_like(positions)
        for i in range(1, n):
            dt = timestamps[i] - timestamps[i - 1]
            vels[i] = (positions[i] - positions[i - 1]) / dt
        vels[0] = vels[1]
        return vels

    @staticmethod
    def _vel_savgol(positions, timestamps):
        from scipy.signal import savgol_filter
        n = len(positions)
        if n < 7:
            return np.zeros_like(positions)
        window = min(31, n // 4 * 2 + 1)
        if window < 5:
            window = 5
        dt = np.mean(np.diff(timestamps))
        vels = np.zeros_like(positions)
        for d in range(3):
            vels[:, d] = savgol_filter(positions[:, d], window, 3, deriv=1, delta=dt)
        return vels

    @staticmethod
    def _vel_kalman(positions, timestamps):
        from src.trajectory.velocity_estimator import KalmanFilterEstimator
        dim = positions.shape[1] if positions.ndim > 1 else 1
        est = KalmanFilterEstimator(dim=dim)
        velocities, _ = est.estimate(positions, timestamps)
        return velocities

    # ---- 接触检测 ----
    def eval_contact(self):
        print("\n" + "=" * 70)
        print("接触检测方案评估")
        print("=" * 70)

        methods = {
            "force_threshold": self._con_force,
            "visual_pose": self._con_visual_pose,
            "velocity_drop": self._con_vel_drop,
        }

        for name, method_fn in methods.items():
            tp = fp = tn = fneg = 0
            for scene_dir in self.scene_dirs:
                data = self.load_scene(scene_dir)
                predictions = method_fn(data)
                gt = data["gt_contacts"]
                min_len = min(len(predictions), len(gt))
                for i in range(min_len):
                    p = bool(predictions[i])
                    g = bool(gt[i])
                    if g and p:
                        tp += 1
                    elif not g and p:
                        fp += 1
                    elif not g and not p:
                        tn += 1
                    else:
                        fneg += 1

            total = tp + fp + tn + fneg
            acc = (tp + tn) / max(total, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fneg, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)

            self.results[name]["contact"] = {
                "accuracy": float(acc), "precision": float(prec),
                "recall": float(rec), "F1": float(f1),
                "TP": tp, "FP": fp, "TN": tn, "FN": fneg,
            }
            print(f"  {name:<18} F1={f1:.3f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}")

        ranked = sorted(methods.keys(), key=lambda k: -self.results[k]["contact"]["F1"])
        print(f"\n  排名: {' → '.join(ranked)}")

    @staticmethod
    def _con_force(data, threshold=0.3):
        forces = data["gt_forces"] + np.random.normal(0, 0.05, len(data["gt_forces"]))
        return forces > threshold

    @staticmethod
    def _con_visual_pose(data, threshold=0.0005):
        positions = data["positions"]
        n = len(positions)
        velocities = np.zeros(n)
        for i in range(1, n):
            velocities[i] = np.linalg.norm(positions[i] - positions[i - 1])
        from scipy.ndimage import uniform_filter1d
        vel_smooth = uniform_filter1d(velocities, size=5)
        predictions = np.zeros(n, dtype=bool)
        for i in range(3, n):
            if vel_smooth[i] < threshold and vel_smooth[i - 2] > threshold * 3:
                predictions[i:] = True
                break
        return predictions

    @staticmethod
    def _con_vel_drop(data, drop_ratio=0.3):
        positions = data["positions"]
        n = len(positions)
        velocities = np.zeros(n)
        for i in range(1, n):
            velocities[i] = np.linalg.norm(positions[i] - positions[i - 1])
        from scipy.ndimage import uniform_filter1d
        vel_smooth = uniform_filter1d(velocities, size=5)
        predictions = np.zeros(n, dtype=bool)
        max_vel_idx = np.argmax(vel_smooth[:n // 2])
        if max_vel_idx > 0:
            peak_vel = vel_smooth[max_vel_idx]
            for i in range(max_vel_idx, n):
                if vel_smooth[i] < peak_vel * drop_ratio:
                    predictions[i:] = True
                    break
        return predictions

    # ---- 滑移检测 ----
    def eval_slip(self):
        print("\n" + "=" * 70)
        print("滑移检测方案评估")
        print("=" * 70)

        methods = {
            "force_variation": self._slip_force_var,
            "velocity_change": self._slip_vel_change,
            "pose_deviation": self._slip_pose_dev,
        }

        for name, method_fn in methods.items():
            tp = fp = tn = fneg = 0
            for scene_dir in self.scene_dirs:
                data = self.load_scene(scene_dir)
                predictions = method_fn(data)
                gt = data["gt_slips"]
                contact_mask = data["gt_contacts"]
                min_len = min(len(predictions), len(gt))
                for i in range(min_len):
                    if not contact_mask[i]:
                        continue
                    p = bool(predictions[i])
                    g = bool(gt[i])
                    if g and p:
                        tp += 1
                    elif not g and p:
                        fp += 1
                    elif not g and not p:
                        tn += 1
                    else:
                        fneg += 1

            total = tp + fp + tn + fneg
            acc = (tp + tn) / max(total, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fneg, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)

            self.results[name]["slip"] = {
                "accuracy": float(acc), "precision": float(prec),
                "recall": float(rec), "F1": float(f1),
                "TP": tp, "FP": fp, "TN": tn, "FN": fneg,
            }
            print(f"  {name:<18} F1={f1:.3f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  (n={total})")

        ranked = sorted(methods.keys(), key=lambda k: -self.results[k]["slip"]["F1"])
        print(f"\n  排名: {' → '.join(ranked)}")

    @staticmethod
    def _slip_force_var(data, threshold=0.05):
        forces = data["gt_forces"]
        n = len(forces)
        predictions = np.zeros(n, dtype=bool)
        for i in range(5, n):
            variation = np.std(forces[max(0, i-5):i+1])
            if data["gt_contacts"][i] and variation > threshold:
                predictions[i] = True
        return predictions

    @staticmethod
    def _slip_vel_change(data, threshold=0.005):
        positions = data["positions"]
        n = len(positions)
        velocities = np.zeros(n)
        for i in range(1, n):
            velocities[i] = np.linalg.norm(positions[i] - positions[i - 1])
        from scipy.ndimage import uniform_filter1d
        vel_smooth = uniform_filter1d(velocities, size=3)
        predictions = np.zeros(n, dtype=bool)
        for i in range(3, n):
            if data["gt_contacts"][i]:
                change = abs(vel_smooth[i] - vel_smooth[i-1])
                if change > threshold:
                    predictions[i] = True
        return predictions

    @staticmethod
    def _slip_pose_dev(data, threshold=0.001):
        positions = data["positions"]
        n = len(positions)
        predictions = np.zeros(n, dtype=bool)
        contact_start = -1
        contact_pos = np.zeros(3)
        for i in range(n):
            if data["gt_contacts"][i]:
                if contact_start < 0:
                    contact_start = i
                    contact_pos = positions[i].copy()
                else:
                    deviation = np.linalg.norm(positions[i] - contact_pos)
                    if deviation > threshold:
                        predictions[i] = True
        return predictions

    # ---- 汇总 ----
    def save(self):
        out = self.output_dir / "full_results.json"
        with open(out, "w") as f:
            json.dump(dict(self.results), f, indent=2)
        print(f"\n结果已保存: {out}")

    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 全方案对比汇总")
        print("=" * 70)

        vel = {k: v["velocity"] for k, v in self.results.items() if "velocity" in v}
        if vel:
            print("\n🏃 速度估计（线速度 RMSE，越低越好）")
            print(f"{'排名':<5}{'方法':<20}{'RMSE(m/s)':<14}{'MAE(m/s)':<14}")
            print("-" * 55)
            for rank, name in enumerate(sorted(vel.keys(), key=lambda k: vel[k]["linear_velocity_rmse_m_s"]), 1):
                r = vel[name]
                print(f"{rank:<5}{name:<20}{r['linear_velocity_rmse_m_s']:<14.4f}{r['linear_velocity_mae_m_s']:<14.4f}")

        con = {k: v["contact"] for k, v in self.results.items() if "contact" in v}
        if con:
            print("\n🤝 接触检测（F1，越高越好）")
            print(f"{'排名':<5}{'方法':<20}{'F1':<10}{'Accuracy':<12}{'Precision':<12}{'Recall':<10}")
            print("-" * 70)
            for rank, name in enumerate(sorted(con.keys(), key=lambda k: -con[k]["F1"]), 1):
                r = con[name]
                print(f"{rank:<5}{name:<20}{r['F1']:<10.3f}{r['accuracy']:<12.3f}{r['precision']:<12.3f}{r['recall']:<10.3f}")

        slip = {k: v["slip"] for k, v in self.results.items() if "slip" in v}
        if slip:
            print("\n🌀 滑移检测（F1，越高越好）")
            print(f"{'排名':<5}{'方法':<20}{'F1':<10}{'Accuracy':<12}{'Precision':<12}{'Recall':<10}")
            print("-" * 70)
            for rank, name in enumerate(sorted(slip.keys(), key=lambda k: -slip[k]["F1"]), 1):
                r = slip[name]
                print(f"{rank:<5}{name:<20}{r['F1']:<10.3f}{r['accuracy']:<12.3f}{r['precision']:<12.3f}{r['recall']:<10.3f}")

        print("\n" + "=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="datasets/ycb_grasp_eval")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation_full")
    args = parser.parse_args()

    ev = FullEvaluation(args.eval_dir, args.output_dir)
    ev.eval_velocity()
    ev.eval_contact()
    ev.eval_slip()
    ev.save()
    ev.print_summary()


if __name__ == "__main__":
    main()
