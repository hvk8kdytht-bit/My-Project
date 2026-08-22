"""
抓取 RGB-D 数据集生成器
在 MuJoCo 中生成带 ground truth 的抓取序列：
    - RGB 图像
    - 深度图像
    - 物体相对夹爪位姿（6DoF）
    - 物体速度、加速度
    - 夹爪状态
    - 接触/滑移状态

生成的数据集格式与 YCB-Video BOP 格式兼容，便于直接用于训练和评估。
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GraspDatasetGenerator:
    """
    MuJoCo 抓取 RGB-D 数据集生成器

    生成的数据集结构:
        mujoco_grasp/
        ├── train/
        │   ├── scene_00001/
        │   │   ├── rgb/              # 彩色图像 .png
        │   │   ├── depth/            # 深度图 .png (16位，毫米)
        │   │   ├── scene_gt.json     # 位姿标注
        │   │   ├── scene_state.json  # 夹爪状态、速度、接触
        │   │   └── camera.json       # 相机参数
        │   └── ...
        ├── test/
        └── camera.json               # 全局相机参数
    """

    def __init__(
        self,
        output_dir: str,
        num_train_scenes: int = 100,
        num_test_scenes: int = 20,
        steps_per_scene: int = 200,
        img_width: int = 640,
        img_height: int = 480,
        object_types: Optional[List[str]] = None,
        randomize_appearance: bool = True,
    ):
        """
        Args:
            output_dir: 输出目录
            num_train_scenes: 训练场景数量
            num_test_scenes: 测试场景数量
            steps_per_scene: 每个场景的步数
            img_width: 图像宽度
            img_height: 图像高度
            object_types: 物体类型列表
            randomize_appearance: 是否随机化物体外观
        """
        self.output_dir = Path(output_dir)
        self.num_train_scenes = num_train_scenes
        self.num_test_scenes = num_test_scenes
        self.steps_per_scene = steps_per_scene
        self.img_width = img_width
        self.img_height = img_height
        self.object_types = object_types or ["box"]
        self.randomize_appearance = randomize_appearance

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 相机参数（与 YCB-Video 近似，便于比较）
        self.camera_params = {
            "fx": 1066.778,
            "fy": 1067.487,
            "cx": 312.9869,
            "cy": 241.3109,
            "width": img_width,
            "height": img_height,
            "depth_scale": 1000.0,  # 深度图缩放因子（毫米）
        }

    def generate(self, split: str = "train"):
        """
        生成数据集

        Args:
            split: 'train' 或 'test'
        """
        from PIL import Image
        from .gripper_env import GripperEnv, _check_mujoco

        if not _check_mujoco():
            print("MuJoCo 不可用，跳过数据集生成")
            print("请先解决 MuJoCo DLL 加载问题后再运行")
            return

        num_scenes = self.num_train_scenes if split == "train" else self.num_test_scenes
        split_dir = self.output_dir / split
        split_dir.mkdir(exist_ok=True)

        # 保存相机参数
        with open(self.output_dir / "camera.json", "w") as f:
            json.dump(self.camera_params, f, indent=2)

        env = GripperEnv(
            img_width=self.img_width,
            img_height=self.img_height,
            render_mode="offscreen",
            randomize_initial_pose=True,
        )

        print(f"生成 {split} 数据集 ({num_scenes} 个场景, 每场景 {self.steps_per_scene} 步)...")

        for scene_idx in range(num_scenes):
            scene_name = f"{scene_idx:05d}"
            scene_dir = split_dir / scene_name
            rgb_dir = scene_dir / "rgb"
            depth_dir = scene_dir / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            # 重置环境
            obs = env.reset()

            # 记录每帧数据
            scene_gt = {}  # 位姿标注
            scene_state = {}  # 状态信息

            # 生成随机抓取动作序列
            action_sequence = self._generate_grasp_sequence(self.steps_per_scene)

            for step_idx in range(self.steps_per_scene):
                # 执行动作
                action = action_sequence[step_idx]
                obs, reward, done, info = env.step(action)

                # 保存 RGB
                rgb_img = Image.fromarray(obs["rgb"])
                rgb_path = rgb_dir / f"{step_idx:06d}.png"
                rgb_img.save(rgb_path)

                # 保存深度（转为16位PNG，单位毫米）
                depth_mm = (obs["depth"] * self.camera_params["depth_scale"]).astype(np.uint16)
                depth_img = Image.fromarray(depth_mm)
                depth_path = depth_dir / f"{step_idx:06d}.png"
                depth_img.save(depth_path)

                # 记录位姿（BOP格式兼容）
                obj_id = 1  # 单个物体
                R = self._quat_to_rotmat(obs["object_quat"])
                t = obs["object_rel_pos"] * 1000.0  # 毫米（BOP格式用毫米）

                scene_gt[str(step_idx)] = [{
                    "obj_id": obj_id,
                    "cam_R_m2c": R.flatten().tolist(),
                    "cam_t_m2c": t.tolist(),
                }]

                # 记录状态信息
                scene_state[str(step_idx)] = {
                    "gripper_pos": obs["gripper_pos"].tolist(),
                    "gripper_vel": obs["gripper_vel"].tolist(),
                    "object_pos": obs["object_pos"].tolist(),
                    "object_quat": obs["object_quat"].tolist(),
                    "object_vel": obs["object_vel"].tolist(),
                    "gripper_width": info["gripper_width"],
                    "has_contact": info["has_contact"],
                    "num_contacts": info["num_contacts"],
                    "contact_forces": info["contact_forces"].tolist(),
                }

                if done:
                    break

            # 保存标注
            with open(scene_dir / "scene_gt.json", "w") as f:
                json.dump(scene_gt, f)

            with open(scene_dir / "scene_state.json", "w") as f:
                json.dump(scene_state, f)

            # 场景级相机参数
            with open(scene_dir / "camera.json", "w") as f:
                json.dump(self.camera_params, f, indent=2)

            if (scene_idx + 1) % 10 == 0:
                print(f"  进度: {scene_idx + 1}/{num_scenes} 场景")

        env.close()
        print(f"{split} 数据集生成完成，保存在: {split_dir}")

    def _generate_grasp_sequence(self, num_steps: int) -> List[np.ndarray]:
        """
        生成抓取动作序列
        阶段:
            1. 接近（夹爪张开，向下移动 - 此处简化为固定夹爪位置）
            2. 闭合抓取
            3. 保持/抬起
            4. 松开
        """
        actions = []
        phase1 = num_steps // 4   # 张开等待
        phase2 = num_steps // 4   # 闭合抓取
        phase3 = num_steps // 2   # 保持/抬起

        # 阶段1: 夹爪张开
        for _ in range(phase1):
            actions.append(np.array([0.04, -0.04]))  # 张开

        # 阶段2: 逐渐闭合
        for i in range(phase2):
            t = i / phase2
            left_pos = 0.04 - 0.04 * t   # 从0.04到0
            right_pos = -0.04 + 0.04 * t  # 从-0.04到0
            actions.append(np.array([left_pos, right_pos]))

        # 阶段3: 保持夹持
        for _ in range(phase3):
            # 小幅振动模拟搬运
            noise = np.random.normal(0, 0.001, 2)
            actions.append(np.array([0.005, -0.005]) + noise)

        # 裁剪到num_steps
        return actions[:num_steps]

    def _quat_to_rotmat(self, quat: np.ndarray) -> np.ndarray:
        """四元数 (w,x,y,z) 转旋转矩阵"""
        w, x, y, z = quat
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
        return R

    def compute_velocity_from_poses(
        self,
        scene_dir: str,
        dt: float = 0.002,
    ) -> Dict:
        """
        从位姿序列计算速度和加速度（后处理，用于验证ground truth）

        Args:
            scene_dir: 场景目录
            dt: 时间步长（秒）

        Returns:
            速度和加速度字典
        """
        with open(Path(scene_dir) / "scene_gt.json", "r") as f:
            scene_gt = json.load(f)

        positions = []
        for frame_idx in sorted(scene_gt.keys(), key=int):
            ann = scene_gt[frame_idx][0]
            t = np.array(ann["cam_t_m2c"]) / 1000.0  # 转米
            positions.append(t)

        positions = np.array(positions)

        # 速度
        velocities = np.diff(positions, axis=0) / dt

        # 加速度
        accelerations = np.diff(velocities, axis=0) / dt

        return {
            "positions": positions,
            "velocities": velocities,
            "accelerations": accelerations,
        }
