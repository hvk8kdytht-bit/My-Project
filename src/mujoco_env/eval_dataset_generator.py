"""
MuJoCo 评估测试集生成器（最终裁判）
使用 YCB 物体真实尺寸，生成带完整 GT 的抓取序列：
- RGB + 深度
- 物体相对夹爪位姿 (6DoF)
- 物体线速度 / 角速度 / 线加速度 / 角加速度
- 接触力、滑移状态
- 夹爪开合状态

所有感知方案（RGB baseline、RGBD baseline、光流法、Kalman滤波法等）
都在这个统一测试集上评估，保证公平。
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


# YCB-Video 21个物体的真实尺寸（半尺寸，单位: 米）
# 基于 models_info.json 中直径估计
YCB_OBJECT_SIZES = {
    1:  (0.035, 0.045, 0.095),    # 002_master_chef_can - 直径103mm
    2:  (0.050, 0.070, 0.120),    # 003_cracker_box - 直径148mm
    3:  (0.045, 0.060, 0.100),    # 004_sugar_box - 直径128mm
    4:  (0.033, 0.042, 0.090),    # 005_tomato_soup_can - 直径99mm
    5:  (0.035, 0.050, 0.105),    # 006_mustard_bottle - 直径112mm
    6:  (0.035, 0.035, 0.035),    # 007_tuna_fish_can - 直径74mm
    7:  (0.050, 0.050, 0.040),    # 008_pudding_box - 直径86mm
    8:  (0.035, 0.050, 0.035),    # 009_gelatin_box - 直径74mm
    9:  (0.040, 0.055, 0.085),    # 010_potted_meat_can - 直径100mm
    10: (0.050, 0.070, 0.050),    # 011_banana - 直径118mm
    11: (0.070, 0.070, 0.100),    # 019_pitcher_base - 直径146mm
    12: (0.055, 0.065, 0.115),    # 021_bleach_cleanser - 直径139mm
    13: (0.055, 0.055, 0.035),    # 024_bowl - 直径112mm
    14: (0.040, 0.060, 0.065),    # 025_mug - 直径92mm
    15: (0.035, 0.120, 0.040),    # 035_power_drill - 直径137mm
    16: (0.050, 0.050, 0.050),    # 036_wood_block - 直径103mm
    17: (0.025, 0.080, 0.025),    # 037_scissors - 直径95mm
    18: (0.015, 0.015, 0.130),    # 040_large_marker - 直径131mm
    19: (0.050, 0.030, 0.120),    # 051_large_clamp - 直径133mm
    20: (0.060, 0.035, 0.160),    # 052_extra_large_clamp - 直径170mm
    21: (0.040, 0.065, 0.030),    # 061_foam_brick - 直径81mm
}

YCB_OBJECT_NAMES = {
    1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box",
    4: "005_tomato_soup_can", 5: "006_mustard_bottle", 6: "007_tuna_fish_can",
    7: "008_pudding_box", 8: "009_gelatin_box", 9: "010_potted_meat_can",
    10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser",
    13: "024_bowl", 14: "025_mug", 15: "035_power_drill",
    16: "036_wood_block", 17: "037_scissors", 18: "040_large_marker",
    19: "051_large_clamp", 20: "052_extra_large_clamp", 21: "061_foam_brick",
}


def build_ycb_mujoco_xml(obj_id: int, img_w: int = 640, img_h: int = 480) -> str:
    """为指定YCB物体构建MuJoCo XML（用box近似，尺寸与真实YCB模型一致）"""
    sx, sy, sz = YCB_OBJECT_SIZES[obj_id]
    name = YCB_OBJECT_NAMES[obj_id]

    return f"""<mujoco model="ycb_grasp_{name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" iterations="50" tolerance="1e-6" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <global offwidth="{img_w}" offheight="{img_h}"/>
  </visual>

  <asset>
    <texture name="table_tex" type="2d" builtin="flat" rgb1="0.8 0.75 0.65" width="100" height="100"/>
    <material name="table_mat" texture="table_tex" texrepeat="5 5" specular="0.1"/>
    <texture name="obj_tex" type="2d" builtin="flat" rgb1="0.9 0.6 0.3" width="10" height="10"/>
    <material name="obj_mat" texture="obj_tex" specular="0.2" shininess="0.5"/>
    <texture name="gripper_tex" type="2d" builtin="flat" rgb1="0.7 0.7 0.7" width="10" height="10"/>
    <material name="gripper_mat" texture="gripper_tex" specular="0.3"/>
  </asset>

  <worldbody>
    <!-- 桌面 -->
    <body name="table" pos="0 0 -0.05">
      <geom type="box" size="0.3 0.3 0.02" material="table_mat" contype="1" conaffinity="1" friction="0.8 0.1 0.01"/>
    </body>

    <!-- 夹爪基座（固定） -->
    <body name="gripper_base" pos="0 0 0.2">
      <geom type="box" size="0.03 0.03 0.02" material="gripper_mat"/>

      <!-- 左指（可动，沿y轴平移） -->
      <body name="left_finger" pos="0 0.04 0">
        <joint name="left_slide" type="slide" axis="0 1 0" range="0.005 0.06"/>
        <geom type="box" size="0.012 0.008 0.045" pos="0 0 0" material="gripper_mat"
              contype="2" conaffinity="2" friction="1.2 0.02 0.001" priority="1" condim="3"/>
        <site name="left_pad_site" type="box" size="0.012 0.001 0.04" pos="0 -0.008 0" rgba="1 0 0 0.3"/>
      </body>

      <!-- 右指（可动，沿y轴平移） -->
      <body name="right_finger" pos="0 -0.04 0">
        <joint name="right_slide" type="slide" axis="0 1 0" range="-0.06 -0.005"/>
        <geom type="box" size="0.012 0.008 0.045" pos="0 0 0" material="gripper_mat"
              contype="2" conaffinity="2" friction="1.2 0.02 0.001" priority="1" condim="3"/>
        <site name="right_pad_site" type="box" size="0.012 0.001 0.04" pos="0 0.008 0" rgba="1 0 0 0.3"/>
      </body>
    </body>

    <!-- YCB物体 -->
    <body name="object" pos="0 0 0.08">
      <freejoint/>
      <geom type="box" size="{sx} {sy} {sz}" material="obj_mat"
            contype="1" conaffinity="1" friction="0.6 0.05 0.01" mass="0.1"/>
    </body>

    <!-- 相机（正前方俯视） -->
    <camera name="front" pos="0 0.0 0.5" zaxis="0 -1 0" fovy="45"/>
  </worldbody>

  <actuator>
    <position name="left_act" joint="left_slide" kp="500" kv="10" forcerange="-50 50"/>
    <position name="right_act" joint="right_slide" kp="500" kv="10" forcerange="-50 50"/>
  </actuator>

  <sensor>
    <touch name="left_touch" site="left_pad_site"/>
    <touch name="right_touch" site="right_pad_site"/>
    <actuatorfrc name="left_force" actuator="left_act"/>
    <actuatorfrc name="right_force" actuator="right_act"/>
  </sensor>
</mujoco>"""


class YCBGraspEvalGenerator:
    """
    YCB 物体抓取评估数据集生成器（MuJoCo 物理仿真）

    生成的数据集是所有感知方案的统一考试（最终裁判）：
    - 每个方案输入 RGB/深度图 → 输出位姿/速度/接触状态
    - 与 MuJoCo GT 比较 → 得出方案排名

    输出结构（与 BOP 格式兼容，额外包含速度/接触 GT）:
        ycb_grasp_eval/
        ├── test/
        │   ├── 00001_obj01/
        │   │   ├── rgb/
        │   │   ├── depth/
        │   │   ├── scene_gt.json        # 位姿GT（BOP兼容）
        │   │   ├── scene_velocity.json  # 速度GT
        │   │   ├── scene_acceleration.json  # 加速度GT
        │   │   ├── scene_contact.json   # 接触/滑移GT
        │   │   ├── scene_gripper.json   # 夹爪状态GT
        │   │   └── scene_camera.json    # 相机参数
        │   └── ...
        └── camera.json
    """

    def __init__(
        self,
        output_dir: str,
        num_scenes_per_object: int = 3,
        steps_per_scene: int = 300,
        img_width: int = 640,
        img_height: int = 480,
        object_ids: Optional[List[int]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.num_scenes_per_object = num_scenes_per_object
        self.steps_per_scene = steps_per_scene
        self.img_width = img_width
        self.img_height = img_height
        self.object_ids = object_ids or [1, 3, 5, 10, 14, 21]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_params = {
            "fx": 1066.778, "fy": 1067.487,
            "cx": 312.9869, "cy": 241.3109,
            "width": img_width, "height": img_height,
            "depth_scale": 1000.0,
        }
        self.dt = 0.002

    def generate(self):
        import mujoco
        from PIL import Image

        test_dir = self.output_dir / "test"
        test_dir.mkdir(exist_ok=True)

        with open(self.output_dir / "camera.json", "w") as f:
            json.dump(self.camera_params, f, indent=2)

        total_scenes = len(self.object_ids) * self.num_scenes_per_object
        scene_idx = 0

        for obj_id in self.object_ids:
            xml = build_ycb_mujoco_xml(obj_id, img_w=self.img_width, img_h=self.img_height)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            rgb_renderer = mujoco.Renderer(model, self.img_height, self.img_width)
            depth_renderer = mujoco.Renderer(model, self.img_height, self.img_width)
            depth_renderer.enable_depth_rendering()

            obj_name = YCB_OBJECT_NAMES[obj_id]
            print(f"\n物体 {obj_name} (obj_id={obj_id}): {self.num_scenes_per_object} 个场景")

            for s in range(self.num_scenes_per_object):
                scene_idx += 1
                scene_name = f"{scene_idx:05d}_{obj_name}"
                scene_dir = test_dir / scene_name
                rgb_dir = scene_dir / "rgb"
                depth_dir = scene_dir / "depth"
                rgb_dir.mkdir(parents=True, exist_ok=True)
                depth_dir.mkdir(parents=True, exist_ok=True)

                # 随机初始位姿
                mujoco.mj_resetData(model, data)
                init_x = np.random.uniform(-0.02, 0.02)
                init_y = np.random.uniform(-0.02, 0.02)
                init_z = np.random.uniform(0.06, 0.12)
                # qpos 顺序: left_slide(1), right_slide(1), object freejoint(7=3t+4q)
                data.qpos[0] = 0.05    # left_slide
                data.qpos[1] = -0.05   # right_slide
                data.qpos[2] = init_x  # obj x
                data.qpos[3] = init_y  # obj y
                data.qpos[4] = init_z  # obj z
                # 随机小旋转（四元数 w,x,y,z）
                from scipy.spatial.transform import Rotation
                roll = np.random.uniform(-0.3, 0.3)
                pitch = np.random.uniform(-0.3, 0.3)
                yaw = np.random.uniform(-1.5, 1.5)
                q = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()  # x,y,z,w
                data.qpos[5:9] = [q[3], q[0], q[1], q[2]]  # MuJoCo: w,x,y,z

                # 生成动作序列
                actions = self._make_grasp_actions(self.steps_per_scene)

                # 记录数据
                poses = []
                velocities = []
                contacts = []
                gripper_states = []

                obj_body_id = model.body("object").id
                base_body_id = model.body("gripper_base").id

                for step_idx in range(self.steps_per_scene):
                    left_act, right_act = actions[step_idx]
                    data.ctrl[0] = left_act
                    data.ctrl[1] = right_act
                    mujoco.mj_step(model, data)

                    # 渲染 RGB
                    rgb_renderer.update_scene(data, camera="front")
                    rgb = rgb_renderer.render()
                    # 渲染深度
                    depth_renderer.update_scene(data, camera="front")
                    depth = depth_renderer.render()

                    # 保存
                    Image.fromarray(rgb).save(rgb_dir / f"{step_idx:06d}.png")
                    depth_mm = (depth * 1000.0).astype(np.uint16)
                    Image.fromarray(depth_mm).save(depth_dir / f"{step_idx:06d}.png")

                    # 物体位姿（世界系）
                    obj_pos = data.xpos[obj_body_id].copy()
                    obj_quat = data.xquat[obj_body_id].copy()

                    # 夹爪中心位置（基于两指的中点 + base的x,z）
                    left_joint_pos = data.qpos[0]
                    right_joint_pos = data.qpos[1]
                    base_pos = data.xpos[base_body_id].copy()
                    gripper_center = np.array([
                        base_pos[0],
                        base_pos[1] + (left_joint_pos + right_joint_pos) / 2,
                        base_pos[2],
                    ])

                    # 相对位姿
                    rel_pos = obj_pos - gripper_center
                    R = self._quat2mat(obj_quat)

                    # 物体速度（世界系）
                    lin_vel = data.qvel[2:5].copy()   # object 线速度 (3,)
                    ang_vel = data.qvel[5:8].copy()   # object 角速度 (3,)

                    # 接触检测
                    n_con = data.ncon
                    has_contact = n_con > 0
                    total_force = 0.0
                    for i in range(min(n_con, 10)):
                        cf = np.zeros(6)
                        mujoco.mj_contactForce(model, data, i, cf)
                        total_force += np.linalg.norm(cf[:3])

                    # 滑移检测：有接触且物体切向速度>阈值
                    is_slipping = False
                    if has_contact and step_idx > 0:
                        if np.linalg.norm(lin_vel[:2]) > 0.005:
                            is_slipping = True

                    poses.append((R, rel_pos))
                    velocities.append((lin_vel, ang_vel))
                    contacts.append({
                        "has_contact": bool(has_contact),
                        "num_contacts": int(n_con),
                        "total_force_N": float(total_force),
                        "is_slipping": bool(is_slipping),
                    })
                    gripper_states.append({
                        "left_pos_m": float(left_joint_pos),
                        "right_pos_m": float(right_joint_pos),
                        "width_m": float(abs(left_joint_pos - right_joint_pos)),
                    })

                # 保存 scene_gt.json（BOP格式）
                scene_gt = {}
                for i, (R, t) in enumerate(poses):
                    scene_gt[str(i)] = [{
                        "obj_id": obj_id,
                        "cam_R_m2c": R.flatten().tolist(),
                        "cam_t_m2c": (t * 1000).tolist(),
                    }]
                with open(scene_dir / "scene_gt.json", "w") as f:
                    json.dump(scene_gt, f)

                # 速度 GT
                scene_vel = {}
                for i, (lv, av) in enumerate(velocities):
                    scene_vel[str(i)] = {
                        "linear_velocity_m_s": lv.tolist(),
                        "angular_velocity_rad_s": av.tolist(),
                    }
                with open(scene_dir / "scene_velocity.json", "w") as f:
                    json.dump(scene_vel, f)

                # 加速度 GT（数值微分）
                scene_acc = {}
                for i in range(1, len(velocities)):
                    lv1, av1 = velocities[i-1]
                    lv2, av2 = velocities[i]
                    lin_acc = (lv2 - lv1) / self.dt
                    ang_acc = (av2 - av1) / self.dt
                    scene_acc[str(i)] = {
                        "linear_acceleration_m_s2": lin_acc.tolist(),
                        "angular_acceleration_rad_s2": ang_acc.tolist(),
                    }
                with open(scene_dir / "scene_acceleration.json", "w") as f:
                    json.dump(scene_acc, f)

                # 接触 GT
                scene_con = {str(i): c for i, c in enumerate(contacts)}
                with open(scene_dir / "scene_contact.json", "w") as f:
                    json.dump(scene_con, f)

                # 夹爪 GT
                scene_grp = {str(i): g for i, g in enumerate(gripper_states)}
                with open(scene_dir / "scene_gripper.json", "w") as f:
                    json.dump(scene_grp, f)

                # 相机参数（逐帧，BOP格式）
                cam_K = [
                    self.camera_params["fx"], 0, self.camera_params["cx"],
                    0, self.camera_params["fy"], self.camera_params["cy"],
                    0, 0, 1,
                ]
                cam_entry = {str(i): {"cam_K": cam_K, "depth_scale": 1.0}
                             for i in range(len(poses))}
                with open(scene_dir / "scene_camera.json", "w") as f:
                    json.dump(cam_entry, f)

                n_contact = sum(1 for c in contacts if c["has_contact"])
                n_slip = sum(1 for c in contacts if c["is_slipping"])
                print(f"  {scene_name}: {self.steps_per_scene}帧, "
                      f"接触{n_contact}帧, 滑移{n_slip}帧")

            rgb_renderer.close()
            depth_renderer.close()

        # 元信息
        meta = {
            "description": "YCB物体抓取评估数据集（MuJoCo仿真，最终裁判）",
            "num_objects": len(self.object_ids),
            "num_scenes": total_scenes,
            "steps_per_scene": self.steps_per_scene,
            "dt_seconds": self.dt,
            "objects": {str(i): YCB_OBJECT_NAMES[i] for i in self.object_ids},
            "gt_modalities": [
                "pose_6DoF", "linear_velocity", "angular_velocity",
                "linear_acceleration", "angular_acceleration",
                "contact_force", "slip_state", "gripper_state",
            ],
            "evaluation_metrics": {
                "pose": ["ADD", "ADI", "projection_error_px"],
                "velocity": ["RMSE_linear_m_s", "RMSE_angular_rad_s"],
                "acceleration": ["RMSE_linear_m_s2", "RMSE_angular_rad_s2"],
                "contact": ["accuracy", "precision", "recall", "F1"],
                "slip": ["accuracy", "precision", "recall", "F1"],
            },
        }
        with open(self.output_dir / "dataset_info.json", "w") as f:
            json.dump(meta, f, indent=2)

        total_frames = total_scenes * self.steps_per_scene
        print(f"\n✅ 评估数据集生成完成: {self.output_dir}")
        print(f"   {total_scenes} 场景 × {self.steps_per_scene} 帧 = {total_frames} 帧")

    def _make_grasp_actions(self, num_steps: int) -> List[tuple]:
        """抓取动作序列：张开等待 → 闭合抓取 → 保持带振动 → 释放"""
        actions = []
        n_open = num_steps // 6
        n_close = num_steps // 6
        n_hold = num_steps // 2
        n_release = num_steps - n_open - n_close - n_hold

        # 张开
        for _ in range(n_open):
            actions.append((0.05, -0.05))

        # 闭合
        for i in range(n_close):
            t = i / n_close
            l = 0.05 - 0.045 * t
            r = -0.05 + 0.045 * t
            actions.append((l, r))

        # 保持（带振动模拟搬运，可能产生滑移）
        hold_l = 0.007
        hold_r = -0.007
        for i in range(n_hold):
            noise = np.random.normal(0, 0.0015, 2)
            actions.append((hold_l + noise[0], hold_r + noise[1]))

        # 释放
        for i in range(n_release):
            t = i / max(n_release, 1)
            l = hold_l + 0.04 * t
            r = hold_r - 0.04 * t
            actions.append((l, r))

        return actions[:num_steps]

    @staticmethod
    def _quat2mat(quat: np.ndarray) -> np.ndarray:
        """MuJoCo 四元数 (w,x,y,z) 转 3x3 旋转矩阵"""
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/ycb_grasp_eval")
    parser.add_argument("--scenes_per_obj", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--img_h", type=int, default=480)
    args = parser.parse_args()

    gen = YCBGraspEvalGenerator(
        output_dir=args.output_dir,
        num_scenes_per_object=args.scenes_per_obj,
        steps_per_scene=args.steps,
        img_width=args.img_w,
        img_height=args.img_h,
    )
    gen.generate()
