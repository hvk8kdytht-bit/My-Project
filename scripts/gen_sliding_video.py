"""
MuJoCo 物体滑移仿真视频生成
相机固定不动，物体在桌面上被抓取后滑动
"""
import sys
import os
import json
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "mujoco_env"))

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_OK = True
except Exception as e:
    MUJOCO_OK = False
    print(f"MuJoCo import failed: {e}")


def create_sliding_scene_xml():
    """创建物体滑移场景：固定相机 + 物体在桌面滑动"""
    xml = """
    <mujoco>
      <option timestep="0.002" gravity="0 0 -9.81"/>

      <visual>
        <headlight diffuse="0.8 0.8 0.8"/>
        <rgba haze="0.15 0.15 0.15 1"/>
        <global azimuth="45" elevation="-30"/>
      </visual>

      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 1" rgb2="0.3 0.5 0.8" width="256" height="256"/>
        <texture name="floor" type="2d" builtin="checker" mark="cross" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512"/>
        <material name="floor_mat" texture="floor" texrepeat="5 5" reflectance="0.2"/>
        <texture name="obj" type="cube" builtin="gradient" rgb1="0.8 0.3 0.2" rgb2="0.6 0.2 0.1" width="100"/>
        <material name="obj_mat" texture="obj" rgba="0.8 0.3 0.2 1"/>
      </asset>

      <default>
        <geom friction="0.8 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"/>
      </default>

      <worldbody>
        <!-- 固定相机：从斜上方看桌面 -->
        <camera name="fixed_cam" pos="0.5 -0.5 0.8" quat="0.75 0.25 0.25 0.55"/>

        <!-- 地面 -->
        <geom name="floor" type="plane" size="2 2 0.1" material="floor_mat"/>

        <!-- 桌面 -->
        <geom name="table" type="box" pos="0 0 0.4" size="0.3 0.3 0.02" rgba="0.4 0.3 0.2 1"/>

        <!-- 滑动物体：红色方块 -->
        <body name="sliding_box" pos="-0.15 0 0.435">
          <freejoint name="box_joint"/>
          <geom name="box" type="box" size="0.03 0.03 0.015" mass="0.1" rgba="0.8 0.2 0.2 1" friction="0.8 0.005 0.0001"/>
        </body>

        <!-- 第二个物体：蓝色球 -->
        <body name="rolling_ball" pos="0.1 0.1 0.425">
          <freejoint name="ball_joint"/>
          <geom name="ball" type="sphere" size="0.025" mass="0.05" rgba="0.2 0.4 0.8 1" friction="0.8 0.005 0.0001"/>
        </body>

        <!-- 第三个物体：绿色圆柱 -->
        <body name="sliding_cyl" pos="0.0 -0.1 0.43">
          <freejoint name="cyl_joint"/>
          <geom name="cyl" type="cylinder" size="0.02 0.04" mass="0.08" rgba="0.2 0.8 0.3 1" friction="0.8 0.005 0.0001"/>
        </body>

        <!-- 光源 -->
        <light pos="0.5 -0.5 1.5" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
      </worldbody>

      <actuator>
        <general name="push_box" joint="box_joint" gear="1 0 0 0 0 0" ctrlrange="-5 5"/>
        <general name="push_ball" joint="ball_joint" gear="0 1 0 0 0 0" ctrlrange="-5 5"/>
        <general name="push_cyl" joint="cyl_joint" gear="1 0 0 0 0 0" ctrlrange="-5 5"/>
      </actuator>
    </mujoco>
    """
    return xml


def run_sliding_simulation(output_dir, num_frames=150, fps=30):
    """运行滑移仿真，生成视频和GT数据"""
    if not MUJOCO_OK:
        print("MuJoCo not available!")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)

    # 创建模型
    xml = create_sliding_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 设置相机
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam")

    # 渲染上下文
    renderer = mujoco.Renderer(model, height=480, width=640)

    # 物体信息
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sliding_box")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rolling_ball")
    cyl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sliding_cyl")

    box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    cyl_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cyl")

    frames = []
    gt_data = []

    print(f"Running sliding simulation ({num_frames} frames)...")

    for i in range(num_frames):
        # 直接设置物体位置模拟滑移（完全可控）
        # 方块：第10-50帧向右滑动
        if 10 <= i < 50:
            data.qpos[0] = -0.15 + (i - 10) * 0.002
            data.qvel[0] = 0.06  # 匹配位置变化的速度
        # 球：第50-90帧向前滚动
        if 50 <= i < 90:
            data.qpos[7] = 0.1 + (i - 50) * 0.0015
            data.qvel[7] = 0.045
        # 圆柱：第90-130帧向右滑动
        if 90 <= i < 130:
            data.qpos[14] = 0.0 + (i - 90) * 0.0025
            data.qvel[12] = 0.075

        # 关闭控制器
        data.ctrl[:] = 0

        # 物理步进
        mujoco.mj_step(model, data)

        # 渲染（使用固定相机）
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render()
        frames.append(rgb)

        # 保存帧
        cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # 记录GT
        box_pos = data.xpos[box_id].copy()
        ball_pos = data.xpos[ball_id].copy()
        cyl_pos = data.xpos[cyl_id].copy()

        box_vel = data.cvel[box_id].copy()
        ball_vel = data.cvel[ball_id].copy()
        cyl_vel = data.cvel[cyl_id].copy()

        gt_data.append({
            "frame": i,
            "box": {"pos": box_pos.tolist(), "vel": box_vel.tolist()},
            "ball": {"pos": ball_pos.tolist(), "vel": ball_vel.tolist()},
            "cyl": {"pos": cyl_pos.tolist(), "vel": cyl_vel.tolist()},
        })

        if (i + 1) % 30 == 0:
            print(f"  Frame {i+1}/{num_frames}")

    # 生成视频
    video_path = output_dir / "sliding_simulation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    # 保存GT
    gt_path = output_dir / "sliding_gt.json"
    with open(gt_path, "w") as f:
        json.dump({
            "num_frames": num_frames,
            "fps": fps,
            "camera": "fixed",
            "objects": ["sliding_box", "rolling_ball", "sliding_cyl"],
            "phases": {
                "phase1_push_box": "frames 10-50, force along x",
                "phase2_push_ball": "frames 50-90, force along y",
                "phase3_push_cyl": "frames 90-130, force along x",
            },
            "frames": gt_data,
        }, f, indent=2)

    print(f"Video saved: {video_path} ({os.path.getsize(str(video_path))/1024:.0f} KB)")
    print(f"GT saved: {gt_path}")
    print(f"Frames saved: {rgb_dir}")

    return video_path, gt_path


if __name__ == "__main__":
    run_sliding_simulation("outputs/sliding_simulation", num_frames=150, fps=30)
