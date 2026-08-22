"""
MuJoCo 两指夹爪抓取环境
- 简单的两指平行夹爪 + 可抓取物体
- 支持 RGB-D 离屏渲染
- 记录物体相对夹爪的位姿、速度、加速度（ground truth）
- 支持多种物体类型（立方体、圆柱体、易碎物、柔软物）

使用方法:
    env = GripperEnv(xml_path="model.xml")
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# MuJoCo 导入（DLL可能被安全策略阻止，延迟处理）
_mujoco_available = None
_mj = None
_mjr = None
_glfw = None


def _check_mujoco():
    """检查MuJoCo是否可用，延迟导入"""
    global _mujoco_available, _mj, _mjr, _glfw
    if _mujoco_available is not None:
        return _mujoco_available
    try:
        import mujoco as mj
        from mujoco import mjr
        _mj = mj
        _mjr = mjr
        _mujoco_available = True
    except ImportError as e:
        print(f"警告: MuJoCo 不可用 ({e})")
        print("请确保 mujoco 已安装且 DLL 未被安全策略阻止")
        _mujoco_available = False
    return _mujoco_available


# MuJoCo XML 模型模板 - 两指夹爪 + 物体
GRIPPER_MODEL_XML = """
<mujoco model="gripper_grasp">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="50" solver="Newton"/>
  <visual>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
    <global offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture name="table_texture" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="100" height="100"/>
    <material name="table_material" texture="table_texture" texrepeat="5 5" specular="0.1"/>
    <texture name="gripper_texture" type="2d" builtin="flat" rgb1="0.7 0.7 0.7" width="10" height="10"/>
    <material name="gripper_material" texture="gripper_texture" specular="0.3"/>
    <texture name="object_texture" type="2d" builtin="flat" rgb1="0.8 0.3 0.3" width="10" height="10"/>
    <material name="object_material" texture="object_texture" specular="0.2"/>
  </asset>

  <worldbody>
    <!-- 桌面 -->
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="0.5 0.5 0.02" pos="0 0 -0.02" material="table_material" condim="3" friction="0.8 0.1 0.1"/>
    </body>

    <!-- 夹爪基座（固定） -->
    <body name="gripper_base" pos="0 0 0.2">
      <geom name="base" type="box" size="0.03 0.03 0.02" material="gripper_material"/>

      <!-- 左指 -->
      <body name="left_finger" pos="0 0.04 0">
        <joint name="left_slide" type="slide" axis="0 1 0" range="0 0.05" damping="5"/>
        <geom name="left_pad" type="box" size="0.015 0.005 0.04" pos="0 0.005 0" material="gripper_material" condim="3" friction="1.0 0.05 0.01" priority="1"/>
        <geom name="left_finger_body" type="box" size="0.01 0.03 0.04" pos="0 0.03 0" material="gripper_material"/>
      </body>

      <!-- 右指 -->
      <body name="right_finger" pos="0 -0.04 0">
        <joint name="right_slide" type="slide" axis="0 1 0" range="-0.05 0" damping="5"/>
        <geom name="right_pad" type="box" size="0.015 0.005 0.04" pos="0 -0.005 0" material="gripper_material" condim="3" friction="1.0 0.05 0.01" priority="1"/>
        <geom name="right_finger_body" type="box" size="0.01 0.03 0.04" pos="0 -0.03 0" material="gripper_material"/>
      </body>
    </body>

    <!-- 可抓取物体 -->
    <body name="object" pos="0 0 0.03" quat="1 0 0 0">
      <freejoint name="object_joint"/>
      <geom name="object_geom" type="box" size="0.02 0.02 0.03" material="object_material" condim="3" friction="0.6 0.05 0.01" mass="0.05" solimp="0.95 0.99 0.01"/>
    </body>
  </worldbody>

  <actuator>
    <!-- 夹爪开合（位置控制） -->
    <position name="left_open" joint="left_slide" kp="50" ctrlrange="0 0.05"/>
    <position name="right_open" joint="right_slide" kp="50" ctrlrange="-0.05 0"/>
  </actuator>

  <sensor>
    <!-- 夹爪位置传感器 -->
    <jointpos name="left_pos" joint="left_slide"/>
    <jointpos name="right_pos" joint="right_slide"/>
    <jointvel name="left_vel" joint="left_slide"/>
    <jointvel name="right_vel" joint="right_slide"/>
  </sensor>
</mujoco>
"""


class GripperEnv:
    """
    两指夹爪抓取仿真环境

    观测空间:
        - rgb: 彩色图像 (480, 640, 3) uint8
        - depth: 深度图 (480, 640) float32 (米)
        - gripper_pos: 夹爪位置 (2,)
        - gripper_vel: 夹爪速度 (2,)
        - object_pose: 物体位姿 (7,) - xyz + 四元数
        - object_vel: 物体速度 (6,) - 线速度 + 角速度

    动作空间:
        - gripper_target: 夹爪开合目标位置 (2,) - 左右指位置
        或
        - gripper_delta: 夹爪开合增量 (1,) - 正为张开，负为闭合
    """

    def __init__(
        self,
        xml_path: Optional[str] = None,
        img_width: int = 640,
        img_height: int = 480,
        render_mode: str = "offscreen",
        object_type: str = "box",
        object_size: Tuple[float, float, float] = (0.02, 0.02, 0.03),
        randomize_initial_pose: bool = True,
    ):
        """
        Args:
            xml_path: 自定义XML模型路径，None则使用内置模板
            img_width: 渲染图像宽度
            img_height: 渲染图像高度
            render_mode: 'offscreen' 离屏渲染 或 'window' 窗口渲染
            object_type: 物体类型 'box', 'cylinder', 'sphere'
            object_size: 物体尺寸 (x, y, z) 米
            randomize_initial_pose: 是否随机化初始物体位姿
        """
        if not _check_mujoco():
            raise RuntimeError("MuJoCo 不可用，请检查安装")

        self.img_width = img_width
        self.img_height = img_height
        self.render_mode = render_mode
        self.object_type = object_type
        self.object_size = object_size
        self.randomize_initial_pose = randomize_initial_pose

        # 加载模型
        if xml_path and Path(xml_path).exists():
            self.model = _mj.MJModel.from_xml_path(xml_path)
        else:
            # 使用内置模板
            xml_content = self._build_xml()
            self.model = _mj.MJModel.from_xml_string(xml_content)

        self.data = _mj.MjData(self.model)

        # 离屏渲染器
        if render_mode == "offscreen":
            self.renderer = _mjr.MjrContext(self.model, _mjr.mjtFontScale.mjFONTSCALE_150.value)
            self._setup_offscreen_camera()
        else:
            self.renderer = None

        # 相机参数
        self.camera_matrix = self._compute_camera_matrix()

        # 物体 ID
        self.object_body_id = self.model.body("object").id
        self.gripper_base_id = self.model.body("gripper_base").id

    def _build_xml(self) -> str:
        """根据参数构建XML模型"""
        # 目前直接使用模板，后续可根据object_type动态修改
        return GRIPPER_MODEL_XML

    def _setup_offscreen_camera(self):
        """设置离屏渲染相机"""
        # 相机位置：正前方，略高于夹爪
        # 在 worldbody 中添加一个相机
        self._cam_pos = np.array([0.0, -0.2, 0.15])
        self._cam_quat = np.array([0.707, 0.0, 0.707, 0.0])  # 看向-Y方向

    def _compute_camera_matrix(self) -> np.ndarray:
        """计算相机内参矩阵（近似）"""
        # 假设FOV约60度
        fov = 60.0
        focal_length = self.img_width / (2 * np.tan(np.radians(fov / 2)))
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        return K

    def reset(self, object_pos: Optional[np.ndarray] = None) -> Dict:
        """
        重置环境

        Args:
            object_pos: 物体初始位置 (3,)，None则随机

        Returns:
            观测字典
        """
        _mj.mj_resetData(self.model, self.data)

        # 随机化物体初始位置
        if self.randomize_initial_pose and object_pos is None:
            object_pos = np.array([
                np.random.uniform(-0.01, 0.01),   # x方向小扰动
                np.random.uniform(-0.01, 0.01),   # y方向小扰动
                0.03 + np.random.uniform(0, 0.01), # z方向高度
            ])
        elif object_pos is None:
            object_pos = np.array([0.0, 0.0, 0.03])

        # 设置物体位置
        obj_body = self.model.body(self.object_body_id)
        self.data.qpos[obj_body.dofadr:obj_body.dofadr + 3] = object_pos
        # 随机旋转（小角度）
        if self.randomize_initial_pose:
            angle = np.random.uniform(-5, 5)
            self.data.qpos[obj_body.dofadr + 3:obj_body.dofadr + 7] = np.array([
                np.cos(np.radians(angle / 2)),
                0, 0, np.sin(np.radians(angle / 2))
            ])

        # 初始夹爪张开
        self.data.ctrl[0] = 0.04  # 左指
        self.data.ctrl[1] = -0.04  # 右指

        # 前向运动学
        _mj.mj_forward(self.model, self.data)

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步仿真

        Args:
            action: 动作，可以是:
                - (2,): 左右指目标位置
                - (1,): 夹爪开合增量（正=张开，负=闭合）

        Returns:
            observation, reward, done, info
        """
        action = np.asarray(action, dtype=np.float32)

        if action.size == 1:
            # 增量模式
            delta = float(action[0])
            current_left = float(self.data.sensordata[0])
            current_right = float(self.data.sensordata[1])
            # 左指向右（负方向）= 闭合
            self.data.ctrl[0] = np.clip(current_left - delta, 0, 0.05)
            self.data.ctrl[1] = np.clip(current_right + delta, -0.05, 0)
        elif action.size == 2:
            # 绝对位置模式
            self.data.ctrl[0] = np.clip(action[0], 0, 0.05)
            self.data.ctrl[1] = np.clip(action[1], -0.05, 0)

        # 步进仿真
        _mj.mj_step(self.model, self.data)

        obs = self._get_observation()
        info = self._get_info()
        reward = self._compute_reward(obs, info)
        done = self._check_done(obs, info)

        return obs, reward, done, info

    def _get_observation(self) -> Dict:
        """获取当前观测"""
        obs = {}

        # RGB-D 图像
        if self.render_mode == "offscreen" and self.renderer:
            rgb, depth = self._render_offscreen()
            obs["rgb"] = rgb
            obs["depth"] = depth

        # 夹爪状态
        obs["gripper_pos"] = np.array([
            float(self.data.sensordata[0]),  # 左指位置
            float(self.data.sensordata[1]),  # 右指位置
        ], dtype=np.float32)
        obs["gripper_vel"] = np.array([
            float(self.data.sensordata[2]),  # 左指速度
            float(self.data.sensordata[3]),  # 右指速度
        ], dtype=np.float32)

        # 物体位姿 (ground truth)
        obj_body = self.model.body(self.object_body_id)
        obs["object_pos"] = self.data.xpos[self.object_body_id].copy()
        obs["object_quat"] = self.data.xquat[self.object_body_id].copy()  # wxyz

        # 物体速度 (ground truth)
        obs["object_vel"] = self.data.cvel[self.object_body_id].copy()  # 6维: 角速度 + 线速度

        return obs

    def _render_offscreen(self) -> Tuple[np.ndarray, np.ndarray]:
        """离屏渲染 RGB-D"""
        # 设置视口
        viewport = _mjr.MjrRect(0, 0, self.img_width, self.img_height)

        # 设置相机
        cam = _mj.MjvCamera()
        cam.type = _mj.mjtCamera.mjCAMERA_FIXED.value
        cam.fixedcamid = 0  # 使用第一个固定相机

        # 渲染设置
        vopt = _mj.MjvOption()
        pert = _mj.MjvPerturb()

        # 创建场景
        scene = _mj.MjvScene(self.model, maxgeom=1000)
        _mj.mjv_updateScene(
            self.model, self.data, vopt, pert, cam,
            _mj.mjtCatBit.mjCAT_ALL.value, scene
        )

        # 渲染 RGB
        rgb_buffer = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        depth_buffer = np.zeros((self.img_height, self.img_width), dtype=np.float32)

        _mjr.mjr_render(viewport, scene, self.renderer)

        # 读取像素
        _mjr.mjr_readPixels(rgb_buffer, depth_buffer, viewport, self.renderer)

        # 翻转（OpenGL坐标系是上下颠倒的）
        rgb = np.flipud(rgb_buffer).copy()
        depth = np.flipud(depth_buffer).copy()

        # 深度值转换（从归一化深度转为米）
        # MuJoCo 的深度是 [0, 1]，需要根据近远平面转换
        z_near = 0.01
        z_far = 10.0
        depth = z_near * z_far / (z_far - depth * (z_far - z_near))

        return rgb, depth

    def _get_info(self) -> Dict:
        """获取额外信息（用于评估）"""
        info = {}

        # 物体相对夹爪的位姿
        gripper_pos = self.data.xpos[self.gripper_base_id]
        object_pos = self.data.xpos[self.object_body_id]

        # 相对位置
        info["object_rel_pos"] = (object_pos - gripper_pos).astype(np.float32)

        # 夹爪开合距离
        left_pos = float(self.data.sensordata[0])
        right_pos = float(self.data.sensordata[1])
        info["gripper_width"] = abs(left_pos - right_pos)

        # 检测接触（滑移状态）
        info["has_contact"] = self.data.ncon > 0
        info["num_contacts"] = int(self.data.ncon)

        # 接触力（粗略估计）
        contact_forces = []
        for i in range(min(self.data.ncon, 10)):
            try:
                c = self.data.contact[i]
                force = np.zeros(6)
                _mj.mj_contactForce(self.model, self.data, i, force)
                contact_forces.append(force[:3])  # 法向力 + 两个切向力
            except:
                pass
        if contact_forces:
            info["contact_forces"] = np.array(contact_forces, dtype=np.float32)
        else:
            info["contact_forces"] = np.zeros((0, 3), dtype=np.float32)

        return info

    def _compute_reward(self, obs: Dict, info: Dict) -> float:
        """简单奖励（用于测试，实际训练用外部reward函数）"""
        # 物体是否被抓取（在空中且有接触）
        object_z = obs["object_pos"][2]
        grasped = object_z > 0.05 and info["has_contact"]
        return 1.0 if grasped else 0.0

    def _check_done(self, obs: Dict, info: Dict) -> bool:
        """检查是否结束"""
        # 物体掉出桌面
        if obs["object_pos"][2] < -0.1:
            return True
        # 物体偏离太远
        if abs(obs["object_pos"][0]) > 0.3 or abs(obs["object_pos"][1]) > 0.3:
            return True
        return False

    def close(self):
        """关闭环境"""
        if self.renderer:
            self.renderer.free()
            self.renderer = None

    def __del__(self):
        self.close()


def get_gripper_relative_pose(
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    gripper_pos: np.ndarray,
) -> Dict:
    """
    计算物体相对夹爪的位姿

    Args:
        object_pos: 物体世界坐标位置 (3,)
        object_quat: 物体世界坐标四元数 (4,) wxyz
        gripper_pos: 夹爪世界坐标位置 (3,)

    Returns:
        relative position (3,), quaternion (4,)
    """
    # 相对位置（假设夹爪朝向固定）
    rel_pos = object_pos - gripper_pos
    return {
        "relative_position": rel_pos.astype(np.float32),
        "relative_quaternion": object_quat.astype(np.float32),
    }
