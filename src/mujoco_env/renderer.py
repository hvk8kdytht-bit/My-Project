"""
离屏渲染工具
封装 MuJoCo 的离屏渲染，提供 RGB-D 渲染接口
"""

import numpy as np
from typing import Tuple, Optional

_renderer_available = None
_mj = None
_mjr = None


def _check_renderer():
    global _renderer_available, _mj, _mjr
    if _renderer_available is not None:
        return _renderer_available
    try:
        import mujoco as mj
        from mujoco import mjr
        _mj = mj
        _mjr = mjr
        _renderer_available = True
    except ImportError:
        _renderer_available = False
    return _renderer_available


class OffscreenRenderer:
    """
    MuJoCo 离屏渲染器

    用法:
        renderer = OffscreenRenderer(width=640, height=480)
        rgb, depth = renderer.render(model, data, camera_pos, camera_quat)
    """

    def __init__(self, width: int = 640, height: int = 480):
        if not _check_renderer():
            raise RuntimeError("MuJoCo 不可用")

        self.width = width
        self.height = height
        self._ctx = None

    def _ensure_context(self, model):
        """确保渲染上下文存在"""
        if self._ctx is None:
            self._ctx = _mjr.MjrContext(model, _mjr.mjtFontScale.mjFONTSCALE_150.value)
        return self._ctx

    def render(
        self,
        model,
        data,
        camera_pos: Optional[np.ndarray] = None,
        camera_quat: Optional[np.ndarray] = None,
        camera_id: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        渲染当前场景

        Args:
            model: MuJoCo 模型
            data: MuJoCo 数据
            camera_pos: 相机位置 (3,)，为None则使用camera_id
            camera_quat: 相机朝向四元数 (4,) wxyz
            camera_id: 固定相机ID（-1表示自由相机）

        Returns:
            rgb: (H, W, 3) uint8
            depth: (H, W) float32 (米)
        """
        ctx = self._ensure_context(model)

        # 视口
        viewport = _mjr.MjrRect(0, 0, self.width, self.height)

        # 相机
        cam = _mj.MjvCamera()
        if camera_id >= 0:
            cam.type = _mj.mjtCamera.mjCAMERA_FIXED.value
            cam.fixedcamid = camera_id
        else:
            cam.type = _mj.mjtCamera.mjCAMERA_FREE.value
            if camera_pos is not None:
                cam.lookat[:] = camera_pos
            if camera_quat is not None:
                # 设置相机朝向（简化处理）
                pass

        # 场景
        scene = _mj.MjvScene(model, maxgeom=2000)
        vopt = _mj.MjvOption()
        pert = _mj.MjvPerturb()

        _mj.mjv_updateScene(
            model, data, vopt, pert, cam,
            _mj.mjtCatBit.mjCAT_ALL.value, scene
        )

        # 渲染
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)

        _mjr.mjr_render(viewport, scene, ctx)
        _mjr.mjr_readPixels(rgb, depth, viewport, ctx)

        # 翻转
        rgb = np.flipud(rgb).copy()
        depth = np.flipud(depth).copy()

        # 深度归一化转米
        z_near = 0.01
        z_far = 10.0
        depth_m = z_near * z_far / (z_far - depth * (z_far - z_near))

        return rgb, depth_m

    def close(self):
        if self._ctx:
            self._ctx.free()
            self._ctx = None

    def __del__(self):
        self.close()
