"""
速度估计模块（多种方案 baseline）

方案汇总:
1. 有限差分法 (FiniteDifferenceEstimator) - 最简基线，位姿序列差分
2. Savitzky-Golay 滤波法 (SavGolVelocityEstimator) - 平滑+微分
3. Kalman 滤波法 (KalmanVelocityEstimator) - 递归估计，带动力学模型
4. 光流法 (OpticalFlowVelocityEstimator) - 纯视觉，无需位姿模型
5. Lucas-Kanade 光流法 (LucasKanadeVelocityEstimator) - 稀疏特征光流

所有方法统一接口: estimate(...) -> dict with 'linear_velocity' / 'angular_velocity'
"""

try:
    from .optical_flow_estimator import (
        OpticalFlowVelocityEstimator,
        LucasKanadeVelocityEstimator,
    )
except ImportError:
    OpticalFlowVelocityEstimator = None
    LucasKanadeVelocityEstimator = None

try:
    from .raft_estimator import RaftOpticalFlowEstimator, RaftSlipDetector, RaftContactDetector
except ImportError:
    RaftOpticalFlowEstimator = None
    RaftSlipDetector = None
    RaftContactDetector = None

__all__ = [
    "OpticalFlowVelocityEstimator",
    "LucasKanadeVelocityEstimator",
    "RaftOpticalFlowEstimator",
    "RaftSlipDetector",
    "RaftContactDetector",
]
