# 易碎/柔软物体安全抓取视触 VLA 项目

> 语言条件下易碎、柔软和可变形物体的安全抓取研究
> 机器人平台：Kinova Gen3 单臂 + 两指夹爪 + RGB-D 视觉 + 夹爪触觉
> 项目周期：约两个月

## 项目简介

本项目研究语言条件下易碎/柔软物体的安全抓取问题。核心思路是：**利用接触前视觉和候选动作预测期望触觉及安全接触区间，并在接触后通过校准式评估选择继续闭合、保持、减小夹持或重新规划。**

研究方案详见：`docs/research_proposal.md`

## 项目结构

```
h:\Program\
├── README.md                     # 本文件
├── docs/                         # 项目文档
│   ├── research_proposal.md      # 研究方案（视触 VLA 安全抓取）
│   ├── datasets_verification.md  # 数据集核验报告
│   └── project_structure.md      # 项目结构说明
├── src/                          # 源代码
│   ├── data/                     # 数据加载器
│   │   ├── ycb_video.py          # YCB-Video BOP格式加载器
│   │   ├── dexycb.py             # DexYCB 抓取数据集加载器
│   │   └── rgbd1k.py             # RGBD1K 跟踪数据集加载器
│   ├── models/                   # 模型定义
│   │   ├── pose_estimator.py     # 6D位姿估计模型（ResNet+测地线损失）
│   │   └── tracker.py            # 视频目标跟踪模型
│   ├── trajectory/               # 轨迹与运动估计
│   │   ├── velocity_estimator.py # 速度/加速度估计（有限差分/Kalman/SavGol）
│   │   └── trajectory_smoother.py# 轨迹平滑、插值、异常值检测
│   ├── mujoco_env/               # MuJoCo 仿真环境
│   │   ├── gripper_env.py        # 两指夹爪抓取环境
│   │   ├── dataset_generator.py  # BOP格式数据集生成器
│   │   └── renderer.py           # 离屏渲染工具
│   ├── utils/                    # 工具函数
│   │   ├── metrics.py            # ADD/ADI/投影误差等评估指标
│   │   └── visualization.py      # 位姿/BBox/深度可视化
│   └── tactile_sensor/           # 触觉传感器驱动与数据采集
│       ├── tactile_sensor.py     # 传感器驱动模块
│       ├── collect_data.py       # 数据采集脚本
│       ├── visualize.py          # 可视化脚本
│       └── demo.py               # 模拟演示脚本
├── scripts/                      # 工具脚本
│   ├── check_env.py              # 环境依赖检查
│   ├── download_datasets.ps1     # 数据集下载脚本（25GB精选子集）
│   ├── train_pose.py             # 位姿估计训练脚本
│   ├── eval_pose.py              # 位姿估计评估脚本
│   ├── test_pipeline.py          # 训练管道测试（合成数据）
│   ├── test_trajectory.py        # 轨迹估计模块测试
│   ├── test_metrics.py           # 评估指标模块测试
│   └── test_visualization.py     # 可视化工具测试
├── output/                       # 测试输出
├── data/                         # 实验数据
├── datasets/                     # 公开数据集（待下载）
│   ├── ycb_video/                # YCB-Video 数据集
│   ├── dexycb/                   # DexYCB 数据集
│   └── rgbd1k/                   # RGBD1K 数据集
└── KinovaSlipProject/            # MuJoCo 仿真环境
```

## 当前进度

### 已完成

**核心模块（已测试通过）**
- [x] 触觉传感器驱动开发（支持 0x11/0x12/0x13/0x14 型号）
- [x] 三组触碰实验数据采集框架（横向滑移、纵向滑移、按压）
- [x] MuJoCo 3.11.0 仿真环境 + 夹爪抓取环境
- [x] 6D 位姿评估指标（ADD / ADI / 投影误差）
- [x] 位姿可视化工具（BBox / 3D点投影 / 预测vsGT对比）
- [x] 速度/加速度估计模块（有限差分 / 卡尔曼滤波 / SavGol）
- [x] 轨迹平滑与异常值检测修复
- [x] YCB-Video / DexYCB / RGBD1K 数据加载器
- [x] 位姿估计模型（ResNet + 测地线损失）
- [x] 训练/评估脚本框架

**已验证可运行（纯 numpy/scipy）**
- [x] 位姿评估指标 — 测试通过
- [x] 可视化工具 — 测试通过
- [x] 轨迹/速度/加速度估计 — 测试通过（Kalman滤波速度误差0.065m/s）
- [x] 异常值检测与修复 — 测试通过

### 待本地环境验证（需 PyTorch + MuJoCo）
- [ ] 位姿模型前向传播与训练
- [ ] MuJoCo 数据集生成
- [ ] 数据加载器真实数据测试

### 待开始
- [ ] 数据集下载（25GB精选子集）
- [ ] 纯视觉 baseline 模型训练（YCB-Video）
- [ ] 触觉预测器与安全评估器开发
- [ ] Kinova 机械臂集成与真实实验

## 环境说明

> **当前环境限制**：Windows 安全策略阻止 PyTorch 和 MuJoCo 的 DLL 加载。代码框架已全部写好，需在本地 Python 环境安装依赖后运行。

### 快速环境检查
```bash
python scripts/check_env.py
```

### 依赖安装（本地环境）
```bash
# 核心依赖
pip install numpy scipy opencv-python pillow matplotlib tqdm

# PyTorch（CPU版，用于测试）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# MuJoCo
pip install mujoco
```

## 基座模型

| 模型 | 用途 | 状态 |
|------|------|------|
| pi0.5 / OpenPI | 论文主基座 VLA | 待集成 |
| T-Rex | 快触觉结构参考 | 待参考 |
| SmolVLA | 工程保底方案 | 待评估 |

## 数据集计划

精选 25GB 子集（推荐优先下载）：

| 数据集 | 用途 | 体积 | 下载状态 |
|--------|------|------|----------|
| YCB-Video BOP | 6D位姿、轨迹、速度估计 | ~15-20GB (train_real + test) | 待下载 |
| DexYCB | 真实抓取场景微调 | ~1.4GB (models + calibration) | 待下载 |
| RGBD1K | 通用RGB-D跟踪预训练 | ~3-5GB (测试集+部分训练集) | 待下载 |
| MuJoCo RGB-D | 最终测试集（夹爪相对位姿真值） | 自建 | 代码就绪 |

下载脚本：`scripts/download_datasets.ps1`
详细信息：`docs/datasets_verification.md`

## 测试运行

```bash
# 位姿评估指标测试
python scripts/test_metrics.py

# 轨迹/速度估计测试
python scripts/test_trajectory.py

# 可视化工具测试
python scripts/test_visualization.py

# 环境检查
python scripts/check_env.py
```

## 硬件

- Kinova Gen3 机械臂
- 两指夹爪
- RGB-D 相机
- 消费级指尖指腹触觉传感器（0x11/0x12/0x13/0x14）

## 核心参考论文

- **VTAM** - 易碎物任务、触觉形变预测
- **T-Rex** - 快触觉 residual、时序 encoder
- **DreamTacVLA** - 视觉/动作条件的未来触觉预测
- **OmniVTA** - prediction-residual baseline
- **AT-VLA** - 自适应触觉注入
- **TacVLA** - 接触门控 baseline
- **ForceVLA2** - 语言到物理安全约束
- **OmniVTLA** - 触觉语义与软硬属性

项目文档更新日期：2026-08-19
