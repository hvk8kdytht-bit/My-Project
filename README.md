# RGB-D 6D 位姿估计与滑移检测方案对比

> 基于 YCB-Video 真实数据 + MuJoCo 仿真评估的多方案 baseline 对比项目
> 目标：比较纯视觉方法在 6D 位姿估计、速度估计、接触检测、滑移检测上的表现

## 项目简介

本项目对比了多种纯视觉/视觉+深度方法在抓取场景中的表现，以 MuJoCo 物理引擎输出的真实位姿作为评估标准，为安全抓取任务选择最优感知方案。

### 对比的方案

| 任务 | 方案 |
|------|------|
| **6D 位姿估计** | RGB ResNet18 / RGBD ResNet18 (4通道) |
| **速度估计** | 有限差分 / Savitzky-Golay / Kalman 滤波 / 稠密光流 / 稀疏光流 / CoTracker3 |
| **接触检测** | 力阈值法 / 速度下降法 / 位姿阈值法 |
| **滑移检测** | 位姿偏移法 / 力波动法 / 速度变化法 |

## 评估结果

### 速度估计排名

| 排名 | 方法 | RMSE (m/s) | 类型 |
|------|------|-----------|------|
| 1 | Savitzky-Golay 滤波 | 0.024 | 位姿序列 |
| 2 | 有限差分法 | 0.032 | 位姿序列 |
| 3 | 稠密光流 Farneback | 0.285 | 纯视觉 |
| 4 | 稀疏光流 Lucas-Kanade | 0.285 | 纯视觉 |
| 5 | Kalman 滤波 | 0.289 | 位姿序列 |

### 接触检测排名

| 排名 | 方法 | F1 | 类型 |
|------|------|-----|------|
| 1 | 力阈值法 | 0.989 | 力觉 |
| 2 | 速度下降法 | 0.914 | 纯视觉 |
| 3 | 位姿阈值法 | 0.000 | 纯视觉 |

### 滑移检测排名

| 排名 | 方法 | F1 | 类型 |
|------|------|-----|------|
| 1 | 位姿偏移法 | 0.644 | 位姿序列 |
| 2 | 力波动法 | 0.613 | 力觉 |
| 3 | 速度变化法 | 0.000 | 位姿序列 |

### CoTracker3 测试结果

| 指标 | 结果 |
|------|------|
| 跟踪精度（点在物体内） | 87.0% |
| 平均可见率 | 35.7% |
| CPU 推理速度 | 8.5s / 50帧 |
| 跟踪视频 | `outputs/cotracker_test/videos/` |

## 项目结构

```
h:\Program\
├── src/                           # 源代码
│   ├── data/                      # 数据加载器
│   │   ├── ycb_video.py           # YCB-Video BOP 格式加载器
│   │   ├── dexycb.py              # DexYCB 抓取数据集加载器
│   │   ├── rgbd1k.py             # RGBD1K 跟踪数据集加载器
│   │   └── transforms.py          # 数据增强
│   ├── models/                    # 模型定义
│   │   ├── pose_estimator.py      # 6D 位姿估计 (ResNet18 + 测地线损失)
│   │   └── tracker.py             # 视频目标跟踪
│   ├── trajectory/               # 轨迹与运动估计
│   │   ├── velocity_estimator.py  # 速度/加速度估计 (差分/Kalman/SavGol)
│   │   └── trajectory_smoother.py # 轨迹平滑与异常值检测
│   ├── velocity/                  # 光流速度估计
│   │   ├── optical_flow_estimator.py  # Farneback + Lucas-Kanade
│   │   ├── cotracker_estimator.py     # CoTracker3 速度估计
│   │   └── raft_estimator.py         # RAFT 光流估计
│   ├── contact/                   # 接触与滑移检测
│   │   ├── contact_detector.py    # 接触检测 (力阈值/速度下降/位姿阈值)
│   │   ├── slip_detector.py       # 滑移检测 (位姿偏移/力波动/速度变化)
│   │   └── cotracker_slip_detector.py # CoTracker3 滑移检测
│   ├── mujoco_env/               # MuJoCo 仿真环境
│   │   ├── gripper_env.py        # 两指夹爪抓取环境
│   │   ├── dataset_generator.py  # BOP 格式数据集生成器
│   │   ├── eval_dataset_generator.py # 评估集生成器
│   │   └── renderer.py           # 离屏渲染工具
│   └── utils/                    # 工具函数
│       ├── metrics.py             # ADD/ADI/投影误差
│       ├── visualization.py      # 位姿可视化
│       └── mask_utils.py          # Mask 工具
├── scripts/                      # 脚本
│   ├── train_pose.py             # 位姿模型训练
│   ├── eval_pose_real.py         # 真实数据评估
│   ├── eval_optical_flow.py      # 光流法评估
│   ├── test_cotracker.py         # CoTracker3 测试
│   ├── full_evaluation.py        # 全方案评估
│   ├── generate_final_report.py  # 报告生成
│   ├── check_env.py              # 环境检查
│   └── download_datasets.ps1     # 数据集下载
├── docs/                         # 文档
│   ├── research_proposal.md      # 研究方案
│   ├── datasets_verification.md  # 数据集核验
│   └── getting_started.md        # 快速开始
├── requirements.txt              # Python 依赖
└── .gitignore
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- PyTorch 2.0+
- MuJoCo 3.0+
- OpenCV 4.0+

### 2. 安装依赖

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 检查环境

```bash
python scripts/check_env.py
```

### 4. 下载数据集（约 25GB）

```powershell
# Windows
.\scripts\download_datasets.ps1
```

或手动下载：
- YCB-Video: https://huggingface.co/datasets/bop-benchmark/ycbv
- DexYCB: https://dex-ycb.github.io/
- RGBD1K: https://github.com/facebookresearch/rgbd1k

### 5. 运行评估

```bash
# 全方案评估
python scripts/full_evaluation.py

# CoTracker3 测试
python scripts/test_cotracker.py --checkpoint checkpoints/cotracker3_offline.pth

# 位姿模型评估
python scripts/eval_pose_real.py --checkpoint outputs/baseline_rgb/checkpoints/best.pth
```

## 数据集

| 数据集 | 来源 | 用途 |
|--------|------|------|
| YCB-Video | BOP Benchmark | 6D 位姿估计训练与评估 |
| DexYCB | 官网 | 真实抓取场景 |
| RGBD1K | GitHub | RGB-D 跟踪 |
| MuJoCo 评估集 | 本项目生成 | 速度/接触/滑移评估真值 |

## 硬件

- CPU: Intel Core Ultra 9 285H (16核)
- GPU: NVIDIA RTX 5060 Laptop (4GB)
- RAM: 32 GB

## License

MIT License - 详见 [LICENSE](LICENSE)
