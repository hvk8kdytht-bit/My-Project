# 本地快速开始指南

> 本文档帮助你在本地 Python 环境中快速搭建项目并跑通完整流程。

## ⚡ 环境已就绪（2026-08-19 配置完成）

虚拟环境已创建在 **`H:\Program\venv`**（基于系统独立 Python 3.14.6，不受 TRAE 沙箱 WDAC 策略影响），所有依赖已安装并测试通过：

| 包 | 版本 |
|---|---|
| torch / torchvision | 2.13.0+cpu / 0.28.0+cpu |
| mujoco | 3.11.0（与 KinovaSlipProject 捆绑版一致） |
| numpy / scipy | 2.5.2 / 1.18.0 |
| opencv-python / Pillow / matplotlib | 5.0.0 / 12.3.0 / 3.11.1 |
| huggingface_hub | 1.28.0 |

**使用方式**（在 H:\Program 目录下）：

```powershell
# 激活环境
H:\Program\venv\Scripts\Activate.ps1

# 或直接用完整路径运行（无需激活）
H:\Program\venv\Scripts\python.exe scripts\test_pipeline.py
```

全部 5 个测试套件已在该环境验证通过（metrics / trajectory / visualization / contact / pipeline）。

## 环境要求

- Python 3.9+
- Windows / Linux / macOS
- 推荐 GPU（可选，CPU 也可运行但较慢）

## 第一步：创建独立 Python 环境

```bash
# 使用 conda（推荐）
conda create -n grasp_vla python=3.10
conda activate grasp_vla

# 或使用 venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
```

## 第二步：安装依赖

```bash
cd h:\Program  # 或你的项目路径

# 核心依赖（必需）
pip install numpy scipy opencv-python pillow matplotlib tqdm pyyaml

# PyTorch（必需，用于模型训练）
# CPU 版（推荐先安装验证）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 版（根据你的CUDA版本选择，参考 pytorch.org）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MuJoCo（必需，用于最终评估和合成数据集生成）
pip install mujoco

# HuggingFace Hub（用于下载数据集）
pip install huggingface_hub
```

### 验证安装

```bash
python scripts/check_env.py
```

预期输出（所有核心包显示 ✅）：
```
=== 核心依赖检查 ===
  numpy                   ✅         1.x.x
  scipy                   ✅         1.x.x
  torch                   ✅         2.x.x
  mujoco                  ✅         3.x.x
  ...
```

## 第三步：下载数据集（约 25GB）

### 方式 A：使用 HuggingFace CLI（推荐）

```bash
# 安装 CLI 工具
pip install -U "huggingface_hub[cli]"

# 下载 YCB-Video BOP 数据集（主数据集，约 15-20GB）
huggingface-cli download bop-benchmark/ycbv --local-dir datasets/ycbv --repo-type dataset --include "ycbv_base.zip" "ycbv_models.zip" "ycbv_test_all.zip" "ycbv_train_real.zip"

# 下载 DexYCB（models + calibration，约 1.4GB）
# 注意：DexYCB 需要单独下载，不在 BOP HuggingFace 上
# 访问: https://dex-ycb.github.io/

# 下载 RGBD1K（测试集，约 3-5GB）
# 访问: https://github.com/facebookresearch/rgbd1k
```

### 方式 B：使用项目脚本

```powershell
# PowerShell
scripts\download_datasets.ps1
```

### 方式 C：手动下载

如果 HuggingFace 连接慢，可以使用镜像站或迅雷下载：

| 数据集 | 下载地址 | 大小 |
|--------|----------|------|
| YCB-Video BOP base | https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_base.zip | ~1MB |
| YCB-Video BOP models | https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_models.zip | ~100MB |
| YCB-Video BOP test | https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_all.zip | ~2GB |
| YCB-Video BOP train_real | https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_train_real.zip | ~15GB |

下载后解压到 `datasets/ycbv/` 目录。

### 验证数据集

```bash
python scripts/verify_data.py
```

## 第四步：跑通测试

### 1. 基础模块测试（不依赖 PyTorch）

```bash
# 位姿评估指标测试
python scripts/test_metrics.py

# 轨迹/速度估计测试
python scripts/test_trajectory.py

# 可视化工具测试
python scripts/test_visualization.py

# 接触/滑移检测测试
python scripts/test_contact.py
```

### 2. 训练管道测试（需要 PyTorch）

```bash
python scripts/test_pipeline.py
```

预期输出：
```
=== 训练管道测试 ===
1. 测试数据加载...
   数据集大小: 100
   ✅ 通过

2. 测试模型前向传播...
   输出形状: torch.Size([4, 3, 4])
   ✅ 通过

3. 测试损失函数...
   损失: 0.xxx
   ✅ 通过

4. 测试反向传播...
   ✅ 通过

5. 测试模型保存/加载...
   ✅ 通过

✅ 训练管道测试全部通过!
```

### 3. MuJoCo 环境测试（需要 mujoco）

```bash
python -c "
import mujoco
print(f'MuJoCo 版本: {mujoco.__version__}')
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name=\"box\" type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>')
print('✅ MuJoCo 工作正常')
"
```

## 第五步：开始训练

### 纯视觉位姿估计 baseline

```bash
# 使用合成数据快速验证
python scripts/train_pose.py --dataset synthetic --epochs 10

# 使用 YCB-Video 真实数据
python scripts/train_pose.py \
    --dataset ycbv \
    --data_path datasets/ycbv \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --output_dir outputs/pose_baseline
```

### 评估模型

```bash
python scripts/eval_pose.py \
    --model_path outputs/pose_baseline/best_model.pth \
    --dataset ycbv \
    --data_path datasets/ycbv \
    --split test
```

## 第六步：生成 MuJoCo 测试集（最终评估）

```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.mujoco_env.dataset_generator import GraspDatasetGenerator

gen = GraspDatasetGenerator(
    output_dir='datasets/mujoco_test',
    num_scenes=10,
    steps_per_scene=200,
)
gen.generate('test')
print('✅ MuJoCo 测试集生成完成')
"
```

## 常见问题

### Q: PyTorch 安装失败（文件名太长）

**A**: 使用 `--target` 安装到短路径目录：

```bash
pip install torch torchvision --target C:\torch_pkg --index-url https://download.pytorch.org/whl/cpu
# 然后在代码开头添加
import sys
sys.path.insert(0, 'C:\\torch_pkg')
```

### Q: MuJoCo 渲染失败（GLFW 错误）

**A**: 确保使用离屏渲染模式，或安装 OpenGL 驱动：

```bash
# Windows: 安装最新显卡驱动
# Linux: apt install libgl1-mesa-glx libglfw3
```

### Q: 数据集下载太慢

**A**: 使用镜像站或下载工具：

- HuggingFace 镜像: `HF_ENDPOINT=https://hf-mirror.com`
- 设置环境变量后再下载：
  ```bash
  set HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download ...
  ```

### Q: 显存不足

**A**: 减小 batch size 或图像分辨率：

```bash
python scripts/train_pose.py --batch_size 4 --img_size 224
```

## 项目结构速查

```
src/
├── data/              # 数据加载器
├── models/            # 模型定义
├── trajectory/        # 轨迹/速度/加速度估计
├── contact/           # 接触/滑移检测
├── mujoco_env/        # MuJoCo 仿真环境
└── utils/             # 评估指标 + 可视化

scripts/
├── check_env.py       # 环境检查
├── test_*.py          # 各模块测试
├── train_pose.py      # 位姿训练
└── eval_pose.py       # 位姿评估
```

## 下一步

- [ ] 跑通所有测试
- [ ] 训练 YCB-Video 位姿 baseline
- [ ] 生成 MuJoCo 测试集
- [ ] 对比不同方案的位姿估计精度
- [ ] 接入触觉预测模块
