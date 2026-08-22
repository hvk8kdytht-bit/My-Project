# 数据集核验报告

> 核验日期：2026-08-18
> 核验对象：YCB-Video、DexYCB、RGBD1K

## 总览

| 数据集 | 主要用途 | 数据规模 | 预估体积 | 官方入口 | 下载方式 |
|--------|----------|----------|----------|----------|----------|
| YCB-Video | 6D 位姿、轨迹、速度、加速度估计 | 92 视频 / 133,827 帧 / 21 物体 | ~265GB（原版）/ ~100GB（BOP版） | 华盛顿大学 PoseCNN 项目页 | Box / HuggingFace |
| DexYCB | 真实抓取场景微调 | 58.2万 RGB-D 帧 / 1000 序列 / 20 物体 | ~119GB | NVIDIA 官方项目页 | Google Drive |
| RGBD1K | 通用 RGB-D 跟踪预训练 | 1,050 序列 / ~250 万帧 | 待确认（仅标注帧已上传） | GitHub 官方仓库 | 百度网盘 / Google Drive |

---

## 1. YCB-Video 数据集

### 基本信息

- **发布方**：华盛顿大学 Yu Xiang 等（PoseCNN 论文）
- **发表会议**：RSS 2018
- **数据内容**：
  - 92 个视频序列
  - 133,827 帧真实 RGB-D 数据
  - 21 个 YCB 日常物体
  - 逐帧 6D 物体位姿标注
  - 深度图、相机内参
  - 另外提供约 80,000 帧合成数据作为训练集扩展

### 官方入口

| 资源 | 链接 | 大小 |
|------|------|------|
| 项目主页 | https://rse-lab.cs.washington.edu/projects/posecnn/ | - |
| 完整数据集（原版） | https://utdallas.box.com/s/r5sx2ghgn62bg1tgjp9ily6jx2fifahl | ~265 GB |
| 3D 模型 | https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/ | ~367 MB |
| 工具箱（GitHub） | https://github.com/yuxng/YCB_Video_toolbox | - |

### BOP 版本（推荐下载）

YCB-Video 也是 BOP (Benchmark for 6D Object Pose Estimation) 的核心数据集之一，BOP 版做了格式统一和筛选，体积更小，约 100GB 左右。

| 资源 | 链接 |
|------|------|
| HuggingFace 数据集页 | https://huggingface.co/datasets/bop-benchmark/ycbv |
| ycbv_base.zip | `https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_base.zip` |
| ycbv_models.zip | `https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_models.zip` |
| ycbv_test_bop19.zip | `https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip` |

下载方法（使用 huggingface_hub CLI）：
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download bop-benchmark/ycbv --local-dir ./ycbv --repo-type dataset
```

### 数据结构

原版目录结构：
```
YCB_Video_Dataset/
├── data/                    # RGB 图像 + 深度图像序列
├── data_syn/                # 合成数据（可选）
├── models/                  # 21 个物体的 3D 模型（.ply）
├── */*-meta/                # 每帧元数据（位姿、内参等 .mat 文件）
├── */*-label/               # 每帧标注和分割掩码
├── image_sets/              # 训练/测试划分（train.txt, val.txt, keyframe.txt）
└── poses/                   # 所有物体 6D 位姿（四元数+平移）
```

BOP 版目录结构：
```
ycbv/
├── train_real/              # 真实训练图像
├── train_pbr/               # PBR 合成训练图像
├── test/                    # 测试集
├── models/                  # 3D 模型
└── camera.json              # 相机参数
```

### 下载建议

- **首选 BOP 版本**：体积更小（~100GB vs ~265GB）、格式标准、社区支持好
- 如果需要完整 92 个视频的所有帧（用于轨迹学习），再下载原版
- 3D 模型文件很小（367MB），可以先下载

---

## 2. DexYCB 数据集

### 基本信息

- **发布方**：NVIDIA Research（Yu-Wei Chao 等）
- **发表会议**：CVPR 2021
- **数据内容**：
  - 58.2 万 RGB-D 帧
  - 1,000 个抓取序列
  - 10 名受试者
  - 20 个 YCB 物体
  - 8 个相机视角
  - 标注：2D 物体检测、关键点、6D 物体位姿、3D 手部姿态（MANO 参数）
- **许可证**：CC BY-NC 4.0

### 官方入口

| 资源 | 链接 | 大小 |
|------|------|------|
| 项目主页 | https://dex-ycb.github.io/ | - |
| 完整数据集（单文件） | https://drive.google.com/file/d/18fD8RtWJM_fBi3bsuQTzdjIKxdRYtl57 | 119 GB |
| 工具箱（GitHub） | https://github.com/NVlabs/dex-ycb-toolkit | - |

### 分卷下载（10 个受试者 + 补充）

每个受试者约 12GB，可分批下载：

| 文件 | 大小 |
|------|------|
| 20200709-subject-01.tar.gz | 12 GB |
| 20200813-subject-02.tar.gz | 12 GB |
| 20200820-subject-03.tar.gz | 12 GB |
| 20200903-subject-04.tar.gz | 12 GB |
| 20200908-subject-05.tar.gz | 12 GB |
| 20200918-subject-06.tar.gz | 12 GB |
| 20200928-subject-07.tar.gz | 12 GB |
| 20201002-subject-08.tar.gz | 12 GB |
| 20201015-subject-09.tar.gz | 12 GB |
| 20201022-subject-10.tar.gz | 12 GB |
| bop.tar.gz | 1.2 GB |
| calibration.tar.gz | 16 KB |
| models.tar.gz | 1.4 GB |

合计：10 × 12GB + 1.2GB + 1.4GB ≈ **122.6 GB**

### 数据结构

```
dex-ycb/
├── 20200709-subject-01/
│   ├── <sequence_id>/
│   │   ├── color/          # 彩色图像
│   │   ├── depth/          # 深度图像
│   │   ├── label/          # 标注文件
│   │   └── ...
│   └── ...
├── 20200813-subject-02/
│   └── ...
├── calibration/             # 相机校准数据
└── models/                  # 物体 3D 模型
```

### 下载建议

- 可先下载 2-3 个受试者做小规模测试
- models.tar.gz 和 calibration.tar.gz 很小，先下载
- Google Drive 大文件下载可能需要 gdown 或 aria2 等工具

---

## 3. RGBD1K 数据集

### 基本信息

- **发布方**：江南大学吴小俊团队（Xue-Feng Zhu 等）
- **发表会议**：AAAI 2023
- **数据内容**：
  - 1,050 个 RGB-D 视频序列
  - 约 250 万总帧数
  - 717,900 帧有标注
  - 1,000 个训练序列 + 50 个测试序列
  - 15 种场景属性
  - 长期跟踪（LT）基准
- **许可证**：CC BY-NC-SA 4.0

### 官方入口

| 资源 | 链接 |
|------|------|
| GitHub 仓库 | https://github.com/xuefeng-zhu5/RGBD1K |
| 百度网盘 | https://pan.baidu.com/s/1JD4RdgCLzWw7GTtAK-7MlA?pwd=xnrs （提取码: xnrs） |
| Google Drive | https://drive.google.com/drive/folders/1Z2PnWEgdZG0KVI2MX5chWddNlbuuEug3 |

### 体积说明

官方页面未明确标注总文件体积。根据数据规模估算：
- 1,050 个序列 × 平均 2,384 帧 = 250 万帧
- 目前仅上传了**有标注的 RGB-D 帧**（约 71.8 万帧）
- 未标注帧后续会发布

预估标注部分体积约 **50-100 GB**（取决于图像分辨率和压缩率），完整数据可能翻倍。

### 数据结构

```
RGBD1K/
├── train/
│   ├── <video_id>/
│   │   ├── rgb/            # RGB 图像
│   │   ├── depth/          # 深度图像
│   │   └── groundtruth.txt # 跟踪标注
│   └── ...
└── test/
    ├── <video_id>/
    │   ├── rgb/
    │   ├── depth/
    │   └── groundtruth.txt
    └── ...
```

### 下载建议

- 国内用户优先使用百度网盘
- 可先下载测试集（50 个序列）做数据格式验证
- 注意：该数据集只有跟踪标注（bbox），没有 6D 物体位姿，**不能作为精确加速度真值来源**

---

## 下载优先级与存储规划

### 磁盘空间现状（2026-08-18 核验）

| 盘符 | 已用 | 可用 | 总容量 |
|------|------|------|--------|
| H: | 0.61 GB | **149.48 GB** | ~150 GB |
| D: | 24.06 GB | **226.14 GB** | ~250 GB |

### 存储位置分配建议

H 盘只有 ~150GB，放不下全部数据集（预估 270-320GB），建议分配如下：

| 数据集 | 建议存储位置 | 预估体积 | 占可用空间比例 |
|--------|-------------|----------|---------------|
| YCB-Video (BOP版) | `H:\Program\datasets\ycb_video\` | ~100 GB | H盘 ~67% |
| DexYCB | `D:\datasets\dexycb\` | ~119 GB | D盘 ~53% |
| RGBD1K | `H:\Program\datasets\rgbd1k\` | ~50 GB（标注部分） | H盘 ~33% |
| **合计** | | **~269 GB** | |

> 注意：项目代码、MuJoCo 仿真环境、模型权重等仍放在 `H:\Program\` 下。
> 如果 H 盘空间紧张，RGBD1K 也可以放到 D 盘。

### 下载脚本

已提供 PowerShell 下载脚本：`scripts/download_datasets.ps1`

使用方法：
```powershell
cd H:\Program
.\scripts\download_datasets.ps1
```

脚本会显示当前磁盘空间，并按批次引导下载。

### 总预估体积
| 数据集 | 预估体积 |
|--------|----------|
| YCB-Video (BOP版) | ~100 GB |
| DexYCB | ~119 GB |
| RGBD1K（标注部分） | ~50-100 GB |
| **合计** | **~270-320 GB** |

### 下载顺序（建议）

| 阶段 | 内容 | 预计体积 | 目的 |
|------|------|----------|------|
| 第一批 | YCB-Video 3D模型 + DexYCB models+calibration + RGBD1K 测试集 | < 5 GB | 验证数据格式、编写预处理代码 |
| 第二批 | RGBD1K 训练集 | ~50-100 GB | 训练通用 RGB-D 跟踪能力 |
| 第三批 | YCB-Video (BOP版) | ~100 GB | 训练 6D 位姿/轨迹/速度估计 |
| 第四批 | DexYCB 全部受试者 | ~119 GB | 抓取场景微调 |

### 注意事项

1. **H 盘可用空间检查**：下载前确认 H 盘剩余空间 > 500GB（数据集 + 模型 + 缓存）
2. **避免下载到 C 盘**：确保所有下载路径指向 H 盘
3. **校验文件完整性**：大文件下载后建议校验 MD5/SHA
4. **磁盘空间不足时**：优先保证 YCB-Video（主数据集），RGBD1K 可只下载标注帧

---

## 暂不纳入的数据集

| 数据集 | 排除原因 |
|--------|----------|
| ContactPose | 连续运动信息不足，数据规模大 |
| GraspNet | 主要是静态抓取场景，不适合学习速度和加速度 |
| SynPick / Kubric / MOVi | 仿真或合成数据，优先使用真实数据 |
