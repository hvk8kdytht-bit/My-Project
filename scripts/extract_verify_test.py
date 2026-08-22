"""
解压 ycbv_test_all.zip 并验证 BOP 测试集结构 + 数据加载器冒烟测试
"""
import os
import sys
import json
import zipfile
import time
import numpy as np

sys.path.insert(0, '.')

BASE = r"H:\Program\datasets\ycbv"
ZIP_PATH = os.path.join(BASE, "ycbv_test_all.zip")

# ========== 1. 解压 ==========
print("=== 1. 解压 ycbv_test_all.zip ===")
if os.path.exists(os.path.join(BASE, "test")):
    print("⏭️  test/ 已存在，跳过解压")
elif not os.path.exists(ZIP_PATH):
    print(f"❌ 未找到 {ZIP_PATH}")
    sys.exit(1)
else:
    print(f"大小: {os.path.getsize(ZIP_PATH)/1024/1024:.0f} MB")
    start = time.time()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        print(f"包含 {len(names)} 个文件")
        zf.extractall(BASE)
    print(f"✅ 解压完成，耗时 {time.time()-start:.0f}s")
print()

# ========== 2. 结构验证 ==========
print("=== 2. 测试集结构验证 ===")
test_dir = os.path.join(BASE, "test")
if not os.path.exists(test_dir):
    # 有的压缩包顶层就是 test_all/，找一下
    for d in os.listdir(BASE):
        if "test" in d.lower() and os.path.isdir(os.path.join(BASE, d)):
            test_dir = os.path.join(BASE, d)
            break

if not os.path.exists(test_dir):
    print(f"❌ 未找到 test 目录，BASE 下有: {os.listdir(BASE)}")
    sys.exit(1)

scenes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
print(f"场景数: {len(scenes)}")
print(f"场景列表: {scenes[:5]} ... {scenes[-2:]}")

# 检查第一个场景的文件结构
scene0 = os.path.join(test_dir, scenes[0])
expected = ["rgb", "depth", "mask_visib", "scene_gt.json", "scene_gt_info.json", "scene_camera.json"]
for e in expected:
    p = os.path.join(scene0, e)
    print(f"  {'✅' if os.path.exists(p) else '❌'} {e}")
print()

# ========== 3. 数据内容抽查 ==========
print("=== 3. 数据内容抽查 ===")
with open(os.path.join(scene0, "scene_gt.json")) as f:
    scene_gt = json.load(f)
with open(os.path.join(scene0, "scene_camera.json")) as f:
    scene_camera = json.load(f)

img_ids = sorted(scene_gt.keys())
print(f"场景 {scenes[0]}: {len(img_ids)} 帧标注")

first = img_ids[0]
anns = scene_gt[first]
print(f"帧 {first}: {len(anns)} 个物体实例")
for i, ann in enumerate(anns[:3]):
    t = ann["cam_t_m2c"]
    print(f"  实例{i}: obj_id={ann['obj_id']}, t=[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]mm")

cam = scene_camera[first]
K = np.array(cam["cam_K"]).reshape(3, 3)
print(f"  cam_K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}, depth_scale={cam.get('depth_scale', 'N/A')}")

# 图像数量统计
rgb_dir = os.path.join(scene0, "rgb")
n_rgb = len(os.listdir(rgb_dir))
depth_dir = os.path.join(scene0, "depth")
n_depth = len(os.listdir(depth_dir)) if os.path.exists(depth_dir) else 0
print(f"  RGB帧数: {n_rgb}, 深度帧数: {n_depth}")
print()

# ========== 4. 数据加载器冒烟测试 ==========
print("=== 4. 数据加载器冒烟测试 (YCBVideoPoseDataset) ===")
import torch
from src.data.ycb_video import YCBVideoPoseDataset, YCBVideoDataset

try:
    ds = YCBVideoPoseDataset(
        root_dir=BASE,
        split="test",
        load_depth=True,
    )
    print(f"✅ 数据集加载成功: {len(ds)} 个位姿样本")

    # 取第一个样本检查
    sample = ds[0]
    print(f"   keys: {list(sample.keys())}")
    print(f"   物体: {sample['obj_name']} (id={sample['obj_id']})")
    print(f"   平移(米): {sample['translation'].numpy().round(4)}")
    print(f"   内参 fx: {sample['camera_intrinsics'][0,0]:.1f}")
    if 'depth' in sample:
        d = sample['depth']
        valid = d[d > 0]
        print(f"   深度范围: {valid.min():.3f} ~ {valid.max():.3f} m ({len(valid)} 有效像素)")
    if 'mask' in sample:
        print(f"   掩码正像素: {int(sample['mask'].sum())}")

    # 抽查中间样本
    mid = ds[len(ds)//2]
    print(f"   中间样本: {mid['obj_name']}, t_z={mid['translation'][2]:.3f}m")

    print()
    print("🎉 测试集解压 + 加载器全链路验证通过!")
except Exception as e:
    import traceback
    print(f"❌ 加载器测试失败: {e}")
    traceback.print_exc()
    sys.exit(1)
