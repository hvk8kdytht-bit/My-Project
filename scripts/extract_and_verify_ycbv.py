"""解压并验证 YCB-Video base + models 数据结构"""
import os
import sys
import zipfile

sys.path.insert(0, '.')

BASE_DIR = r"H:\Program\datasets\ycbv"

for zip_name in ["ycbv_base.zip", "ycbv_models.zip"]:
    zip_path = os.path.join(BASE_DIR, zip_name)
    if not os.path.exists(zip_path):
        print(f"❌ 未找到 {zip_path}")
        continue

    print(f"--- 解压 {zip_name} ---")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        print(f"   包含 {len(names)} 个文件")
        # 显示顶层结构
        top_dirs = set()
        for n in names:
            parts = n.split('/')
            if len(parts) > 1 and parts[0]:
                top_dirs.add(parts[0])
        print(f"   顶层目录: {sorted(top_dirs)[:30]}")

        zf.extractall(BASE_DIR)
    print(f"   ✅ 解压完成")
    print()

print("=== 验证目录结构 ===")
for root, dirs, files in os.walk(BASE_DIR):
    depth = root.replace(BASE_DIR, '').count(os.sep)
    if depth > 2:
        dirs[:] = []
        continue
    indent = '  ' * depth
    print(f"{indent}{os.path.basename(root)}/  ({len(files)} 个文件)")
    if depth >= 1 and len(dirs) > 20:
        print(f"{indent}  ... 共 {len(dirs)} 个子目录")
        dirs[:] = dirs[:5]

print()
print("=== 检查关键文件 ===")
key_files = [
    "models/models_info.json",       # 模型信息（直径等）
    "models/obj_000001.ply",         # 第一个物体模型
    "models/obj_000001_points.npy",  # 第一个物体点云
    "ycbv_bop19_rgb_test_targets.csv",
    "camera_params.json",
]
for kf in key_files:
    full = os.path.join(BASE_DIR, kf)
    exists = os.path.exists(full)
    if exists:
        size = os.path.getsize(full)
        print(f"  {'✅' if exists else '❌'} {kf}  ({size/1024:.1f} KB)")
    else:
        print(f"  ❌ {kf}")

# 读取 models_info.json 检查内容
info_path = os.path.join(BASE_DIR, "models", "models_info.json")
if os.path.exists(info_path):
    import json
    with open(info_path) as f:
        info = json.load(f)
    print()
    print(f"=== models_info.json 内容 ===")
    print(f"物体数量: {len(info)}")
    for obj_id, obj_info in sorted(info.items(), key=lambda x: int(x[0]))[:5]:
        diameter_mm = obj_info.get('diameter', 0)
        print(f"  obj_{int(obj_id):06d}: 直径 {diameter_mm:.1f}mm = {diameter_mm/10:.1f}cm")
    print(f"  ... 共 {len(info)} 个物体")

print()
print("✅ 数据结构验证完成")
