"""调试 mask 生成"""
import sys
sys.path.insert(0, "H:/Program")

from src.utils.mask_utils import get_object_model_size
from pathlib import Path
import json

models_dir = Path("datasets/ycbv/models")
info_path = models_dir / "models_info.json"
if info_path.exists():
    with open(info_path) as f:
        info = json.load(f)
    for obj_id in [1, 2, 3, 4, 5]:
        key = str(obj_id)
        if key in info:
            s = info[key].get("size", "N/A")
            d = info[key].get("diameter", "N/A")
            print(f"物体 {obj_id}: size={s} mm, diameter={d} mm")

print("\n--- 从函数读取 ---")
for obj_id in [2, 3]:
    size, diameter = get_object_model_size(str(models_dir), obj_id)
    print(f"物体 {obj_id}: size_mm={size*1000}, diameter_mm={diameter*1000}")

# 检查第一个场景的位姿
from scripts.eval_cotracker import load_scene_data
test_dir = Path("datasets/ycbv/test")
scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
scene_dir = test_dir / scenes[-1]
data = load_scene_data(str(scene_dir), max_frames=5)
print(f"\n场景 {scenes[-1]}:")
print(f"  物体 ID: {data['obj_id']}")
pose = data["poses"][0]
print(f"  平移 t (mm): {pose['t']}")
print(f"  深度 z (mm): {pose['t'][2]}")
print(f"  相机 K:\n{data['camera_k']}")
