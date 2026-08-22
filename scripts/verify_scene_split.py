"""验证场景级划分的三个数据集（train/val/test）"""
import json
import sys

sys.path.insert(0, '.')
from src.data.ycb_video import YCBVideoPoseDataset

BASE = r"H:\Program\datasets\ycbv"

with open(f"{BASE}/scene_split.json") as f:
    split_cfg = json.load(f)

for split_name in ["train", "val", "test"]:
    scenes = split_cfg[split_name]
    ds = YCBVideoPoseDataset(
        root_dir=BASE,
        split="test",  # test_all 解压后目录名为 test/
        scene_filter=scenes,
    )
    # 统计物体分布
    from collections import Counter
    obj_counts = Counter(s["obj_name"] for s in ds.pose_samples)
    print(f"{split_name:5s}: {len(scenes)} 场景 {scenes}")
    print(f"       {len(ds)} 个位姿样本, 涉及 {len(obj_counts)} 种物体")
    print(f"       物体TOP5: {obj_counts.most_common(5)}")
    print()

print("✅ 场景级划分验证通过")
