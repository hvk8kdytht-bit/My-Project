"""扫描所有场景，找出物体运动最大的场景"""
import json
import numpy as np
from pathlib import Path

test_dir = Path("h:/Program/datasets/ycbv/test")
scenes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

results = []

for scene in scenes:
    gt_path = test_dir / scene / "scene_gt.json"
    if not gt_path.exists():
        continue
    with open(gt_path) as f:
        gt = json.load(f)

    keys = sorted(gt.keys())
    if len(keys) < 10:
        continue

    # 获取物体ID
    obj_id = gt[keys[0]][0]["obj_id"]

    translations = []
    for k in keys:
        for item in gt[k]:
            if item["obj_id"] == obj_id:
                translations.append(item["cam_t_m2c"])
                break

    if len(translations) < 10:
        continue

    trans = np.array(translations) / 1000.0
    total_disp = np.linalg.norm(trans[-1] - trans[0])

    # 计算平均速度
    frame_disps = []
    for i in range(1, len(trans)):
        d = np.linalg.norm(trans[i] - trans[i-1])
        frame_disps.append(d)
    avg_speed = np.mean(frame_disps) if frame_disps else 0
    max_speed = max(frame_disps) if frame_disps else 0

    results.append({
        "scene": scene,
        "obj_id": obj_id,
        "frames": len(trans),
        "total_disp_mm": total_disp * 1000,
        "avg_speed_mms": avg_speed * 1000,
        "max_speed_mms": max_speed * 1000,
    })

# 按总位移排序
results.sort(key=lambda x: x["total_disp_mm"], reverse=True)

print(f"{'Scene':<12} {'ObjID':<8} {'Frames':<8} {'Total(mm)':<12} {'AvgSpeed(mm/s)':<16} {'MaxSpeed(mm/s)':<16}")
print("-" * 80)
for r in results[:15]:
    print(f"{r['scene']:<12} {r['obj_id']:<8} {r['frames']:<8} {r['total_disp_mm']:<12.1f} {r['avg_speed_mms']:<16.1f} {r['max_speed_mms']:<16.1f}")

print(f"\nTop 5 most dynamic scenes:")
for i, r in enumerate(results[:5]):
    print(f"  {i+1}. Scene {r['scene']} - Total disp: {r['total_disp_mm']:.1f}mm, Avg speed: {r['avg_speed_mms']:.1f}mm/s")
