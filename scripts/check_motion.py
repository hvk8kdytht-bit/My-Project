import json
import numpy as np
from pathlib import Path

for scene in ['000058', '000059']:
    gt_path = Path(f'h:/Program/datasets/ycbv/test/{scene}/scene_gt.json')
    if not gt_path.exists():
        print(f'{scene}: scene_gt not found')
        continue
    with open(gt_path) as f:
        gt = json.load(f)

    keys = sorted(gt.keys())[:60]
    first_obj = gt[keys[0]][0]
    obj_id = first_obj['obj_id']

    translations = []
    for k in keys:
        for item in gt[k]:
            if item['obj_id'] == obj_id:
                translations.append(item['cam_t_m2c'])
                break

    translations = translations[:60]
    if len(translations) < 2:
        print(f'{scene}: too few frames')
        continue

    trans = np.array(translations) / 1000.0
    total_disp = np.linalg.norm(trans[-1] - trans[0])
    max_disp = 0
    for i in range(1, len(trans)):
        d = np.linalg.norm(trans[i] - trans[i-1])
        if d > max_disp:
            max_disp = d

    moving = "YES" if total_disp > 5 else "NO (static)"
    print(f"Scene {scene} (obj_id={obj_id}):")
    print(f"  Frames: {len(trans)}")
    print(f"  Start pos: ({trans[0][0]:.3f}, {trans[0][1]:.3f}, {trans[0][2]:.3f}) m")
    print(f"  End pos:   ({trans[-1][0]:.3f}, {trans[-1][1]:.3f}, {trans[-1][2]:.3f}) m")
    print(f"  Total displacement: {total_disp*1000:.1f} mm")
    print(f"  Max frame-to-frame: {max_disp*1000:.1f} mm")
    print(f"  Moving: {moving}")
    print()
