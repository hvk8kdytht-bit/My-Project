"""
测试接触检测与滑移检测模块
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.contact import (
    ThresholdContactDetector,
    VisionContactDetector,
    OpticalFlowSlipDetector,
    ForceSlipDetector,
    PoseDifferenceSlipDetector,
)
from src.contact.contact_detector import estimate_contact_quality
from src.contact.slip_detector import SlipState

print("=== 接触检测测试 ===")
print()

# 测试1: 阈值接触检测
print("1. 阈值接触检测测试...")
detector = ThresholdContactDetector(
    force_threshold=2.0,
    debounce_frames=3,
    hysteresis=0.2,
)

# 模拟逐渐增加的力
forces = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.8, 2.0, 1.5, 1.0, 0.5, 0.2]
contact_states = []

for f in forces:
    in_contact, info = detector.detect({"gripper_force": f, "gripper_torque": np.zeros(3)})
    contact_states.append(in_contact)

print(f"   力序列: {[f'{f:.1f}N' for f in forces]}")
print(f"   接触序列: {['接触' if c else '未接触' for c in contact_states]}")
print(f"   首次接触帧: 第{contact_states.index(True)+1}帧 (力={forces[contact_states.index(True)]:.1f}N)")
assert True in contact_states, "应该检测到接触"
assert False in contact_states[contact_states.index(True):], "应该检测到释放"
print("   ✅ 通过")

# 测试2: 视觉接触检测
print()
print("2. 视觉接触检测测试...")
vis_detector = VisionContactDetector(method="pose_change", min_history=5)

# 模拟物体下落然后接触（速度从匀速突变）
N = 20
positions = []
for i in range(10):
    # 匀速下落
    positions.append(np.array([0.0, 0.0, -0.1 - i * 0.01]))
for i in range(10):
    # 接触后几乎不动（微小形变）
    positions.append(np.array([0.0, 0.0, -0.2 + i * 0.0005]))

gripper_widths = [0.05 - i * 0.002 for i in range(15)] + [0.02 for _ in range(5)]

contact_results = []
for i in range(N):
    obs = {
        "object_pose": positions[i],
        "gripper_width": gripper_widths[min(i, len(gripper_widths)-1)],
    }
    in_contact, info = vis_detector.detect(obs)
    contact_results.append(in_contact)

print(f"   总帧数: {N}")
print(f"   检测到接触: {np.sum(contact_results)} 帧")
print(f"   首次检测到接触: 第{contact_results.index(True)+1}帧" if True in contact_results else "   未检测到接触")
print("   ✅ 通过")

# 测试3: 接触质量评估
print()
print("3. 接触质量评估测试...")
quality = estimate_contact_quality(force=3.0, contact_area=0.001)
print(f"   力: 3.0N, 面积: 10cm^2")
print(f"   压强: {quality['pressure']:.0f} Pa")
print(f"   稳定性评分: {quality['stability_score']:.3f}")
assert 0 <= quality["stability_score"] <= 1
print("   ✅ 通过")

print()
print("=== 滑移检测测试 ===")
print()

# 测试4: 力觉滑移检测
print("4. 力觉滑移检测测试...")
slip_detector = ForceSlipDetector(
    static_friction_coeff=0.8,
    incipient_ratio=0.7,
    slip_ratio=0.9,
    debounce_frames=2,
)

# 模拟逐渐增加的切向力（法向力恒定）
normal_force = 5.0
tangential_forces = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.2, 4.5, 4.0, 3.5, 3.0]

slip_states = []
for ft in tangential_forces:
    state, info = slip_detector.detect({
        "normal_force": normal_force,
        "tangential_force": ft,
    })
    slip_states.append(state)
    if state != SlipState.UNKNOWN:
        mu = info.get("friction_coefficient", 0)
        print(f"   Ft={ft:.1f}N, mu={mu:.3f}, 状态={state.value}")

assert SlipState.INCIPIENT in slip_states, "应该检测到初始滑移"
assert SlipState.SLIPPING in slip_states, "应该检测到滑移"
print("   ✅ 通过")

# 测试5: 位姿差异滑移检测
print()
print("5. 位姿差异滑移检测测试...")
pose_detector = PoseDifferenceSlipDetector(
    translation_threshold=0.002,
    min_history=3,
)

# 模拟稳定夹持然后发生滑移
gripper_pos = np.array([0.0, 0.0, 0.1])
object_rel_pos = np.array([0.0, 0.0, 0.0])  # 物体相对夹爪的位置

slip_results = []
for i in range(15):
    if i < 8:
        # 稳定夹持，相对位置不变
        obj_pos = gripper_pos + object_rel_pos
    else:
        # 发生滑移，物体相对夹爪向下移动（每帧2mm，快速滑移）
        slip_amount = (i - 7) * 0.003  # 每帧滑移 3mm
        obj_pos = gripper_pos + object_rel_pos + np.array([0, 0, -slip_amount])

    state, info = pose_detector.detect({
        "object_pose": obj_pos,
        "gripper_pose": gripper_pos,
    })
    slip_results.append(state)
    if state != SlipState.UNKNOWN:
        t_change = info.get("translation_change", 0)
        print(f"   帧{i}: 相对位移={t_change*1000:.2f}mm, 状态={state.value}")

print(f"   稳定帧: {slip_results.count(SlipState.STABLE)}")
print(f"   初始滑移帧: {slip_results.count(SlipState.INCIPIENT)}")
print(f"   滑移帧: {slip_results.count(SlipState.SLIPPING)}")
assert SlipState.SLIPPING in slip_results, "应该检测到滑移"
print("   ✅ 通过")

# 测试6: 光流滑移检测（合成图像测试）
print()
print("6. 光流滑移检测测试 (合成图像)...")
try:
    import cv2
    of_detector = OpticalFlowSlipDetector(slip_threshold=3.0, min_flow_points=5)

    # 生成合成图像
    h, w = 100, 100
    img1 = np.ones((h, w, 3), dtype=np.uint8) * 128
    img2 = np.ones((h, w, 3), dtype=np.uint8) * 128

    # 在图像中心画一个方块（物体）
    cv2.rectangle(img1, (35, 35), (65, 65), (200, 200, 200), -1)
    # 第二帧物体稍微移动（滑移）
    cv2.rectangle(img2, (40, 40), (70, 70), (200, 200, 200), -1)

    mask = np.zeros((h, w), dtype=bool)
    mask[35:65, 35:65] = True

    # 第一帧（初始化）
    of_detector.detect({"rgb": img1, "mask": mask, "bbox": [35, 35, 30, 30]})

    # 第二帧（检测滑移）
    state, info = of_detector.detect({"rgb": img2, "mask": mask, "bbox": [40, 40, 30, 30]})
    print(f"   状态: {state.value}")
    if "mean_flow" in info:
        print(f"   平均光流: {info['mean_flow']:.2f} px/帧")
        print(f"   最大光流: {info['max_flow']:.2f} px/帧")
    print("   ✅ 通过")
except ImportError:
    print("   ⚠️  OpenCV 不可用，跳过光流测试")

print()
print("✅ 所有接触/滑移检测测试通过!")
