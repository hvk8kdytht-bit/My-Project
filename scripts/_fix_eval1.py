path = 'scripts/full_evaluation.py'
with open(path, 'r') as f:
    c = f.read()

# 修复 eval_contact_methods 中的 fn 变量冲突
c = c.replace(
    "tp = fp = tn = fn = 0\n            for scene_dir in self.scene_dirs:\n                data = self.load_scene(scene_dir)\n                predictions = fn(data)\n\n                gt = data[\"gt_contacts\"]",
    "tp = fp = tn = fneg = 0\n            for scene_dir in self.scene_dirs:\n                data = self.load_scene(scene_dir)\n                predictions = method_fn(data)\n\n                gt = data[\"gt_contacts\"]"
)
# 但我需要先把 fn 重命名为 method_fn
# 让我用更精确的替换

with open(path, 'w') as f:
    f.write(c)
print('attempt 1 - may need manual fix')
