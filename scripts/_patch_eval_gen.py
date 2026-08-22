"""修补 eval_dataset_generator.py 的三处 MuJoCo API 问题"""

path = 'src/mujoco_env/eval_dataset_generator.py'
with open(path, 'r') as f:
    content = f.read()

# 1. 单 renderer -> 双 renderer (RGB + Depth)
old1 = '            renderer = mujoco.Renderer(model, self.img_height, self.img_width)'
new1 = (
    '            rgb_renderer = mujoco.Renderer(model, self.img_height, self.img_width)\n'
    '            depth_renderer = mujoco.Renderer(model, self.img_height, self.img_width)\n'
    '            depth_renderer.enable_depth_rendering()'
)
assert old1 in content, 'old1 not found'
content = content.replace(old1, new1)
print('1/3 done: dual renderers')

# 2. 渲染逻辑
old2 = (
    '                    # 渲染\n'
    '                    renderer.update_scene(data, camera="front")\n'
    '                    rgb = renderer.render()\n'
    '                    depth = renderer.render(depth=True)'
)
new2 = (
    '                    # 渲染 RGB\n'
    '                    rgb_renderer.update_scene(data, camera="front")\n'
    '                    rgb = rgb_renderer.render()\n'
    '                    # 渲染深度\n'
    '                    depth_renderer.update_scene(data, camera="front")\n'
    '                    depth = depth_renderer.render()'
)
assert old2 in content, 'old2 not found'
content = content.replace(old2, new2)
print('2/3 done: render calls')

# 3. close
old3 = '            renderer.close()'
new3 = '            rgb_renderer.close()\n            depth_renderer.close()'
assert old3 in content, 'old3 not found'
content = content.replace(old3, new3)
print('3/3 done: close calls')

with open(path, 'w') as f:
    f.write(content)
print('All patches applied')
