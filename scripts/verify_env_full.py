import torch, torchvision, mujoco, numpy, scipy, cv2, PIL, matplotlib

print(f'torch        {torch.__version__}')
print(f'torchvision  {torchvision.__version__}')
print(f'mujoco       {mujoco.__version__}')
print(f'numpy        {numpy.__version__}')
print(f'scipy        {scipy.__version__}')
print(f'opencv       {cv2.__version__}')
print(f'Pillow       {PIL.__version__}')
print(f'matplotlib   {matplotlib.__version__}')

x = torch.randn(2, 3)
y = (x @ x.T).sum()
print(f'tensor 计算: {y.item():.4f}')

xml = '<mujoco><worldbody><body><geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print('MuJoCo 仿真步进: OK')
print()
print('=== 全部依赖就绪 ===')
