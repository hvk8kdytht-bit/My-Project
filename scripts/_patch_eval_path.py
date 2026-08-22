"""修补脚本路径问题"""
path = 'scripts/run_evaluation.py'
with open(path, 'r') as f:
    content = f.read()

old = 'import os\nimport json\nimport numpy as np'
new = 'import os\nimport sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nimport json\nimport numpy as np'

assert old in content
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('done')
