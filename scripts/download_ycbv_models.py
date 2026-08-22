"""下载 YCB-Video base + models 子集（通过 hf-mirror 镜像）"""
import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, '.')

from huggingface_hub import hf_hub_download

REPO = "bop-benchmark/ycbv"
LOCAL_DIR = r"H:\Program\datasets\ycbv"

files_to_download = [
    "ycbv_base.zip",     # ~1MB   基础信息（相机参数、物体列表）
    "ycbv_models.zip",   # ~100MB 3D模型（网格+点云+直径）
]

os.makedirs(LOCAL_DIR, exist_ok=True)
print(f"仓库: {REPO}")
print(f"保存到: {LOCAL_DIR}")
print(f"镜像: {os.environ['HF_ENDPOINT']}")
print()

for filename in files_to_download:
    print(f"--- 下载 {filename} ---")
    start = time.time()
    try:
        path = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
        )
        elapsed = time.time() - start
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"✅ 完成: {path}")
        print(f"   大小: {size_mb:.1f} MB, 耗时: {elapsed:.1f}s, 速度: {size_mb/elapsed:.1f} MB/s")
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {str(e)[:150]}")
        sys.exit(1)
    print()

print("✅ 全部下载完成")
