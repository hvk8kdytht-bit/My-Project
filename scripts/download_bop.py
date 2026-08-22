"""
通用 BOP 数据集下载脚本（通过 hf-mirror 镜像）
用法: python download_bop.py <filename> [local_dir]
示例: python download_bop.py ycbv_test_all.zip
"""
import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

REPO = "bop-benchmark/ycbv"
DEFAULT_DIR = r"H:\Program\datasets\ycbv"


def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    if len(sys.argv) < 2:
        print("用法: python download_bop.py <filename> [local_dir]")
        print("可用文件: ycbv_base.zip ycbv_models.zip ycbv_test_all.zip "
              "ycbv_test_bop19.zip ycbv_train_real.zip ycbv_train_real.z01 "
              "ycbv_train_pbr.zip ycbv_train_synt.zip")
        sys.exit(1)

    filename = sys.argv[1]
    local_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DIR

    os.makedirs(local_dir, exist_ok=True)
    print(f"下载: {filename}")
    print(f"仓库: {REPO}  (镜像: {os.environ['HF_ENDPOINT']})")
    print(f"保存: {local_dir}")
    print()

    start = time.time()
    try:
        path = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
        )
    except Exception as e:
        print(f"❌ 下载失败: {type(e).__name__}: {str(e)[:200]}")
        sys.exit(1)

    elapsed = time.time() - start
    size = os.path.getsize(path)
    print(f"✅ 完成: {path}")
    print(f"   大小: {human_size(size)}, 耗时: {elapsed:.0f}s, 平均速度: {human_size(size/elapsed)}/s")


if __name__ == "__main__":
    main()
