"""通过 hf-mirror.com 镜像测试 HuggingFace 连通性"""
import os
import sys
sys.path.insert(0, '.')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print(f"使用镜像: {os.environ['HF_ENDPOINT']}")
print()

print("=== 查询 bop-benchmark/ycbv 仓库结构 ===")
try:
    from huggingface_hub import list_repo_files
    files = list_repo_files("bop-benchmark/ycbv", repo_type="dataset")
    print(f"✅ 镜像可访问! 仓库共 {len(files)} 个文件")
    print()
    print("文件列表:")
    for f in sorted(files):
        print(f"  {f}")
except Exception as e:
    print(f"❌ 镜像查询失败: {type(e).__name__}: {str(e)[:150]}")
    sys.exit(1)
