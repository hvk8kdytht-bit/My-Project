"""测试 HuggingFace 连通性并查询 ycbv 数据集结构"""
import sys
sys.path.insert(0, '.')

print("=== 1. HuggingFace 连通性测试 ===")
try:
    from huggingface_hub import HfApi
    api = HfApi(timeout=15)
    info = api.whoami()
    print(f"✅ HuggingFace 可访问")
except Exception as e:
    print(f"❌ whoami 失败: {type(e).__name__}: {str(e)[:100]}")
    print("   (whoami 失败不代表无法匿名下载，继续尝试...)")

print()
print("=== 2. 查询 bop-benchmark/ycbv 仓库结构 ===")
try:
    from huggingface_hub import list_repo_files
    files = list_repo_files("bop-benchmark/ycbv", repo_type="dataset")
    print(f"✅ 仓库可访问，共 {len(files)} 个文件")
    print()
    print("文件列表:")
    for f in sorted(files):
        print(f"  {f}")
except Exception as e:
    print(f"❌ 查询失败: {type(e).__name__}: {str(e)[:150]}")
    sys.exit(1)
