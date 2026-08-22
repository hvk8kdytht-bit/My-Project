"""
环境检查与依赖安装指南
检查当前Python环境并提供安装建议

用法:
    python scripts/check_env.py
"""

import sys
import importlib


def check_package(name: str, min_version: str = None) -> dict:
    """检查包是否安装"""
    try:
        pkg = importlib.import_module(name)
        version = getattr(pkg, "__version__", "unknown")
        return {"available": True, "version": version, "name": name, "error": None}
    except ImportError:
        return {"available": False, "version": None, "name": name, "error": "not_installed"}
    except Exception as e:
        # 捕获DLL加载失败等其他错误
        return {"available": False, "version": None, "name": name, "error": str(e)[:80]}


def main():
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print()

    # 核心依赖
    packages = [
        ("numpy", "1.24"),
        ("scipy", "1.10"),
        ("PIL", "10.0"),  # Pillow
        ("cv2", "4.8"),    # opencv-python
        ("matplotlib", "3.7"),
        ("torch", "2.0"),
        ("torchvision", "0.15"),
        ("mujoco", "3.0"),
        ("huggingface_hub", "0.17"),
        ("tqdm", "4.65"),
        ("serial", "3.5"),  # pyserial
        ("pandas", "2.0"),
    ]

    print(f"{'包名':<25} {'状态':<10} {'版本':<15}")
    print("-" * 60)

    all_ok = True
    for name, min_ver in packages:
        result = check_package(name)
        status = "✅" if result["available"] else "❌"
        if result["available"]:
            ver = result["version"]
        elif result["error"] == "not_installed":
            ver = "未安装"
        else:
            ver = "加载失败"
        print(f"  {name:<23} {status:<10} {ver:<15}")
        if not result["available"]:
            all_ok = False
            if result["error"] and result["error"] != "not_installed":
                print(f"    原因: {result['error']}")

    print()
    print("=" * 60)

    if all_ok:
        print("✅ 所有依赖已安装!")
    else:
        print("⚠️  部分依赖未安装")
        print()
        print("安装建议:")
        print()
        print("  1. 核心依赖（必须）:")
        print("     pip install numpy scipy pillow opencv-python matplotlib tqdm")
        print()
        print("  2. PyTorch（必须，用于模型训练）:")
        print("     注意: 当前Python路径较深，可能遇到长路径问题")
        print("     建议在独立Python环境中安装:")
        print("       # CPU版")
        print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print()
        print("       # GPU版 (CUDA 12.1)")
        print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("  3. MuJoCo（仿真环境）:")
        print("     pip install mujoco")
        print("     注意: Windows 可能遇到安全策略阻止DLL的问题")
        print("     解决方法: 以管理员权限运行，或将 mujoco DLL 添加到信任列表")
        print()
        print("  4. 传感器驱动（可选）:")
        print("     pip install pyserial")
        print()

    # 磁盘空间
    print("=" * 60)
    print("磁盘空间:")
    try:
        import os
        for drive in ["H:", "D:", "C:"]:
            try:
                # 简单的磁盘空间检查
                import shutil
                usage = shutil.disk_usage(drive + "\\")
                free_gb = usage.free / (1024**3)
                total_gb = usage.total / (1024**3)
                print(f"  {drive}\  {free_gb:.1f} GB 可用 / {total_gb:.1f} GB 总计")
            except Exception as e:
                print(f"  {drive}\  无法读取: {e}")
    except:
        pass

    print()
    print("=" * 60)
    print("项目结构验证:")
    print("-" * 60)

    from pathlib import Path
    root = Path(__file__).parent.parent

    key_files = [
        "README.md",
        "requirements.txt",
        "src/data/__init__.py",
        "src/data/ycb_video.py",
        "src/data/dexycb.py",
        "src/data/rgbd1k.py",
        "src/data/transforms.py",
        "src/models/__init__.py",
        "src/models/pose_estimator.py",
        "src/models/tracker.py",
        "src/utils/__init__.py",
        "src/utils/metrics.py",
        "src/utils/visualization.py",
        "src/mujoco_env/__init__.py",
        "src/mujoco_env/gripper_env.py",
        "src/mujoco_env/dataset_generator.py",
        "scripts/verify_data.py",
        "scripts/train_pose.py",
        "scripts/eval_pose.py",
        "scripts/test_pipeline.py",
        "scripts/download_datasets.ps1",
        "docs/research_proposal.md",
        "docs/datasets_verification.md",
    ]

    for f in key_files:
        path = root / f
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✅ {f} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {f} (缺失)")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
