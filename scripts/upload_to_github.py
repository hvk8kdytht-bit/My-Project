"""
通过 GitHub REST API 上传项目文件到仓库
不需要安装 git
"""

import os
import sys
import json
import base64
import time
import urllib.request
import urllib.error
from pathlib import Path

# 配置
GITHUB_TOKEN = ""  # 需要用户提供 Personal Access Token
REPO_OWNER = "hvk8kdytht-bit"
REPO_NAME = "My-Project"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

# 要上传的文件模式
INCLUDE_DIRS = ["src", "scripts", "docs", "notebooks"]
INCLUDE_FILES = [
    "README.md", "requirements.txt", "PROGRESS.md",
    ".gitignore", "collect_data.py", "demo.py",
    "force_3d_view.py", "force_3d_web.html",
    "tactile_sensor.py", "visualize.py", "run_all_tests.bat",
]

# 排除模式
EXCLUDE_PATTERNS = [
    "__pycache__", ".pyc", ".pyo",
    ".log", ".tmp",
    "co-tracker.zip", "RGB+RGB-D.zip",
]


def should_exclude(path):
    """判断文件是否应该排除"""
    for pat in EXCLUDE_PATTERNS:
        if pat in str(path):
            return True
    # 排除大于 10MB 的文件
    try:
        if path.stat().st_size > 10 * 1024 * 1024:
            return True
    except:
        return True
    return False


def collect_files():
    """收集所有要上传的文件"""
    root = Path("h:/Program")
    files = []

    # 指定目录
    for d in INCLUDE_DIRS:
        dir_path = root / d
        if not dir_path.exists():
            continue
        for f in dir_path.rglob("*"):
            if f.is_file() and not should_exclude(f):
                files.append(f)

    # 指定文件
    for f in INCLUDE_FILES:
        fp = root / f
        if fp.exists() and not should_exclude(fp):
            files.append(fp)

    return root, files


def github_api(method, url, data=None, token=None):
    """调用 GitHub API"""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  API 错误 {e.code}: {e.read().decode()[:200]}")
        return None


def upload_file(local_path, remote_path, token, root):
    """上传单个文件到 GitHub"""
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{remote_path}"

    # 读取文件内容并 base64 编码
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    data = {
        "message": f"Add {remote_path}",
        "content": content,
        "branch": "main",
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }

    payload = json.dumps(data).encode()
    req = urllib.request.Request(api_url, data=payload, headers=headers, method="PUT")

    try:
        with urllib.request.urlopen(req) as resp:
            return True
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode()
        if "already exists" in error_msg:
            print(f"  跳过（已存在）: {remote_path}")
            return True
        print(f"  上传失败 {remote_path}: {error_msg[:100]}")
        return False


def main():
    print("=" * 60)
    print("GitHub 上传工具（无需 git）")
    print("=" * 60)

    # 获取 token
    token = GITHUB_TOKEN or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("\n需要 GitHub Personal Access Token")
        print("获取方式：GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)")
        print("  -> Generate new token -> 勾选 'repo' 权限")
        token = input("\n请输入你的 token: ").strip()

    if not token:
        print("未提供 token，退出")
        return

    # 收集文件
    root, files = collect_files()
    print(f"\n共 {len(files)} 个文件待上传")

    # 显示文件列表
    total_size = 0
    for f in files:
        rel = f.relative_to(root)
        size = f.stat().st_size
        total_size += size
        print(f"  {rel} ({size/1024:.1f}KB)")
    print(f"\n总大小: {total_size/1024:.1f}KB")

    # 确认
    confirm = input(f"\n确认上传 {len(files)} 个文件到 {REPO_URL}? (y/n): ").strip()
    if confirm.lower() != "y":
        print("取消")
        return

    # 上传
    success = 0
    failed = 0
    for i, f in enumerate(files):
        rel_path = str(f.relative_to(root)).replace("\\", "/")
        print(f"[{i+1}/{len(files)}] 上传 {rel_path}...", end=" ")
        if upload_file(f, rel_path, token, root):
            print("OK")
            success += 1
        else:
            print("FAILED")
            failed += 1
        # 避免 API 限流
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"完成: 成功 {success}, 失败 {failed}")
    print(f"仓库地址: {REPO_URL}")


if __name__ == "__main__":
    main()
