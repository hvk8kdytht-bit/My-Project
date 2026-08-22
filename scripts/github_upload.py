"""上传项目文件到 GitHub（通过 REST API，不需要 git）"""
import os
import json
import base64
import time
import urllib.request
import urllib.error
from pathlib import Path

TOKEN = os.environ.get("GITHUB_TOKEN", "")
OWNER = "hvk8kdytht-bit"
REPO = "My-Project"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"

def api_test():
    """测试 API 连接"""
    req = urllib.request.Request(API, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            print(f"仓库: {data.get('full_name', '?')}")
            print(f"默认分支: {data.get('default_branch', '?')}")
            print(f"大小: {data.get('size', 0)} KB")
            print("API 连接成功!\n")
            return True
    except Exception as e:
        print(f"API 连接失败: {e}\n")
        return False

def upload_file(local_path, remote_path):
    """上传单个文件"""
    url = f"{API}/contents/{remote_path}"
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    data = json.dumps({
        "message": f"Add {remote_path}",
        "content": content,
        "branch": "main",
    }).encode()
    req = urllib.request.Request(url, data=data, method="PUT", headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return True, ""
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        if "already exists" in err:
            return True, "exists"
        return False, err[:100]

def main():
    print("=" * 60)
    print("GitHub 上传工具")
    print("=" * 60)

    if not api_test():
        return

    root = Path("h:/Program")
    include_dirs = ["src", "scripts", "docs"]
    include_files = ["README.md", "requirements.txt", ".gitignore"]

    files = []
    for d in include_dirs:
        p = root / d
        if p.exists():
            for f in p.rglob("*"):
                if f.is_file() and "__pycache__" not in str(f) and ".pyc" not in str(f) and ".log" not in str(f):
                    files.append(f)
    for f in include_files:
        fp = root / f
        if fp.exists():
            files.append(fp)

    total_size = sum(f.stat().st_size for f in files)
    print(f"共 {len(files)} 个文件, {total_size/1024:.1f} KB")
    print()

    success = 0
    failed = 0
    for i, f in enumerate(sorted(files)):
        rel = str(f.relative_to(root)).replace("\\", "/")
        ok, msg = upload_file(f, rel)
        status = "OK" if ok and msg != "exists" else ("SKIP" if msg == "exists" else "FAIL")
        if ok:
            success += 1
        else:
            failed += 1
        print(f"[{i+1}/{len(files)}] {status} {rel}" + (f" ({msg})" if msg and msg != "exists" else ""))
        time.sleep(0.3)

    print()
    print(f"完成: 成功 {success}, 失败 {failed}")
    print(f"仓库: https://github.com/{OWNER}/{REPO}")

if __name__ == "__main__":
    main()
