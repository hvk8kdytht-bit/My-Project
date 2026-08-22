<#
.SYNOPSIS
    安全抓取视触 VLA 项目 - 数据集下载脚本
.DESCRIPTION
    下载 YCB-Video / DexYCB / RGBD1K 精选子集（约 25GB）
    支持 HuggingFace 镜像加速和中断后继续下载
#>

param(
    [ValidateSet("ycbv", "dexycb", "rgbd1k", "all")]
    [string]$Dataset = "all",

    [string]$OutputDir = "datasets",

    [string]$Mirror = "",   # hf-mirror.com 等镜像站

    [switch]$DryRun         # 只显示将要下载的内容，不实际下载
)

$ErrorActionPreference = "Stop"

# 确保在项目根目录
Set-Location "$PSScriptRoot\.."
$ProjectRoot = Get-Location
$OutputPath = Join-Path $ProjectRoot $OutputDir

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  数据集下载脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "项目目录: $ProjectRoot"
Write-Host "输出目录: $OutputPath"
Write-Host ""

# 创建输出目录
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
}

# 设置 HuggingFace 镜像
if ($Mirror) {
    $env:HF_ENDPOINT = "https://$Mirror"
    Write-Host "使用镜像: $Mirror" -ForegroundColor Yellow
    Write-Host ""
}

function Test-HuggingFaceCli {
    $result = Get-Command huggingface-cli -ErrorAction SilentlyContinue
    return $null -ne $result
}

function Invoke-HuggingFaceDownload {
    param(
        [string]$Repo,
        [string]$LocalDir,
        [string[]]$IncludeFiles
    )

    Write-Host "下载仓库: $Repo" -ForegroundColor Green
    Write-Host "保存到: $LocalDir" -ForegroundColor Gray

    if ($DryRun) {
        Write-Host "  [DryRun] 将下载以下文件:" -ForegroundColor Yellow
        foreach ($f in $IncludeFiles) {
            Write-Host "    - $f"
        }
        return
    }

    if (-not (Test-Path $LocalDir)) {
        New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null
    }

    $includeArgs = @()
    foreach ($f in $IncludeFiles) {
        $includeArgs += "--include"
        $includeArgs += $f
    }

    & huggingface-cli download $Repo `
        --local-dir $LocalDir `
        --repo-type dataset `
        @includeArgs

    if ($LASTEXITCODE -ne 0) {
        throw "下载失败: $Repo"
    }

    Write-Host "✅ $Repo 下载完成" -ForegroundColor Green
    Write-Host ""
}

function Expand-ZipFiles {
    param([string]$Dir)

    $zipFiles = Get-ChildItem -Path $Dir -Filter "*.zip" -Recurse
    if ($zipFiles.Count -eq 0) {
        Write-Host "  没有找到 zip 文件"
        return
    }

    Write-Host "解压 $($zipFiles.Count) 个压缩文件..."

    foreach ($zip in $zipFiles) {
        $destDir = $zip.DirectoryName
        Write-Host "  解压: $($zip.Name)"

        # 优先使用 7z（支持分卷）
        $has7z = $null -ne (Get-Command 7z -ErrorAction SilentlyContinue)
        if ($has7z) {
            & 7z x $zip.FullName -o"$destDir" -y | Out-Null
        }
        else {
            Expand-Archive -Path $zip.FullName -DestinationPath $destDir -Force
        }
    }

    Write-Host "  ✅ 解压完成" -ForegroundColor Green
}

# ============================================================
#  YCB-Video BOP
# ============================================================
function Download-YCBVideo {
    Write-Host ""
    Write-Host "--- YCB-Video BOP 数据集 ---" -ForegroundColor Cyan
    Write-Host ""

    $ycbvDir = Join-Path $OutputPath "ycbv"

    # 检查是否已下载
    if (Test-Path $ycbvDir) {
        $existing = Get-ChildItem $ycbvDir -ErrorAction SilentlyContinue
        if ($existing.Count -gt 0) {
            Write-Host "⚠️  YCB-Video 目录已存在，跳过" -ForegroundColor Yellow
            return
        }
    }

    if (-not (Test-HuggingFaceCli)) {
        Write-Host "⚠️  未安装 huggingface-cli" -ForegroundColor Yellow
        Write-Host "  安装命令: pip install -U 'huggingface_hub[cli]'"
        return
    }

    $include = @(
        "ycbv_base.zip",           # 基础信息（~1MB）
        "ycbv_models.zip",         # 3D模型（~100MB）
        "ycbv_test_all.zip",       # 测试集（~2GB）
        "ycbv_train_real.zip"      # 真实训练集（~15GB）
    )

    Write-Host "将下载:"
    foreach ($f in $include) {
        Write-Host "  - $f"
    }
    Write-Host "预估总大小: ~17-20 GB"
    Write-Host ""

    Invoke-HuggingFaceDownload -Repo "bop-benchmark/ycbv" -LocalDir $ycbvDir -IncludeFiles $include

    if (-not $DryRun) {
        Expand-ZipFiles -Dir $ycbvDir
    }
}

# ============================================================
#  DexYCB
# ============================================================
function Download-DexYCB {
    Write-Host ""
    Write-Host "--- DexYCB 数据集 ---" -ForegroundColor Cyan
    Write-Host ""

    $dexycbDir = Join-Path $OutputPath "dexycb"

    if (Test-Path $dexycbDir) {
        $existing = Get-ChildItem $dexycbDir -ErrorAction SilentlyContinue
        if ($existing.Count -gt 0) {
            Write-Host "⚠️  DexYCB 目录已存在，跳过" -ForegroundColor Yellow
            return
        }
    }

    Write-Host "DexYCB 需要手动下载" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "下载地址:"
    Write-Host "  官网: https://dex-ycb.github.io/"
    Write-Host "  模型文件 (~1.4GB): https://dex-ycb.github.io/download.html"
    Write-Host ""
    Write-Host "下载后解压到: $dexycbDir"
    Write-Host ""
    Write-Host "推荐下载的最小子集:"
    Write-Host "  - models/          (3D物体模型)"
    Write-Host "  - calibration/     (相机标定参数)"
    Write-Host "  - 20200709-subject-01/  (1个受试者的抓取数据，约5GB)"
    Write-Host ""
}

# ============================================================
#  RGBD1K
# ============================================================
function Download-RGBD1K {
    Write-Host ""
    Write-Host "--- RGBD1K 数据集 ---" -ForegroundColor Cyan
    Write-Host ""

    $rgbd1kDir = Join-Path $OutputPath "rgbd1k"

    if (Test-Path $rgbd1kDir) {
        $existing = Get-ChildItem $rgbd1kDir -ErrorAction SilentlyContinue
        if ($existing.Count -gt 0) {
            Write-Host "⚠️  RGBD1K 目录已存在，跳过" -ForegroundColor Yellow
            return
        }
    }

    Write-Host "RGBD1K 需要手动下载" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "下载地址:"
    Write-Host "  GitHub: https://github.com/facebookresearch/rgbd1k"
    Write-Host "  数据: https://github.com/facebookresearch/rgbd1k#download"
    Write-Host ""
    Write-Host "下载后解压到: $rgbd1kDir"
    Write-Host ""
    Write-Host "推荐下载的最小子集:"
    Write-Host "  - 测试集 (约1-2GB)"
    Write-Host "  - 部分训练序列 (约3-5GB)"
    Write-Host ""
}

# ============================================================
#  主流程
# ============================================================
Write-Host "下载计划:" -ForegroundColor White
switch ($Dataset) {
    "ycbv"   { Write-Host "  - YCB-Video BOP (~17-20GB)" }
    "dexycb" { Write-Host "  - DexYCB (手动下载, ~1.4GB+)" }
    "rgbd1k" { Write-Host "  - RGBD1K (手动下载, ~3-5GB)" }
    "all"    {
        Write-Host "  - YCB-Video BOP (~17-20GB)"
        Write-Host "  - DexYCB (手动下载, ~1.4GB+)"
        Write-Host "  - RGBD1K (手动下载, ~3-5GB)"
        Write-Host "  合计: ~22-27GB"
    }
}

Write-Host ""
if ($DryRun) {
    Write-Host "[DryRun 模式] 不会实际下载" -ForegroundColor Yellow
    Write-Host ""
}

# 检查 huggingface-cli
if (($Dataset -eq "ycbv" -or $Dataset -eq "all") -and -not (Test-HuggingFaceCli)) {
    Write-Host "⚠️  未检测到 huggingface-cli" -ForegroundColor Yellow
    Write-Host "  正在安装..."
    pip install -U "huggingface_hub[cli]"
    Write-Host "  ✅ 安装完成"
    Write-Host ""
}

# 执行下载
switch ($Dataset) {
    "ycbv"   { Download-YCBVideo }
    "dexycb" { Download-DexYCB }
    "rgbd1k" { Download-RGBD1K }
    "all"    {
        Download-YCBVideo
        Download-DexYCB
        Download-RGBD1K
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  下载完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "验证数据集: python scripts\verify_data.py"
Write-Host ""
