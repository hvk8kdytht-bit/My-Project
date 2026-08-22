# GitHub 上传脚本
# 在普通 PowerShell 终端中运行（不要在 TRAE 里运行）
# 
# 使用方法：
# 1. 打开普通 PowerShell（Win+X -> Terminal / PowerShell）
# 2. cd H:\Program
# 3. .\upload_github.ps1
#
# 如果提示执行策略限制，先运行：
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$TOKEN = $env:GITHUB_TOKEN
$OWNER = "hvk8kdytht-bit"
$REPO = "My-Project"
$API = "https://api.github.com/repos/$OWNER/$REPO"

# 测试连接
Write-Host "测试 GitHub API 连接..." -ForegroundColor Cyan
try {
    $resp = Invoke-RestMethod -Uri $API -Headers @{Authorization = "Bearer $TOKEN"; Accept = "application/vnd.github+json"} -TimeoutSec 15
    Write-Host "连接成功! 仓库: $($resp.full_name), 默认分支: $($resp.default_branch)" -ForegroundColor Green
} catch {
    Write-Host "连接失败: $_" -ForegroundColor Red
    exit 1
}

# 收集文件
$root = "H:\Program"
$includeDirs = @("src", "scripts", "docs")
$includeFiles = @("README.md", "requirements.txt", ".gitignore")
$excludePatterns = @("__pycache__", ".pyc", ".pyo", ".log")

$files = @()
foreach ($d in $includeDirs) {
    $dirPath = Join-Path $root $d
    if (Test-Path $dirPath) {
        $files += Get-ChildItem $dirPath -Recurse -File | Where-Object {
            $skip = $false
            foreach ($p in $excludePatterns) {
                if ($_.FullName -like "*$p*") { $skip = $true; break }
            }
            -not $skip
        }
    }
}
foreach ($f in $includeFiles) {
    $fp = Join-Path $root $f
    if (Test-Path $fp) {
        $files += Get-Item $fp
    }
}

$totalSize = ($files | Measure-Object -Property Length -Sum).Sum
Write-Host "`n共 $($files.Count) 个文件, $([math]::Round($totalSize/1024, 1)) KB" -ForegroundColor Cyan

# 逐个上传
$success = 0
$failed = 0
$idx = 0
foreach ($file in $files) {
    $idx++
    $rel = $file.FullName.Substring($root.Length + 1).Replace("\", "/")
    
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $base64 = [Convert]::ToBase64String($bytes)
    
    $body = @{
        message = "Add $rel"
        content = $base64
        branch = "main"
    } | ConvertTo-Json
    
    try {
        $result = Invoke-RestMethod -Uri "$API/contents/$rel" -Method Put -Headers @{
            Authorization = "Bearer $TOKEN"
            Accept = "application/vnd.github+json"
        } -Body $body -ContentType "application/json" -TimeoutSec 30
        Write-Host "[$idx/$($files.Count)] OK   $rel" -ForegroundColor Green
        $success++
    } catch {
        $errMsg = $_.ErrorDetails.Message
        if ($errMsg -like "*already exists*") {
            Write-Host "[$idx/$($files.Count)] SKIP $rel (已存在)" -ForegroundColor Yellow
            $success++
        } else {
            Write-Host "[$idx/$($files.Count)] FAIL $rel : $errMsg" -ForegroundColor Red
            $failed++
        }
    }
    Start-Sleep -Milliseconds 300
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "完成: 成功 $success, 失败 $failed" -ForegroundColor $(if ($failed -eq 0) {"Green"} else {"Yellow"})
Write-Host "仓库: https://github.com/$OWNER/$REPO" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
