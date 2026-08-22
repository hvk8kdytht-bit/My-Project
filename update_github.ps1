# Upload new/updated files to GitHub
# Run: $env:GITHUB_TOKEN = "your_token"; .\update_github.ps1

$TOKEN = $env:GITHUB_TOKEN
$OWNER = "hvk8kdytht-bit"
$REPO = "My-Project"
$API = "https://api.github.com/repos/$OWNER/$REPO"
$root = "H:\Program"

# All files to upload
$files = @(
    "scripts\test_cotracker.py",
    "scripts\test_cotracker_v2.py",
    "scripts\github_upload.py",
    "upload_github.ps1",
    "fix_upload.ps1",
    "update_github.ps1",
    "README.md",
    "LICENSE",
    "requirements.txt",
    ".gitignore"
)

Write-Host "Uploading $($files.Count) files..." -ForegroundColor Cyan

$success = 0
$failed = 0
$skipped = 0
$idx = 0

foreach ($rel in $files) {
    $idx++
    $filePath = Join-Path $root $rel
    $remotePath = $rel.Replace("\", "/")

    if (-not (Test-Path $filePath)) {
        Write-Host "[$idx] SKIP $rel (not found)" -ForegroundColor Yellow
        $skipped++
        continue
    }

    # Get SHA if file exists on GitHub
    $sha = $null
    try {
        $existing = Invoke-RestMethod -Uri "$API/contents/$remotePath" -Headers @{
            Authorization = "Bearer $TOKEN"
            Accept = "application/vnd.github+json"
        } -TimeoutSec 15 -ErrorAction Stop
        $sha = $existing.sha
    } catch {}

    # Read and encode
    $bytes = [System.IO.File]::ReadAllBytes($filePath)
    $base64 = [Convert]::ToBase64String($bytes)

    $bodyHash = @{
        message = "Update $remotePath"
        content = $base64
        branch = "main"
    }
    if ($sha) { $bodyHash.sha = $sha }
    $body = $bodyHash | ConvertTo-Json -Depth 3

    try {
        Invoke-RestMethod -Uri "$API/contents/$remotePath" -Method Put -Headers @{
            Authorization = "Bearer $TOKEN"
            Accept = "application/vnd.github+json"
        } -Body $body -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop | Out-Null

        if ($sha) {
            Write-Host "[$idx] UPDATE $remotePath" -ForegroundColor Green
        } else {
            Write-Host "[$idx] NEW     $remotePath" -ForegroundColor Green
        }
        $success++
    } catch {
        $errMsg = $_.ErrorDetails.Message
        if ($errMsg -like "*already exists*") {
            Write-Host "[$idx] EXISTS  $remotePath" -ForegroundColor Yellow
            $skipped++
        } else {
            Write-Host "[$idx] FAIL    $remotePath : $errMsg" -ForegroundColor Red
            $failed++
        }
    }
    Start-Sleep -Milliseconds 400
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Done: Success=$success Failed=$failed Skipped=$skipped" -ForegroundColor $(if ($failed -eq 0) {"Green"} else {"Yellow"})
Write-Host "Repo: https://github.com/$OWNER/$REPO" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
