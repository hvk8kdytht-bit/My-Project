# Fix upload - upload remaining files to GitHub
# Run in PowerShell: cd H:\Program; .\fix_upload.ps1

$TOKEN = $env:GITHUB_TOKEN
$OWNER = "hvk8kdytht-bit"
$REPO = "My-Project"
$API = "https://api.github.com/repos/$OWNER/$REPO"
$root = "H:\Program"

# Collect all files to check
$filesToUpload = @()

# scripts dir
$scriptsDir = Join-Path $root "scripts"
if (Test-Path $scriptsDir) {
    $filesToUpload += Get-ChildItem $scriptsDir -File | ForEach-Object {
        $_.FullName.Substring($root.Length + 1)
    }
}

# docs dir
$docsDir = Join-Path $root "docs"
if (Test-Path $docsDir) {
    $filesToUpload += Get-ChildItem $docsDir -File | ForEach-Object {
        $_.FullName.Substring($root.Length + 1)
    }
}

# individual files
$extraFiles = @(".gitignore", "README.md", "requirements.txt")
foreach ($f in $extraFiles) {
    $fullPath = Join-Path $root $f
    if (Test-Path $fullPath) {
        $filesToUpload += $f
    }
}

# Remove duplicates
$filesToUpload = $filesToUpload | Sort-Object -Unique

Write-Host "Checking $($filesToUpload.Count) files..." -ForegroundColor Cyan

$success = 0
$failed = 0
$skipped = 0
$idx = 0

foreach ($rel in $filesToUpload) {
    $idx++
    $filePath = Join-Path $root $rel
    $remotePath = $rel.Replace("\", "/")
    
    if (-not (Test-Path $filePath)) {
        $skipped++
        continue
    }
    
    # Check if file exists on GitHub
    $sha = $null
    try {
        $existing = Invoke-RestMethod -Uri "$API/contents/$remotePath" -Headers @{
            Authorization = "Bearer $TOKEN"
            Accept = "application/vnd.github+json"
        } -TimeoutSec 15 -ErrorAction Stop
        $sha = $existing.sha
    } catch {
        # File doesn't exist, will create new
    }
    
    # Read and encode file
    $bytes = [System.IO.File]::ReadAllBytes($filePath)
    $base64 = [Convert]::ToBase64String($bytes)
    
    # Build request body
    $bodyHash = @{
        message = "Update $remotePath"
        content = $base64
        branch = "main"
    }
    if ($sha) {
        $bodyHash.sha = $sha
    }
    $body = $bodyHash | ConvertTo-Json -Depth 3
    
    try {
        $result = Invoke-RestMethod -Uri "$API/contents/$remotePath" -Method Put -Headers @{
            Authorization = "Bearer $TOKEN"
            Accept = "application/vnd.github+json"
        } -Body $body -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
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
