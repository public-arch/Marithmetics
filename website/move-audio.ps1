# Move NotebookLM audio files from Downloads to website/public/audio/
# Run from: website/ directory

$downloads = "$env:USERPROFILE\Downloads"
$audioDir = ".\public\audio"

# Ensure audio directory exists
if (!(Test-Path $audioDir)) { New-Item -ItemType Directory -Path $audioDir -Force }

# Find all demo-*.m4a files in Downloads
$files = Get-ChildItem -Path $downloads -Filter "demo-*.m4a" -ErrorAction SilentlyContinue

if ($files.Count -eq 0) {
    Write-Host "No demo-*.m4a files found in $downloads" -ForegroundColor Yellow
    exit
}

foreach ($f in $files) {
    $dest = Join-Path $audioDir $f.Name
    Copy-Item $f.FullName $dest -Force
    Write-Host "Copied: $($f.Name)" -ForegroundColor Green
}

Write-Host "`nDone! $($files.Count) audio files copied to $audioDir" -ForegroundColor Cyan
Write-Host "Files:" -ForegroundColor Cyan
Get-ChildItem $audioDir -Filter "*.m4a" | ForEach-Object { Write-Host "  $_" }
