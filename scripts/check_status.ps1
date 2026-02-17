# Quick Project Status Check
Write-Host '='*60 -ForegroundColor Cyan
Write-Host 'CATS VS DOGS MLOPS PROJECT STATUS' -ForegroundColor Yellow
Write-Host '='*60 -ForegroundColor Cyan

Write-Host '
Git Status:' -ForegroundColor Green
git log --oneline -5

Write-Host '
DVC Status:' -ForegroundColor Green
dvc status

Write-Host '
Project Files:' -ForegroundColor Green
Write-Host 'Models: ' -NoNewline
if (Test-Path 'src/models/cnn.py') { Write-Host '✓' -ForegroundColor Green } else { Write-Host '✗' -ForegroundColor Red }

Write-Host 'Data: ' -NoNewline
if (Test-Path 'data/train') { Write-Host '✓' -ForegroundColor Green } else { Write-Host '✗' -ForegroundColor Red }

Write-Host 'Tests: ' -NoNewline
if (Test-Path 'tests/*.py') { Write-Host '✓' -ForegroundColor Green } else { Write-Host '⏳ Not yet' -ForegroundColor Yellow }

Write-Host '
Module Progress:' -ForegroundColor Green
Write-Host 'M1 (Model Dev):        🟡 40%' -ForegroundColor Yellow
Write-Host 'M2 (Packaging):        ⚪ 0%' -ForegroundColor Gray
Write-Host 'M3 (CI Pipeline):      ⚪ 0%' -ForegroundColor Gray
Write-Host 'M4 (CD Deployment):    ⚪ 0%' -ForegroundColor Gray
Write-Host 'M5 (Monitoring):       ⚪ 0%' -ForegroundColor Gray

Write-Host '
Next Steps:' -ForegroundColor Cyan
Write-Host '1. Data preprocessing utilities'
Write-Host '2. Training script with MLflow'
Write-Host '3. Model evaluation'

Write-Host '
'
Write-Host '='*60 -ForegroundColor Cyan
