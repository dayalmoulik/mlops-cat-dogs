Write-Host "
=== Git Tracking Check ===" -ForegroundColor Cyan
Write-Host "Files tracked by Git in data/:"
git ls-files | Select-String "data"

Write-Host "
=== DVC Files ===" -ForegroundColor Cyan
Get-ChildItem data/*.dvc | Select-Object Name

Write-Host "
=== Git Status ===" -ForegroundColor Cyan
git status --short

Write-Host "
=== DVC Status ===" -ForegroundColor Cyan
dvc status

Write-Host "
✓ If you see only .dvc files tracked by Git, you're good!" -ForegroundColor Green
