Write-Host "="*70 -ForegroundColor Cyan
Write-Host "Grafana Dashboard Setup" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

# 1. Configure data source
Write-Host "`n1. Configuring Prometheus data source..." -ForegroundColor Yellow
powershell scripts/setup_grafana.ps1

Start-Sleep -Seconds 3

# 2. Import dashboard
Write-Host "`n2. Importing dashboard..." -ForegroundColor Yellow
powershell scripts/import_grafana_dashboard.ps1

Start-Sleep -Seconds 3

# 3. Generate traffic
Write-Host "`n3. Generating traffic for metrics..." -ForegroundColor Yellow
for ($i = 1; $i -le 20; $i++) {
    curl http://localhost:8000/health 2>$null | Out-Null
    curl http://localhost:8000/model/info 2>$null | Out-Null
    Write-Host "  Batch $i/20"
    Start-Sleep -Seconds 1
}

Write-Host "`n="*70 -ForegroundColor Cyan
Write-Host "✓ Setup Complete!" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan

Write-Host "`nGrafana Dashboard: http://localhost:3000" -ForegroundColor Yellow
Write-Host "Username: admin" -ForegroundColor White
Write-Host "Password: admin" -ForegroundColor White

Start-Process "http://localhost:3000"
