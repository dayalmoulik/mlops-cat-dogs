$grafanaUrl = "http://localhost:3000"
$auth = "admin:Dayal@040394"
$base64Auth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($auth))
$headers = @{
    "Authorization" = "Basic $base64Auth"
    "Content-Type" = "application/json"
}

Write-Host "Importing Grafana Dashboard..." -ForegroundColor Cyan

# Read dashboard JSON
$dashboardJson = Get-Content "grafana-dashboard-complete.json" -Raw

try {
    $response = Invoke-RestMethod -Uri "$grafanaUrl/api/dashboards/db" -Method Post -Headers $headers -Body $dashboardJson
    Write-Host "✓ Dashboard imported successfully" -ForegroundColor Green
    Write-Host "  Dashboard URL: $grafanaUrl$($response.url)" -ForegroundColor Yellow
    
    # Open dashboard
    Start-Process "$grafanaUrl$($response.url)"
} catch {
    Write-Host "✗ Error importing dashboard: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message
}
