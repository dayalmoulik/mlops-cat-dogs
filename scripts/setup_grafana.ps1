$grafanaUrl = "http://localhost:3000"
$auth = "admin:Dayal@040394"
$base64Auth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($auth))
$headers = @{
    "Authorization" = "Basic $base64Auth"
    "Content-Type" = "application/json"
}

# Add Prometheus data source
$datasource = @{
    name = "Prometheus"
    type = "prometheus"
    url = "http://prometheus:9090"
    access = "proxy"
    isDefault = $true
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources" -Method Post -Headers $headers -Body $datasource
    Write-Host "✓ Data source added successfully" -ForegroundColor Green
} catch {
    if ($_.Exception.Response.StatusCode -eq 409) {
        Write-Host "✓ Data source already exists" -ForegroundColor Yellow
    } else {
        Write-Host "✗ Error: $_" -ForegroundColor Red
    }
}
