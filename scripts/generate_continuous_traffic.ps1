Write-Host "Generating continuous traffic for dashboard..." -ForegroundColor Cyan

$endpoints = @(
    "http://localhost:8000/health",
    "http://localhost:8000/",
    "http://localhost:8000/model/info"
)

for ($i = 1; $i -le 30; $i++) {
    foreach ($endpoint in $endpoints) {
        try {
            Invoke-RestMethod -Uri $endpoint -Method Get -ErrorAction SilentlyContinue | Out-Null
        } catch {}
    }
    Write-Host "Request batch $i/30 sent" -NoNewline -ForegroundColor Yellow
    Write-Host "`r" -NoNewline
    Start-Sleep -Seconds 2
}

Write-Host "`n✓ Traffic generation complete" -ForegroundColor Green
Write-Host "Check dashboard: http://localhost:3000" -ForegroundColor Yellow
