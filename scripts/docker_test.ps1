# Test Docker Container
Write-Host 'Testing Docker container...' -ForegroundColor Cyan
Write-Host ''

# Check if container is running
\ = docker ps -f name=cats-dogs-api -q

if (-not \) {
    Write-Host '✗ Container is not running!' -ForegroundColor Red
    Write-Host 'Start it with: powershell scripts/docker_run.ps1'
    exit 1
}

Write-Host 'Container is running. Running tests...' -ForegroundColor Green
Write-Host ''

python scripts/test_docker_api.py
