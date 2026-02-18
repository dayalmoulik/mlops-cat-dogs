# Run Docker Container
Write-Host 'Starting Docker container...' -ForegroundColor Cyan

# Stop existing container if running
docker stop cats-dogs-api 2>$null
docker rm cats-dogs-api 2>$null

# Run new container
docker run -d \`
    --name cats-dogs-api \`
    -p 8000:8000 \`
    cats-dogs-classifier:latest

if (\0 -eq 0) {
    Write-Host '✓ Container started successfully!' -ForegroundColor Green
    Write-Host ''
    Write-Host 'Waiting for API to be ready...'
    Start-Sleep -Seconds 5
    
    Write-Host ''
    Write-Host 'Container status:'
    docker ps -f name=cats-dogs-api
    
    Write-Host ''
    Write-Host 'API is available at: http://localhost:8000' -ForegroundColor Yellow
    Write-Host 'Docs available at: http://localhost:8000/docs' -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'Test with: python scripts/test_docker_api.py'
} else {
    Write-Host '✗ Container failed to start!' -ForegroundColor Red
}
