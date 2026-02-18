# Stop Docker Container
Write-Host 'Stopping Docker container...' -ForegroundColor Cyan

docker-compose down 2>$null
docker stop cats-dogs-api 2>$null
docker rm cats-dogs-api 2>$null

Write-Host '✓ Container stopped and removed' -ForegroundColor Green
