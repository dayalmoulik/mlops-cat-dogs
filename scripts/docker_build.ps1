# Build Docker Image
Write-Host 'Building Docker image...' -ForegroundColor Cyan
docker build -t cats-dogs-classifier:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host '✓ Image built successfully!' -ForegroundColor Green
    Write-Host ''
    Write-Host 'Image details:'
    docker images cats-dogs-classifier:latest
} else {
    Write-Host '✗ Build failed!' -ForegroundColor Red
}
