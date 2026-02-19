# Kubernetes Deployment Script (PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deploying Cats vs Dogs Classifier to K8s" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if kubectl is available
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Host "Error: kubectl not found. Please install kubectl first." -ForegroundColor Red
    exit 1
}

# Create namespace
Write-Host "`nCreating namespace..." -ForegroundColor Yellow
kubectl apply -f k8s/namespace.yaml

# Apply ConfigMap
Write-Host "`nApplying ConfigMap..." -ForegroundColor Yellow
kubectl apply -f k8s/configmap.yaml -n ml-models

# Apply Deployment
Write-Host "`nApplying Deployment..." -ForegroundColor Yellow
kubectl apply -f k8s/deployment.yaml -n ml-models

# Apply Service
Write-Host "`nApplying Service..." -ForegroundColor Yellow
kubectl apply -f k8s/service.yaml -n ml-models

# Apply HPA
Write-Host "`nApplying Horizontal Pod Autoscaler..." -ForegroundColor Yellow
kubectl apply -f k8s/hpa.yaml -n ml-models

# Wait for deployment
Write-Host "`nWaiting for deployment to be ready..." -ForegroundColor Yellow
kubectl wait --for=condition=available --timeout=300s deployment/cats-dogs-classifier -n ml-models

# Get deployment status
Write-Host "`nDeployment Status:" -ForegroundColor Green
kubectl get deployments -n ml-models
Write-Host ""
kubectl get pods -n ml-models
Write-Host ""
kubectl get services -n ml-models

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
