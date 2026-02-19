# Kubernetes Deployment Guide

## Prerequisites

### Local Development (Minikube)
```powershell
# Install minikube
choco install minikube

# Start minikube
minikube start

# Enable metrics server for HPA
minikube addons enable metrics-server
```

### Cloud Deployment
- Google Kubernetes Engine (GKE)
- Amazon Elastic Kubernetes Service (EKS)
- Azure Kubernetes Service (AKS)

## Deployment Steps

### 1. Build and Push Docker Image
```powershell
# Build image
docker build -t cats-dogs-classifier:latest .

# Tag for registry (update with your registry)
docker tag cats-dogs-classifier:latest YOUR_USERNAME/cats-dogs-classifier:latest

# Push to registry
docker push YOUR_USERNAME/cats-dogs-classifier:latest
```

### 2. Update Deployment Image
Edit `k8s/deployment.yaml` and update image:
```yaml
image: YOUR_USERNAME/cats-dogs-classifier:latest
```

### 3. Deploy to Kubernetes
```powershell
# Using PowerShell script
powershell scripts/deploy-k8s.ps1

# Or manually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml -n ml-models
kubectl apply -f k8s/deployment.yaml -n ml-models
kubectl apply -f k8s/service.yaml -n ml-models
kubectl apply -f k8s/hpa.yaml -n ml-models
```

### 4. Verify Deployment
```powershell
# Check pods
kubectl get pods -n ml-models

# Check service
kubectl get svc -n ml-models

# Check HPA
kubectl get hpa -n ml-models

# View logs
kubectl logs -f deployment/cats-dogs-classifier -n ml-models
```

### 5. Run Smoke Tests
```powershell
# Get service URL
kubectl get svc cats-dogs-service -n ml-models

# Run smoke tests
python scripts/smoke_test.py http://SERVICE_IP:80
```

## Local Testing with Minikube
```powershell
# Start minikube
minikube start

# Use minikube docker daemon
minikube docker-env | Invoke-Expression

# Build image (will be available in minikube)
docker build -t cats-dogs-classifier:latest .

# Deploy
powershell scripts/deploy-k8s.ps1

# Get service URL
minikube service cats-dogs-service -n ml-models --url

# Test
curl $(minikube service cats-dogs-service -n ml-models --url)/health
```

## Scaling

### Manual Scaling
```powershell
kubectl scale deployment cats-dogs-classifier --replicas=5 -n ml-models
```

### Auto-scaling (HPA)
Automatically scales based on CPU and memory usage:
- Min replicas: 2
- Max replicas: 10
- Target CPU: 70%
- Target Memory: 80%

## Updating Deployment

### Rolling Update
```powershell
# Update image
kubectl set image deployment/cats-dogs-classifier \
  cats-dogs-api=YOUR_USERNAME/cats-dogs-classifier:v2 \
  -n ml-models

# Check rollout status
kubectl rollout status deployment/cats-dogs-classifier -n ml-models
```

### Rollback
```powershell
# Rollback to previous version
kubectl rollout undo deployment/cats-dogs-classifier -n ml-models

# Rollback to specific revision
kubectl rollout undo deployment/cats-dogs-classifier --to-revision=2 -n ml-models
```

## Monitoring

### View Logs
```powershell
# All pods
kubectl logs -f deployment/cats-dogs-classifier -n ml-models

# Specific pod
kubectl logs -f POD_NAME -n ml-models

# Previous logs
kubectl logs POD_NAME --previous -n ml-models
```

### Pod Status
```powershell
# Describe pod
kubectl describe pod POD_NAME -n ml-models

# Get events
kubectl get events -n ml-models --sort-by='.lastTimestamp'
```

## Cleanup
```powershell
# Delete deployment
kubectl delete -f k8s/deployment.yaml -n ml-models

# Delete service
kubectl delete -f k8s/service.yaml -n ml-models

# Delete namespace (removes everything)
kubectl delete namespace ml-models
```

## Troubleshooting

### Pods Not Starting
```powershell
# Check pod status
kubectl describe pod POD_NAME -n ml-models

# Check logs
kubectl logs POD_NAME -n ml-models

# Common issues:
# - Image pull errors: Check image name and registry access
# - Resource limits: Check if cluster has enough resources
# - Health checks failing: Check /health endpoint
```

### Service Not Accessible
```powershell
# Check service
kubectl get svc cats-dogs-service -n ml-models

# Check endpoints
kubectl get endpoints cats-dogs-service -n ml-models

# Port forward for testing
kubectl port-forward svc/cats-dogs-service 8000:80 -n ml-models
```

## Best Practices

1. **Resource Limits**: Always set resource requests and limits
2. **Health Checks**: Configure liveness and readiness probes
3. **Replicas**: Run at least 2 replicas for high availability
4. **Auto-scaling**: Use HPA for automatic scaling
5. **Rolling Updates**: Use rolling updates for zero-downtime deployments
6. **Monitoring**: Set up logging and monitoring
7. **Secrets**: Use Kubernetes Secrets for sensitive data

---

**Status**: Kubernetes manifests ready for deployment
**Next**: Set up CD pipeline in GitHub Actions
