#!/bin/bash
# Kubernetes Deployment Script

set -e

echo "=========================================="
echo "Deploying Cats vs Dogs Classifier to K8s"
echo "=========================================="

# Create namespace
echo "Creating namespace..."
kubectl apply -f k8s/namespace.yaml

# Apply ConfigMap
echo "Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml -n ml-models

# Apply Deployment
echo "Applying Deployment..."
kubectl apply -f k8s/deployment.yaml -n ml-models

# Apply Service
echo "Applying Service..."
kubectl apply -f k8s/service.yaml -n ml-models

# Apply HPA
echo "Applying Horizontal Pod Autoscaler..."
kubectl apply -f k8s/hpa.yaml -n ml-models

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/cats-dogs-classifier -n ml-models

# Get deployment status
echo ""
echo "Deployment Status:"
kubectl get deployments -n ml-models
echo ""
kubectl get pods -n ml-models
echo ""
kubectl get services -n ml-models

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
