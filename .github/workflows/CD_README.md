# Continuous Deployment (CD) Pipeline

## Overview

Automated deployment pipeline that deploys the application to Kubernetes after successful CI builds.

## Workflows

### 1. cd-pipeline.yml (Automatic CD)
Automatically deploys after CI pipeline succeeds.

**Triggers:**
- Successful completion of CI pipeline on `main` branch
- Manual trigger via workflow_dispatch

**Steps:**
1. Validate Kubernetes manifests
2. Deploy to Kubernetes cluster
3. Wait for rollout completion
4. Run smoke tests
5. Notify deployment status

### 2. cd-production.yml (Manual Production Deploy)
Manual deployment to staging or production.

**Triggers:**
- Manual workflow dispatch with environment selection

**Steps:**
1. Validate manifests
2. Configure kubectl for cloud provider
3. Apply all Kubernetes resources
4. Wait for rollout
5. Run smoke tests
6. Generate deployment summary

**Environments:**
- `staging`: For testing
- `production`: For live deployment

### 3. rollback.yml (Rollback)
Rollback to previous or specific revision.

**Triggers:**
- Manual workflow dispatch

**Input:**
- `revision`: Optional revision number (empty = previous)

## Setting Up CD

### Prerequisites

1. **Kubernetes Cluster**
   - Minikube (local)
   - GKE, EKS, or AKS (cloud)

2. **kubectl Access**
   - Configured with cluster credentials

3. **GitHub Secrets** (for cloud deployment)
   - AWS: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - GCP: `GCP_SA_KEY`
   - Azure: `AZURE_CREDENTIALS`

### Configuration Steps

#### For AWS EKS:

1. Create IAM user with EKS access
2. Add secrets to GitHub:
```
   AWS_ACCESS_KEY_ID
   AWS_SECRET_ACCESS_KEY
   AWS_REGION
   EKS_CLUSTER_NAME
```

3. Uncomment AWS section in `cd-production.yml`

#### For GCP GKE:

1. Create service account with GKE access
2. Download JSON key
3. Add secret to GitHub:
```
   GCP_SA_KEY (JSON content)
   GCP_PROJECT_ID
   GKE_CLUSTER_NAME
   GKE_ZONE
```

4. Uncomment GCP section in `cd-production.yml`

#### For Azure AKS:

1. Create service principal
2. Add secret to GitHub:
```
   AZURE_CREDENTIALS (JSON)
```

3. Uncomment Azure section in `cd-production.yml`

## Manual Deployment

### Deploy to Staging
1. Go to Actions tab
2. Select "CD Pipeline - Production Deploy"
3. Click "Run workflow"
4. Select "staging"
5. Click "Run workflow"

### Deploy to Production
1. Go to Actions tab
2. Select "CD Pipeline - Production Deploy"
3. Click "Run workflow"
4. Select "production"
5. Click "Run workflow"

### Rollback
1. Go to Actions tab
2. Select "Rollback Deployment"
3. Click "Run workflow"
4. (Optional) Enter revision number
5. Click "Run workflow"

## Deployment Process
```
┌─────────────┐
│   CI Pass   │
└──────┬──────┘
       │
       ↓
┌─────────────────┐
│ Validate K8s    │
│ Manifests       │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ Apply:          │
│ - Namespace     │
│ - ConfigMap     │
│ - Deployment    │
│ - Service       │
│ - HPA           │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ Wait for        │
│ Rollout (5min)  │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ Run Smoke Tests │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ Notify Success  │
└─────────────────┘
```

## Smoke Tests

Automated tests run after deployment:
- Health endpoint check
- API endpoint validation
- Response time check
- Model loading verification

**Location:** `scripts/smoke_test.py`

## Monitoring Deployments

### View Logs
```powershell
# GitHub Actions logs
# Go to Actions → Select workflow run → View logs

# Kubernetes logs
kubectl logs -f deployment/cats-dogs-classifier -n ml-models
```

### Check Status
```powershell
# Deployment status
kubectl get deployments -n ml-models

# Pod status
kubectl get pods -n ml-models

# Service status
kubectl get svc -n ml-models
```

### Rollout History
```powershell
kubectl rollout history deployment/cats-dogs-classifier -n ml-models
```

## Troubleshooting

### Deployment Fails

1. **Check manifest validation:**
```powershell
   kubectl apply --dry-run=client -f k8s/
```

2. **Check pod status:**
```powershell
   kubectl describe pod POD_NAME -n ml-models
```

3. **View logs:**
```powershell
   kubectl logs POD_NAME -n ml-models
```

### Rollout Stuck

1. **Check rollout status:**
```powershell
   kubectl rollout status deployment/cats-dogs-classifier -n ml-models
```

2. **Describe deployment:**
```powershell
   kubectl describe deployment cats-dogs-classifier -n ml-models
```

3. **Force new rollout:**
```powershell
   kubectl rollout restart deployment/cats-dogs-classifier -n ml-models
```

### Smoke Tests Fail

1. Check service is accessible
2. Verify health endpoint responds
3. Check pod logs for errors
4. Ensure model is loaded correctly

## Best Practices

1. **Always test in staging first**
2. **Monitor deployments closely**
3. **Have rollback plan ready**
4. **Run smoke tests after deployment**
5. **Keep rollout history**
6. **Use blue-green or canary deployments for production**
7. **Set up alerts for deployment failures**

## Security

- Never commit secrets to repository
- Use GitHub Secrets for credentials
- Rotate credentials regularly
- Use least-privilege access
- Enable audit logging

## Next Steps

1. ✅ CD workflows created
2. ⏳ Configure cloud provider
3. ⏳ Add GitHub secrets
4. ⏳ Test deployment to staging
5. ⏳ Deploy to production

---

**Status:** CD pipeline ready for configuration
**Next:** Module 5 - Monitoring & Logging
