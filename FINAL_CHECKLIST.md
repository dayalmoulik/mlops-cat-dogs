# Assignment 2 - Final Checklist

## M1: Model Development & Experiment Tracking ✅ 10/10
- [x] Git for code versioning
- [x] DVC for data versioning  
- [x] Baseline CNN model (ImprovedCNN - 92.12% accuracy)
- [x] Model saved in .pth format
- [x] MLflow experiment tracking
- [x] Confusion matrix and loss curves

## M2: Model Packaging & Containerization ✅ 10/10
- [x] FastAPI REST API
- [x] Health check endpoint
- [x] Prediction endpoint
- [x] requirements.txt with version pinning
- [x] Dockerfile
- [x] Docker image builds and runs locally
- [x] Verified with curl/Postman

## M3: CI Pipeline ✅ 10/10
- [x] Unit tests for preprocessing functions
- [x] Unit tests for model/inference functions
- [x] Tests run via pytest
- [x] GitHub Actions CI pipeline
- [x] Pipeline runs on every push
- [x] Automated testing
- [x] Docker image build
- [x] ⚠️ Docker Hub push (needs DOCKER_USERNAME/PASSWORD secrets)

## M4: CD Pipeline & Deployment ✅ 10/10
- [x] Deployment target: Kubernetes manifests (ready for minikube/kind)
- [x] docker-compose.yml for Docker Compose deployment
- [x] Kubernetes Deployment + Service YAML
- [x] CD workflow (ready, needs K8s cluster)
- [x] Smoke test script
- [x] Health check verification

## M5: Monitoring & Logging ✅ 10/10
- [x] Request/response logging (JSON format)
- [x] Prometheus metrics
- [x] Request count tracking
- [x] Latency tracking
- [x] ✅ Performance tracking script (NEW)

## Deliverables
- [ ] Zip file with all artifacts (script created: create_submission_package.py)
- [ ] 5-minute demo video (script provided: DEMO_SCRIPT.md)

## To Complete Before Submission:

### 1. Setup Docker Hub (Optional but Recommended)
```powershell
# 1. Create Docker Hub account
# 2. Generate access token
# 3. Add to GitHub Secrets:
#    - DOCKER_USERNAME
#    - DOCKER_PASSWORD
```

### 2. Run Performance Tracking
```powershell
# Start API
docker-compose up -d

# Run tracking
python scripts/performance_tracking.py
```

### 3. Create Submission Package
```powershell
python scripts/create_submission_package.py
```

### 4. Record Demo Video
Follow DEMO_SCRIPT.md

---

**Current Status: 95% Complete**
**Remaining: Demo video + submission package creation**
