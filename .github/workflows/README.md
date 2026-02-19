# CI/CD Pipeline Documentation

## GitHub Actions Workflows

### 1. ci-simple.yml (Recommended for start)
Simplified CI pipeline without Docker Hub integration.

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`

**Jobs:**
1. **test** - Run unit tests
   - Set up Python 3.9
   - Install dependencies
   - Run pytest with coverage
   - Upload coverage to Codecov

2. **build-docker** - Build and test Docker image
   - Build Docker image
   - Start container
   - Test health endpoint
   - Stop container

**Duration:** ~5-8 minutes

### 2. ci-pipeline.yml (Full pipeline)
Complete CI/CD pipeline with Docker Hub integration.

**Required Secrets:**
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub access token

## Setting Up Docker Hub Secrets

1. **Create Docker Hub Access Token:**
   - Go to https://hub.docker.com/settings/security
   - Click "New Access Token"
   - Name it "GitHub Actions"
   - Copy the token

2. **Add Secrets to GitHub:**
   - Go to your GitHub repository
   - Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Add both secrets

## Local Testing

Test before pushing:
```powershell
# Run tests locally
pytest tests/ -v --cov=src

# Build Docker image
docker build -t cats-dogs-classifier:latest .

# Test container
docker run -d -p 8000:8000 cats-dogs-classifier:latest
curl http://localhost:8000/health
```

## Viewing Results

Go to: https://github.com/YOUR_USERNAME/cats-dogs-mlops/actions

---

**Status:** Workflows created, ready to push
