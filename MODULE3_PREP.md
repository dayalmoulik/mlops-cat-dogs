# Module 3 Preparation - CI Pipeline

## What We'll Build

### Objective
Implement Continuous Integration to automatically test, package, and build container images.

### Tasks (10 marks)

1. **Automated Testing** (3 marks)
   - Write unit tests for data preprocessing
   - Write unit tests for model inference
   - Run tests with pytest
   - Generate coverage reports

2. **CI Setup** (4 marks)
   - GitHub Actions workflow
   - Trigger on push/pull request
   - Run automated tests
   - Build Docker image

3. **Artifact Publishing** (3 marks)
   - Push Docker image to registry
   - Docker Hub or GitHub Container Registry
   - Tag with version/commit SHA

## Prerequisites

### GitHub Secrets Needed
- \DOCKER_USERNAME\ - Docker Hub username
- \DOCKER_PASSWORD\ - Docker Hub password/token

### Files to Create
- \.github/workflows/ci-pipeline.yml\
- \	ests/test_preprocessing.py\ (enhance existing)
- \	ests/test_inference.py\ (new)
- \	ests/test_api.py\ (new)

## Estimated Time
- Writing tests: 1-1.5 hours
- GitHub Actions setup: 1-1.5 hours  
- Testing & debugging: 0.5-1 hour
- Documentation: 0.5 hour
- **Total: 3-4 hours**

## Current Status
- ✅ Module 1 Complete (Model Development)
- ✅ Module 2 Complete (Packaging & Docker)
- 🎯 Ready to start Module 3

---

**Ready to proceed?** We'll start by enhancing the unit tests, then set up GitHub Actions! 🚀
