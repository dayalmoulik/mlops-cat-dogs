# Cats vs Dogs MLOps Pipeline

**Binary image classification MLOps pipeline for pet adoption platform**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-orange.svg)](https://dvc.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

This project implements a complete MLOps pipeline for classifying images of cats and dogs, designed for a pet adoption platform. It covers the full machine learning lifecycle from data versioning to model deployment.

### Assignment Context
- **Course**: MLOps (S1-25_AIMLCZG523)
- **Assignment**: Assignment 2
- **Total Marks**: 50
- **Modules**: 5 (M1-M5, each 10 marks)

---

## 🏗️ Architecture

\\\
┌─────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📊 Data          🧠 Model         🚀 Deploy        📈 Monitor│
│  ├─ DVC           ├─ PyTorch      ├─ Docker        ├─ Prometheus│
│  ├─ Kaggle        ├─ SimpleCNN    ├─ K8s           ├─ MLflow   │
│  └─ 25K images    └─ ImprovedCNN  └─ FastAPI       └─ Grafana  │
│                                                               │
│              🔄 CI/CD (GitHub Actions)                       │
│              ✅ Automated Testing                            │
│              📦 Container Registry                           │
└─────────────────────────────────────────────────────────────┘
\\\

---

## ✅ Current Progress: 40%

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed progress tracking.

### Completed:
- ✅ Project setup and Git repository
- ✅ Anaconda environment with all dependencies
- ✅ Data downloaded and organized (25,000 images)
- ✅ DVC for data versioning
- ✅ Train/Val/Test split (80/10/10) with no overlap
- ✅ CNN model architectures (SimpleCNN & ImprovedCNN)

### In Progress:
- 🚧 Data preprocessing utilities
- 🚧 Training script with MLflow

### To Do:
- ⏳ FastAPI inference service
- ⏳ Dockerization
- ⏳ CI/CD pipeline
- ⏳ Kubernetes deployment
- ⏳ Monitoring and logging

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Anaconda
- Git & DVC
- Kaggle API credentials

### Setup

\\\powershell
# Clone repository
git clone https://github.com/YOUR_USERNAME/cats-dogs-mlops.git
cd cats-dogs-mlops

# Create conda environment
conda create -n mlops python=3.9 -y
conda activate mlops

# Install dependencies
conda install pytorch torchvision numpy pandas scikit-learn pillow matplotlib seaborn -c pytorch
pip install fastapi uvicorn[standard] pydantic python-multipart mlflow dvc prometheus-client python-json-logger pytest pytest-cov pytest-asyncio httpx python-dotenv pyyaml requests

# Download dataset (requires Kaggle API setup)
python scripts/download_data.py

# Test model
python src/models/cnn.py
\\\

---

## 📊 Dataset

- **Source**: [Kaggle - Dog and Cat Classification](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Total Images**: 24,998 (after validation)
- **Classes**: 2 (Cats and Dogs)
- **Image Size**: 224x224 RGB
- **Split**: 80% Train, 10% Validation, 10% Test

| Split | Cats | Dogs | Total |
|-------|------|------|-------|
| Train | 9,999 | 9,999 | 19,998 |
| Validation | 1,250 | 1,250 | 2,500 |
| Test | 1,250 | 1,250 | 2,500 |

See [DATA_README.md](DATA_README.md) for more details.

---

## 🧠 Models

### SimpleCNN
- **Parameters**: 11.2M
- **Architecture**: 3 Conv blocks + 2 FC layers
- **Use Case**: Baseline model, quick experiments

### ImprovedCNN
- **Parameters**: 2.8M
- **Architecture**: ResNet-style with skip connections
- **Use Case**: Production deployment, better accuracy

See [src/models/README.md](src/models/README.md) for architecture details.

---

## 📁 Project Structure

\\\
cats-dogs-mlops/
├── .dvc/                    # DVC configuration
├── .github/workflows/       # CI/CD pipelines (to do)
├── data/                    # Dataset (DVC tracked)
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── api/                 # FastAPI service (to do)
│   ├── models/              # ✅ Model architectures
│   ├── training/            # Training scripts (in progress)
│   └── utils/               # Utilities (in progress)
├── tests/                   # Unit tests (to do)
├── k8s/                     # Kubernetes manifests (to do)
├── scripts/
│   └── download_data.py     # ✅ Data preparation
├── requirements.txt         # ✅ Dependencies
└── PROJECT_STATUS.md        # ✅ Progress tracker
\\\

---

## 🧪 Testing

\\\powershell
# Run tests (when implemented)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
\\\

---

## 📦 Module Breakdown

### M1: Model Development & Experiment Tracking (10M)
- [x] Git & DVC setup
- [x] Data versioning
- [x] Model architecture
- [ ] MLflow tracking
- [ ] Training pipeline

### M2: Model Packaging & Containerization (10M)
- [ ] FastAPI REST API
- [ ] Dockerfile
- [ ] requirements.txt ✅
- [ ] Local testing

### M3: CI Pipeline (10M)
- [ ] Unit tests
- [ ] GitHub Actions
- [ ] Automated Docker builds
- [ ] Container registry push

### M4: CD Pipeline & Deployment (10M)
- [ ] Kubernetes manifests
- [ ] Automated deployment
- [ ] Smoke tests
- [ ] Rollback strategy

### M5: Monitoring & Logging (10M)
- [ ] Request/response logging
- [ ] Prometheus metrics
- [ ] Performance tracking
- [ ] Documentation

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **ML Framework** | PyTorch, torchvision |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **API Framework** | FastAPI, Uvicorn |
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Kubernetes |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus, Grafana |
| **Testing** | pytest, pytest-cov |

---

## 🤝 Contributing

This is an academic project. For suggestions or issues, please create a GitHub issue.

---

## 📄 License

This project is for educational purposes (BITS Pilani MLOps Assignment).

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 📝 Notes

- Data is tracked with DVC, not Git
- Models tested and working
- Following MLOps best practices
- Comprehensive documentation

---

**Last Updated**: 2026-02-17

**Status**: 🚧 Work in Progress - 40% Complete

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed progress.

## 📦 Module 2: Model Packaging & Containerization ✅ COMPLETE

### Deliverables
- ✅ FastAPI REST API with inference endpoints
- ✅ Health check and prediction endpoints  
- ✅ requirements.txt with pinned versions
- ✅ Dockerfile with multi-stage build
- ✅ Docker Compose configuration
- ✅ Local testing verified
- ✅ Swagger UI documentation

### API Endpoints
\\\
GET  /health       - Health check
POST /predict      - Image classification
GET  /model/info   - Model details
GET  /docs         - Interactive API docs
\\\

### Quick Start
\\\powershell
# Local API
python src/api/main.py

# Docker
docker build -t cats-dogs-classifier:latest .
docker run -d -p 8000:8000 cats-dogs-classifier:latest

# Docker Compose
docker-compose up -d
\\\

### Test Results
- ✅ All endpoints working
- ✅ Model predictions: 92.12% accuracy
- ✅ Container builds and runs successfully
- ✅ Health checks pass

---
