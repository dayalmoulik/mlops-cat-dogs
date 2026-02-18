# FastAPI Inference Service

## Overview
REST API for serving the Cats vs Dogs classifier model.

## Features
- ✅ Health check endpoint
- ✅ Image prediction endpoint
- ✅ Model information endpoint
- ✅ Auto-generated API docs (Swagger UI)
- ✅ Request logging
- ✅ Error handling
- ✅ CORS enabled

## Running the API

### Local Development
\\\powershell
# Start the server
python src/api/main.py

# Server will start on http://localhost:8000
\\\

### Production
\\\powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
\\\

## API Endpoints

### 1. Root - GET /
Returns API information and available endpoints

**Response:**
\\\json
{
  \"message\": \"Cats vs Dogs Classifier API\",
  \"version\": \"1.0.0\",
  \"status\": \"running\",
  \"endpoints\": {...}
}
\\\

### 2. Health Check - GET /health
Check if the API and model are running

**Response:**
\\\json
{
  \"status\": \"healthy\",
  \"model_loaded\": true,
  \"device\": \"cuda\",
  \"timestamp\": \"2024-02-18T10:30:00\"
}
\\\

### 3. Predict - POST /predict
Classify an uploaded image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image)

**Example:**
\\\powershell
curl -X POST -F \"file=@cat.jpg\" http://localhost:8000/predict
\\\

**Response:**
\\\json
{
  \"prediction\": \"cat\",
  \"confidence\": 0.9567,
  \"probabilities\": {
    \"cat\": 0.9567,
    \"dog\": 0.0433
  },
  \"timestamp\": \"2024-02-18T10:30:00\"
}
\\\

### 4. Model Info - GET /model/info
Get information about the loaded model

**Response:**
\\\json
{
  \"model_loaded\": true,
  \"model_type\": \"ImprovedCNN\",
  \"num_parameters\": 2768386,
  \"device\": \"cuda\",
  \"classes\": [\"cat\", \"dog\"],
  \"test_accuracy\": \"92.12%\"
}
\\\

## Interactive Documentation

### Swagger UI
Navigate to: http://localhost:8000/docs

### ReDoc
Navigate to: http://localhost:8000/redoc

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | models/checkpoints/best_model.pth | Path to model file |
| MODEL_NAME | improved | Model architecture name |
| PORT | 8000 | Port to run server on |

**Example:**
\\\powershell
\='models/improved/model_e15.pth'
\=5000
python src/api/main.py
\\\

## Testing

### Manual Testing
\\\powershell
# Run test script
python scripts/test_api.py
\\\

### Using curl
\\\powershell
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST -F \"file=@path/to/image.jpg\" http://localhost:8000/predict
\\\

### Using Python
\\\python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Prediction
with open('cat.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
print(response.json())
\\\

## Error Handling

### Status Codes
- **200**: Success
- **400**: Bad Request (invalid file type)
- **500**: Server Error (prediction failed)
- **503**: Service Unavailable (model not loaded)

### Error Response Format
\\\json
{
  \"detail\": \"Error message here\"
}
\\\

## Performance

- **Response Time**: ~100-500ms per prediction (CPU)
- **Response Time**: ~50-100ms per prediction (GPU)
- **Throughput**: ~10-20 requests/second (single worker)

## Logging

Logs include:
- Request timestamps
- Image dimensions
- Predictions and confidence scores
- Errors and exceptions

View logs in console output.

## Next Steps

1. ✅ API implemented
2. 🚀 Containerize with Docker
3. ☸️ Deploy to Kubernetes

