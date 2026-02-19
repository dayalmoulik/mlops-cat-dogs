from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from PIL import Image
import io
import logging
from datetime import datetime
from pathlib import Path
import os
import time

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.models.cnn import get_model
from src.utils.preprocessing import preprocess_image
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level=os.getenv('LOG_LEVEL', 'INFO'))
logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['predicted_class']
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Prediction confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time taken to load the model'
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests'
)

# Create FastAPI app
app = FastAPI(
    title='Cats vs Dogs Classifier API',
    description='Binary image classification API for cats and dogs using deep learning',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Global variables
MODEL = None
DEVICE = None
CLASS_NAMES = ['cat', 'dog']
MODEL_LOADED = False

# Response models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    timestamp: str
    processing_time_ms: float


@app.middleware("http")
async def add_process_time_and_logging(request: Request, call_next):
    """Middleware for logging and metrics"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    # Log request
    logger.info(
        "Request received",
        extra={
            'method': request.method,
            'url': str(request.url),
            'client': request.client.host if request.client else None
        }
    )
    
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    ACTIVE_REQUESTS.dec()
    
    # Log response
    logger.info(
        "Request completed",
        extra={
            'method': request.method,
            'url': str(request.url),
            'status': response.status_code,
            'duration': duration
        }
    )
    
    # Add custom header
    response.headers["X-Process-Time"] = str(duration)
    
    return response


@app.on_event('startup')
async def load_model():
    '''Load the trained model on startup'''
    global MODEL, DEVICE, MODEL_LOADED
    
    start_time = time.time()
    
    try:
        logger.info('Loading model...')
        
        # Set device
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {DEVICE}')
        
        # Model path from environment variable or default
        model_path = os.getenv('MODEL_PATH', 'models/checkpoints/best_model.pth')
        model_name = os.getenv('MODEL_NAME', 'improved')
        
        logger.info(f'Loading model from: {model_path}')
        
        if not Path(model_path).exists():
            logger.error(f'Model file not found: {model_path}')
            raise FileNotFoundError(f'Model file not found: {model_path}')
        
        # Load model
        MODEL = get_model(model_name, num_classes=2)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            MODEL.load_state_dict(checkpoint['model_state_dict'])
        else:
            MODEL.load_state_dict(checkpoint)
        
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        
        MODEL_LOADED = True
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        
        logger.info(f'Model loaded successfully in {load_time:.2f}s')
        
    except Exception as e:
        logger.error(f'Failed to load model: {str(e)}', exc_info=True)
        MODEL_LOADED = False
        raise


@app.get('/', response_model=dict)
async def root():
    '''Root endpoint with API information'''
    logger.debug('Root endpoint accessed')
    return {
        'message': 'Cats vs Dogs Classifier API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'metrics': '/metrics',
            'docs': '/docs',
            'redoc': '/redoc'
        },
        'model_info': {
            'type': 'ImprovedCNN',
            'classes': CLASS_NAMES,
            'input_size': '224x224',
            'accuracy': '92.12%'
        }
    }


@app.get('/health', response_model=HealthResponse)
async def health_check():
    '''Health check endpoint'''
    return HealthResponse(
        status='healthy' if MODEL_LOADED else 'unhealthy',
        model_loaded=MODEL_LOADED,
        device=str(DEVICE) if DEVICE else 'unknown',
        timestamp=datetime.now().isoformat()
    )


@app.get('/metrics')
async def metrics():
    '''Prometheus metrics endpoint'''
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post('/predict', response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    '''Predict whether the uploaded image is a cat or dog'''
    start_time = time.time()
    
    # Check if model is loaded
    if not MODEL_LOADED or MODEL is None:
        logger.error('Prediction attempted but model not loaded')
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        logger.warning(f'Invalid file type: {file.content_type}')
        raise HTTPException(
            status_code=400,
            detail=f'Invalid file type. Expected image, got {file.content_type}'
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        logger.info(f'Processing image: {file.filename}, size: {image.size}')
        
        # Preprocess
        input_tensor = preprocess_image(image, augment=False).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get results
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()
        
        probs_dict = {
            'cat': float(probabilities[0][0].item()),
            'dog': float(probabilities[0][1].item())
        }
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Record metrics
        PREDICTION_COUNT.labels(predicted_class=predicted_class).inc()
        PREDICTION_CONFIDENCE.observe(confidence_score)
        
        logger.info(
            f'Prediction: {predicted_class} (confidence: {confidence_score:.4f})',
            extra={
                'prediction': predicted_class,
                'confidence': confidence_score,
                'processing_time_ms': processing_time
            }
        )
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence_score,
            probabilities=probs_dict,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f'Prediction error: {str(e)}', exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f'Prediction failed: {str(e)}'
        )


@app.get('/model/info')
async def model_info():
    '''Get information about the loaded model'''
    if not MODEL_LOADED or MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    try:
        num_params = sum(p.numel() for p in MODEL.parameters())
        
        return {
            'model_loaded': MODEL_LOADED,
            'model_type': MODEL.__class__.__name__,
            'num_parameters': num_params,
            'device': str(DEVICE),
            'classes': CLASS_NAMES,
            'input_size': '224x224',
            'test_accuracy': '92.12%'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    
    port = int(os.getenv('PORT', 8000))
    
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=port,
        reload=True,
        log_level='info'
    )
