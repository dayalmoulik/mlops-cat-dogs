from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
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

from src.models.cnn import get_model
from src.utils.preprocessing import preprocess_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str


@app.on_event('startup')
async def load_model():
    '''Load the trained model on startup'''
    global MODEL, DEVICE, MODEL_LOADED
    
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
        logger.info('✓ Model loaded successfully')
        
    except Exception as e:
        logger.error(f'Failed to load model: {str(e)}')
        MODEL_LOADED = False
        raise


@app.get('/', response_model=dict)
async def root():
    '''Root endpoint with API information'''
    return {
        'message': 'Cats vs Dogs Classifier API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
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
    '''
    Health check endpoint
    
    Returns the current status of the API and model
    '''
    return HealthResponse(
        status='healthy' if MODEL_LOADED else 'unhealthy',
        model_loaded=MODEL_LOADED,
        device=str(DEVICE) if DEVICE else 'unknown',
        timestamp=datetime.now().isoformat()
    )


@app.post('/predict', response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    '''
    Predict whether the uploaded image is a cat or dog
    
    Args:
        file: Image file (JPG, JPEG, PNG)
    
    Returns:
        Prediction with class name, confidence, and probabilities
    
    Example:
        curl -X POST -F \"file=@cat.jpg\" http://localhost:8000/predict
    '''
    
    # Check if model is loaded
    if not MODEL_LOADED or MODEL is None:
        logger.error('Model not loaded')
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
        
        logger.info(f'Prediction: {predicted_class} (confidence: {confidence_score:.4f})')
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence_score,
            probabilities=probs_dict,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f'Prediction error: {str(e)}')
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
