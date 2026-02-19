import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.api.main import app
import io
from PIL import Image

# Create test client
client = TestClient(app)


class TestAPIEndpoints:
    '''Test all API endpoints'''
    
    def test_root_endpoint(self):
        '''Test root endpoint returns API information'''
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
        assert data['version'] == '1.0.0'
    
    def test_health_endpoint(self):
        '''Test health check endpoint'''
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'device' in data
        assert 'timestamp' in data
    
    def test_model_info_endpoint(self):
        '''Test model info endpoint'''
        response = client.get('/model/info')
        assert response.status_code in [200, 503]  # 503 if model not loaded
        
        if response.status_code == 200:
            data = response.json()
            assert 'model_type' in data
            assert 'num_parameters' in data
            assert 'classes' in data
    
    def test_predict_endpoint_with_valid_image(self):
        '''Test prediction with valid image'''
        # Create a test image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = client.post('/predict', files=files)
        
        # Should succeed or return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'prediction' in data
            assert 'confidence' in data
            assert 'probabilities' in data
            assert data['prediction'] in ['cat', 'dog']
            assert 0 <= data['confidence'] <= 1
    
    def test_predict_endpoint_with_invalid_file(self):
        '''Test prediction with invalid file type'''
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        response = client.post('/predict', files=files)
        # Could be 400 (invalid file) or 503 (model not loaded)
        assert response.status_code in [400, 503]
    
    def test_predict_endpoint_without_file(self):
        '''Test prediction without file'''
        response = client.post('/predict')
        assert response.status_code == 422  # Unprocessable Entity


class TestResponseFormats:
    '''Test response formats and schemas'''
    
    def test_health_response_format(self):
        '''Test health response has correct format'''
        response = client.get('/health')
        data = response.json()
        
        # Check required fields
        required_fields = ['status', 'model_loaded', 'device', 'timestamp']
        for field in required_fields:
            assert field in data
        
        # Check field types
        assert isinstance(data['status'], str)
        assert isinstance(data['model_loaded'], bool)
        assert isinstance(data['device'], str)
        assert isinstance(data['timestamp'], str)
    
    def test_prediction_response_format(self):
        '''Test prediction response has correct format'''
        # Create test image
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = client.post('/predict', files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields
            required_fields = ['prediction', 'confidence', 'probabilities', 'timestamp']
            for field in required_fields:
                assert field in data
            
            # Check field types
            assert isinstance(data['prediction'], str)
            assert isinstance(data['confidence'], float)
            assert isinstance(data['probabilities'], dict)
            assert isinstance(data['timestamp'], str)
            
            # Check probabilities
            assert 'cat' in data['probabilities']
            assert 'dog' in data['probabilities']
            assert isinstance(data['probabilities']['cat'], float)
            assert isinstance(data['probabilities']['dog'], float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

