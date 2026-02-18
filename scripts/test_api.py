import requests
from pathlib import Path
import json

API_URL = 'http://localhost:8000'

print('='*70)
print(' '*20 + 'API TESTING')
print('='*70)

# Test 1: Health Check
print('\n1. Testing Health Endpoint')
print('-'*70)
try:
    response = requests.get(f'{API_URL}/health')
    print(f'Status Code: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
except Exception as e:
    print(f'Error: {e}')

# Test 2: Root Endpoint
print('\n2. Testing Root Endpoint')
print('-'*70)
try:
    response = requests.get(f'{API_URL}/')
    print(f'Status Code: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
except Exception as e:
    print(f'Error: {e}')

# Test 3: Model Info
print('\n3. Testing Model Info Endpoint')
print('-'*70)
try:
    response = requests.get(f'{API_URL}/model/info')
    print(f'Status Code: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
except Exception as e:
    print(f'Error: {e}')

# Test 4: Prediction with Sample Image
print('\n4. Testing Prediction Endpoint')
print('-'*70)

# Find a test image
test_image_paths = list(Path('data/test/cats').glob('*.jpg'))[:1] + \
                   list(Path('data/test/dogs').glob('*.jpg'))[:1]

if test_image_paths:
    for img_path in test_image_paths:
        print(f'\nTesting with: {img_path.name}')
        try:
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                response = requests.post(f'{API_URL}/predict', files=files)
            
            print(f'Status Code: {response.status_code}')
            if response.status_code == 200:
                result = response.json()
                print(f'Prediction: {result["prediction"]}')
                print(f'Confidence: {result["confidence"]:.4f}')
                print(f'Probabilities: {result["probabilities"]}')
            else:
                print(f'Error: {response.text}')
        except Exception as e:
            print(f'Error: {e}')
else:
    print('No test images found. Please ensure data/test/ exists.')

print('\n' + '='*70)
print('✓ API Testing Complete')
print('='*70)
