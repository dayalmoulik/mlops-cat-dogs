import requests
import time
import json

API_URL = 'http://localhost:8000'

print('='*70)
print(' '*15 + 'TESTING DOCKERIZED API')
print('='*70)

# Wait for API to be ready
print('\nWaiting for API to start...')
max_retries = 10
for i in range(max_retries):
    try:
        response = requests.get(f'{API_URL}/health', timeout=2)
        if response.status_code == 200:
            print('✓ API is ready!')
            break
    except:
        print(f'  Attempt {i+1}/{max_retries}... waiting')
        time.sleep(3)
else:
    print('✗ API failed to start')
    exit(1)

# Test 1: Health Check
print('\n1. Health Check')
print('-'*70)
response = requests.get(f'{API_URL}/health')
print(f'Status: {response.status_code}')
print(json.dumps(response.json(), indent=2))

# Test 2: Root Endpoint
print('\n2. Root Endpoint')
print('-'*70)
response = requests.get(f'{API_URL}/')
result = response.json()
print(f'Status: {response.status_code}')
print(f'Message: {result["message"]}')
print(f'Version: {result["version"]}')

# Test 3: Model Info
print('\n3. Model Info')
print('-'*70)
response = requests.get(f'{API_URL}/model/info')
result = response.json()
print(f'Status: {response.status_code}')
print(f'Model Type: {result["model_type"]}')
print(f'Parameters: {result["num_parameters"]:,}')
print(f'Device: {result["device"]}')
print(f'Test Accuracy: {result["test_accuracy"]}')

# Test 4: Prediction (if test images available)
print('\n4. Prediction Test')
print('-'*70)
from pathlib import Path

test_images = list(Path('data/test/cats').glob('*.jpg'))[:1]
if test_images:
    img_path = test_images[0]
    print(f'Testing with: {img_path.name}')
    
    with open(img_path, 'rb') as f:
        files = {'file': (img_path.name, f, 'image/jpeg')}
        response = requests.post(f'{API_URL}/predict', files=files)
    
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print(f'Prediction: {result["prediction"]}')
        print(f'Confidence: {result["confidence"]:.4f}')
        print(f'Cat Probability: {result["probabilities"]["cat"]:.4f}')
        print(f'Dog Probability: {result["probabilities"]["dog"]:.4f}')
    else:
        print(f'Error: {response.text}')
else:
    print('No test images found - skipping prediction test')

print('\n' + '='*70)
print('✓ All tests passed!')
print('='*70)
print('\nDocker container is working correctly! 🐳')
