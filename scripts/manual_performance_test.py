"""
Manual Performance Tracking Script
Run this after deploying the API locally with: docker-compose up -d
"""

import requests
import json
from datetime import datetime
from pathlib import Path

def manual_performance_test():
    """Simple manual performance test"""
    
    api_url = "http://localhost:8000"
    
    print("="*60)
    print("Manual Performance Tracking")
    print("="*60)
    
    # 1. Check API is running
    print("\n1. Checking API health...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ API is running")
            health = response.json()
            print(f"   ✓ Model loaded: {health['model_loaded']}")
            print(f"   ✓ Device: {health['device']}")
        else:
            print(f"   ✗ API health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ✗ Cannot connect to API: {e}")
        print("   Please start API with: docker-compose up -d")
        return
    
    # 2. Find test images
    print("\n2. Finding test images...")
    test_dir = Path("data/test")
    
    if not test_dir.exists():
        print("   ✗ Test directory not found")
        print("   Please ensure data/test exists with cat and dog images")
        return
    
    # Find images
    cat_images = list(test_dir.rglob("*cat*.jpg"))[:10]
    dog_images = list(test_dir.rglob("*dog*.jpg"))[:10]
    
    print(f"   ✓ Found {len(cat_images)} cat images")
    print(f"   ✓ Found {len(dog_images)} dog images")
    
    if not cat_images and not dog_images:
        print("   ✗ No images found")
        return
    
    # 3. Make predictions
    print("\n3. Making predictions...")
    results = []
    
    for img_path in cat_images:
        try:
            with open(img_path, 'rb') as f:
                files = {'file': ('image.jpg', f, 'image/jpeg')}
                response = requests.post(f"{api_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                correct = result['prediction'] == 'cat'
                results.append({
                    'true': 'cat',
                    'predicted': result['prediction'],
                    'confidence': result['confidence'],
                    'correct': correct
                })
                status = "✓" if correct else "✗"
                print(f"   {status} Cat: predicted={result['prediction']}, conf={result['confidence']:.3f}")
            else:
                print(f"   ✗ Error {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    for img_path in dog_images:
        try:
            with open(img_path, 'rb') as f:
                files = {'file': ('image.jpg', f, 'image/jpeg')}
                response = requests.post(f"{api_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                correct = result['prediction'] == 'dog'
                results.append({
                    'true': 'dog',
                    'predicted': result['prediction'],
                    'confidence': result['confidence'],
                    'correct': correct
                })
                status = "✓" if correct else "✗"
                print(f"   {status} Dog: predicted={result['prediction']}, conf={result['confidence']:.3f}")
            else:
                print(f"   ✗ Error {response.status_code}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # 4. Calculate metrics
    if results:
        print("\n4. Performance Metrics")
        print("="*60)
        accuracy = sum(r['correct'] for r in results) / len(results)
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        
        print(f"Total Predictions: {len(results)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average Confidence: {avg_conf:.4f}")
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(results),
            'accuracy': accuracy,
            'avg_confidence': avg_conf,
            'predictions': results
        }
        
        with open('performance_tracking.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: performance_tracking.json")
    else:
        print("\n✗ No successful predictions")
    
    print("="*60)

if __name__ == '__main__':
    manual_performance_test()
