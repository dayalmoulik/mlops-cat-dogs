import requests
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd


class PerformanceTracker:
    """Track model performance post-deployment"""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.predictions = []
        
    def make_prediction(self, image_path, true_label):
        """Make prediction and record result"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                response = requests.post(f"{self.api_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Record prediction
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'image': str(image_path),
                    'true_label': true_label,
                    'predicted_label': result['prediction'],
                    'confidence': result['confidence'],
                    'correct': result['prediction'] == true_label,
                    'processing_time_ms': result.get('processing_time_ms', 0)
                }
                
                self.predictions.append(record)
                return record
            else:
                print(f"Error {response.status_code}: {response.text[:100]}")
                return None
                
        except Exception as e:
            print(f"Error making prediction for {image_path.name}: {e}")
            return None
    
    def collect_batch_predictions(self, test_dir, num_samples=50):
        """Collect predictions on a batch of test images"""
        test_path = Path(test_dir)
        
        if not test_path.exists():
            print(f"Error: Test directory not found: {test_dir}")
            return
        
        print(f"Collecting predictions from {test_dir}...")
        print(f"Target: {num_samples} samples ({num_samples//2} cats, {num_samples//2} dogs)")
        print()
        
        # Get cat images - try different possible directory structures
        cat_dirs = [
            test_path / 'cats',
            test_path / 'Cat',
            test_path
        ]
        
        dog_dirs = [
            test_path / 'dogs',
            test_path / 'Dog',
            test_path
        ]
        
        cat_images = []
        for cat_dir in cat_dirs:
            if cat_dir.exists():
                # Try different patterns
                cat_images.extend(list(cat_dir.glob('*cat*.jpg')))
                cat_images.extend(list(cat_dir.glob('*Cat*.jpg')))
                cat_images.extend(list(cat_dir.glob('cat*.jpg')))
                cat_images.extend(list(cat_dir.glob('test_cats*.jpg')))
                if cat_images:
                    break
        
        dog_images = []
        for dog_dir in dog_dirs:
            if dog_dir.exists():
                # Try different patterns
                dog_images.extend(list(dog_dir.glob('*dog*.jpg')))
                dog_images.extend(list(dog_dir.glob('*Dog*.jpg')))
                dog_images.extend(list(dog_dir.glob('dog*.jpg')))
                dog_images.extend(list(dog_dir.glob('test_dogs*.jpg')))
                if dog_images:
                    break
        
        # Remove duplicates
        cat_images = list(set(cat_images))[:num_samples//2]
        dog_images = list(set(dog_images))[:num_samples//2]
        
        print(f"Found {len(cat_images)} cat images")
        print(f"Found {len(dog_images)} dog images")
        print()
        
        if not cat_images and not dog_images:
            print("Error: No images found!")
            print(f"Please check that images exist in: {test_dir}")
            return
        
        # Make predictions on cat images
        success_count = 0
        for i, img in enumerate(cat_images, 1):
            result = self.make_prediction(img, 'cat')
            if result:
                status = "✓" if result['correct'] else "✗"
                print(f"{status} Cat {i}/{len(cat_images)}: Predicted={result['predicted_label']}, Confidence={result['confidence']:.3f}")
                success_count += 1
            time.sleep(0.1)
        
        # Make predictions on dog images
        for i, img in enumerate(dog_images, 1):
            result = self.make_prediction(img, 'dog')
            if result:
                status = "✓" if result['correct'] else "✗"
                print(f"{status} Dog {i}/{len(dog_images)}: Predicted={result['predicted_label']}, Confidence={result['confidence']:.3f}")
                success_count += 1
            time.sleep(0.1)
        
        print()
        print(f"Successfully processed: {success_count}/{len(cat_images) + len(dog_images)} images")
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.predictions:
            print("No predictions to analyze")
            return None
        
        df = pd.DataFrame(self.predictions)
        
        metrics = {
            'total_predictions': len(df),
            'accuracy': df['correct'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'avg_processing_time_ms': df['processing_time_ms'].mean(),
            'predictions_per_class': df['predicted_label'].value_counts().to_dict(),
            'correct_per_class': df.groupby('true_label')['correct'].mean().to_dict()
        }
        
        return metrics
    
    def save_results(self, output_file='performance_tracking.json'):
        """Save tracking results"""
        metrics = self.calculate_metrics()
        
        if metrics is None:
            print("\n✗ No results to save")
            return
        
        results = {
            'tracking_date': datetime.now().isoformat(),
            'api_url': self.api_url,
            'metrics': metrics,
            'predictions': self.predictions
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Performance Tracking Results")
        print('='*60)
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Avg Confidence: {metrics['avg_confidence']:.4f}")
        print(f"Avg Processing Time: {metrics['avg_processing_time_ms']:.2f}ms")
        print(f"\nPredictions per class:")
        for class_name, count in metrics['predictions_per_class'].items():
            print(f"  {class_name}: {count}")
        print(f"\nAccuracy per class:")
        for class_name, acc in metrics['correct_per_class'].items():
            print(f"  {class_name}: {acc:.2%}")
        print('='*60)
        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    import sys
    
    # Check if API is running
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("="*60)
    print("Performance Tracking")
    print("="*60)
    print(f"API URL: {api_url}")
    
    # Test API connection
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ API is reachable")
            health = response.json()
            print(f"✓ Model loaded: {health.get('model_loaded', 'unknown')}")
        else:
            print(f"✗ API returned status {response.status_code}")
            print("Please ensure the API is running: docker-compose up -d")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print("Please ensure the API is running: docker-compose up -d")
        sys.exit(1)
    
    print()
    
    # Get test directory
    test_dir = sys.argv[2] if len(sys.argv) > 2 else "data/test"
    
    # Create tracker and collect predictions
    tracker = PerformanceTracker(api_url)
    tracker.collect_batch_predictions(test_dir, num_samples=50)
    tracker.save_results()
