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
                files = {'file': f}
                response = requests.post(f"{self.api_url}/predict", files=files)
            
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
                print(f"Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def collect_batch_predictions(self, test_dir, num_samples=50):
        """Collect predictions on a batch of test images"""
        print(f"Collecting {num_samples} predictions...")
        
        # Get cat images
        cat_images = list(Path(test_dir).glob('cats/*.jpg'))[:num_samples//2]
        dog_images = list(Path(test_dir).glob('dogs/*.jpg'))[:num_samples//2]
        
        # Make predictions
        for img in cat_images:
            result = self.make_prediction(img, 'cat')
            if result:
                print(f"✓ Predicted: {result['predicted_label']}, True: cat, Correct: {result['correct']}")
            time.sleep(0.1)
        
        for img in dog_images:
            result = self.make_prediction(img, 'dog')
            if result:
                print(f"✓ Predicted: {result['predicted_label']}, True: dog, Correct: {result['correct']}")
            time.sleep(0.1)
    
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
        print(f"Predictions per class: {metrics['predictions_per_class']}")
        print('='*60)
        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    import sys
    
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_dir = sys.argv[2] if len(sys.argv) > 2 else "data/test"
    
    tracker = PerformanceTracker(api_url)
    tracker.collect_batch_predictions(test_dir, num_samples=50)
    tracker.save_results()
