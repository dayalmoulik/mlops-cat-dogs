import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.training.evaluate import load_trained_model, ModelEvaluator
from src.utils.dataset import CatsDogsDataset
from src.utils.preprocessing import get_val_transforms
from torch.utils.data import DataLoader

# Configuration
MODEL_PATH = 'models/checkpoints/best_model.pth'  # Change this to your model path
MODEL_NAME = 'simple'  # or 'improved'
TEST_DIR = 'data/test'

print('Quick Model Evaluation')
print('='*60)

# Check if model exists
if not Path(MODEL_PATH).exists():
    print(f'✗ Model not found: {MODEL_PATH}')
    print('  Please train a model first or update MODEL_PATH')
    sys.exit(1)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load model
print(f'\nLoading model: {MODEL_PATH}')
model = load_trained_model(MODEL_PATH, MODEL_NAME, device)

# Load test data
print('Loading test dataset...')
test_dataset = CatsDogsDataset(TEST_DIR, transform=get_val_transforms())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f'✓ Test samples: {len(test_dataset)}')

# Evaluate
evaluator = ModelEvaluator(model, test_loader, device)
results = evaluator.evaluate()

# Generate reports
print('\nGenerating visualizations...')
output_dir = Path('evaluation_results')
output_dir.mkdir(exist_ok=True)

evaluator.plot_confusion_matrix(
    results['labels'], 
    results['predictions'],
    save_path=output_dir / 'confusion_matrix.png'
)

evaluator.plot_roc_curve(
    results['labels'],
    results['probabilities'],
    save_path=output_dir / 'roc_curve.png'
)

evaluator.generate_classification_report(
    results['labels'],
    results['predictions'],
    save_path=output_dir / 'classification_report.txt'
)

evaluator.save_results(results, save_path=output_dir / 'test_results.json')

print(f'\n✓ Evaluation complete! Results saved to {output_dir}')
