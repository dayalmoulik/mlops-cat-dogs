# Training Guide

## Quick Start

\\\powershell
# Train with default settings
python src/training/train.py
\\\

## MLflow Tracking

Start MLflow UI to view experiments:

\\\powershell
mlflow ui --port 5000
# Open: http://localhost:5000
\\\

## Configuration

Edit the config dictionary in \src/training/train.py\:

\\\python
config = {
    'model_name': 'simple',      # 'simple' or 'improved'
    'num_epochs': 10,            # Number of training epochs
    'batch_size': 32,            # Batch size
    'learning_rate': 0.001,      # Learning rate
    'num_workers': 4,            # Data loading workers
    'image_size': (224, 224),    # Input image size
    'data_augmentation': True,   # Enable augmentation
}
\\\

## Training Output

After training, you'll find:

- **models/checkpoints/best_model.pth** - Best model weights
- **models/checkpoints/last_checkpoint.pth** - Last epoch weights
- **models/final_model.pth** - Final model
- **training_curves.png** - Loss and accuracy plots
- **confusion_matrix.png** - Confusion matrix
- **classification_report.txt** - Detailed metrics
- **mlruns/** - MLflow experiment data

## Logged Metrics

MLflow tracks:
- Train/val loss per epoch
- Train/val accuracy per epoch
- Best validation accuracy
- Training time
- Model parameters

## Expected Results

With SimpleCNN (10 epochs):
- Training accuracy: ~95-98%
- Validation accuracy: ~85-90%

With ImprovedCNN (15 epochs):
- Training accuracy: ~97-99%
- Validation accuracy: ~90-93%

## Tips

1. **Monitor for overfitting**: Watch for validation loss increasing while training loss decreases
2. **Adjust learning rate**: If loss plateaus, try reducing learning rate
3. **Use GPU**: Training is much faster on GPU
4. **Batch size**: Larger batch size = faster training but needs more memory

## Troubleshooting

**Out of memory:**
- Reduce batch_size to 16 or 8
- Reduce num_workers

**Training too slow:**
- Use GPU if available
- Increase num_workers
- Increase batch_size

**Low accuracy:**
- Train for more epochs
- Adjust learning rate
- Try ImprovedCNN model
- Enable data augmentation

## CLI Training (Recommended)

### Quick Start
\\\powershell
# Train SimpleCNN
python src/training/train_cli.py --model simple --epochs 10

# Train ImprovedCNN
python src/training/train_cli.py --model improved --epochs 15 --lr 0.0005
\\\

### All Options
\\\powershell
python src/training/train_cli.py --help
\\\

### Using Convenience Scripts
\\\powershell
# SimpleCNN
powershell scripts/train_simple.ps1

# ImprovedCNN
powershell scripts/train_improved.ps1

# Quick test
powershell scripts/train_quick_test.ps1
\\\

See [TRAINING_EXAMPLES.md](TRAINING_EXAMPLES.md) for more examples.
