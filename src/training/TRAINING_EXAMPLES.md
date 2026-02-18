# Training Examples and Quick Reference

## Basic Usage

### Train SimpleCNN (default)
python src/training/train_cli.py

### Train ImprovedCNN
python src/training/train_cli.py --model improved

## Recommended Configurations

### SimpleCNN (Quick Training)
python src/training/train_cli.py \
    --model simple \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001

### ImprovedCNN (Better Accuracy)
python src/training/train_cli.py \
    --model improved \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.0005

### Production Training (High Accuracy)
python src/training/train_cli.py \
    --model improved \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.0001 \
    --workers 8

### Quick Test (Fast Iteration)
python src/training/train_cli.py \
    --model simple \
    --epochs 2 \
    --batch-size 16 \
    --workers 0

## Advanced Options

### Disable Data Augmentation
python src/training/train_cli.py --no-augmentation

### Custom Image Size
python src/training/train_cli.py --image-size 256

### Custom Checkpoint Directory
python src/training/train_cli.py --checkpoint-dir models/my_experiment

### Custom MLflow Run Name
python src/training/train_cli.py --run-name my_awesome_model

### Force CPU Training
python src/training/train_cli.py --device cpu

## View All Options
python src/training/train_cli.py --help

## Comparison: SimpleCNN vs ImprovedCNN

| Parameter      | SimpleCNN     | ImprovedCNN   | Notes                    |
|----------------|---------------|---------------|--------------------------|
| Model Size     | 11.2M params  | 2.8M params   | Improved is more efficient|
| Epochs         | 10            | 15            | Improved needs more epochs|
| Learning Rate  | 0.001         | 0.0005        | Lower LR = more stable   |
| Expected Acc   | 85-90%        | 90-93%        | On validation set        |
| Training Time  | ~30 min (CPU) | ~40 min (CPU) | Approximate              |

## Monitoring

View training progress in MLflow UI:
mlflow ui --port 5000
# Open: http://localhost:5000
