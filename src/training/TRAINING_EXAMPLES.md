# Training Examples

## Quick Training

### Train SimpleCNN (10 epochs)
```powershell
python src/training/train_cli.py --model simple --epochs 10
```

Results will be saved to: `evaluation_results/train_results/training_YYYYMMDD_HHMMSS/`

### Train ImprovedCNN (15 epochs)
```powershell
python src/training/train_cli.py --model improved --epochs 15 --lr 0.0005
```

## Custom Training

### With Custom Experiment Name
```powershell
python src/training/train_cli.py \
    --model improved \
    --epochs 20 \
    --experiment-name "improved_v2_20epochs"
```

Results will be saved to: `evaluation_results/train_results/improved_v2_20epochs/`

### With Custom Results Directory
```powershell
python src/training/train_cli.py \
    --model simple \
    --epochs 10 \
    --checkpoint-dir "my_experiments/run1"
```

### Without Data Augmentation
```powershell
python src/training/train_cli.py --model simple --no-augmentation
```

## Output Files

After training, you will find these files in `evaluation_results/train_results/{experiment_name}/`:
```
evaluation_results/train_results/training_20260220_103045/
├── best_model.pth              # Best model checkpoint
├── last_checkpoint.pth         # Latest checkpoint
├── training_curves.png         # Loss and accuracy plots
├── confusion_matrix.png        # Validation confusion matrix
├── classification_report.txt   # Detailed metrics
└── training_history.json       # Complete training history
```

## All Command Line Options
```
--model              Model architecture (simple/improved)
--epochs             Number of training epochs (default: 10)
--batch-size         Batch size (default: 32)
--lr                 Learning rate (default: 0.001)
--train-dir          Training data directory (default: data/train)
--val-dir            Validation data directory (default: data/validation)
--num-workers        Data loading workers (default: 4)
--image-size         Input image size (default: 224)
--no-augmentation    Disable data augmentation
--checkpoint-dir     Results directory (default: evaluation_results/train_results)
--experiment-name    Custom experiment name
--device             Device to use (cuda/cpu, auto-detect if not specified)
```

## Recommended Settings

### SimpleCNN
```powershell
python src/training/train_cli.py \
    --model simple \
    --epochs 10 \
    --lr 0.001 \
    --batch-size 32
```

### ImprovedCNN
```powershell
python src/training/train_cli.py \
    --model improved \
    --epochs 15 \
    --lr 0.0005 \
    --batch-size 32
```

## Monitoring Training

Training progress is shown with progress bars and epoch summaries:
```
Epoch 5/10
------------------------------------------------------------
Training: 100%|████████████████| 625/625 [02:15<00:00, 4.61it/s, loss=0.3245, acc=87.45%]
Validation: 100%|████████████████| 79/79 [00:15<00:00, 5.23it/s, loss=0.2891, acc=89.12%]

Epoch 5 Summary:
  Train Loss: 0.3245 | Train Acc: 87.45%
  Val Loss: 0.2891   | Val Acc: 89.12%
  ✓ New best model saved! (Val Acc: 89.12%)
```

## Resume Training

To resume from a checkpoint, you can load the saved model:
```python
checkpoint = torch.load('evaluation_results/train_results/training_20260220_103045/last_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```
