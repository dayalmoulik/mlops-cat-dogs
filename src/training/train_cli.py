import argparse
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnn import get_model
from src.utils.dataset import create_dataloaders
from src.training.train import create_trainer


def main():
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs Classifier')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'improved'],
                       help='Model architecture to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, default='data/train',
                       help='Path to training data')
    parser.add_argument('--val-dir', type=str, default='data/validation',
                       help='Path to validation data')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size for training')
    
    # Augmentation
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='evaluation_results/train_results',
                       help='Directory to save checkpoints and results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment (auto-generated if not provided)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print('='*70)
    print(' '*20 + 'TRAINING CONFIGURATION')
    print('='*70)
    print(f'Model: {args.model}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch Size: {args.batch_size}')
    print(f'Learning Rate: {args.lr}')
    print(f'Device: {device}')
    print(f'Data Augmentation: {not args.no_augmentation}')
    print(f'Results Directory: {args.checkpoint_dir}')
    print('='*70)
    
    # Create data loaders
    print('\nLoading data...')
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation=not args.no_augmentation
    )
    
    print(f'✓ Training samples: {len(train_loader.dataset):,}')
    print(f'✓ Validation samples: {len(val_loader.dataset):,}')
    print(f'✓ Training batches: {len(train_loader)}')
    print(f'✓ Validation batches: {len(val_loader)}')
    
    # Create model
    print(f'\nCreating {args.model} model...')
    model = get_model(args.model, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'✓ Model created with {total_params:,} parameters')
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    print('\n' + '='*70)
    print('Training Complete!')
    print(f'Best Validation Accuracy: {trainer.best_val_acc:.2f}%')
    print(f'Results saved to: {trainer.exp_dir}')
    print('='*70)


if __name__ == '__main__':
    main()
