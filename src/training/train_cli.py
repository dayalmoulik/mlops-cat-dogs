import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from src.models.cnn import get_model
from src.utils.dataset import create_dataloaders
from src.utils.preprocessing import get_train_transforms, get_val_transforms
from src.training.train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs Classifier')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'improved'],
                       help='Model architecture to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Data arguments
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    # Paths
    parser.add_argument('--train-dir', type=str, default='data/train',
                       help='Path to training data')
    parser.add_argument('--val-dir', type=str, default='data/validation',
                       help='Path to validation data')
    parser.add_argument('--test-dir', type=str, default='data/test',
                       help='Path to test data')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config from arguments
    config = {
        'model_name': args.model,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_workers': args.workers,
        'image_size': (224, 224),
        'data_augmentation': not args.no_augmentation,
    }
    
    print('='*60)
    print('Training Configuration')
    print('='*60)
    for key, value in config.items():
        print(f'{key:20}: {value}')
    print('='*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Start MLflow run
    mlflow.set_experiment('cats-vs-dogs-classification')
    
    with mlflow.start_run(run_name=f'{config["model_name"]}_e{config["num_epochs"]}_bs{config["batch_size"]}'):
        
        # Log configuration
        mlflow.log_params(config)
        
        # Create dataloaders
        print('Loading datasets...')
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            train_transform=get_train_transforms(
                image_size=config['image_size'],
                augmentation=config['data_augmentation']
            ),
            val_transform=get_val_transforms(image_size=config['image_size'])
        )
        
        print(f'✓ Train: {len(train_loader.dataset)} images')
        print(f'✓ Val: {len(val_loader.dataset)} images')
        print(f'✓ Test: {len(test_loader.dataset)} images')
        
        # Create model
        print(f'\nCreating {config["model_name"]} model...')
        model = get_model(config['model_name'], num_classes=2)
        model = model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'✓ Model created with {num_params:,} parameters')
        mlflow.log_param('num_parameters', num_params)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=config['num_epochs']
        )
        
        # Train
        best_val_acc = trainer.train()
        
        # Save final model
        model_dir = Path('models') / config['model_name']
        model_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = model_dir / 'final_model.pth'
        
        torch.save(model.state_dict(), final_model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, 'model')
        mlflow.log_artifact(str(final_model_path))
        
        print(f'\n✓ Model saved to {final_model_path}')
        print(f'✓ MLflow run complete')
        print(f'✓ Best validation accuracy: {best_val_acc:.2f}%')


if __name__ == '__main__':
    main()
