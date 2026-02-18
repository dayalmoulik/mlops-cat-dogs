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


class Trainer:
    '''
    Trainer class for Cats vs Dogs classification
    
    Handles training loop, validation, MLflow logging, and model checkpointing
    '''
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 10,
                 checkpoint_dir: str = 'models/checkpoints'):
        '''
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on (cuda/cpu)
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save model checkpoints
        '''
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self, epoch: int) -> tuple:
        '''Train for one epoch'''
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> tuple:
        '''Validate the model'''
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]  ')
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        '''Save model checkpoint'''
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f'✓ Saved best model with val_acc: {val_acc:.2f}%')
    
    def plot_training_curves(self, save_path: str = 'training_curves.png'):
        '''Plot and save training curves'''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Saved training curves to {save_path}')
        return save_path
    
    def plot_confusion_matrix(self, labels, predictions, 
                             class_names=['Cat', 'Dog'],
                             save_path: str = 'confusion_matrix.png'):
        '''Plot and save confusion matrix'''
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Saved confusion matrix to {save_path}')
        return save_path
    
    def train(self):
        '''Main training loop'''
        print('='*60)
        print('Starting Training')
        print('='*60)
        print(f'Device: {self.device}')
        print(f'Epochs: {self.num_epochs}')
        print(f'Train batches: {len(self.train_loader)}')
        print(f'Val batches: {len(self.val_loader)}')
        print('='*60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, step=epoch)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{self.num_epochs} Summary:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best)
            print('-'*60)
        
        # Training complete
        total_time = time.time() - start_time
        print('\n' + '='*60)
        print('Training Complete!')
        print('='*60)
        print(f'Total time: {total_time/60:.2f} minutes')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')
        
        # Plot and log curves
        curves_path = self.plot_training_curves()
        mlflow.log_artifact(curves_path)
        
        # Plot and log confusion matrix (using last epoch)
        cm_path = self.plot_confusion_matrix(val_labels, val_preds)
        mlflow.log_artifact(cm_path)
        
        # Generate classification report
        report = classification_report(val_labels, val_preds, 
                                      target_names=['Cat', 'Dog'])
        report_path = 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        print(f'\nClassification Report:\n{report}')
        
        # Log final metrics
        mlflow.log_metrics({
            'final_train_acc': self.train_accs[-1],
            'final_val_acc': self.val_accs[-1],
            'best_val_acc': self.best_val_acc,
            'training_time_minutes': total_time / 60,
        })
        
        return self.best_val_acc


def main():
    '''Main training function'''
    
    # Configuration
    config = {
        'model_name': 'improved',  # 'simple' or 'improved'
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'num_workers': 4,
        'image_size': (224, 224),
        'data_augmentation': True,
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Start MLflow run
    mlflow.set_experiment('cats-vs-dogs-classification')
    
    with mlflow.start_run(run_name=f'training_{config["model_name"]}_epochs{config["num_epochs"]}'):
        
        # Log configuration
        mlflow.log_params(config)
        
        # Create dataloaders
        print('\nLoading datasets...')
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dir='data/train',
            val_dir='data/validation',
            test_dir='data/test',
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
        final_model_path = 'models/final_model.pth'
        Path('models').mkdir(exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, 'model')
        mlflow.log_artifact(final_model_path)
        
        print(f'\n✓ Model saved to {final_model_path}')
        print(f'✓ MLflow run complete')
        print(f'✓ Best validation accuracy: {best_val_acc:.2f}%')


if __name__ == '__main__':
    main()
