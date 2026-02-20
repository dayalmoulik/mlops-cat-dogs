import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime


class Trainer:
    """
    Training class for CNN models
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "evaluation_results/train_results",
        experiment_name: str = None
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name for this training run
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment name with timestamp
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"training_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.exp_dir = self.checkpoint_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
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
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs: int):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f'\nStarting training for {num_epochs} epochs...')
        print(f'Results will be saved to: {self.exp_dir}')
        print('='*60)
        
        for epoch in range(1, num_epochs + 1):
            print(f'\nEpoch {epoch}/{num_epochs}')
            print('-' * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f'\nEpoch {epoch} Summary:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f'  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)')
            
            # Save last checkpoint
            self.save_checkpoint('last_checkpoint.pth', epoch, val_acc)
        
        print('\n' + '='*60)
        print('Training completed!')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')
        print('='*60)
        
        # Generate and save plots
        self.plot_training_curves()
        self.plot_confusion_matrix(val_labels, val_preds)
        self.save_training_history()
        
        print(f'\nAll results saved to: {self.exp_dir}')
    
    def save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint_path = self.exp_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.exp_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Training curves saved to: {save_path}')
    
    def plot_confusion_matrix(self, labels, predictions):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Cat', 'Dog'],
                   yticklabels=['Cat', 'Dog'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Validation Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Save plot
        save_path = self.exp_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Confusion matrix saved to: {save_path}')
        
        # Save classification report
        report = classification_report(labels, predictions, 
                                      target_names=['Cat', 'Dog'],
                                      digits=4)
        report_path = self.exp_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write('Validation Classification Report\n')
            f.write('='*60 + '\n\n')
            f.write(report)
        
        print(f'✓ Classification report saved to: {report_path}')
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = self.exp_dir / 'training_history.json'
        
        history_dict = {
            'experiment_name': self.experiment_name,
            'best_val_acc': self.best_val_acc,
            'num_epochs': len(self.history['train_loss']),
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_acc': self.history['val_acc'][-1],
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f'✓ Training history saved to: {history_path}')


def create_trainer(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate=0.001,
    checkpoint_dir="evaluation_results/train_results",
    experiment_name=None
):
    """
    Create a trainer instance with default settings
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory to save results
        experiment_name: Name for this experiment
    
    Returns:
        Trainer instance
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name
    )
    
    return trainer
