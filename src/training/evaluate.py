import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from tqdm import tqdm
import json

from src.models.cnn import get_model
from src.utils.dataset import CatsDogsDataset
from src.utils.preprocessing import get_val_transforms


class ModelEvaluator:
    '''
    Evaluate trained model on test set
    '''
    
    def __init__(self, 
                 model: nn.Module,
                 test_loader: DataLoader,
                 device: torch.device,
                 class_names: list = ['Cat', 'Dog']):
        '''
        Initialize evaluator
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            device: Device to run evaluation on
            class_names: List of class names
        '''
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        
        self.model.eval()
    
    def evaluate(self):
        '''
        Run evaluation on test set
        
        Returns:
            Dictionary with all evaluation metrics
        '''
        print('='*60)
        print('Running Model Evaluation on Test Set')
        print('='*60)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Accumulate results
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = 100. * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        print(f'\n✓ Evaluation complete!')
        print(f'  Test Accuracy: {accuracy:.2f}%')
        print(f'  Test Precision: {precision:.4f}')
        print(f'  Test Recall: {recall:.4f}')
        print(f'  Test F1-Score: {f1:.4f}')
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_confusion_matrix(self, labels, predictions, save_path='test_confusion_matrix.png'):
        '''Plot confusion matrix for test set'''
        cm = confusion_matrix(labels, predictions)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        
        # Plot percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Percentage (%)'})
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Confusion matrix saved to {save_path}')
        return save_path
    
    def plot_roc_curve(self, labels, probabilities, save_path='test_roc_curve.png'):
        '''Plot ROC curve for binary classification'''
        # For binary classification, use probability of positive class (dog = 1)
        fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'✓ ROC curve saved to {save_path}')
        return save_path, roc_auc
    
    def plot_per_class_metrics(self, labels, predictions, save_path='test_per_class_metrics.png'):
        '''Plot per-class precision, recall, and F1-score'''
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(x - width, precision, width, label='Precision', color='skyblue')
        ax.bar(x, recall, width, label='Recall', color='lightcoral')
        ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics - Test Set', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.02, f'{p:.3f}', ha='center', fontsize=9)
            ax.text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=9)
            ax.text(i + width, f + 0.02, f'{f:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'✓ Per-class metrics saved to {save_path}')
        return save_path
    
    def generate_classification_report(self, labels, predictions, save_path='test_classification_report.txt'):
        '''Generate detailed classification report'''
        report = classification_report(
            labels, predictions, 
            target_names=self.class_names,
            digits=4
        )
        
        with open(save_path, 'w') as f:
            f.write('='*60 + '\n')
            f.write('CLASSIFICATION REPORT - TEST SET\n')
            f.write('='*60 + '\n\n')
            f.write(report)
            f.write('\n' + '='*60 + '\n')
        
        print(f'✓ Classification report saved to {save_path}')
        print(f'\nClassification Report:\n{report}')
        return save_path
    
    def save_results(self, results, save_path='test_results.json'):
        '''Save evaluation results to JSON'''
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'num_samples': len(results['labels']),
            'class_names': self.class_names
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        print(f'✓ Results saved to {save_path}')
        return save_path


def load_trained_model(model_path: str, model_name: str = 'simple', device: torch.device = None):
    '''
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        model_name: Model architecture name ('simple' or 'improved')
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(model_name, num_classes=2)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f'✓ Model loaded from {model_path}')
    return model


def main():
    '''Main evaluation function'''
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-name', type=str, default='simple',
                       choices=['simple', 'improved'],
                       help='Model architecture')
    parser.add_argument('--test-dir', type=str, default='data/test',
                       help='Path to test data')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Load model
    print('Loading trained model...')
    model = load_trained_model(args.model_path, args.model_name, device)
    
    # Create test dataset and loader
    print('\nLoading test dataset...')
    test_dataset = CatsDogsDataset(
        args.test_dir,
        transform=get_val_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'✓ Test set: {len(test_dataset)} images\n')
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, device)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Generate visualizations and reports
    print('\nGenerating evaluation artifacts...')
    
    cm_path = evaluator.plot_confusion_matrix(
        results['labels'], 
        results['predictions'],
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    roc_path, roc_auc = evaluator.plot_roc_curve(
        results['labels'],
        results['probabilities'],
        save_path=output_dir / 'roc_curve.png'
    )
    
    metrics_path = evaluator.plot_per_class_metrics(
        results['labels'],
        results['predictions'],
        save_path=output_dir / 'per_class_metrics.png'
    )
    
    report_path = evaluator.generate_classification_report(
        results['labels'],
        results['predictions'],
        save_path=output_dir / 'classification_report.txt'
    )
    
    results_path = evaluator.save_results(
        results,
        save_path=output_dir / 'test_results.json'
    )
    
    # Print summary
    print('\n' + '='*60)
    print('EVALUATION SUMMARY')
    print('='*60)
    print(f'Model: {args.model_name}')
    print(f'Test Samples: {len(results["labels"])}')
    print(f'Accuracy: {results["accuracy"]:.2f}%')
    print(f'Precision: {results["precision"]:.4f}')
    print(f'Recall: {results["recall"]:.4f}')
    print(f'F1-Score: {results["f1_score"]:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'\nResults saved to: {output_dir}')
    print('='*60)


if __name__ == '__main__':
    main()
