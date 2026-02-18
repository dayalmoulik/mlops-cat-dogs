import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from src.training.evaluate import ModelEvaluator
from src.models.cnn import SimpleCNN
from torch.utils.data import DataLoader, TensorDataset

class TestModelEvaluator:
    '''Test cases for ModelEvaluator'''
    
    @pytest.fixture
    def dummy_model(self):
        '''Create a dummy model for testing'''
        model = SimpleCNN(num_classes=2)
        model.eval()
        return model
    
    @pytest.fixture
    def dummy_loader(self):
        '''Create a dummy data loader'''
        # Create random data
        images = torch.randn(100, 3, 224, 224)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        return loader
    
    def test_evaluator_initialization(self, dummy_model, dummy_loader):
        '''Test evaluator initialization'''
        device = torch.device('cpu')
        evaluator = ModelEvaluator(dummy_model, dummy_loader, device)
        
        assert evaluator.model is not None
        assert evaluator.test_loader is not None
        assert evaluator.device == device
        assert len(evaluator.class_names) == 2
    
    def test_evaluate_returns_dict(self, dummy_model, dummy_loader):
        '''Test that evaluate returns a dictionary with expected keys'''
        device = torch.device('cpu')
        evaluator = ModelEvaluator(dummy_model, dummy_loader, device)
        
        results = evaluator.evaluate()
        
        assert isinstance(results, dict)
        assert 'predictions' in results
        assert 'labels' in results
        assert 'probabilities' in results
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
    
    def test_evaluate_shapes(self, dummy_model, dummy_loader):
        '''Test that predictions have correct shapes'''
        device = torch.device('cpu')
        evaluator = ModelEvaluator(dummy_model, dummy_loader, device)
        
        results = evaluator.evaluate()
        
        assert len(results['predictions']) == 100
        assert len(results['labels']) == 100
        assert results['probabilities'].shape == (100, 2)
    
    def test_metrics_range(self, dummy_model, dummy_loader):
        '''Test that metrics are in valid ranges'''
        device = torch.device('cpu')
        evaluator = ModelEvaluator(dummy_model, dummy_loader, device)
        
        results = evaluator.evaluate()
        
        assert 0 <= results['accuracy'] <= 100
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
