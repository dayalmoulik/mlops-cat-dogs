import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.models.cnn import SimpleCNN


class TestInferenceBasic:
    '''Basic inference tests with model'''
    
    @pytest.fixture
    def dummy_model(self):
        '''Create a dummy model for testing'''
        model = SimpleCNN(num_classes=2)
        model.eval()
        return model
    
    @pytest.fixture
    def test_image(self):
        '''Create a test image'''
        return Image.new('RGB', (224, 224), color='green')
    
    def test_model_forward_pass(self, dummy_model):
        '''Test model forward pass works'''
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = dummy_model(x)
        
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_prediction_format(self, dummy_model):
        '''Test model prediction returns valid format'''
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = dummy_model(x)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Check shapes
        assert probabilities.shape == (1, 2)
        assert confidence.shape == (1,)
        assert predicted.shape == (1,)
        
        # Check values
        assert 0 <= confidence.item() <= 1
        assert predicted.item() in [0, 1]
    
    def test_model_probabilities_sum_to_one(self, dummy_model):
        '''Test that softmax probabilities sum to 1'''
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = dummy_model(x)
            probabilities = F.softmax(output, dim=1)
        
        prob_sum = probabilities.sum().item()
        assert abs(prob_sum - 1.0) < 0.001
    
    def test_model_batch_inference(self, dummy_model):
        '''Test model handles batch inference'''
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = dummy_model(x)
        
        assert output.shape == (batch_size, 2)
    
    def test_model_output_range(self, dummy_model):
        '''Test model outputs are in reasonable range'''
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = dummy_model(x)
        
        # Logits can be any value, but shouldn't be extreme
        assert output.abs().max() < 100


class TestModelProperties:
    '''Test model properties and info'''
    
    @pytest.fixture
    def model(self):
        return SimpleCNN(num_classes=2)
    
    def test_model_has_parameters(self, model):
        '''Test model has trainable parameters'''
        params = list(model.parameters())
        assert len(params) > 0
    
    def test_model_parameter_count(self, model):
        '''Test parameter count is positive'''
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_model_eval_mode(self, model):
        '''Test model can be set to eval mode'''
        model.eval()
        assert not model.training
    
    def test_model_train_mode(self, model):
        '''Test model can be set to train mode'''
        model.train()
        assert model.training


class TestInferenceWithPreprocessing:
    '''Test inference pipeline with preprocessing'''
    
    @pytest.fixture
    def model(self):
        model = SimpleCNN(num_classes=2)
        model.eval()
        return model
    
    def test_inference_with_pil_image(self, model):
        '''Test inference works with PIL image'''
        from src.utils.preprocessing import preprocess_image
        
        # Create test image
        img = Image.new('RGB', (300, 300), color='red')
        
        # Preprocess
        tensor = preprocess_image(img, augment=False)
        
        # Inference
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)
        
        assert probabilities.shape == (1, 2)
        assert 0 <= probabilities.max() <= 1
    
    def test_inference_different_image_sizes(self, model):
        '''Test inference works with different input sizes after preprocessing'''
        from src.utils.preprocessing import preprocess_image
        
        sizes = [(100, 100), (500, 500), (224, 224), (640, 480)]
        
        for size in sizes:
            img = Image.new('RGB', size, color='blue')
            tensor = preprocess_image(img, augment=False)
            
            with torch.no_grad():
                output = model(tensor)
            
            # Should always produce same output shape
            assert output.shape == (1, 2)


class TestOutputValidation:
    '''Test output validation logic'''
    
    def test_valid_output_shape(self):
        '''Test validation accepts correct shape'''
        output = torch.randn(4, 2)
        
        # Check basic properties
        assert output.dim() == 2
        assert output.size(1) == 2
    
    def test_detect_nan_in_output(self):
        '''Test detection of NaN values'''
        output = torch.tensor([[1.0, float('nan')], [0.5, 0.5]])
        assert torch.isnan(output).any()
    
    def test_detect_inf_in_output(self):
        '''Test detection of Inf values'''
        output = torch.tensor([[1.0, float('inf')], [0.5, 0.5]])
        assert torch.isinf(output).any()
    
    def test_valid_output_no_nan_inf(self):
        '''Test valid output has no NaN or Inf'''
        output = torch.randn(4, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
