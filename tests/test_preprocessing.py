import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.utils.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    preprocess_image,
    validate_image,
    denormalize_tensor,
    batch_preprocess,
    tensor_to_image,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class TestPreprocessing:
    '''Test cases for data preprocessing functions'''
    
    @pytest.fixture
    def dummy_image(self):
        '''Create a dummy test image'''
        return Image.new('RGB', (300, 300), color='red')
    
    @pytest.fixture
    def small_image(self):
        '''Create a small test image'''
        return Image.new('RGB', (50, 50), color='blue')
    
    def test_get_train_transforms_returns_compose(self):
        '''Test that get_train_transforms returns Compose object'''
        transform = get_train_transforms()
        assert isinstance(transform, transforms.Compose)
    
    def test_get_train_transforms_with_augmentation(self):
        '''Test train transforms with augmentation enabled'''
        transform = get_train_transforms(augmentation=True)
        assert isinstance(transform, transforms.Compose)
    
    def test_get_train_transforms_without_augmentation(self):
        '''Test train transforms with augmentation disabled'''
        transform = get_train_transforms(augmentation=False)
        assert isinstance(transform, transforms.Compose)
    
    def test_get_val_transforms_returns_compose(self):
        '''Test that get_val_transforms returns Compose object'''
        transform = get_val_transforms()
        assert isinstance(transform, transforms.Compose)
    
    def test_preprocess_image_shape(self, dummy_image):
        '''Test that preprocessing returns correct tensor shape'''
        tensor = preprocess_image(dummy_image)
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_image_dtype(self, dummy_image):
        '''Test that preprocessing returns correct data type'''
        tensor = preprocess_image(dummy_image)
        assert tensor.dtype == torch.float32
    
    def test_preprocess_image_normalization(self, dummy_image):
        '''Test that image values are normalized'''
        tensor = preprocess_image(dummy_image)
        # Normalized values should be in a reasonable range
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0
    
    def test_preprocess_image_custom_size(self, dummy_image):
        '''Test preprocessing with custom image size'''
        tensor = preprocess_image(dummy_image, target_size=(256, 256))
        assert tensor.shape == (1, 3, 256, 256)
    
    def test_validate_image_valid_rgb(self, dummy_image):
        '''Test validation of valid RGB image'''
        assert validate_image(dummy_image) == True
    
    def test_validate_image_invalid_mode(self):
        '''Test validation rejects non-RGB image'''
        grayscale = Image.new('L', (224, 224), color=128)
        assert validate_image(grayscale) == False
    
    def test_validate_image_min_size(self, small_image):
        '''Test validation with minimum size constraint'''
        assert validate_image(small_image, min_size=(100, 100)) == False
        assert validate_image(small_image, min_size=(40, 40)) == True
    
    def test_validate_image_max_size(self, dummy_image):
        '''Test validation with maximum size constraint'''
        assert validate_image(dummy_image, max_size=(200, 200)) == False
        assert validate_image(dummy_image, max_size=(400, 400)) == True
    
    def test_denormalize_tensor_shape(self):
        '''Test denormalization preserves shape'''
        tensor = torch.randn(3, 224, 224)
        denorm = denormalize_tensor(tensor)
        assert denorm.shape == tensor.shape
    
    def test_denormalize_tensor_range(self):
        '''Test denormalized values are in [0, 1]'''
        tensor = torch.randn(3, 224, 224)
        denorm = denormalize_tensor(tensor)
        assert denorm.min() >= 0.0
        assert denorm.max() <= 1.0
    
    def test_denormalize_tensor_batch(self):
        '''Test denormalization with batch dimension'''
        tensor = torch.randn(4, 3, 224, 224)
        denorm = denormalize_tensor(tensor)
        assert denorm.shape == tensor.shape
    
    def test_batch_preprocess_returns_tensor(self, dummy_image):
        '''Test batch preprocessing returns tensor'''
        images = [dummy_image for _ in range(4)]
        batch = batch_preprocess(images)
        assert isinstance(batch, torch.Tensor)
    
    def test_batch_preprocess_shape(self, dummy_image):
        '''Test batch preprocessing returns correct shape'''
        images = [dummy_image for _ in range(4)]
        batch = batch_preprocess(images)
        assert batch.shape == (4, 3, 224, 224)
    
    def test_batch_preprocess_empty_list(self):
        '''Test batch preprocessing with empty list raises error'''
        with pytest.raises(RuntimeError):
            # Empty list should raise error (expected behavior)
            batch = batch_preprocess([])
    
    def test_tensor_to_image_returns_pil(self, dummy_image):
        '''Test tensor to image conversion returns PIL Image'''
        tensor = preprocess_image(dummy_image)
        reconstructed = tensor_to_image(tensor[0], denormalize=True)
        assert isinstance(reconstructed, Image.Image)
    
    def test_tensor_to_image_mode(self, dummy_image):
        '''Test reconstructed image is RGB'''
        tensor = preprocess_image(dummy_image)
        reconstructed = tensor_to_image(tensor[0], denormalize=True)
        assert reconstructed.mode == 'RGB'
    
    def test_tensor_to_image_size(self, dummy_image):
        '''Test reconstructed image has correct size'''
        tensor = preprocess_image(dummy_image)
        reconstructed = tensor_to_image(tensor[0], denormalize=True)
        assert reconstructed.size == (224, 224)
    
    def test_imagenet_constants(self):
        '''Test ImageNet normalization constants'''
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert all(0 <= m <= 1 for m in IMAGENET_MEAN)
        assert all(0 < s <= 1 for s in IMAGENET_STD)


class TestTransformConsistency:
    '''Test consistency between different transform functions'''
    
    @pytest.fixture
    def test_image(self):
        return Image.new('RGB', (224, 224), color='green')
    
    def test_train_and_val_transforms_compatible(self, test_image):
        '''Test that train and val transforms produce compatible outputs'''
        train_transform = get_train_transforms(augmentation=False)
        val_transform = get_val_transforms()
        
        train_output = train_transform(test_image)
        val_output = val_transform(test_image)
        
        # Both should have same shape
        assert train_output.shape == val_output.shape
        assert train_output.dtype == val_output.dtype
    
    def test_preprocess_image_matches_transforms(self, test_image):
        '''Test that preprocess_image matches manual transform application'''
        manual_transform = get_val_transforms()
        manual_output = manual_transform(test_image).unsqueeze(0)
        
        preprocess_output = preprocess_image(test_image, augment=False)
        
        # Shapes should match
        assert manual_output.shape == preprocess_output.shape


class TestEdgeCases:
    '''Test edge cases and error handling'''
    
    def test_very_small_image(self):
        '''Test preprocessing very small image'''
        tiny_image = Image.new('RGB', (10, 10), color='yellow')
        tensor = preprocess_image(tiny_image)
        # Should still produce 224x224 output
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_very_large_image(self):
        '''Test preprocessing very large image'''
        large_image = Image.new('RGB', (4000, 4000), color='purple')
        tensor = preprocess_image(large_image)
        # Should resize to 224x224
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_non_square_image(self):
        '''Test preprocessing non-square image'''
        rect_image = Image.new('RGB', (640, 480), color='orange')
        tensor = preprocess_image(rect_image)
        # Should resize to square 224x224
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_validate_image_none_constraints(self):
        '''Test validation with None constraints'''
        image = Image.new('RGB', (100, 100), color='cyan')
        assert validate_image(image, min_size=None, max_size=None) == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

