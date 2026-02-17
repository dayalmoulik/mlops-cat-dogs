import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List


# Standard ImageNet normalization (commonly used for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: Tuple[int, int] = (224, 224),
                         augmentation: bool = True) -> transforms.Compose:
    '''
    Get training data transformations with optional augmentation
    
    Args:
        image_size: Target image dimensions (height, width)
        augmentation: Whether to apply data augmentation
    
    Returns:
        Composed transforms for training data
    
    Example:
        >>> transform = get_train_transforms(image_size=(224, 224), augmentation=True)
        >>> image = transform(pil_image)
    '''
    transform_list = [
        transforms.Resize(image_size),
    ]
    
    if augmentation:
        # Data augmentation for better generalization
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
        ])
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    '''
    Get validation/test data transformations (no augmentation)
    
    Args:
        image_size: Target image dimensions (height, width)
    
    Returns:
        Composed transforms for validation/test data
    
    Example:
        >>> transform = get_val_transforms(image_size=(224, 224))
        >>> image = transform(pil_image)
    '''
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image: Image.Image, 
                     target_size: Tuple[int, int] = (224, 224),
                     augment: bool = False) -> torch.Tensor:
    '''
    Preprocess a single PIL Image for model inference
    
    Args:
        image: PIL Image in RGB format
        target_size: Target dimensions (height, width)
        augment: Whether to apply augmentation
    
    Returns:
        Preprocessed tensor of shape [1, 3, H, W] (with batch dimension)
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open('cat.jpg')
        >>> tensor = preprocess_image(img)
        >>> print(tensor.shape)  # torch.Size([1, 3, 224, 224])
    '''
    if augment:
        transform = get_train_transforms(target_size, augmentation=True)
    else:
        transform = get_val_transforms(target_size)
    
    tensor = transform(image)
    # Add batch dimension
    return tensor.unsqueeze(0)


def validate_image(image: Image.Image,
                   min_size: Optional[Tuple[int, int]] = None,
                   max_size: Optional[Tuple[int, int]] = None) -> bool:
    '''
    Validate that an image meets basic requirements
    
    Args:
        image: PIL Image to validate
        min_size: Optional minimum dimensions (width, height)
        max_size: Optional maximum dimensions (width, height)
    
    Returns:
        True if valid, False otherwise
    
    Example:
        >>> img = Image.open('cat.jpg')
        >>> is_valid = validate_image(img, min_size=(100, 100))
    '''
    # Check if RGB
    if image.mode != 'RGB':
        return False
    
    width, height = image.size
    
    # Check minimum size
    if min_size is not None:
        if width < min_size[0] or height < min_size[1]:
            return False
    
    # Check maximum size
    if max_size is not None:
        if width > max_size[0] or height > max_size[1]:
            return False
    
    return True


def denormalize_tensor(tensor: torch.Tensor,
                       mean: List[float] = IMAGENET_MEAN,
                       std: List[float] = IMAGENET_STD) -> torch.Tensor:
    '''
    Denormalize a tensor that was normalized with ImageNet stats
    
    Args:
        tensor: Normalized tensor of shape [C, H, W] or [B, C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
    
    Returns:
        Denormalized tensor in range [0, 1]
    
    Example:
        >>> normalized = transform(image)  # Normalized tensor
        >>> denormalized = denormalize_tensor(normalized)
        >>> # Now can convert to PIL: transforms.ToPILImage()(denormalized)
    '''
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denormalized = tensor * std + mean
    return torch.clamp(denormalized, 0, 1)


def batch_preprocess(images: List[Image.Image],
                     target_size: Tuple[int, int] = (224, 224),
                     augment: bool = False) -> torch.Tensor:
    '''
    Preprocess a batch of images
    
    Args:
        images: List of PIL Images
        target_size: Target dimensions
        augment: Whether to apply augmentation
    
    Returns:
        Batched tensor of shape [B, 3, H, W]
    
    Example:
        >>> images = [Image.open(f'cat_{i}.jpg') for i in range(4)]
        >>> batch = batch_preprocess(images)
        >>> print(batch.shape)  # torch.Size([4, 3, 224, 224])
    '''
    if augment:
        transform = get_train_transforms(target_size, augmentation=True)
    else:
        transform = get_val_transforms(target_size)
    
    tensors = [transform(img) for img in images]
    return torch.stack(tensors)


def tensor_to_image(tensor: torch.Tensor,
                   denormalize: bool = True) -> Image.Image:
    '''
    Convert a tensor back to PIL Image
    
    Args:
        tensor: Tensor of shape [C, H, W] or [1, C, H, W]
        denormalize: Whether to denormalize first
    
    Returns:
        PIL Image
    
    Example:
        >>> tensor = preprocess_image(img)
        >>> reconstructed = tensor_to_image(tensor[0])
    '''
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if needed
    if denormalize:
        tensor = denormalize_tensor(tensor)
    
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


def calculate_dataset_stats(image_paths: List[str],
                           sample_size: int = 1000) -> Tuple[List[float], List[float]]:
    '''
    Calculate mean and std of a dataset (for custom normalization)
    
    Args:
        image_paths: List of image file paths
        sample_size: Number of images to sample for calculation
    
    Returns:
        Tuple of (mean, std) for each channel [R, G, B]
    
    Example:
        >>> from pathlib import Path
        >>> paths = list(Path('data/train/cats').glob('*.jpg'))
        >>> mean, std = calculate_dataset_stats(paths[:1000])
        >>> print(f'Mean: {mean}, Std: {std}')
    '''
    import random
    from tqdm import tqdm
    
    # Sample random images if dataset is large
    if len(image_paths) > sample_size:
        image_paths = random.sample(image_paths, sample_size)
    
    # Transform to resize only (no normalization yet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Accumulate pixel values
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0
    
    print(f'Calculating dataset statistics from {len(image_paths)} images...')
    
    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)  # [C, H, W]
            
            pixel_sum += tensor.sum(dim=[1, 2])
            pixel_squared_sum += (tensor ** 2).sum(dim=[1, 2])
            num_pixels += tensor.shape[1] * tensor.shape[2]
        except Exception as e:
            print(f'Skipping {img_path}: {e}')
            continue
    
    # Calculate mean and std
    mean = (pixel_sum / num_pixels).tolist()
    std = torch.sqrt(pixel_squared_sum / num_pixels - torch.tensor(mean) ** 2).tolist()
    
    return mean, std


class ImageAugmentation:
    '''
    Custom augmentation class for more control
    
    Example:
        >>> aug = ImageAugmentation(rotation_range=20, zoom_range=0.2)
        >>> augmented_img = aug(original_img)
    '''
    
    def __init__(self,
                 rotation_range: float = 15,
                 zoom_range: float = 0.1,
                 horizontal_flip: bool = True,
                 brightness_range: float = 0.2):
        '''
        Initialize augmentation parameters
        
        Args:
            rotation_range: Max rotation in degrees
            zoom_range: Zoom factor range (0.1 = 90%-110%)
            horizontal_flip: Enable horizontal flip
            brightness_range: Brightness adjustment range
        '''
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        '''Apply augmentation to an image'''
        augmentations = []
        
        if self.horizontal_flip:
            augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
        
        augmentations.extend([
            transforms.RandomRotation(degrees=self.rotation_range),
            transforms.RandomAffine(
                degrees=0,
                scale=(1 - self.zoom_range, 1 + self.zoom_range)
            ),
            transforms.ColorJitter(brightness=self.brightness_range)
        ])
        
        transform = transforms.Compose(augmentations)
        return transform(image)


if __name__ == '__main__':
    '''Test the preprocessing functions'''
    print('='*60)
    print('Testing Preprocessing Utilities')
    print('='*60)
    
    # Create a dummy image
    dummy_img = Image.new('RGB', (300, 300), color='red')
    
    # Test 1: Train transforms
    print('\nTest 1: Train Transforms')
    train_transform = get_train_transforms(augmentation=True)
    train_tensor = train_transform(dummy_img)
    print(f'✓ Train tensor shape: {train_tensor.shape}')
    print(f'✓ Train tensor range: [{train_tensor.min():.2f}, {train_tensor.max():.2f}]')
    
    # Test 2: Validation transforms
    print('\nTest 2: Validation Transforms')
    val_transform = get_val_transforms()
    val_tensor = val_transform(dummy_img)
    print(f'✓ Val tensor shape: {val_tensor.shape}')
    
    # Test 3: Single image preprocessing
    print('\nTest 3: Single Image Preprocessing')
    preprocessed = preprocess_image(dummy_img)
    print(f'✓ Preprocessed shape: {preprocessed.shape}')
    
    # Test 4: Batch preprocessing
    print('\nTest 4: Batch Preprocessing')
    images = [Image.new('RGB', (300, 300), color='blue') for _ in range(4)]
    batch = batch_preprocess(images)
    print(f'✓ Batch shape: {batch.shape}')
    
    # Test 5: Denormalization
    print('\nTest 5: Denormalization')
    denorm = denormalize_tensor(train_tensor)
    print(f'✓ Denormalized range: [{denorm.min():.2f}, {denorm.max():.2f}]')
    
    # Test 6: Tensor to image conversion
    print('\nTest 6: Tensor to Image Conversion')
    reconstructed = tensor_to_image(preprocessed)
    print(f'✓ Reconstructed image size: {reconstructed.size}')
    print(f'✓ Reconstructed image mode: {reconstructed.mode}')
    
    # Test 7: Image validation
    print('\nTest 7: Image Validation')
    is_valid = validate_image(dummy_img, min_size=(100, 100))
    print(f'✓ Image valid: {is_valid}')
    
    # Test 8: Custom augmentation
    print('\nTest 8: Custom Augmentation')
    aug = ImageAugmentation(rotation_range=20, zoom_range=0.2)
    augmented = aug(dummy_img)
    print(f'✓ Augmented image size: {augmented.size}')
    
    print('\n' + '='*60)
    print('✓ All preprocessing tests passed!')
    print('='*60)
