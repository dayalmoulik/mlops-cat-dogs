# Data Preprocessing Utilities

## Overview

This module provides utilities for image preprocessing, data augmentation, and dataset loading for the Cats vs Dogs classification project.

## Modules

### preprocessing.py

Functions for image transformation and preprocessing.

#### Key Functions:

**get_train_transforms(image_size, augmentation)**
- Creates training data transformations
- Optional data augmentation (rotation, flip, color jitter, etc.)
- Returns composed PyTorch transforms

**get_val_transforms(image_size)**
- Creates validation/test transformations
- No augmentation (only resize, normalize)

**preprocess_image(image, target_size, augment)**
- Preprocess a single PIL Image
- Returns tensor with batch dimension [1, 3, H, W]

**batch_preprocess(images, target_size, augment)**
- Preprocess multiple images at once
- Returns batched tensor [B, 3, H, W]

**validate_image(image, min_size, max_size)**
- Check if image meets requirements
- Validates RGB mode and dimensions

**denormalize_tensor(tensor, mean, std)**
- Reverse normalization for visualization
- Converts back to [0, 1] range

#### Usage Example:

\\\python
from PIL import Image
from src.utils.preprocessing import get_train_transforms, preprocess_image

# Load image
img = Image.open('cat.jpg')

# Get transforms
transform = get_train_transforms(image_size=(224, 224), augmentation=True)

# Apply transform
tensor = transform(img)
print(tensor.shape)  # torch.Size([3, 224, 224])

# Or use convenience function
tensor = preprocess_image(img, augment=True)
print(tensor.shape)  # torch.Size([1, 3, 224, 224])
\\\

### dataset.py

PyTorch Dataset classes for loading data.

#### CatsDogsDataset

Custom Dataset class for Cats vs Dogs data.

**Features:**
- Loads images from class folders
- Applies transforms
- Handles errors gracefully
- Provides class distribution statistics

**Usage:**

\\\python
from src.utils.dataset import CatsDogsDataset
from src.utils.preprocessing import get_train_transforms

# Create dataset
transform = get_train_transforms()
dataset = CatsDogsDataset('data/train', transform=transform)

# Get sample
image, label = dataset[0]
print(f'Image: {image.shape}, Label: {label}')

# Get class distribution
distribution = dataset.get_class_distribution()
print(distribution)  # {'cats': 10000, 'dogs': 10000}
\\\

#### create_dataloaders()

Factory function to create train/val/test dataloaders.

**Usage:**

\\\python
from src.utils.dataset import create_dataloaders
from src.utils.preprocessing import get_train_transforms, get_val_transforms

train_loader, val_loader, test_loader = create_dataloaders(
    train_dir='data/train',
    val_dir='data/validation',
    test_dir='data/test',
    batch_size=32,
    num_workers=4,
    train_transform=get_train_transforms(),
    val_transform=get_val_transforms()
)

# Iterate through batches
for images, labels in train_loader:
    # images: torch.Tensor [32, 3, 224, 224]
    # labels: torch.Tensor [32]
    pass
\\\

## Data Augmentation

Training data uses the following augmentations:
- **Random Horizontal Flip**: 50% probability
- **Random Rotation**: ±15 degrees
- **Color Jitter**: Brightness, contrast, saturation (±20%), hue (±10%)
- **Random Affine**: Translation (±10%), scale (90%-110%)

Validation/test data only uses:
- **Resize**: To target size (224x224)
- **Normalization**: ImageNet mean and std

## Normalization

All images are normalized using ImageNet statistics:
- **Mean**: [0.485, 0.456, 0.406]
- **Std**: [0.229, 0.224, 0.225]

This allows for better transfer learning if needed later.

## Performance Tips

1. **Use num_workers > 0**: Parallel data loading
2. **pin_memory=True**: Faster GPU transfer
3. **Batch size**: Start with 32, adjust based on GPU memory
4. **Data augmentation**: Only for training, not validation/test

## Complete Training Example

\\\python
import torch
from torch.utils.data import DataLoader
from src.models.cnn import SimpleCNN
from src.utils.dataset import create_dataloaders
from src.utils.preprocessing import get_train_transforms, get_val_transforms

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    'data/train', 'data/validation', 'data/test',
    batch_size=32,
    num_workers=4,
    train_transform=get_train_transforms(augmentation=True),
    val_transform=get_val_transforms()
)

# Create model
model = SimpleCNN(num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass
    outputs = model(images)
    # ... rest of training code
\\\

## Testing

Run tests with:
\\\powershell
python src/utils/preprocessing.py
python src/utils/dataset.py
\\\

## Next Steps

- Implement training script using these utilities
- Add MLflow tracking
- Create evaluation metrics
