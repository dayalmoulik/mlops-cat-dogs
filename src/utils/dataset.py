import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from typing import Optional, Callable, Tuple, List
import numpy as np


class CatsDogsDataset(Dataset):
    '''
    PyTorch Dataset for Cats vs Dogs classification
    
    Expected directory structure:
        root/
            cats/
                cat_00000.jpg
                cat_00001.jpg
                ...
            dogs/
                dog_00000.jpg
                dog_00001.jpg
                ...
    
    Example:
        >>> from src.utils.preprocessing import get_train_transforms
        >>> transform = get_train_transforms()
        >>> dataset = CatsDogsDataset('data/train', transform=transform)
        >>> image, label = dataset[0]
    '''
    
    def __init__(self,
                 root_dir: str,
                 transform: Optional[Callable] = None,
                 class_names: List[str] = ['cats', 'dogs']):
        '''
        Initialize dataset
        
        Args:
            root_dir: Root directory containing class folders
            transform: Optional transform to apply to images
            class_names: List of class folder names
        '''
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f'Found 0 images in {root_dir}')
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        '''Load all image paths and their labels'''
        samples = []
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f'Warning: Class directory {class_dir} does not exist')
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Find all image files
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in class_dir.glob(ext):
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        '''Return the total number of samples'''
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        '''
        Get a sample by index
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image_tensor, label)
        '''
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        '''Get the distribution of classes in the dataset'''
        distribution = {class_name: 0 for class_name in self.class_names}
        
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        
        return distribution
    
    def get_sample_path(self, idx: int) -> Path:
        '''Get the file path of a sample'''
        return self.samples[idx][0]


def create_dataloaders(train_dir: str,
                       val_dir: str,
                       test_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       train_transform: Optional[Callable] = None,
                       val_transform: Optional[Callable] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Create DataLoaders for train, validation, and test sets
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        test_dir: Path to test data
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes for data loading
        train_transform: Transform for training data
        val_transform: Transform for validation and test data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Example:
        >>> from src.utils.preprocessing import get_train_transforms, get_val_transforms
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     'data/train', 'data/validation', 'data/test',
        ...     batch_size=32,
        ...     train_transform=get_train_transforms(),
        ...     val_transform=get_val_transforms()
        ... )
    '''
    # Create datasets
    train_dataset = CatsDogsDataset(train_dir, transform=train_transform)
    val_dataset = CatsDogsDataset(val_dir, transform=val_transform)
    test_dataset = CatsDogsDataset(test_dir, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info(dataset: CatsDogsDataset) -> dict:
    '''
    Get information about a dataset
    
    Args:
        dataset: CatsDogsDataset instance
    
    Returns:
        Dictionary with dataset information
    '''
    distribution = dataset.get_class_distribution()
    
    return {
        'total_samples': len(dataset),
        'class_distribution': distribution,
        'class_names': dataset.class_names,
        'class_to_idx': dataset.class_to_idx
    }


if __name__ == '__main__':
    '''Test the dataset classes'''
    from src.utils.preprocessing import get_train_transforms, get_val_transforms
    
    print('='*60)
    print('Testing Dataset Classes')
    print('='*60)
    
    # Check if data exists
    train_dir = Path('data/train')
    if not train_dir.exists():
        print('\n⚠ Warning: data/train directory not found')
        print('Please run: python scripts/download_data.py')
        print('\nCreating dummy test...')
        
        # Create dummy dataset for testing
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create dummy structure
        (temp_dir / 'cats').mkdir()
        (temp_dir / 'dogs').mkdir()
        
        # Create dummy images
        for i in range(5):
            dummy_img = Image.new('RGB', (224, 224), color='red')
            dummy_img.save(temp_dir / 'cats' / f'cat_{i}.jpg')
            
            dummy_img = Image.new('RGB', (224, 224), color='blue')
            dummy_img.save(temp_dir / 'dogs' / f'dog_{i}.jpg')
        
        train_dir = temp_dir
    
    try:
        # Test 1: Create dataset
        print('\nTest 1: Create Dataset')
        train_transform = get_train_transforms(augmentation=True)
        dataset = CatsDogsDataset(train_dir, transform=train_transform)
        print(f'✓ Dataset created with {len(dataset)} samples')
        
        # Test 2: Get sample
        print('\nTest 2: Get Sample')
        image, label = dataset[0]
        print(f'✓ Image shape: {image.shape}')
        print(f'✓ Label: {label} ({dataset.class_names[label]})')
        
        # Test 3: Class distribution
        print('\nTest 3: Class Distribution')
        distribution = dataset.get_class_distribution()
        for class_name, count in distribution.items():
            print(f'✓ {class_name}: {count} images')
        
        # Test 4: Dataset info
        print('\nTest 4: Dataset Info')
        info = get_dataset_info(dataset)
        print(f'✓ Total samples: {info["total_samples"]}')
        print(f'✓ Classes: {info["class_names"]}')
        
        # Test 5: Create dataloaders
        print('\nTest 5: Create DataLoaders')
        if (Path('data/train').exists() and 
            Path('data/validation').exists() and 
            Path('data/test').exists()):
            
            train_loader, val_loader, test_loader = create_dataloaders(
                'data/train',
                'data/validation',
                'data/test',
                batch_size=4,
                num_workers=0,  # Use 0 for testing
                train_transform=get_train_transforms(),
                val_transform=get_val_transforms()
            )
            
            print(f'✓ Train loader: {len(train_loader)} batches')
            print(f'✓ Val loader: {len(val_loader)} batches')
            print(f'✓ Test loader: {len(test_loader)} batches')
            
            # Test batch loading
            print('\nTest 6: Load Batch')
            images, labels = next(iter(train_loader))
            print(f'✓ Batch images shape: {images.shape}')
            print(f'✓ Batch labels shape: {labels.shape}')
            print(f'✓ Labels in batch: {labels.tolist()}')
        else:
            print('⚠ Skipping dataloader test - full dataset not available')
        
        print('\n' + '='*60)
        print('✓ All dataset tests passed!')
        print('='*60)
        
    except Exception as e:
        print(f'\n✗ Error during testing: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temp directory if created
        if 'temp_dir' in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir)
