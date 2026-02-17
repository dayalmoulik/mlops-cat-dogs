# Model Architecture

## SimpleCNN

A straightforward convolutional neural network for binary image classification.

### Architecture
- **Conv Block 1**: 3→32 channels, 224×224 → 112×112
- **Conv Block 2**: 32→64 channels, 112×112 → 56×56
- **Conv Block 3**: 64→128 channels, 56×56 → 28×28
- **FC Layer 1**: Flatten → 256 features
- **FC Layer 2**: 256 → 2 classes (cat/dog)

### Parameters
- Total: ~11.2M trainable parameters
- Dropout: 0.5 (configurable)

### Usage
\\\python
from src.models.cnn import SimpleCNN

model = SimpleCNN(num_classes=2, dropout_rate=0.5)
output = model(input_tensor)  # input: (batch, 3, 224, 224)
\\\

## ImprovedCNN

An enhanced CNN with residual connections for better gradient flow and accuracy.

### Architecture
- **Initial Conv**: 3→64 channels, 224×224 → 56×56
- **Residual Block 1**: 64 channels (2 blocks)
- **Residual Block 2**: 64→128 channels (2 blocks)
- **Residual Block 3**: 128→256 channels (2 blocks)
- **Global Average Pooling**
- **FC Layer**: 256 → 2 classes

### Parameters
- Total: ~2.8M trainable parameters
- More efficient than SimpleCNN

### Usage
\\\python
from src.models.cnn import ImprovedCNN

model = ImprovedCNN(num_classes=2)
output = model(input_tensor)
\\\

## Factory Function

Use the factory function for easy model creation:

\\\python
from src.models.cnn import get_model

# Get SimpleCNN
model = get_model('simple', num_classes=2, dropout_rate=0.5)

# Get ImprovedCNN
model = get_model('improved', num_classes=2)
\\\

## Model Selection Guide

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| SimpleCNN | 11.2M | Fast | Good | Quick experiments, baseline |
| ImprovedCNN | 2.8M | Medium | Better | Production, better accuracy |

## Next Steps

1. Train the model using \src/training/train.py\
2. Evaluate on test set
3. Export for deployment
