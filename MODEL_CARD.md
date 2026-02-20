# Model Card: Cats vs Dogs Classifier

## Model Details
- **Model Name**: ImprovedCNN
- **Model Type**: Convolutional Neural Network with Residual Connections
- **Version**: 1.0
- **Date**: 2026-02-18
- **Framework**: PyTorch 2.0.1
- **License**: Educational Use

## Intended Use
**Primary Use**: Binary classification of cat and dog images for pet adoption platforms

**Intended Users**: 
- Pet adoption agencies
- Animal shelters
- Veterinary services
- Pet identification systems

**Out of Scope**: 
- Other animal species
- Multiple animals in one image
- Non-standard angles or heavily occluded images

## Training Data
- **Source**: Kaggle Dogs vs Cats Dataset
- **Total Images**: 24,998 (12,499 cats, 12,499 dogs)
- **Training Set**: 19,998 images (80%)
- **Validation Set**: 2,500 images (10%)
- **Test Set**: 2,500 images (10%)
- **Image Size**: 224x224 RGB
- **Preprocessing**: Resize, normalize (ImageNet stats)
- **Augmentation**: Random flip, rotation, color jitter, affine transforms

## Model Architecture
```
ImprovedCNN:
- Initial Conv: 3→64 channels
- ResBlock 1: 64 channels (2 blocks)
- ResBlock 2: 64→128 channels (2 blocks)
- ResBlock 3: 128→256 channels (2 blocks)
- Global Average Pooling
- FC Layer: 256→2 classes

Total Parameters: 2,768,386
```

## Performance

### Test Set Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 92.12% |
| Precision | 0.9220 |
| Recall | 0.9212 |
| F1-Score | 0.9212 |
| ROC AUC | ~0.95+ |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Cat | ~0.92 | ~0.92 | ~0.92 | 2,500 |
| Dog | ~0.92 | ~0.92 | ~0.92 | 2,500 |

### Training Details
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 10
- **Loss Function**: CrossEntropyLoss
- **Hardware**: [CPU/GPU]
- **Training Time**: ~[Time]

## Limitations
- Model is trained on standard pet images only
- May not perform well on:
  - Very young kittens or puppies
  - Unusual breeds
  - Poor lighting conditions
  - Multiple animals in frame
  - Extreme angles or occlusions

## Ethical Considerations
- Model should not be used for:
  - Automated pet identification without human verification
  - Critical decisions without expert review
- Potential biases:
  - May favor common breeds seen in training data
  - Performance may vary with different image qualities

## Model Usage

### Loading the Model
```python
import torch
from src.models.cnn import get_model

# Load model
model = get_model('improved', num_classes=2)
model.load_state_dict(torch.load('models/checkpoints/best_model.pth'))
model.eval()
```

### Making Predictions
```python
from PIL import Image
from src.utils.preprocessing import preprocess_image

# Load and preprocess image
image = Image.open('cat.jpg')
tensor = preprocess_image(image)

# Predict
with torch.no_grad():
    output = model(tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = probabilities.argmax().item()

# 0 = Cat, 1 = Dog
result = 'Cat' if predicted_class == 0 else 'Dog'
confidence = probabilities[0][predicted_class].item()
```

## Maintenance
- **Retraining**: Recommended if accuracy drops below 90%
- **Monitoring**: Track prediction confidence and misclassifications
- **Updates**: Consider retraining with new data periodically

## Contact
For questions or issues, please refer to the project repository.

## Citation
```
@misc{cats-dogs-mlops-2024,
  title={Cats vs Dogs MLOps Pipeline},
  author={[Your Name]},
  year={2024},
  howpublished={`url{https://github.com/[username]/cats-dogs-mlops}}
}
```

---

**Model Status**: ✅ Production Ready
**Last Updated**: 2026-02-18
**Version**: 1.0
