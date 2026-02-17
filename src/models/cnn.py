import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    '''
    Simple CNN for binary image classification (Cats vs Dogs)
    
    Architecture:
    - 3 Convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
    - 2 Fully connected layers with Dropout
    - Binary classification output (2 classes)
    
    Input: (batch_size, 3, 224, 224) - RGB images
    Output: (batch_size, 2) - class logits
    '''
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        
        # Calculate flattened size: 128 channels * 28 * 28
        self.flatten_size = 128 * 28 * 28
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        '''
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        '''
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self):
        '''Return total number of trainable parameters'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ImprovedCNN(nn.Module):
    '''
    Improved CNN with residual connections for better accuracy
    
    Architecture:
    - Initial convolution layer
    - 3 Residual blocks
    - Global Average Pooling
    - Fully connected classifier
    '''
    
    def __init__(self, num_classes=2):
        super(ImprovedCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        '''Create a residual layer with multiple blocks'''
        layers = []
        
        # First block may have stride
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.fc(x)
        
        return x
    
    def get_num_parameters(self):
        '''Return total number of trainable parameters'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    '''Basic residual block for ImprovedCNN'''
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


def get_model(model_name='simple', num_classes=2, **kwargs):
    '''
    Factory function to get model by name
    
    Args:
        model_name: 'simple' or 'improved'
        num_classes: Number of output classes (default: 2)
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    
    Example:
        >>> model = get_model('simple', num_classes=2, dropout_rate=0.5)
        >>> model = get_model('improved', num_classes=2)
    '''
    if model_name.lower() == 'simple':
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif model_name.lower() == 'improved':
        return ImprovedCNN(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model: {model_name}. Choose from: simple, improved')


if __name__ == '__main__':
    # Test the models
    print('='*60)
    print('Testing SimpleCNN')
    print('='*60)
    
    model = SimpleCNN(num_classes=2)
    print(f'Total parameters: {model.get_num_parameters():,}')
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    output = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Output (logits):\n{output}')
    
    # Test with probabilities
    probs = F.softmax(output, dim=1)
    print(f'Probabilities:\n{probs}')
    
    print('\n' + '='*60)
    print('Testing ImprovedCNN')
    print('='*60)
    
    model_improved = ImprovedCNN(num_classes=2)
    print(f'Total parameters: {model_improved.get_num_parameters():,}')
    
    output_improved = model_improved(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {output_improved.shape}')
    
    print('\n✓ All models working correctly!')
