import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.cnn import SimpleCNN
from src.utils.dataset import create_dataloaders
from src.utils.preprocessing import get_train_transforms, get_val_transforms

print('Quick Training Test (2 epochs)')
print('='*60)

# Small config for testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Create small dataloaders
train_loader, val_loader, _ = create_dataloaders(
    'data/train', 'data/validation', 'data/test',
    batch_size=16,
    num_workers=0,  # Use 0 for testing
    train_transform=get_train_transforms(),
    val_transform=get_val_transforms()
)

# Create model
model = SimpleCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Quick train loop
model.train()
for epoch in range(2):
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Train on first 10 batches only
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 10:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / min(10, len(train_loader))
    epoch_acc = 100. * correct / total
    
    print(f'Epoch {epoch+1}/2 - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

print('\n✓ Training test passed!')
