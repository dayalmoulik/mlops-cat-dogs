import torch
from pathlib import Path
from src.models.cnn import get_model

print("Creating dummy model for CI...")

# Create directory
Path("models/checkpoints").mkdir(parents=True, exist_ok=True)

# Create model
model = get_model("improved", num_classes=2)

# Save model
model_path = "models/checkpoints/best_model.pth"
torch.save(model.state_dict(), model_path)

# Verify
file_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
print(f"✓ Model created: {model_path}")
print(f"✓ Size: {file_size:.2f} MB")
