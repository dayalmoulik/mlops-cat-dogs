# Train ImprovedCNN with optimized settings
Write-Host 'Training ImprovedCNN...' -ForegroundColor Cyan
python src/training/train_cli.py \
    --model improved \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.0005
