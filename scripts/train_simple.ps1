# Train SimpleCNN with default settings
Write-Host 'Training SimpleCNN...' -ForegroundColor Cyan
python src/training/train_cli.py \
    --model simple \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001
