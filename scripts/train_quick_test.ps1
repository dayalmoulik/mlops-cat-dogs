# Quick training test (2 epochs)
Write-Host 'Running quick training test...' -ForegroundColor Yellow
python src/training/train_cli.py \
    --model simple \
    --epochs 2 \
    --batch-size 16 \
    --workers 0
