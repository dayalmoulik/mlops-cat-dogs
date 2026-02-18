# Model Evaluation Guide

## Quick Evaluation

After training, evaluate your model on the test set:

\\\powershell
# Using the CLI
python src/training/evaluate.py \
    --model-path models/checkpoints/best_model.pth \
    --model-name simple \
    --test-dir data/test

# Using the quick script
python scripts/quick_evaluate.py
\\\

## Output Files

Evaluation generates the following files in \evaluation_results/\:

1. **confusion_matrix.png** - Visual confusion matrix (counts & percentages)
2. **roc_curve.png** - ROC curve with AUC score
3. **per_class_metrics.png** - Bar chart of precision/recall/F1 per class
4. **classification_report.txt** - Detailed metrics report
5. **test_results.json** - JSON file with all metrics

## Evaluation Metrics

### Accuracy
Overall correctness: (Correct Predictions / Total Predictions) × 100

### Precision
Of all predicted cats/dogs, how many were actually cats/dogs?

### Recall
Of all actual cats/dogs, how many did we correctly identify?

### F1-Score
Harmonic mean of precision and recall (balanced metric)

### ROC AUC
Area under the ROC curve (higher is better, 1.0 is perfect)

## Expected Results

### SimpleCNN
- Test Accuracy: 85-90%
- Test F1-Score: 0.85-0.90
- ROC AUC: 0.90-0.95

### ImprovedCNN
- Test Accuracy: 90-93%
- Test F1-Score: 0.90-0.93
- ROC AUC: 0.95-0.97

## CLI Options

\\\
--model-path      Path to model checkpoint (required)
--model-name      Model architecture (simple/improved)
--test-dir        Path to test data directory
--batch-size      Batch size for evaluation
--num-workers     Number of data loading workers
--output-dir      Directory to save results
\\\

## Examples

### Evaluate SimpleCNN
\\\powershell
python src/training/evaluate.py \
    --model-path models/checkpoints/best_model.pth \
    --model-name simple
\\\

### Evaluate ImprovedCNN
\\\powershell
python src/training/evaluate.py \
    --model-path models/improved/model_e15.pth \
    --model-name improved \
    --output-dir evaluation_results/improved
\\\

### Custom batch size
\\\powershell
python src/training/evaluate.py \
    --model-path models/checkpoints/best_model.pth \
    --model-name simple \
    --batch-size 64 \
    --num-workers 8
\\\

## Interpreting Results

### Confusion Matrix
- **Top-left**: True Negatives (correctly identified cats)
- **Top-right**: False Positives (dogs predicted as cats)
- **Bottom-left**: False Negatives (cats predicted as dogs)
- **Bottom-right**: True Positives (correctly identified dogs)

### ROC Curve
- Closer to top-left corner = better performance
- AUC = 1.0 is perfect
- AUC = 0.5 is random guessing

### Per-Class Metrics
- Check if model performs equally well on both classes
- Large differences may indicate class imbalance or bias

## Troubleshooting

**Model file not found:**
- Check the path to your trained model
- Default location: \models/checkpoints/best_model.pth\

**Out of memory:**
- Reduce batch size: \--batch-size 16\
- Reduce workers: \--num-workers 0\

**Low accuracy:**
- Model may need more training
- Check training curves for overfitting
- Try ImprovedCNN architecture
