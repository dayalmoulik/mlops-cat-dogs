import json
from pathlib import Path

print('='*70)
print(' '*20 + 'EVALUATION RESULTS VIEWER')
print('='*70)

results_path = Path('evaluation_results/test_results.json')

if not results_path.exists():
    print('\n✗ Results file not found. Please run evaluation first.')
    exit(1)

with open(results_path, 'r') as f:
    results = json.load(f)

print(f'\n📊 Test Set Performance:')
print(f'   Samples: {results["num_samples"]:,}')
print(f'   Classes: {", ".join(results["class_names"])}')
print()
print(f'   Accuracy:  {results["accuracy"]:.2f}%')
print(f'   Precision: {results["precision"]:.4f}')
print(f'   Recall:    {results["recall"]:.4f}')
print(f'   F1-Score:  {results["f1_score"]:.4f}')

# Read classification report
report_path = Path('evaluation_results/classification_report.txt')
if report_path.exists():
    print(f'\n📋 Detailed Classification Report:')
    print('='*70)
    with open(report_path, 'r') as f:
        print(f.read())

print('\n📈 Visualizations:')
for file in Path('evaluation_results').glob('*.png'):
    print(f'   ✓ {file.name}')

print('\n✅ View the PNG files to see confusion matrix, ROC curve, and per-class metrics')
print('='*70)
