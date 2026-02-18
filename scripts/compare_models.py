import json
import pandas as pd
from pathlib import Path

print('='*70)
print(' '*20 + 'MODEL COMPARISON')
print('='*70)

results = {}

for model_dir in ['simple', 'improved']:
    results_file = Path(f'evaluation_results/{model_dir}/test_results.json')
    if results_file.exists():
        with open(results_file, 'r') as f:
            results[model_dir] = json.load(f)

if len(results) == 0:
    print('\n✗ No results found. Run evaluation first.')
    exit(1)

# Create comparison dataframe
comparison = []
for model_name, data in results.items():
    comparison.append({
        'Model': model_name.upper(),
        'Accuracy (%)': f'{data["accuracy"]:.2f}',
        'Precision': f'{data["precision"]:.4f}',
        'Recall': f'{data["recall"]:.4f}',
        'F1-Score': f'{data["f1_score"]:.4f}',
    })

df = pd.DataFrame(comparison)
print('\n')
print(df.to_string(index=False))
print('\n')

# Determine winner
if len(results) > 1:
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f'🏆 Best Model: {best_model[0].upper()} with {best_model[1]["accuracy"]:.2f}% accuracy')

print('='*70)
