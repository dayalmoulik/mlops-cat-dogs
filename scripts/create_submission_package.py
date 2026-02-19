import os
import zipfile
from pathlib import Path
from datetime import datetime

def create_submission_package():
    """Create zip file with all required artifacts"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"MLOPS_Assignment2_Submission_{timestamp}.zip"
    
    print(f"Creating submission package: {zip_filename}")
    print("="*60)
    
    # Files and directories to include
    include_patterns = [
        # Source code
        'src/**/*.py',
        'tests/**/*.py',
        'scripts/**/*.py',
        
        # Configuration files
        'requirements.txt',
        'requirements-test.txt',
        'requirements-docker.txt',
        'setup.py',
        'pytest.ini',
        
        # Docker
        'Dockerfile',
        '.dockerignore',
        'docker-compose.yml',
        
        # CI/CD
        '.github/workflows/*.yml',
        
        # Kubernetes
        'k8s/*.yaml',
        
        # DVC
        '.dvc/config',
        'data/*.dvc',
        
        # Documentation
        '*.md',
        
        # Model artifacts (if small enough)
        'models/checkpoints/*.pth',
        
        # Results
        'performance_tracking.json',
        'evaluation_results/*.png',
        'evaluation_results/*.txt',
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pattern in include_patterns:
            for file_path in Path('.').glob(pattern):
                if file_path.is_file():
                    print(f"Adding: {file_path}")
                    zipf.write(file_path)
    
    zip_size = Path(zip_filename).stat().st_size / (1024 * 1024)  # MB
    
    print("="*60)
    print(f"✓ Submission package created: {zip_filename}")
    print(f"✓ Size: {zip_size:.2f} MB")
    print("="*60)
    
    return zip_filename

if __name__ == '__main__':
    create_submission_package()
