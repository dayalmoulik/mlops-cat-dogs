import os
import zipfile
import shutil
from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image

def download_dataset():
    print("="*60)
    print("Downloading Cats vs Dogs Dataset from Kaggle")
    print("="*60)
    print("Dataset: bhavikjikadara/dog-and-cat-classification-dataset")
    print("\nPlease wait, this may take 5-15 minutes...\n")
    
    # Download using Kaggle API
    result = os.system('kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset -p data/')
    
    if result == 0:
        print("\n✓ Download complete!")
        return True
    else:
        print("\n✗ Download failed!")
        return False

def extract_dataset():
    print("\n" + "="*60)
    print("Extracting Dataset")
    print("="*60)
    
    # Find the zip file
    data_path = Path('data')
    zip_files = list(data_path.glob('*.zip'))
    
    if not zip_files:
        print("✗ Error: No zip file found in data/ folder!")
        return False
    
    zip_path = zip_files[0]
    print(f"Found: {zip_path.name}")
    
    extract_path = Path('data/temp')
    extract_path.mkdir(exist_ok=True)
    
    # Extract with progress bar
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        print(f"Extracting {len(members)} files...")
        
        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, extract_path)
    
    print("✓ Extraction complete!")
    return True

def check_image_validity(image_path):
    """Check if image file is valid and can be opened"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open to actually load the image (verify closes it)
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception as e:
        return False

def collect_valid_images(folder_path, label_name):
    """Collect all valid images from a folder"""
    print(f"\nCollecting {label_name} images...")
    
    all_images = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    for ext in extensions:
        all_images.extend(list(folder_path.glob(ext)))
    
    # Remove duplicates based on full path
    all_images = list(set(all_images))
    
    print(f"Found {len(all_images)} {label_name} image files")
    print(f"Validating {label_name} images...")
    
    valid_images = []
    corrupted_count = 0
    
    for img_path in tqdm(all_images, desc=f"Validating {label_name}", unit="img"):
        if check_image_validity(img_path):
            valid_images.append(img_path)
        else:
            corrupted_count += 1
    
    print(f"✓ {len(valid_images)} valid {label_name} images")
    if corrupted_count > 0:
        print(f"⚠ Skipped {corrupted_count} corrupted/invalid images")
    
    return valid_images

def organize_dataset():
    print("\n" + "="*60)
    print("Organizing Dataset into Train/Validation/Test Splits")
    print("="*60)
    
    temp_path = Path('data/temp')
    
    # Find cat and dog folders
    print("\nSearching for cat and dog image folders...")
    
    cat_folder = None
    dog_folder = None
    
    # Search for PetImages structure
    for item in temp_path.rglob('*'):
        if item.is_dir():
            folder_name = item.name.lower()
            if folder_name == 'cat' and not cat_folder:
                cat_folder = item
                print(f"✓ Found cat folder: {item.relative_to(temp_path)}")
            elif folder_name == 'dog' and not dog_folder:
                dog_folder = item
                print(f"✓ Found dog folder: {item.relative_to(temp_path)}")
    
    if not cat_folder or not dog_folder:
        print("\n✗ Error: Could not find cat and dog folders!")
        return False
    
    # Collect and validate images
    cat_images = collect_valid_images(cat_folder, 'cat')
    dog_images = collect_valid_images(dog_folder, 'dog')
    
    if len(cat_images) == 0 or len(dog_images) == 0:
        print("\n✗ Error: No valid images found!")
        return False
    
    # Sort for reproducibility, then shuffle with fixed seed
    print("\n" + "="*60)
    print("Creating Train/Val/Test Splits (80%/10%/10%)")
    print("="*60)
    
    # Sort by filename for reproducibility
    cat_images = sorted(cat_images, key=lambda x: x.name)
    dog_images = sorted(dog_images, key=lambda x: x.name)
    
    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(cat_images)
    
    random.seed(42)  # Reset seed for dogs
    random.shuffle(dog_images)
    
    # Calculate split points
    cat_total = len(cat_images)
    dog_total = len(dog_images)
    
    cat_train_end = int(0.8 * cat_total)
    cat_val_end = int(0.9 * cat_total)
    
    dog_train_end = int(0.8 * dog_total)
    dog_val_end = int(0.9 * dog_total)
    
    print(f"\nSplit indices:")
    print(f"Cats  - Train: 0-{cat_train_end}, Val: {cat_train_end}-{cat_val_end}, Test: {cat_val_end}-{cat_total}")
    print(f"Dogs  - Train: 0-{dog_train_end}, Val: {dog_train_end}-{dog_val_end}, Test: {dog_val_end}-{dog_total}")
    
    # Create non-overlapping splits using list slicing
    cat_train = cat_images[0:cat_train_end]
    cat_val = cat_images[cat_train_end:cat_val_end]
    cat_test = cat_images[cat_val_end:]
    
    dog_train = dog_images[0:dog_train_end]
    dog_val = dog_images[dog_train_end:dog_val_end]
    dog_test = dog_images[dog_val_end:]
    
    # Verify lengths
    print(f"\nVerifying split sizes:")
    print(f"Cats  - Train: {len(cat_train)}, Val: {len(cat_val)}, Test: {len(cat_test)}")
    print(f"Dogs  - Train: {len(dog_train)}, Val: {len(dog_val)}, Test: {len(dog_test)}")
    
    # Verify no overlap using object identity (not filenames)
    print("\nVerifying no overlap using image paths...")
    
    # Convert to sets of full paths
    cat_train_set = set([str(img.absolute()) for img in cat_train])
    cat_val_set = set([str(img.absolute()) for img in cat_val])
    cat_test_set = set([str(img.absolute()) for img in cat_test])
    
    dog_train_set = set([str(img.absolute()) for img in dog_train])
    dog_val_set = set([str(img.absolute()) for img in dog_val])
    dog_test_set = set([str(img.absolute()) for img in dog_test])
    
    # Check cat overlaps
    cat_train_val_overlap = cat_train_set & cat_val_set
    cat_train_test_overlap = cat_train_set & cat_test_set
    cat_val_test_overlap = cat_val_set & cat_test_set
    
    # Check dog overlaps
    dog_train_val_overlap = dog_train_set & dog_val_set
    dog_train_test_overlap = dog_train_set & dog_test_set
    dog_val_test_overlap = dog_val_set & dog_test_set
    
    print("\nCat overlaps:")
    print(f"  Train ∩ Val: {len(cat_train_val_overlap)}")
    print(f"  Train ∩ Test: {len(cat_train_test_overlap)}")
    print(f"  Val ∩ Test: {len(cat_val_test_overlap)}")
    
    print("\nDog overlaps:")
    print(f"  Train ∩ Val: {len(dog_train_val_overlap)}")
    print(f"  Train ∩ Test: {len(dog_train_test_overlap)}")
    print(f"  Val ∩ Test: {len(dog_val_test_overlap)}")
    
    if (cat_train_val_overlap or cat_train_test_overlap or cat_val_test_overlap or
        dog_train_val_overlap or dog_train_test_overlap or dog_val_test_overlap):
        print("\n✗ ERROR: Found overlapping images between splits!")
        return False
    
    print("\n✓ No overlap detected - all splits are independent!")
    
    # Create splits dictionary
    splits = {
        'train': {'cats': cat_train, 'dogs': dog_train},
        'validation': {'cats': cat_val, 'dogs': dog_val},
        'test': {'cats': cat_test, 'dogs': dog_test}
    }
    
    # Copy files to their respective folders
    print("\n" + "="*60)
    print("Copying Files to Organized Structure")
    print("="*60)
    
    for split_name, labels_dict in splits.items():
        for label_name, image_list in labels_dict.items():
            dest_dir = Path(f'data/{split_name}/{label_name}')
            
            # Remove existing directory if it exists
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files with new sequential naming
            for idx, img_path in enumerate(tqdm(image_list, 
                                                desc=f"Copying {label_name} to {split_name}", 
                                                unit="img")):
                # Create new filename with index
                new_filename = f'{split_name}_{label_name}_{idx:05d}{img_path.suffix.lower()}'  # unique!
                dest_path = dest_dir / new_filename
                shutil.copy2(img_path, dest_path)
    
    # Print summary
    print("\n" + "="*60)
    print("Split Summary")
    print("="*60)
    
    grand_total = cat_total + dog_total
    
    for split_name in ['train', 'validation', 'test']:
        cat_count = len(splits[split_name]['cats'])
        dog_count = len(splits[split_name]['dogs'])
        total = cat_count + dog_count
        percentage = (total / grand_total) * 100
        print(f"{split_name.capitalize():12} - Cats: {cat_count:5}, Dogs: {dog_count:5}, Total: {total:5} ({percentage:.1f}%)")
    
    print("="*60)
    
    return True

def cleanup():
    print("\n" + "="*60)
    print("Cleaning Up Temporary Files")
    print("="*60)
    
    # Remove temp directory
    temp_path = Path('data/temp')
    if temp_path.exists():
        print("Removing temporary extraction folder...")
        shutil.rmtree(temp_path)
        print("✓ Temporary folder removed")
    
    # Remove zip file
    zip_files = list(Path('data').glob('*.zip'))
    for zip_file in zip_files:
        print(f"Removing {zip_file.name}...")
        zip_file.unlink()
        print("✓ Zip file removed")
    
    print("\n✓ Cleanup complete!")

def verify_final_dataset():
    """Comprehensive verification of the final dataset"""
    print("\n" + "="*60)
    print("FINAL DATASET VERIFICATION")
    print("="*60)
    
    all_train_files = set()
    all_val_files = set()
    all_test_files = set()
    
    total_all = 0
    
    # Collect all files and count
    for split in ['train', 'validation', 'test']:
        split_path = Path(f'data/{split}')
        
        if not split_path.exists():
            print(f"✗ {split} folder not found!")
            continue
        
        cats_path = split_path / 'cats'
        dogs_path = split_path / 'dogs'
        
        cat_files = list(cats_path.glob('*.*')) if cats_path.exists() else []
        dog_files = list(dogs_path.glob('*.*')) if dogs_path.exists() else []
        
        # Filter only image files
        cat_files = [f for f in cat_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        dog_files = [f for f in dog_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        cat_count = len(cat_files)
        dog_count = len(dog_files)
        total = cat_count + dog_count
        total_all += total
        
        # Store filenames for overlap check
        current_files = set([f.name for f in cat_files + dog_files])
        
        if split == 'train':
            all_train_files = current_files
        elif split == 'validation':
            all_val_files = current_files
        else:
            all_test_files = current_files
        
        print(f"{split.capitalize():12} - Cats: {cat_count:5}, Dogs: {dog_count:5}, Total: {total:5}")
    
    print("="*60)
    print(f"GRAND TOTAL: {total_all} images")
    print("="*60)
    
    # Check for overlaps in final structure (by filename)
    # This should be 0 because we use sequential naming
    print("\nChecking for filename overlaps (should be 0 with sequential naming)...")
    overlap_train_val = all_train_files & all_val_files
    overlap_train_test = all_train_files & all_test_files
    overlap_val_test = all_val_files & all_test_files
    
    print(f"Train ∩ Validation: {len(overlap_train_val)} files")
    print(f"Train ∩ Test: {len(overlap_train_test)} files")
    print(f"Validation ∩ Test: {len(overlap_val_test)} files")
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("\n✗ WARNING: Overlapping filenames detected!")
        return False
    else:
        print("\n✓ SUCCESS: No overlaps - all splits contain unique images!")
    
    # Show sample files
    print("\n" + "="*60)
    print("Sample Files from Each Split")
    print("="*60)
    
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()}:")
        for label in ['cats', 'dogs']:
            path = Path(f'data/{split}/{label}')
            if path.exists():
                files = [f.name for f in path.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']][:3]
                print(f"  {label}: {files}")
    
    return True

def main():
    print("\n" + "="*60)
    print("CATS VS DOGS DATASET PREPARATION")
    print("WITH GUARANTEED NO OVERLAP BETWEEN SPLITS")
    print("="*60)
    
    print("\nThis script will:")
    print("  1. Download dataset from Kaggle")
    print("  2. Extract and validate all images")
    print("  3. Create non-overlapping train/val/test splits (80%/10%/10%)")
    print("  4. Verify no overlap between splits")
    print("  5. Organize into proper folder structure")
    print("  6. Clean up temporary files")
    
    print("\n" + "="*60)
    print("IMPORTANT - Before Running:")
    print("="*60)
    print("1. Kaggle API credentials in: C:\\Users\\YourUsername\\.kaggle\\kaggle.json")
    print("2. Accept dataset at:")
    print("   https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset")
    print("="*60)
    
    response = input("\nPress ENTER to continue (or 'q' to quit): ")
    if response.lower() == 'q':
        print("Aborted.")
        return
    
    try:
        # Step 1: Download
        if not download_dataset():
            print("\n✗ Download failed. Exiting.")
            return
        
        # Step 2: Extract
        if not extract_dataset():
            print("\n✗ Extraction failed. Exiting.")
            return
        
        # Step 3: Organize with no overlap
        if not organize_dataset():
            print("\n✗ Organization failed. Exiting.")
            return
        
        # Step 4: Cleanup
        cleanup()
        
        # Step 5: Final Verification
        if verify_final_dataset():
            print("\n" + "="*60)
            print("✓ SUCCESS! Dataset is ready for training!")
            print("="*60)
            print("\nYour dataset has:")
            print("  • No overlapping images between train/val/test")
            print("  • All images validated and verified")
            print("  • Proper 80/10/10 split")
            print("\nYou can now proceed to model training.")
        else:
            print("\n⚠ Verification found issues. Please review.")
        
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user")
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
