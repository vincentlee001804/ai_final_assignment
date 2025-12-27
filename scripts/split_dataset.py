"""
Script to split skin cancer dataset into training (70%), validation (15%), and testing (15%) sets.
Maintains class balance across all splits.
"""

import os
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

def split_dataset(source_dir='archive', output_dir='data', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        source_dir: Source directory containing the dataset
        output_dir: Output directory for the split dataset
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
    """
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['benign', 'malignant']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Collect all images, using filename as unique identifier to avoid duplicates
    benign_images_dict = {}  # filename -> path
    malignant_images_dict = {}  # filename -> path
    
    # Check archive/data structure (preferred source)
    data_train_benign = source_path / 'data' / 'train' / 'benign'
    data_train_malignant = source_path / 'data' / 'train' / 'malignant'
    data_test_benign = source_path / 'data' / 'test' / 'benign'
    data_test_malignant = source_path / 'data' / 'test' / 'malignant'
    
    # Collect benign images from data directory
    if data_train_benign.exists():
        for img_path in data_train_benign.glob('*.jpg'):
            benign_images_dict[img_path.name] = img_path
    if data_test_benign.exists():
        for img_path in data_test_benign.glob('*.jpg'):
            if img_path.name not in benign_images_dict:  # Only add if not already found
                benign_images_dict[img_path.name] = img_path
    
    # Collect malignant images from data directory
    if data_train_malignant.exists():
        for img_path in data_train_malignant.glob('*.jpg'):
            malignant_images_dict[img_path.name] = img_path
    if data_test_malignant.exists():
        for img_path in data_test_malignant.glob('*.jpg'):
            if img_path.name not in malignant_images_dict:  # Only add if not already found
                malignant_images_dict[img_path.name] = img_path
    
    # Also check archive/train and archive/test structure (if exists and not in data)
    archive_train_benign = source_path / 'train' / 'benign'
    archive_train_malignant = source_path / 'train' / 'malignant'
    archive_test_benign = source_path / 'test' / 'benign'
    archive_test_malignant = source_path / 'test' / 'malignant'
    
    if archive_train_benign.exists():
        for img_path in archive_train_benign.glob('*.jpg'):
            if img_path.name not in benign_images_dict:
                benign_images_dict[img_path.name] = img_path
    if archive_test_benign.exists():
        for img_path in archive_test_benign.glob('*.jpg'):
            if img_path.name not in benign_images_dict:
                benign_images_dict[img_path.name] = img_path
    if archive_train_malignant.exists():
        for img_path in archive_train_malignant.glob('*.jpg'):
            if img_path.name not in malignant_images_dict:
                malignant_images_dict[img_path.name] = img_path
    if archive_test_malignant.exists():
        for img_path in archive_test_malignant.glob('*.jpg'):
            if img_path.name not in malignant_images_dict:
                malignant_images_dict[img_path.name] = img_path
    
    # Convert to lists
    benign_images = list(benign_images_dict.values())
    malignant_images = list(malignant_images_dict.values())
    
    # Shuffle the lists
    random.shuffle(benign_images)
    random.shuffle(malignant_images)
    
    print(f"Total benign images found: {len(benign_images)}")
    print(f"Total malignant images found: {len(malignant_images)}")
    print(f"Total images: {len(benign_images) + len(malignant_images)}")
    
    # Split each class separately to maintain class balance
    def split_class(images, train_ratio, val_ratio, test_ratio):
        """Split a list of images into train, val, and test sets."""
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        return train_images, val_images, test_images
    
    # Split benign images
    benign_train, benign_val, benign_test = split_class(benign_images, train_ratio, val_ratio, test_ratio)
    
    # Split malignant images
    malignant_train, malignant_val, malignant_test = split_class(malignant_images, train_ratio, val_ratio, test_ratio)
    
    print("\nSplitting results:")
    print(f"Benign - Train: {len(benign_train)}, Val: {len(benign_val)}, Test: {len(benign_test)}")
    print(f"Malignant - Train: {len(malignant_train)}, Val: {len(malignant_val)}, Test: {len(malignant_test)}")
    print(f"Total - Train: {len(benign_train) + len(malignant_train)}, "
          f"Val: {len(benign_val) + len(malignant_val)}, "
          f"Test: {len(benign_test) + len(malignant_test)}")
    
    # Copy files to new structure
    def copy_files(file_list, dest_dir, class_name):
        """Copy files to destination directory."""
        for img_path in file_list:
            dest_path = dest_dir / class_name / img_path.name
            shutil.copy2(img_path, dest_path)
    
    print("\nCopying files...")
    
    # Copy benign images
    copy_files(benign_train, output_path / 'train', 'benign')
    copy_files(benign_val, output_path / 'val', 'benign')
    copy_files(benign_test, output_path / 'test', 'benign')
    
    # Copy malignant images
    copy_files(malignant_train, output_path / 'train', 'malignant')
    copy_files(malignant_val, output_path / 'val', 'malignant')
    copy_files(malignant_test, output_path / 'test', 'malignant')
    
    print("Dataset split completed successfully!")
    print(f"\nOutput directory structure:")
    print(f"  {output_dir}/train/benign/ ({len(benign_train)} images)")
    print(f"  {output_dir}/train/malignant/ ({len(malignant_train)} images)")
    print(f"  {output_dir}/val/benign/ ({len(benign_val)} images)")
    print(f"  {output_dir}/val/malignant/ ({len(malignant_val)} images)")
    print(f"  {output_dir}/test/benign/ ({len(benign_test)} images)")
    print(f"  {output_dir}/test/malignant/ ({len(malignant_test)} images)")

if __name__ == "__main__":
    split_dataset(
        source_dir='archive',
        output_dir='data',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

