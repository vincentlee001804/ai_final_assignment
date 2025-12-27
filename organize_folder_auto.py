"""
Script to automatically organize the assignment folder structure
"""
import os
import shutil

# Define folder structure
FOLDER_STRUCTURE = {
    'scripts': [
        'train_all_models.py',
        'evaluate_all_models.py',
        'compare_models.py',
        'hyperparameter_tuning.py',
        'run_complete_pipeline.py',
        'generate_report_data.py',
    ],
    'models': [
        'train_alexnet.py',
        'train_googlenet.py',
        'train_resnet18.py',
        'train_resnet50.py',
        'train_resnet101.py',
        'train_densenet169.py',
        'train_mobilenet_v2.py',
        'train_mobilenet_v3_small.py',
        'train_mobilenet_v3_large.py',
        'train_vgg16.py',
        'train_vgg19.py',
        'model.py',
    ],
    'utils': [
        'check_gpu.py',
        'classify.py',
        'evaluate.py',
        'train.py',
        'create_individual_train_scripts.py',
    ],
    'samples': [
        'sampletrain.py',
        'samplemodel.py',
        'sampleclassify.py',
    ],
    'docs': [
        'README.md',
        'ASSIGNMENT_CHECKLIST.md',
        'Assignment.pdf',
    ],
}

def organize():
    """Organize files into folders"""
    print("Organizing folder structure...")
    print("="*60)
    
    # Create directories
    for folder in FOLDER_STRUCTURE.keys():
        os.makedirs(folder, exist_ok=True)
        print(f"Created/Verified: {folder}/")
    
    # Create results directory (for outputs)
    os.makedirs('results', exist_ok=True)
    print("Created/Verified: results/")
    
    # Move files to appropriate folders
    moved_count = 0
    not_found = []
    
    for folder, files in FOLDER_STRUCTURE.items():
        for file in files:
            if os.path.exists(file):
                try:
                    dest = os.path.join(folder, file)
                    if not os.path.exists(dest):  # Don't move if already there
                        shutil.move(file, dest)
                        print(f"  Moved: {file} -> {folder}/")
                        moved_count += 1
                    else:
                        print(f"  Already exists: {folder}/{file}")
                except Exception as e:
                    print(f"  Error moving {file}: {e}")
            else:
                not_found.append(file)
    
    if not_found:
        print(f"\nFiles not found (may already be moved): {', '.join(not_found)}")
    
    print(f"\n{'='*60}")
    print(f"Organization complete! Moved {moved_count} files.")
    print(f"{'='*60}")
    print("\nNew folder structure:")
    print("  scripts/     - Main training and evaluation scripts")
    print("  models/      - Individual model training scripts")
    print("  utils/       - Utility scripts")
    print("  samples/     - Sample code files")
    print("  docs/        - Documentation")
    print("  results/     - Output files (created when scripts run)")
    print("  data/        - Dataset (train/val/test)")
    print("\nNote: Scripts are organized but may need import path updates.")

if __name__ == "__main__":
    organize()

