# Folder Structure Organization

## Recommended Structure

```
finalassignment/
├── data/                    # Dataset (70/15/15 split)
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   ├── val/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
│
├── scripts/                 # Main scripts
│   ├── train_all_models.py
│   ├── evaluate_all_models.py
│   ├── compare_models.py
│   ├── run_complete_pipeline.py
│   └── generate_report_data.py
│
├── models/                  # Individual model training scripts
│   ├── train_alexnet.py
│   ├── train_googlenet.py
│   ├── train_resnet18.py
│   ├── train_resnet50.py
│   ├── train_resnet101.py
│   ├── train_densenet169.py
│   ├── train_mobilenet_v2.py
│   ├── train_mobilenet_v3_small.py
│   ├── train_mobilenet_v3_large.py
│   ├── train_vgg16.py
│   ├── train_vgg19.py
│   └── model.py
│
├── utils/                   # Utility scripts
│   ├── check_gpu.py
│   ├── classify.py
│   ├── evaluate.py
│   ├── train.py
│   └── create_individual_train_scripts.py
│
├── samples/                  # Sample code (reference)
│   ├── sampletrain.py
│   ├── samplemodel.py
│   └── sampleclassify.py
│
├── docs/                     # Documentation
│   ├── README.md
│   ├── ASSIGNMENT_CHECKLIST.md
│   └── Assignment.pdf
│
├── results/                  # Output files (generated)
│   ├── trained_models/       # Model checkpoints
│   ├── evaluation_results.json
│   ├── model_comparison.csv
│   ├── comparison_plots/
│   ├── roc_curves/
│   └── best_model_recommendation.txt
│
├── requirements.txt          # Dependencies
└── FOLDER_STRUCTURE.md       # This file
```

## How to Organize

### Option 1: Automatic Organization
Run the organization script:
```bash
python organize_folder.py
```

### Option 2: Manual Organization
Manually move files according to the structure above.

## After Organization

### Update Import Paths
If you organize files, you may need to update imports in some scripts:

1. **In `scripts/train_all_models.py`**: 
   - Add to imports: `import sys; sys.path.append('..')`
   - Or update: `from models.model import MyCNNModel` → `from model import MyCNNModel`

2. **In `models/train_*.py`**:
   - May need to update paths to `data/` directory

3. **In `scripts/evaluate_all_models.py`**:
   - Update: `from train_all_models import get_model` → `from scripts.train_all_models import get_model`

### Alternative: Keep Current Structure
If you prefer to keep files in root for easier imports, that's also fine! The current structure works well too.

## Current Working Structure (Alternative)

If you keep files in root:
```
finalassignment/
├── data/              # Dataset
├── train_*.py         # Individual model scripts (11 files)
├── train_all_models.py # Main training script
├── evaluate_all_models.py
├── compare_models.py
├── model.py
├── requirements.txt
├── README.md
└── ... (other files)
```

This structure is simpler and avoids import path issues.

