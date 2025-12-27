# Quick Start Guide

## Folder Structure

```
finalassignment/
├── data/              # Dataset (train/val/test)
├── scripts/           # Main scripts
├── models/            # Individual model training scripts
├── utils/             # Utility scripts
├── samples/           # Sample code (reference)
├── docs/              # Documentation
├── results/           # Output files (auto-generated)
└── requirements.txt   # Dependencies
```

## Running Scripts

### From Root Directory

All scripts should be run from the root directory (`finalassignment/`):

```bash
# Main training script
python scripts/train_all_models.py --model all

# Individual model training
python models/train_alexnet.py
python models/train_resnet50.py
# ... etc

# Evaluation
python scripts/evaluate_all_models.py --model all

# Complete pipeline
python scripts/run_complete_pipeline.py
```

### Or Add to PATH

You can also add scripts to your PATH or use relative imports.

## Important Notes

1. **Run from root**: Always run scripts from the `finalassignment/` root directory
2. **Data path**: Scripts look for `data/train`, `data/val`, `data/test`
3. **Outputs**: Results will be saved in `results/` folder (auto-created)

## Quick Commands

```bash
# Check GPU
python utils/check_gpu.py

# Train all models
python scripts/train_all_models.py --model all

# Train specific model
python models/train_resnet50.py

# Evaluate all
python scripts/evaluate_all_models.py --model all

# Compare models
python scripts/compare_models.py

# Run complete pipeline
python scripts/run_complete_pipeline.py
```

