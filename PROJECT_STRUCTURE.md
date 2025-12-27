# Project Structure and Execution

## 📁 Project Structure

```
finalassignment/
├── archive/                    # Raw Kaggle dataset (source data)
│   └── [original dataset structure]
│
├── data/                       # Processed dataset (70/15/15 split)
│   ├── train/ (benign, malignant)
│   ├── val/ (benign, malignant)
│   └── test/ (benign, malignant)
│
├── scripts/                    # Main execution scripts
│   ├── split_dataset.py        # Creates data/train, data/val, data/test from archive/
│   ├── train_all_models.py    # Train all models (with fixed hyperparameters)
│   ├── evaluate_all_models.py # Evaluate all models on test set
│   ├── compare_models.py      # Compare and recommend best model
│   ├── run_complete_pipeline.py
│   └── generate_report_data.py
│
├── models/                     # Individual model training scripts
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
├── utils/                      # Utility functions and helpers
│   ├── train.py               # Data loaders, training and validation loops
│   ├── evaluate.py            # Evaluation functions
│   ├── classify.py             # Classification utilities
│   ├── check_gpu.py            # GPU verification
│   └── [checkpoint saving/loading functions]
│
├── samples/                    # Reference sample code
│   ├── sampletrain.py
│   ├── samplemodel.py
│   └── sampleclassify.py
│
├── docs/                       # Documentation
│   ├── README.md
│   ├── ASSIGNMENT_CHECKLIST.md
│   ├── EARLY_STOPPING_INFO.md
│   ├── FOLDER_STRUCTURE.md
│   ├── HYPERPARAMETER_TUNING_STATUS.md
│   └── Assignment.pdf
│
├── results/                    # Output files (auto-generated)
│   ├── trained_models/         # Model checkpoints
│   │   ├── {model}_best.pt     # Best model (lowest validation loss)
│   │   ├── {model}_last.pt     # Final model (last epoch)
│   │   └── {model}_history.json
│   ├── evaluation_results.json # All evaluation metrics
│   ├── model_comparison.csv    # Comparison table
│   ├── roc_curves/             # ROC curve plots
│   │   └── {model}_roc.png
│   ├── comparison_plots/       # Visualization plots
│   ├── hyperparameter_results/ # Hyperparameter tuning results
│   └── best_model_recommendation.txt
│
├── requirements.txt            # Python dependencies
├── class_name.txt              # Generated class names file
├── README.md                   # Main README
├── ORGANIZATION_SUMMARY.md     # Organization details
├── QUICK_START.md              # Quick reference
└── PROJECT_STRUCTURE.md        # This file
```

## 🔄 Execution Flow

### 1. Data Preparation

**Step:** Split raw dataset into train/val/test splits

```bash
python scripts/split_dataset.py
```

**What it does:**
- Reads raw Kaggle dataset from `archive/` folder
- Splits into 70% train, 15% validation, 15% test
- Maintains class balance across splits
- Creates `data/train/`, `data/val/`, and `data/test/` folders

**Output:**
- `data/train/` (benign, malignant)
- `data/val/` (benign, malignant)
- `data/test/` (benign, malignant)

### 2. Model Training

**Option A: Train all models**
```bash
python scripts/train_all_models.py --model all
```

**Option B: Train individual models**
```bash
python models/train_alexnet.py
python models/train_resnet50.py
# ... etc
```

**What happens:**
- Each `train_*.py` script trains one model
- Uses data loaders and training/validation loops from `utils/train.py`
- Saves checkpoints using checkpoint saving functions
- Implements early stopping to prevent overfitting

**Output:**
- `results/trained_models/{model}_best.pt` - Best model (lowest validation loss)
- `results/trained_models/{model}_last.pt` - Final model (last epoch)
- `results/trained_models/{model}_history.json` - Training history

### 3. Model Evaluation

**Evaluate all models:**
```bash
python scripts/evaluate_all_models.py --model all
```

**Evaluate specific model:**
```bash
python scripts/evaluate_all_models.py --model resnet50
```

**What it does:**
- Loads all `*_best.pt` files from `results/trained_models/`
- Evaluates each model on `data/test/`
- Calculates comprehensive metrics:
  - Accuracy
  - Precision (per class)
  - Recall (per class)
  - True Negative Rate (TNR/Specificity)
  - Macro Precision
  - Macro Recall
  - ROC Curve and AUC
  - Confusion Matrix

**Output:**
- `results/evaluation_results.json` - All evaluation metrics (JSON format)
- `results/roc_curves/{model}_roc.png` - Individual ROC curve for each model

**Note:** The evaluation script currently outputs JSON. If you need CSV format, you can convert the JSON or modify the script to output CSV directly.

### 4. Model Comparison

```bash
python scripts/compare_models.py
```

**Output:**
- `results/model_comparison.csv` - Detailed comparison table
- `results/comparison_plots/model_comparison.png` - Visualization
- `results/best_model_recommendation.txt` - Final recommendation

### 5. Complete Pipeline

Run everything in sequence:
```bash
python scripts/run_complete_pipeline.py
```

This executes:
1. Train all models with fixed hyperparameters (Task 3)
2. Evaluate all models (Task 5)
3. Compare and recommend (Task 6)
4. Generate report data (Task 7)

## 📝 Key Components

### Data Loaders and Training Utilities

**Location:** `utils/train.py` and `utils/evaluate.py`

**Functions:**
- Data loaders for train/val/test sets
- Training loop with forward pass, loss calculation, backpropagation
- Validation loop with metrics calculation
- Checkpoint saving/loading functions
- Early stopping implementation

### Model Checkpoints

**Format:** PyTorch `.pt` files (state_dict)

**Files saved:**
- `{model}_best.pt` - Best model based on validation loss
- `{model}_last.pt` - Model from final epoch
- `{model}_history.json` - Training metrics history

**Location:** `results/trained_models/`

### Evaluation Metrics

**Binary Classification Metrics:**
- Accuracy
- Precision (per class: benign, malignant)
- Recall (per class: benign, malignant)
- True Negative Rate (TNR/Specificity)
- Macro Precision (average)
- Macro Recall (average)
- ROC Curve and AUC
- Confusion Matrix

**Output Format:**
- JSON: `results/evaluation_results.json`
- CSV: `results/model_comparison.csv` (via compare_models.py)
- Plots: `results/roc_curves/{model}_roc.png`

## 🚀 Quick Start

1. **Prepare data:**
   ```bash
   python scripts/split_dataset.py
   ```

2. **Train models:**
   ```bash
   python scripts/train_all_models.py --model all
   ```

3. **Evaluate models:**
   ```bash
   python scripts/evaluate_all_models.py --model all
   ```

4. **Compare and get recommendation:**
   ```bash
   python scripts/compare_models.py
   ```

Or run everything at once:
```bash
python scripts/run_complete_pipeline.py
```

## 📊 Output Files Summary

| File/Folder | Description |
|------------|-------------|
| `results/trained_models/{model}_best.pt` | Best model checkpoint |
| `results/trained_models/{model}_last.pt` | Final epoch checkpoint |
| `results/trained_models/{model}_history.json` | Training history |
| `results/evaluation_results.json` | All evaluation metrics |
| `results/model_comparison.csv` | Comparison table |
| `results/roc_curves/{model}_roc.png` | ROC curves per model |
| `results/comparison_plots/model_comparison.png` | Comparison visualization |
| `results/best_model_recommendation.txt` | Best model recommendation |

