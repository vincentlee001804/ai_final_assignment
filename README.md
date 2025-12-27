# Skin Cancer Classification Assignment

Complete implementation for training and evaluating multiple deep learning models for skin cancer classification.

## 📁 Folder Structure

```
finalassignment/
├── data/                    # Dataset (70/15/15 split)
│   ├── train/ (benign, malignant)
│   ├── val/ (benign, malignant)
│   └── test/ (benign, malignant)
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
│   └── train.py
│
├── samples/                 # Sample code (reference)
│   ├── sampletrain.py
│   ├── samplemodel.py
│   └── sampleclassify.py
│
├── docs/                    # Documentation
│   ├── README.md
│   ├── ASSIGNMENT_CHECKLIST.md
│   └── Assignment.pdf
│
└── results/                 # Output files (auto-generated)
    ├── trained_models/
    ├── hyperparameter_results/
    ├── comparison_plots/
    └── roc_curves/
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Verify GPU (Optional)

```bash
python utils/check_gpu.py
```

## 📋 Usage

### Option 1: Run Complete Pipeline (Recommended)

Run all steps automatically in the correct order:
```bash
python scripts/run_complete_pipeline.py
```

**This will execute:**
1. Task 3: Train all 11 models (with fixed hyperparameters)
2. Task 5: Evaluate all models
3. Task 6: Compare and recommend best model
4. Task 7: Generate report data

### Option 2: Run Steps Individually

#### Step 1: Train All Models (Task 3)

**Train all 11 models:**
```bash
python scripts/train_all_models.py --model all
```

**Train a specific model:**
```bash
python scripts/train_all_models.py --model resnet50
```

**Or use individual scripts:**
```bash
python models/train_alexnet.py
python models/train_resnet50.py
# ... etc
```

**Output:**
- `results/trained_models/`: All trained model checkpoints (`{model_name}_best.pt`)
- `results/trained_models/`: Training history JSON files

#### Step 2: Evaluate All Models (Task 5)

**Evaluate all models on test set:**
```bash
python scripts/evaluate_all_models.py --model all
```

**Evaluate a specific model:**
```bash
python scripts/evaluate_all_models.py --model resnet50
```

**Metrics Calculated (for binary classification):**
- Accuracy
- Precision (per class)
- Recall (per class)
- True Negative Rate (TNR/Specificity)
- Macro Precision
- Macro Recall
- ROC Curve and AUC

**Output:**
- `results/evaluation_results.json`: All evaluation metrics
- `results/roc_curves/`: ROC curve plots for each model
- Console output with detailed metrics

#### Step 3: Compare Models (Task 6)

Compare all models and get recommendation:
```bash
python scripts/compare_models.py
```

**Output:**
- `results/model_comparison.csv`: Detailed comparison table
- `results/comparison_plots/model_comparison.png`: Visualization plots
- `results/best_model_recommendation.txt`: Final recommendation

#### Step 4: Generate Report Data (Task 7)

Generate formatted data for your technical report:
```bash
python scripts/generate_report_data.py
```

**Output:**
- `results/report_data.txt`: Formatted data for report sections

## 📊 Models Implemented

1. AlexNet
2. GoogleNet
3. ResNet18
4. ResNet50
5. ResNet101
6. DenseNet169
7. MobileNetV2
8. MobileNetV3 Small
9. MobileNetV3 Large
10. VGG16
11. VGG19

## 📈 Evaluation Metrics

For binary classification (benign vs malignant):
- ✅ Accuracy
- ✅ Recall (Sensitivity)
- ✅ True Negative Rate (TNR/Specificity)
- ✅ Precision
- ✅ ROC Curve and AUC

## 📁 Output Files

All outputs are saved in `results/` folder:
- `results/trained_models/` - Model checkpoints
- `results/evaluation_results.json` - All metrics
- `results/model_comparison.csv` - Comparison table
- `results/comparison_plots/` - Visualizations
- `results/roc_curves/` - ROC curves
- `results/best_model_recommendation.txt` - Recommendation
- `results/report_data.txt` - Report data

## 📝 Assignment Tasks

✅ Task 1: Dataset (data/)  
✅ Task 2: Data split (70/15/15)  
✅ Task 3: Train all 11 models (with fixed hyperparameters)  
✅ Task 5: Evaluation with all metrics  
✅ Task 6: Compare and recommend  
✅ Task 7: Report data generation  

## 🔧 Notes

- All scripts follow `samples/sampletrain.py` pattern
- Uses SGD with momentum (0.9)
- Simple transforms (Resize + ToTensor)
- Early stopping prevents overfitting
- All outputs organized in `results/` folder

## 📚 Documentation

- `docs/README.md` - Full documentation
- `docs/ASSIGNMENT_CHECKLIST.md` - Task checklist
- `QUICK_START.md` - Quick reference guide
- `ORGANIZATION_SUMMARY.md` - Folder organization details

