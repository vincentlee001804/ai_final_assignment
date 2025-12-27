# Skin Cancer Classification Assignment

Complete implementation for training and evaluating multiple deep learning models for skin cancer classification.

## Assignment Tasks Completed

✅ **Task 1**: Dataset structure (data/train, data/val, data/test)  
✅ **Task 2**: Dataset split (70% train, 15% val, 15% test)  
✅ **Task 3**: Training scripts for all 11 models  
✅ **Task 5**: Comprehensive evaluation with all required metrics  
✅ **Task 6**: Model comparison and recommendation  

## Models Implemented

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

## Project Structure

```
.
├── data/
│   ├── train/          # Training set (70%)
│   │   ├── benign/
│   │   └── malignant/
│   ├── val/            # Validation set (15%)
│   │   ├── benign/
│   │   └── malignant/
│   └── test/           # Test set (15%)
│       ├── benign/
│       └── malignant/
├── train_all_models.py      # Train all 11 models (with fixed hyperparameters)
├── evaluate_all_models.py    # Evaluate all models
├── compare_models.py        # Compare and recommend best model
├── run_complete_pipeline.py # Run entire pipeline
├── model.py                 # Custom CNN (if needed)
├── train.py                 # Simple training script
├── classify.py              # Single image classification
├── evaluate.py              # Simple evaluation
└── requirements.txt         # Dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify GPU setup (optional):
```bash
python check_gpu.py
```

## Usage

### Option 1: Run Complete Pipeline (Recommended)

Run all steps automatically:
```bash
python run_complete_pipeline.py
```

This will:
1. Train all 11 models
2. Evaluate all models on test set
3. Compare models and generate recommendations

### Option 2: Run Steps Individually

#### Step 1: Train All Models

**Option A: Train all models together:**
```bash
python train_all_models.py --model all
```

**Option B: Train a specific model using main script:**
```bash
python train_all_models.py --model resnet50
```

**Option C: Train each model separately (individual scripts):**
```bash
python train_alexnet.py
python train_googlenet.py
python train_resnet18.py
python train_resnet50.py
python train_resnet101.py
python train_densenet169.py
python train_mobilenet_v2.py
python train_mobilenet_v3_small.py
python train_mobilenet_v3_large.py
python train_vgg16.py
python train_vgg19.py
```

**Note:** Individual scripts allow you to:
- Run models in parallel on different GPUs
- Train models separately at different times
- Easier debugging for specific models

**Parameters:**
- `--model`: Model name or 'all' (default: all)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 50)
- `--patience`: Early stopping patience (default: 10)
- `--data_dir`: Data directory (default: data)

**Output:**
- `trained_models/`: All trained model checkpoints (`{model_name}_best.pt`)
- `trained_models/`: Training history JSON files

#### Step 2: Evaluate All Models

Evaluate all models on test set:
```bash
python evaluate_all_models.py --model all
```

Evaluate a specific model:
```bash
python evaluate_all_models.py --model resnet50
```

**Output:**
- `evaluation_results.json`: All evaluation metrics
- `roc_curves/`: ROC curve plots for each model
- Console output with detailed metrics

**Metrics Calculated (for binary classification):**
- Accuracy
- Precision (per class)
- Recall (per class)
- True Negative Rate (TNR/Specificity)
- Macro Precision
- Macro Recall
- ROC Curve and AUC

#### Step 4: Compare Models

Compare all models and get recommendation:
```bash
python compare_models.py
```

**Output:**
- `model_comparison.csv`: Detailed comparison table
- `comparison_plots/model_comparison.png`: Visualization plots
- `best_model_recommendation.txt`: Final recommendation

## Evaluation Metrics

For **binary classification** (benign vs malignant):
- ✅ Accuracy
- ✅ Recall (Sensitivity)
- ✅ True Negative Rate (TNR/Specificity)
- ✅ Precision
- ✅ ROC Curve and AUC

For **multi-class classification**:
- ✅ Accuracy
- ✅ Macro Recall
- ✅ Macro Precision

## Generated Files

After running the pipeline, you'll have:

```
trained_models/
├── alexnet_best.pt
├── googlenet_best.pt
├── resnet18_best.pt
├── ... (all 11 models)
└── {model_name}_history.json (training history)

evaluation_results.json          # All evaluation metrics
model_comparison.csv             # Comparison table
comparison_plots/
└── model_comparison.png         # Visualization
roc_curves/
├── alexnet_roc.png
├── googlenet_roc.png
└── ... (ROC curves for all models)

best_model_recommendation.txt    # Final recommendation
```

## Model Architecture Details

All models use **transfer learning** with ImageNet pretrained weights:
- Final classification layer is replaced for 2-class classification
- Models are fine-tuned on the skin cancer dataset
- Data augmentation applied during training (random flip, rotation, color jitter)
- Early stopping prevents overfitting

## Hyperparameters

Default hyperparameters (fixed in each model):
- Learning Rate: 0.001
- Batch Size: 32
- Optimizer: Adam
- Loss Function: Cross Entropy
- Early Stopping Patience: 10 epochs
- Learning Rate Scheduler: ReduceLROnPlateau

## Technical Report Preparation

The generated files provide all data needed for your report:

1. **Methodology**: Use training scripts and model configurations
2. **Results**: Use `evaluation_results.json` and `model_comparison.csv`
3. **Discussions**: Use `best_model_recommendation.txt` and comparison plots
4. **Appendix**: Include all Python scripts

## Quick Start Example

```bash
# 1. Train all models (this may take several hours)
python train_all_models.py --model all --epochs 50

# 2. Evaluate all models
python evaluate_all_models.py --model all

# 3. Compare and get recommendation
python compare_models.py

# 4. View results
cat best_model_recommendation.txt
```

## Notes

- Training all 11 models may take several hours depending on your hardware
- Models automatically use GPU if available
- Early stopping saves training time
- All results are saved in JSON/CSV format for easy analysis
- ROC curves are automatically generated for binary classification

## Troubleshooting

**Out of Memory Error:**
- Reduce batch size: `--batch_size 16`
- Train models one at a time

**Model Not Found:**
- Ensure you've trained the model first using `train_all_models.py`

**Missing Dependencies:**
- Run `pip install -r requirements.txt`

## Citation

For your report references, cite:
- PyTorch: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
- torchvision models: Various papers (ResNet, VGG, MobileNet, etc.)
