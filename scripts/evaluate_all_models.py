"""
Evaluate all trained models on test set with comprehensive metrics
For binary classification: accuracy, recall, TNR, precision, ROC curve
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import sys
import os
from math import ceil, sqrt

# Try to import seaborn, fallback to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_all_models import get_model, MODEL_CONFIGS

def evaluate_model(model_name, test_loader, num_classes, device, class_names):
    """Evaluate a single model on test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load model (following sampletrain.py pattern - load best.pt)
    model_path = os.path.join("results", "trained_models", f"{model_name}_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join("trained_models", f"{model_name}_best.pt")  # Fallback
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    # Collect predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # For binary classification
    if num_classes == 2:
        # Precision, Recall for each class
        precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        
        # True Negative Rate (Specificity)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Overall precision and recall (macro average)
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        
        # ROC curve for binary classification
        # Use probability of positive class (class 1)
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            'accuracy': float(accuracy),
            'loss': float(avg_loss),
            'precision_class_0': float(precision[0]),
            'precision_class_1': float(precision[1]),
            'recall_class_0': float(recall[0]),
            'recall_class_1': float(recall[1]),
            'true_negative_rate': float(tnr),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
        
        # Print results
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nPer-class Precision: {precision}")
        print(f"Per-class Recall: {recall}")
        print(f"True Negative Rate (TNR): {tnr:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Save ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name.upper()}')
        plt.legend(loc="lower right")
        
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        roc_dir = os.path.join(results_dir, "roc_curves")
        os.makedirs(roc_dir, exist_ok=True)
        plt.savefig(os.path.join(roc_dir, f"{model_name}_roc.png"))
        plt.close()
        
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Count'})
        else:
            im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, label='Count')
            plt.xticks(range(len(class_names)), class_names)
            plt.yticks(range(len(class_names)), class_names)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')
        plt.title(f'Confusion Matrix - {model_name.upper()}\nAccuracy: {accuracy:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        confusion_dir = os.path.join(results_dir, "confusion_matrices")
        os.makedirs(confusion_dir, exist_ok=True)
        plt.savefig(os.path.join(confusion_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
        
    else:
        # Multi-class classification
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'loss': float(avg_loss),
            'macro_precision': float(precision),
            'macro_recall': float(recall),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all models on test set')
    parser.add_argument('--model', type=str, default='all', help='Model to evaluate (all or specific model)')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class names
    with open("class_name.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    
    # Test data path
    test_path = os.path.join(args.data_dir, "test")
    
    # Data transform (following sampletrain.py pattern - simple transforms)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    test_dataset = ImageFolder(root=test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Determine which models to evaluate
    if args.model.lower() == 'all':
        models_to_evaluate = list(MODEL_CONFIGS.keys())
    else:
        if args.model.lower() not in MODEL_CONFIGS:
            print(f"Error: Model '{args.model}' not found.")
            return
        models_to_evaluate = [args.model.lower()]
    
    # Evaluate all models
    all_results = {}
    for model_name in models_to_evaluate:
        try:
            metrics = evaluate_model(model_name, test_loader, num_classes, device, class_names)
            if metrics:
                all_results[model_name] = metrics
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    # Save all results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate combined confusion matrices grid
    if all_results:
        try:
            num_models = len(all_results)
            # Calculate grid dimensions (approximately square)
            cols = int(ceil(sqrt(num_models)))
            rows = int(ceil(num_models / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            if num_models == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, (model_name, metrics) in enumerate(all_results.items()):
                if 'confusion_matrix' in metrics:
                    cm = np.array(metrics['confusion_matrix'])
                    accuracy = metrics.get('accuracy', 0)
                    
                    if HAS_SEABORN:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                                    xticklabels=class_names, yticklabels=class_names,
                                    cbar_kws={'label': 'Count'})
                    else:
                        im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
                        plt.colorbar(im, ax=axes[idx], label='Count')
                        axes[idx].set_xticks(range(len(class_names)))
                        axes[idx].set_xticklabels(class_names)
                        axes[idx].set_yticks(range(len(class_names)))
                        axes[idx].set_yticklabels(class_names)
                        for i in range(len(class_names)):
                            for j in range(len(class_names)):
                                axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center',
                                              color='white' if cm[i, j] > cm.max() / 2 else 'black')
                    axes[idx].set_title(f'{model_name.upper()}\nAcc: {accuracy:.4f}', fontsize=10)
                    axes[idx].set_ylabel('True Label', fontsize=9)
                    axes[idx].set_xlabel('Predicted Label', fontsize=9)
            
            # Hide unused subplots
            for idx in range(num_models, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('Confusion Matrices - All Models', fontsize=16, y=0.995)
            plt.tight_layout()
            
            confusion_dir = os.path.join(results_dir, "confusion_matrices")
            os.makedirs(confusion_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, "confusion_matrices_all_models.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nCombined confusion matrices saved to: {os.path.join(results_dir, 'confusion_matrices_all_models.png')}")
        except Exception as e:
            print(f"Warning: Could not generate combined confusion matrices: {str(e)}")
    
    # Print summary comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro Precision':<18} {'Macro Recall':<15} {'ROC AUC':<10}")
    print("-" * 80)
    
    for model_name, metrics in all_results.items():
        acc = metrics.get('accuracy', 0)
        macro_prec = metrics.get('macro_precision', 0)
        macro_rec = metrics.get('macro_recall', 0)
        roc_auc = metrics.get('roc_auc', 0)
        
        print(f"{model_name:<20} {acc:<12.4f} {macro_prec:<18.4f} {macro_rec:<15.4f} {roc_auc:<10.4f}")
    
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()

