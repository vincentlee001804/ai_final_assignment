"""
Generate confusion matrix visualizations from evaluation_results.json
Creates both individual confusion matrices and a combined grid
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt

# Try to import seaborn, fallback to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not available, using matplotlib for visualization")

def load_class_names():
    """Load class names from class_name.txt"""
    class_name_file = "class_name.txt"
    if not os.path.exists(class_name_file):
        print(f"Warning: {class_name_file} not found. Using default class names.")
        return ['benign', 'malignant']
    
    with open(class_name_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def generate_confusion_matrices_from_json(json_path='results/evaluation_results.json'):
    """Generate confusion matrix plots from evaluation_results.json"""
    
    # Load results
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please run evaluation first.")
        return
    
    with open(json_path, 'r') as f:
        all_results = json.load(f)
    
    if not all_results:
        print("Error: No results found in JSON file.")
        return
    
    # Load class names
    class_names = load_class_names()
    
    results_dir = "results"
    confusion_dir = os.path.join(results_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    
    print(f"Generating confusion matrices for {len(all_results)} models...")
    
    # Generate individual confusion matrices
    for model_name, metrics in all_results.items():
        if 'confusion_matrix' not in metrics:
            print(f"Warning: No confusion matrix found for {model_name}")
            continue
        
        cm = np.array(metrics['confusion_matrix'])
        accuracy = metrics.get('accuracy', 0)
        
        # Create individual confusion matrix plot
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
        
        individual_path = os.path.join(confusion_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {individual_path}")
    
    # Generate combined grid
    print("\nGenerating combined confusion matrices grid...")
    num_models = len([m for m in all_results.values() if 'confusion_matrix' in m])
    
    if num_models == 0:
        print("No confusion matrices to combine.")
        return
    
    # Calculate grid dimensions (approximately square)
    cols = int(ceil(sqrt(num_models)))
    rows = int(ceil(num_models / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    idx = 0
    for model_name, metrics in all_results.items():
        if 'confusion_matrix' not in metrics:
            continue
        
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
        idx += 1
    
    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, y=0.995)
    plt.tight_layout()
    
    combined_path = os.path.join(results_dir, "confusion_matrices_all_models.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {combined_path}")
    
    print(f"\n[SUCCESS] All confusion matrices generated successfully!")
    print(f"  Individual matrices: {confusion_dir}/")
    print(f"  Combined grid: {combined_path}")

if __name__ == "__main__":
    generate_confusion_matrices_from_json()

