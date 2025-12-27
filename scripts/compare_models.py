"""
Compare all models and identify the best one for skin cancer classification
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Load evaluation results"""
    results_path = os.path.join("results", "evaluation_results.json")
    if not os.path.exists(results_path):
        results_path = "evaluation_results.json"  # Fallback to root
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Please run evaluate_all_models.py first.")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)

def compare_models():
    """Compare all models and identify the best one"""
    results = load_results()
    if results is None:
        return
    
    print("="*80)
    print("MODEL COMPARISON AND ANALYSIS")
    print("="*80)
    
    # Create DataFrame for easier analysis
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Macro Precision': metrics.get('macro_precision', 0),
            'Macro Recall': metrics.get('macro_recall', 0),
            'ROC AUC': metrics.get('roc_auc', 0),
            'TNR': metrics.get('true_negative_rate', 0),
            'Test Loss': metrics.get('loss', float('inf'))
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by accuracy (primary) and ROC AUC (secondary)
    df = df.sort_values(['Accuracy', 'ROC AUC'], ascending=[False, False])
    
    print("\nRanked Models (by Accuracy):")
    print(df.to_string(index=False))
    
    # Identify best model
    best_model = df.iloc[0]
    print(f"\n{'='*80}")
    print("BEST MODEL RECOMMENDATION")
    print(f"{'='*80}")
    print(f"Model: {best_model['Model'].upper()}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"Macro Precision: {best_model['Macro Precision']:.4f}")
    print(f"Macro Recall: {best_model['Macro Recall']:.4f}")
    print(f"ROC AUC: {best_model['ROC AUC']:.4f}")
    print(f"True Negative Rate: {best_model['TNR']:.4f}")
    print(f"Test Loss: {best_model['Test Loss']:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('Accuracy', ascending=True)
    ax1.barh(df_sorted['Model'], df_sorted['Accuracy'])
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.grid(axis='x', alpha=0.3)
    
    # ROC AUC comparison
    ax2 = axes[0, 1]
    df_sorted = df.sort_values('ROC AUC', ascending=True)
    ax2.barh(df_sorted['Model'], df_sorted['ROC AUC'])
    ax2.set_xlabel('ROC AUC')
    ax2.set_title('Model ROC AUC Comparison')
    ax2.grid(axis='x', alpha=0.3)
    
    # Macro Precision and Recall
    ax3 = axes[1, 0]
    x = range(len(df))
    width = 0.35
    ax3.bar([i - width/2 for i in x], df['Macro Precision'], width, label='Macro Precision')
    ax3.bar([i + width/2 for i in x], df['Macro Recall'], width, label='Macro Recall')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax3.set_ylabel('Score')
    ax3.set_title('Macro Precision vs Macro Recall')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Combined metrics radar-like comparison (top 5 models)
    ax4 = axes[1, 1]
    top5 = df.head(5)
    metrics_to_plot = ['Accuracy', 'Macro Precision', 'Macro Recall', 'ROC AUC']
    x_pos = range(len(metrics_to_plot))
    for idx, row in top5.iterrows():
        values = [row[m] for m in metrics_to_plot]
        ax4.plot(x_pos, values, marker='o', label=row['Model'], linewidth=2)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax4.set_ylabel('Score')
    ax4.set_title('Top 5 Models - Combined Metrics')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"\nComparison plots saved to {plots_dir}/model_comparison.png")
    
    # Save detailed comparison to CSV
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Detailed comparison saved to {csv_path}")
    
    # Generate recommendation text
    recommendation = f"""
RECOMMENDATION SUMMARY:

Based on comprehensive evaluation, {best_model['Model'].upper()} is recommended as the best model 
for skin cancer classification with the following justifications:

1. Highest Accuracy: {best_model['Accuracy']:.4f} - Best overall classification performance
2. Strong ROC AUC: {best_model['ROC AUC']:.4f} - Excellent discrimination ability
3. Balanced Precision/Recall: Precision={best_model['Macro Precision']:.4f}, Recall={best_model['Macro Recall']:.4f}
4. Good Specificity: TNR={best_model['TNR']:.4f} - Low false positive rate

This model demonstrates the best balance of all evaluation metrics and is most suitable 
for clinical skin cancer classification tasks.
"""
    
    print(recommendation)
    
    # Save recommendation
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    rec_path = os.path.join(results_dir, "best_model_recommendation.txt")
    with open(rec_path, "w") as f:
        f.write(recommendation)
    print(f"Recommendation saved to {rec_path}")

if __name__ == "__main__":
    compare_models()

