"""
Generate formatted data for technical report
Extracts and formats results for easy copy-paste into report
"""
import json
import os
import pandas as pd

def generate_report_sections():
    """Generate formatted text for report sections"""
    
    # Load evaluation results
    results_path = os.path.join("results", "evaluation_results.json")
    if not os.path.exists(results_path):
        results_path = "evaluation_results.json"  # Fallback to root
    if not os.path.exists(results_path):
        print("Error: evaluation_results.json not found. Run evaluate_all_models.py first.")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load comparison data
    csv_path = os.path.join("results", "model_comparison.csv")
    if not os.path.exists(csv_path):
        csv_path = "model_comparison.csv"  # Fallback to root
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("Error: model_comparison.csv not found. Run compare_models.py first.")
        return
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False)
    
    output = []
    output.append("="*80)
    output.append("TECHNICAL REPORT DATA")
    output.append("="*80)
    
    # Results Section
    output.append("\n\n## RESULTS SECTION")
    output.append("-"*80)
    output.append("\n### Table: Model Performance Comparison\n")
    
    # Create formatted table
    table_data = []
    for _, row in df.iterrows():
        table_data.append({
            'Model': row['Model'],
            'Accuracy': f"{row['Accuracy']:.4f}",
            'Macro Precision': f"{row['Macro Precision']:.4f}",
            'Macro Recall': f"{row['Macro Recall']:.4f}",
            'ROC AUC': f"{row['ROC AUC']:.4f}" if 'ROC AUC' in row else "N/A"
        })
    
    table_df = pd.DataFrame(table_data)
    output.append(table_df.to_string(index=False))
    
    # Best Model
    best = df.iloc[0]
    output.append(f"\n\n### Best Model: {best['Model'].upper()}")
    output.append(f"- Accuracy: {best['Accuracy']:.4f}")
    output.append(f"- Macro Precision: {best['Macro Precision']:.4f}")
    output.append(f"- Macro Recall: {best['Macro Recall']:.4f}")
    if 'ROC AUC' in best:
        output.append(f"- ROC AUC: {best['ROC AUC']:.4f}")
    
    # Detailed Results
    output.append("\n\n### Detailed Results for All Models\n")
    for model_name, metrics in results.items():
        output.append(f"\n**{model_name.upper()}:**")
        output.append(f"  - Accuracy: {metrics.get('accuracy', 0):.4f}")
        output.append(f"  - Test Loss: {metrics.get('loss', 0):.4f}")
        
        if 'precision_class_0' in metrics:
            output.append(f"  - Precision (Benign): {metrics['precision_class_0']:.4f}")
            output.append(f"  - Precision (Malignant): {metrics['precision_class_1']:.4f}")
            output.append(f"  - Recall (Benign): {metrics['recall_class_0']:.4f}")
            output.append(f"  - Recall (Malignant): {metrics['recall_class_1']:.4f}")
            output.append(f"  - True Negative Rate: {metrics.get('true_negative_rate', 0):.4f}")
            output.append(f"  - ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        
        output.append(f"  - Macro Precision: {metrics.get('macro_precision', 0):.4f}")
        output.append(f"  - Macro Recall: {metrics.get('macro_recall', 0):.4f}")
    
    # Confusion Matrices
    output.append("\n\n### Confusion Matrices\n")
    for model_name, metrics in results.items():
        if 'confusion_matrix' in metrics:
            output.append(f"\n**{model_name.upper()}:**")
            cm = metrics['confusion_matrix']
            if len(cm) == 2 and len(cm[0]) == 2:
                output.append(f"  True Negatives: {cm[0][0]}")
                output.append(f"  False Positives: {cm[0][1]}")
                output.append(f"  False Negatives: {cm[1][0]}")
                output.append(f"  True Positives: {cm[1][1]}")
    
    # Save to file
    output_text = "\n".join(output)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "report_data.txt")
    with open(report_path, "w") as f:
        f.write(output_text)
    
    print(output_text)
    print("\n\n" + "="*80)
    print(f"Report data saved to: {report_path}")
    print("="*80)

if __name__ == "__main__":
    generate_report_sections()

