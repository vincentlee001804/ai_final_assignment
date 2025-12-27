"""
Complete pipeline script to run all steps of the assignment
Following assignment task requirements:
1. Dataset (already prepared)
2. Data split 70/15/15 (already done)
3. Train all 11 models
4. Hyperparameter tuning for all models
5. Evaluate all models with required metrics
6. Compare and recommend best model
7. Generate report data
"""
import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"TASK: {description}")
    print(f"{'='*80}")
    print(f"Running: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    else:
        print(result.stdout)
        return True

def main():
    print("="*80)
    print("COMPLETE PIPELINE FOR SKIN CANCER CLASSIFICATION ASSIGNMENT")
    print("Following all assignment task requirements")
    print("="*80)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("Error: 'data' directory not found. Please ensure your dataset is in data/train, data/val, data/test")
        sys.exit(1)
    
    steps = [
        # Task 4: Hyperparameter tuning (should be done before training)
        ("python scripts/hyperparameter_tuning.py --model all", 
         "Task 4: Perform hyperparameter tunings on all models"),
        
        # Task 3: Train all models
        ("python scripts/train_all_models.py --model all", 
         "Task 3: Train all 11 models (AlexNet, GoogleNet, ResNet18, ResNet50, ResNet101, DenseNet169, MobileNetV2, MobileNetV3 Small/Large, VGG16, VGG19)"),
        
        # Task 5: Evaluate all models
        ("python scripts/evaluate_all_models.py --model all", 
         "Task 5: Evaluate all models on test set (accuracy, recall, TNR, precision, ROC curve)"),
        
        # Task 6: Compare models
        ("python scripts/compare_models.py", 
         "Task 6: Compare results and propose best model"),
        
        # Task 7: Generate report data
        ("python scripts/generate_report_data.py", 
         "Task 7: Generate data for technical report")
    ]
    
    print("\nThis pipeline will:")
    print("1. Perform hyperparameter tuning for all models")
    print("2. Train all 11 models with best hyperparameters")
    print("3. Evaluate all models with required metrics")
    print("4. Compare models and recommend the best one")
    print("5. Generate report data")
    print("\nNote: This will take several hours to complete.")
    print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
    
    try:
        import time
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nPipeline cancelled by user.")
        sys.exit(0)
    
    for command, description in steps:
        success = run_command(command, description)
        if not success:
            print(f"\nPipeline stopped at: {description}")
            print("Please check the error messages above.")
            sys.exit(1)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files for your report:")
    print("  - trained_models/: All trained model checkpoints")
    print("  - hyperparameter_results/: Best hyperparameters for each model")
    print("  - evaluation_results.json: All evaluation metrics")
    print("  - model_comparison.csv: Detailed comparison table")
    print("  - comparison_plots/: Visualization plots")
    print("  - roc_curves/: ROC curves for all models")
    print("  - best_model_recommendation.txt: Final recommendation")
    print("  - report_data.txt: Formatted data for your report")
    print("\nYou can now use these results to prepare your technical report (Task 7).")
    print("Remember to format references in IEEE style (Task 8).")

if __name__ == "__main__":
    main()
