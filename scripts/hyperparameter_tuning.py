"""
Hyperparameter tuning script for all models
Following the pattern from sampletrain.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import os
import json
import itertools
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_all_models import get_model

def tune_hyperparameters(model_name, train_loader, val_loader, num_classes, device):
    """Perform hyperparameter tuning for a model following sampletrain.py pattern"""
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Model: {model_name.upper()}")
    print(f"Device: {device}")
    
    # Hyperparameter search space (Task 4: Perform hyperparameter tunings)
    # Reduced search space for faster tuning: 8 combinations instead of 27
    learning_rates = [0.0001, 0.001]  # Reduced from 3 to 2 (most common range)
    batch_sizes = [32, 64]  # Reduced from 3 to 2 (common sizes)
    momentum_values = [0.9, 0.95]  # Reduced from 3 to 2 (most effective values)
    
    best_config = None
    best_val_loss = float('inf')
    results = []
    
    total_combinations = len(learning_rates) * len(batch_sizes) * len(momentum_values)
    print(f"Total combinations to test: {total_combinations}", flush=True)
    print(f"Starting hyperparameter search...", flush=True)
    
    combination_num = 0
    for lr, batch_size, momentum in itertools.product(learning_rates, batch_sizes, momentum_values):
        combination_num += 1
        print(f"\n[{combination_num}/{total_combinations}] Testing: LR={lr}, Batch Size={batch_size}, Momentum={momentum}", flush=True)
        print(f"  Training... (this may take a few minutes)", flush=True)
        
        # Get model
        model = get_model(model_name, num_classes)
        model.to(device)
        
        # Setup optimizer (following sampletrain.py pattern - SGD with momentum)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        
        # Quick training (fewer epochs for tuning, following sampletrain.py structure)
        best_val_loss_this = float('inf')
        patience = 5  # Shorter patience for tuning
        counter = 0
        
        for epoch in range(20):  # Reduced epochs for faster tuning
            # Training phase (following sampletrain.py pattern)
            model.train()
            total_training_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs) # Feedforward
                loss = criterion(outputs, labels) # Loss calculation (using cross entropy loss function)
                loss.backward() # Backpropagation
                optimizer.step() # Update weights
                total_training_loss = total_training_loss + loss.item()
            
            avg_train_loss = total_training_loss / len(train_loader)
            
            # Validation phase (following sampletrain.py pattern)
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    total_val_loss = total_val_loss + criterion(outputs, labels).item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss_this:
                best_val_loss_this = avg_val_loss
                counter = 0
            else:
                counter += 1
            
            # Early stopping for tuning
            if counter >= patience:
                break
        
        # Flush output to ensure progress is visible
        sys.stdout.flush()
        
        results.append({
            'lr': lr,
            'batch_size': batch_size,
            'momentum': momentum,
            'val_loss': best_val_loss_this
        })
        
        print(f"  Result: Val Loss = {best_val_loss_this:.4f}", flush=True)
        
        if best_val_loss_this < best_val_loss:
            best_val_loss = best_val_loss_this
            best_config = {
                'lr': lr,
                'batch_size': batch_size,
                'momentum': momentum
            }
            print(f"  -> New best configuration!", flush=True)
    
    print(f"\nBest configuration for {model_name}:")
    print(f"  Learning Rate: {best_config['lr']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Momentum: {best_config['momentum']}")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    hyper_dir = os.path.join(results_dir, "hyperparameter_results")
    os.makedirs(hyper_dir, exist_ok=True)
    results_path = os.path.join(hyper_dir, f"{model_name}_tuning.json")
    with open(results_path, 'w') as f:
        json.dump({
            'best_config': best_config,
            'best_val_loss': best_val_loss,
            'all_results': results
        }, f, indent=2)
    
    return best_config

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for models (Task 4)')
    parser.add_argument('--model', type=str, default='all', 
                       help='Model name to tune (all, alexnet, googlenet, etc.)')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    current_directory = os.getcwd()
    train_path = os.path.join(current_directory, args.data_dir, "train")
    val_path = os.path.join(current_directory, args.data_dir, "val")
    
    # Data transforms (following sampletrain.py pattern - simple transforms)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = ImageFolder(root=train_path, transform=train_transform)
    val_dataset = ImageFolder(root=val_path, transform=train_transform)
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # Determine which models to tune
    from train_all_models import MODEL_CONFIGS
    if args.model.lower() == 'all':
        models_to_tune = list(MODEL_CONFIGS.keys())
    else:
        if args.model.lower() not in MODEL_CONFIGS:
            print(f"Error: Model '{args.model}' not found. Available models: {list(MODEL_CONFIGS.keys())}")
            return
        models_to_tune = [args.model.lower()]
    
    # Perform tuning for all models
    all_best_configs = {}
    total_models = len(models_to_tune)
    print(f"\n{'='*80}")
    print(f"STARTING HYPERPARAMETER TUNING FOR {total_models} MODEL(S)")
    print(f"{'='*80}")
    
    for idx, model_name in enumerate(models_to_tune, 1):
        print(f"\n{'#'*80}")
        print(f"# STATUS: Tuning Model {idx}/{total_models}: {model_name.upper()}")
        print(f"# Progress: {idx}/{total_models} ({idx*100//total_models}%)")
        print(f"{'#'*80}")
        try:
            # Create data loaders with different batch sizes will be handled in tuning
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            best_config = tune_hyperparameters(
                model_name, train_loader, val_loader, num_classes, device
            )
            all_best_configs[model_name] = best_config
            print(f"\n[SUCCESS] {model_name.upper()} hyperparameter tuning completed successfully!")
        except Exception as e:
            print(f"\n[ERROR] Error tuning {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    hyper_dir = os.path.join(results_dir, "hyperparameter_results")
    os.makedirs(hyper_dir, exist_ok=True)
    summary_path = os.path.join(hyper_dir, "all_models_best_configs.json")
    with open(summary_path, 'w') as f:
        json.dump(all_best_configs, f, indent=2)
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*80}")
    print(f"Total models tuned: {len(all_best_configs)}/{total_models}")
    for model_name, config in all_best_configs.items():
        print(f"  [OK] {model_name}: LR={config['lr']}, Batch={config['batch_size']}, Momentum={config['momentum']}")
    print(f"\nAll results saved to: results/hyperparameter_results/")

if __name__ == "__main__":
    main()
