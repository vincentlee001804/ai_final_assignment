"""
Training script for all models: AlexNet, GoogleNet, ResNet18, ResNet50, ResNet101,
DenseNet169, MobileNetV2, MobileNetV3 Small/Large, VGG16, VGG19
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
import argparse

# Model configurations
MODEL_CONFIGS = {
    'alexnet': {'model': models.alexnet, 'pretrained': True},
    'googlenet': {'model': models.googlenet, 'pretrained': True},
    'resnet18': {'model': models.resnet18, 'pretrained': True},
    'resnet50': {'model': models.resnet50, 'pretrained': True},
    'resnet101': {'model': models.resnet101, 'pretrained': True},
    'densenet169': {'model': models.densenet169, 'pretrained': True},
    'mobilenet_v2': {'model': models.mobilenet_v2, 'pretrained': True},
    'mobilenet_v3_small': {'model': models.mobilenet_v3_small, 'pretrained': True},
    'mobilenet_v3_large': {'model': models.mobilenet_v3_large, 'pretrained': True},
    'vgg16': {'model': models.vgg16, 'pretrained': True},
    'vgg19': {'model': models.vgg19, 'pretrained': True},
}

def get_model(model_name, num_classes):
    """Get and modify model for transfer learning"""
    config = MODEL_CONFIGS[model_name.lower()]
    model = config['model'](pretrained=config['pretrained'])
    
    # Modify the final layer based on model architecture
    if 'resnet' in model_name.lower():
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'densenet' in model_name.lower():
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'mobilenet' in model_name.lower():
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'vgg' in model_name.lower() or 'alexnet' in model_name.lower():
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'googlenet' in model_name.lower():
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_model(model_name, train_loader, val_loader, num_classes, device):
    """Train a single model following sampletrain.py pattern"""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Model: {model_name.upper()}")
    print(f"Device: {device}")
    print(f"Starting training...")
    
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    num_epochs = 1000
    MOMENTUM = 0.9
    patience = 10
    
    # Get model
    model = get_model(model_name, num_classes)
    model.to(device)
    
    # Loss and optimizer (SGD with momentum)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
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
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_val_loss = total_val_loss + criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions = correct_predictions + (predicted == labels).sum().item()
                total_samples = total_samples + labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        validation_accuracy = correct_predictions / total_samples
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(validation_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")
        
        # Save last model
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        model_dir = os.path.join(results_dir, "trained_models")
        os.makedirs(model_dir, exist_ok=True)
        last_model_path = os.path.join(model_dir, f"{model_name}_last.pt")
        torch.save(model.state_dict(), last_model_path)
        
        # Early stopping and best model saving
        if avg_val_loss < best_val_loss: # Model is getting better/ improving.
            best_val_loss = avg_val_loss
            counter = 0
            best_model_path = os.path.join(model_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"{model_name}_best.pt has been saved!")
        else:
            counter = counter + 1
            
        if counter >= patience:
            print(f"Suspecting overfitting! Early stopping triggered after {patience} epochs of no improvement.")
            break # Stop the training loop.
    
    # Save training history
    history_path = os.path.join(model_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train all models for skin cancer classification')
    parser.add_argument('--model', type=str, default='all', 
                       help='Model to train (all, alexnet, googlenet, etc.)')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    # Setup
    current_directory = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    train_data_path = os.path.join(current_directory, args.data_dir, "train")
    val_data_path = os.path.join(current_directory, args.data_dir, "val")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
    val_dataset = ImageFolder(root=val_data_path, transform=train_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    class_names = train_dataset.classes
    with open("class_name.txt", "w") as file:
        for class_name in class_names:
            file.write(class_name + "\n")
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Determine which models to train
    if args.model.lower() == 'all':
        models_to_train = list(MODEL_CONFIGS.keys())
    else:
        if args.model.lower() not in MODEL_CONFIGS:
            print(f"Error: Model '{args.model}' not found. Available models: {list(MODEL_CONFIGS.keys())}")
            return
        models_to_train = [args.model.lower()]
    
    # Train all models
    results = {}
    total_models = len(models_to_train)
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING FOR {total_models} MODEL(S)")
    print(f"{'='*80}")
    
    for idx, model_name in enumerate(models_to_train, 1):
        print(f"\n{'#'*80}")
        print(f"# STATUS: Training Model {idx}/{total_models}: {model_name.upper()}")
        print(f"# Progress: {idx}/{total_models} ({idx*100//total_models}%)")
        print(f"{'#'*80}")
        try:
            model, history = train_model(
                model_name, train_loader, val_loader, num_classes, device
            )
            results[model_name] = {
                'best_val_loss': min(history['val_loss']),
                'best_val_accuracy': max(history['val_accuracy'])
            }
            print(f"\n[SUCCESS] {model_name.upper()} training completed successfully!")
        except Exception as e:
            print(f"\n[ERROR] Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total models trained: {len(results)}/{total_models}")
    for model_name, metrics in results.items():
        print(f"  [OK] {model_name}: Val Loss={metrics['best_val_loss']:.4f}, Val Acc={metrics['best_val_accuracy']:.4f}")
    print(f"\nAll models saved to: results/trained_models/")

if __name__ == "__main__":
    main()
