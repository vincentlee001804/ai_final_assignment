"""
Training script for RESNET50
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

current_directory = os.getcwd()

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
num_epochs = 1000
MOMENTUM = 0.9
patience = 10

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data_path = os.path.join(current_directory, "data", "train")
val_data_path = os.path.join(current_directory, "data", "val")

train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
val_dataset = ImageFolder(root=val_data_path, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
with open("class_name.txt", "w") as file:
    for class_name in class_names:
        file.write(class_name + "\n")

num_classes = len(train_dataset.classes)

# Get RESNET50 model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
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

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.join(results_dir, "trained_models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "resnet50_last.pt"))

    if avg_val_loss < best_val_loss: # Model is getting better/ improving.
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(model_dir, "resnet50_best.pt"))
        print("resnet50_best.pt has been saved!")
    else:
        counter = counter + 1
        
    if counter >= patience:
        print(f"Suspecting overfitting! Early stopping triggered after {patience} epochs of no improvement.")
        break # Stop the training loop.
