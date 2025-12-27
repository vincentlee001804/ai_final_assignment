import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import MyCNNModel
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

current_directory = os.getcwd()

# Load class names
file_path = "class_name.txt"
if not os.path.exists(file_path):
    print(f"Error: Class names file '{file_path}' not found. Please train the model first using train.py")
    exit(1)

with open(file_path, "r") as file:
    class_names = [line.strip() for line in file.readlines()]

num_classes = len(class_names)

# Load model
trainedModel = "best.pt"
if not os.path.exists(trainedModel):
    print(f"Error: Model file '{trainedModel}' not found. Please train the model first using train.py")
    exit(1)

model = MyCNNModel(num_classes)
model.load_state_dict(torch.load(trainedModel, weights_only=True))
model.eval()

# Test data path
test_data_path = os.path.join(current_directory, "data", "test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = ImageFolder(root=test_data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

criterion = nn.CrossEntropyLoss()

# Evaluate on test set
all_predictions = []
all_labels = []
total_test_loss = 0.0
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
test_accuracy = correct_predictions / total_samples

print("\n" + "="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({correct_predictions}/{total_samples})")
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))
print("="*60)

