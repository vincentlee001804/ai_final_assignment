import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import MyCNNModel
import os
import sys

# Get image path from command line argument or use default
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "./data/test/benign/1000.jpg"  # Default test image

trainedModel = "best.pt"

# Check if model file exists
if not os.path.exists(trainedModel):
    print(f"Error: Model file '{trainedModel}' not found. Please train the model first using train.py")
    sys.exit(1)

# Check if class names file exists
file_path = "class_name.txt"
if not os.path.exists(file_path):
    print(f"Error: Class names file '{file_path}' not found. Please train the model first using train.py")
    sys.exit(1)

with open(file_path, "r") as file:
    rows = file.readlines()

rows = [row.strip() for row in rows]
num_classes = len(rows)

model = MyCNNModel(num_classes)
model.load_state_dict(torch.load(trainedModel, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Check if image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    sys.exit(1)

input_image = Image.open(image_path).convert("RGB")
input_image = transform(input_image)
input_batch = input_image.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)

with torch.no_grad():
    output = model(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(probabilities).item()
confidence_score = probabilities[predicted_class].item()
predicted_label = rows[predicted_class]

print(f"Image: {image_path}")
print(f"Predicted class: {predicted_label}")
print(f"Confidence: {confidence_score:.4f}")

