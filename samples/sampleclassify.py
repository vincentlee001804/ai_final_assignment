import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import MyCNNModel

image_path = "./test_data/48.png"
trainedModel = "best.pt"

file_path = "class_name.txt"
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

print(f"Predicted class: {predicted_label}")
print(f"Confidence: {confidence_score:.4f}")
