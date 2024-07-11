import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model architecture
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = 10  # replace with the actual number of classes
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the saved weights
model.load_state_dict(torch.load('color_detection_model.pth'))
model.eval()  # Set the model to evaluation mode

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

image_path = '/path/to/your/image.jpg'
image = preprocess_image(image_path)

# Make predictions
with torch.no_grad():
    outputs = model(image.to(device))
    _, preds = torch.max(outputs, 1)

# Get the predicted class
predicted_class = preds.item()
class_names = image_datasets['train'].classes  # replace with your actual class names
print(f'The predicted color is: {class_names[predicted_class]}')