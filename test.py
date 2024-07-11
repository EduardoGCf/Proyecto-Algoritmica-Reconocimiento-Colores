import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Ensure your trained model file path is correct
MODEL_PATH = 'color_detection_with_masks_model.pth'

# Define the correct model architecture to match the saved model
class ColorDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ColorDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, image, mask):
        x = torch.cat((image, mask), dim=1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorDetectionModel(num_classes=9)  # Update num_classes if needed
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
classes = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Violet', 'White', 'Yellow']

def predict_image(image, mask):
    image = transform(image).unsqueeze(0).to(device)
    mask = transform(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image, mask)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Capture from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    crop_size = 100  # Define the size of the crop around the center

    # Calculate the coordinates of the region to crop
    x1 = max(center_x - crop_size // 2, 0)
    y1 = max(center_y - crop_size // 2, 0)
    x2 = min(center_x + crop_size // 2, width)
    y2 = min(center_y + crop_size // 2, height)
    
    # Crop the region from the frame
    cropped_frame = frame[y1:y2, x1:x2]
    
    # Convert the cropped frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    
    # Create a dummy mask (as the original model expects a mask as well)
    mask = Image.new('RGB', pil_image.size, (255, 255, 255))
    
    # Predict the color
    predicted_class = predict_image(pil_image, mask)
    color_name = classes[predicted_class]
    
    # Display the prediction on the frame
    cv2.putText(frame, f"Color: {color_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw a crosshair in the center of the frame
    cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, 
                   markerSize=20, thickness=2)
    
    # Draw the rectangle around the target region
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Webcam - Color Detection', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
