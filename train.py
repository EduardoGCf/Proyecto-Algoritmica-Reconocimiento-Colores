import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class MaskedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size, classes, subset):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.classes = classes
        self.subset = subset
        self.image_paths = []
        self.mask_paths = []
        self.labels = []

        for idx, cls in enumerate(classes):
            img_dir = os.path.join(image_dir, cls)
            msk_dir = os.path.join(mask_dir, cls)
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Image directory {img_dir} does not exist")
            if not os.path.exists(msk_dir):
                raise FileNotFoundError(f"Mask directory {msk_dir} does not exist")
            img_paths = sorted(os.listdir(img_dir))
            mask_paths = sorted(os.listdir(msk_dir))
            for img, mask in zip(img_paths, mask_paths):
                self.image_paths.append(os.path.join(img_dir, img))
                self.mask_paths.append(os.path.join(msk_dir, mask))
                self.labels.append(idx)
        
        # Split dataset into training and validation sets
        split = int(len(self.image_paths) * 0.8)
        if subset == 'training':
            self.image_paths = self.image_paths[:split]
            self.mask_paths = self.mask_paths[:split]
            self.labels = self.labels[:split]
        else:
            self.image_paths = self.image_paths[split:]
            self.mask_paths = self.mask_paths[split:]
            self.labels = self.labels[split:]
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image, mask, label


class ColorDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(ColorDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjust for 128x128 input size
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, image, mask):
        x = torch.cat((image, mask), dim=1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Adjust for 128x128 input size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Absolute paths
image_dir = r'D:\Users\kodo\Documents\NUR\2024 - I\Web\arduino stuff\Second try\data\images'
mask_dir = r'D:\Users\kodo\Documents\NUR\2024 - I\Web\arduino stuff\Second try\data\masks'
image_size = (128, 128)  # Consistent size for resizing
batch_size = 16
classes = ['Black', 'Blue', 'Brown', 'Green', 'Orange', 'Red', 'Violet', 'White', 'Yellow']

# Verify directories
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory {image_dir} does not exist")
if not os.path.exists(mask_dir):
    raise FileNotFoundError(f"Mask directory {mask_dir} does not exist")

# Load data
train_dataset = MaskedDataset(image_dir, mask_dir, image_size, classes, 'training')
validation_dataset = MaskedDataset(image_dir, mask_dir, image_size, classes, 'validation')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorDetectionModel(num_classes=len(classes)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks, labels in train_loader:
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks, labels in validation_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Validation Accuracy: {100 * correct / total}%")

# Save the model
torch.save(model.state_dict(), 'color_detection_with_masks_model.pth')
