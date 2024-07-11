import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
            img_paths = sorted(os.listdir(os.path.join(image_dir, cls)))
            mask_paths = sorted(os.listdir(os.path.join(mask_dir, cls)))
            for img, mask in zip(img_paths, mask_paths):
                self.image_paths.append(os.path.join(image_dir, cls, img))
                self.mask_paths.append(os.path.join(mask_dir, cls, mask))
                self.labels.append(idx)
        
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
