"""
Train Deepfake Detector Model
Trains EfficientNet-B0 to classify real vs. deepfake/synthetic faces

For demo purposes, we'll use a simple dataset:
- Real: Clean ID document faces (from clean folder)
- Fake: Placeholder for deepfake/synthetic faces (using forged for now)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import time
import cv2
import numpy as np

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Using device: {device}")

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
CLEAN_DIR = DATASET_DIR / "clean"
FORGED_DIR = DATASET_DIR / "forged"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEEPFAKE_MODEL_PATH = MODEL_DIR / "deepfake_detector.pt"

# Simple dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load real images (label 0 = real)
        real_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        for ext in real_exts:
            for img_path in real_dir.glob(f"*{ext}"):
                self.samples.append((str(img_path), 0))
        
        # Load fake images (label 1 = fake)
        for ext in real_exts:
            for img_path in fake_dir.glob(f"*{ext}"):
                self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len([s for s in self.samples if s[1]==0])} real, {len([s for s in self.samples if s[1]==1])} fake images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image with OpenCV and convert to RGB
        img = cv2.imread(img_path)
        if img is None:
            # Fallback to PIL for special formats
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
print("Loading dataset...")
full_dataset = DeepfakeDataset(CLEAN_DIR, FORGED_DIR, transform=train_transform)

# Split: 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Update validation set transform
val_dataset.dataset.transform = val_transform

# DataLoaders
batch_size = 32 if torch.cuda.is_available() else 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"Batch size: {batch_size}")

# Build model - EfficientNet-B0
print("Building EfficientNet-B0 model...")
model = models.efficientnet_b0(pretrained=True)

# Replace classifier
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()  # Binary classification
)

# Unfreeze last few layers for fine-tuning
for param in model.features[-3:].parameters():
    param.requires_grad = True

model = model.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

# Training loop
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(loader), correct / total

# Train
print("\n" + "="*60)
print("TRAINING DEEPFAKE DETECTOR")
print("="*60)

num_epochs = 10
best_val_acc = 0.0
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), DEEPFAKE_MODEL_PATH)
        print(f"  ✓ Best model saved (val_acc: {val_acc*100:.2f}%)")

total_time = time.time() - start_time

print("\n" + "="*60)
print(f"TRAINING COMPLETE - {total_time/60:.2f} minutes")
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"Model saved to: {DEEPFAKE_MODEL_PATH}")
print("="*60)
