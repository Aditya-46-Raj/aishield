"""
Train a lightweight document forgery classifier
- Model: MobileNetV2 (transfer learning)
- Input: 256x256 RGB images
- Output: Binary classification (0=clean, 1=forged)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Dataset paths
DATASET_ROOT = Path("dataset")
CLEAN_DIR = DATASET_ROOT / "clean"
FORGED_DIR = DATASET_ROOT / "forged"
MODEL_PATH = Path("models/doc_classifier.pt")

class DocumentDataset(Dataset):
    """Custom dataset for document forgery detection"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Handle HEIC format - convert to RGB
        try:
            # Try to open with PIL (works for JPG, PNG, etc.)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If HEIC or unsupported format, try alternative loading
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
                image = Image.open(img_path).convert('RGB')
            except ImportError:
                print(f"Warning: Could not load {img_path}, skipping...")
                # Return a blank image as fallback
                image = Image.new('RGB', (256, 256), color=(0, 0, 0))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms():
    """Define augmentation transforms"""
    
    # Training augmentations - aggressive for better generalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(20),  # More rotation
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))
        ], p=0.4),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # More color variation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))  # Translation and scale
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0)  # Perspective distortion
        ], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # Random erasing/cutout
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_dataset(train_split=0.8):
    """Load and split dataset"""
    
    print("Loading dataset...")
    
    # Get all image paths - support multiple formats
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.heic', '*.HEIC']
    
    clean_images = []
    for ext in image_extensions:
        clean_images.extend(CLEAN_DIR.glob(ext))
    
    forged_images = []
    for ext in image_extensions:
        forged_images.extend(FORGED_DIR.glob(ext))
    
    print(f"  Found {len(clean_images)} clean images")
    print(f"  Found {len(forged_images)} forged images")
    
    # Create labels (0=clean, 1=forged)
    clean_labels = [0] * len(clean_images)
    forged_labels = [1] * len(forged_images)
    
    # Combine
    all_images = clean_images + forged_images
    all_labels = clean_labels + forged_labels
    
    # Shuffle
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)
    
    # Split
    split_idx = int(len(all_images) * train_split)
    train_images = all_images[:split_idx]
    train_labels = all_labels[:split_idx]
    val_images = all_images[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"  Total samples: {len(all_images)}")
    print(f"  Training: {len(train_images)} (clean: {train_labels.count(0)}, forged: {train_labels.count(1)})")
    print(f"  Validation: {len(val_images)} (clean: {val_labels.count(0)}, forged: {val_labels.count(1)})")
    
    return train_images, train_labels, val_images, val_labels

def create_model():
    """Create EfficientNet-B0 model with partially unfrozen layers"""
    
    print("\nCreating EfficientNet-B0 model...")
    
    # Load pretrained EfficientNet-B0 (stronger than MobileNetV2)
    model = models.efficientnet_b0(pretrained=True)
    
    # Freeze early layers, unfreeze last few blocks for fine-tuning
    for i, param in enumerate(model.features.parameters()):
        # Unfreeze last 40 parameters (approximately last 3-4 blocks)
        if i < len(list(model.features.parameters())) - 40:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Replace classifier with deeper head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=15):
    """Train the model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    model = model.to(device)
    criterion = nn.BCELoss()
    # Train all unfrozen parameters - higher learning rate for faster convergence
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
    
    # Early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = 8
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend((outputs > 0.5).cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Step scheduler
        scheduler.step()
        
        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ New best model saved (val_acc: {val_acc:.4f})")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    return history

def evaluate_model(model, val_loader):
    """Evaluate model with detailed metrics"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_preds.extend((outputs > 0.5).cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n" + "="*80)
    print("FINAL EVALUATION METRICS")
    print("="*80)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    return accuracy, precision, recall, f1

def main():
    print("="*80)
    print("DOCUMENT FORGERY CLASSIFIER TRAINING")
    print("="*80)
    
    # Load dataset
    train_images, train_labels, val_images, val_labels = load_dataset()
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = DocumentDataset(train_images, train_labels, train_transform)
    val_dataset = DocumentDataset(val_images, val_labels, val_transform)
    
    # Create data loaders with larger batch size for GPU and multi-workers for speed
    batch_size = 64 if torch.cuda.is_available() else 32
    num_workers = 4 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    # Create model
    model = create_model()
    
    # Train model
    history = train_model(model, train_loader, val_loader, num_epochs=15)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Evaluate
    accuracy, precision, recall, f1 = evaluate_model(model, val_loader)
    
    # Save model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'history': history
    }, MODEL_PATH)
    
    print(f"\n✓ Model saved to: {MODEL_PATH}")
    print(f"  File size: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")
    
    # Print loss curves summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print("\nLoss Curves:")
    print("  Epoch | Train Loss | Val Loss | Train Acc | Val Acc")
    print("  " + "-"*55)
    for i in range(len(history['train_loss'])):
        print(f"  {i+1:5d} | {history['train_loss'][i]:10.4f} | {history['val_loss'][i]:8.4f} | {history['train_acc'][i]:9.4f} | {history['val_acc'][i]:7.4f}")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
