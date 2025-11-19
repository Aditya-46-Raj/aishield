"""
Generate a balanced document forgery dataset with augmentations
- Clean samples: synthetic documents with variations
- Forged samples: patch forgery with multiple techniques
"""
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
import os
from pathlib import Path

# Dataset directories
DATASET_ROOT = Path("dataset")
CLEAN_DIR = DATASET_ROOT / "clean"
FORGED_DIR = DATASET_ROOT / "forged"

def create_directories():
    """Create dataset directory structure"""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    FORGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directories:")
    print(f"  - {CLEAN_DIR}")
    print(f"  - {FORGED_DIR}")

def create_synthetic_image(width=800, height=600, seed=42):
    """Create a synthetic image with random patterns"""
    np.random.seed(seed)
    # Create a random RGB image
    img_array = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Add gradient for realism
    for i in range(height):
        img_array[i, :, :] += int((i / height) * 50)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array, 'RGB')

def apply_augmentations(img, aug_type='none'):
    """Apply various augmentations to simulate real-world variations"""
    img_aug = img.copy()
    
    if aug_type == 'blur':
        # Slight blur
        img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    elif aug_type == 'rotate':
        # Small rotation (-5 to +5 degrees)
        angle = random.uniform(-5, 5)
        img_aug = img_aug.rotate(angle, fillcolor=(200, 200, 200), expand=False)
    
    elif aug_type == 'brightness':
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(img_aug)
        factor = random.uniform(0.8, 1.2)
        img_aug = enhancer.enhance(factor)
    
    elif aug_type == 'contrast':
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(img_aug)
        factor = random.uniform(0.8, 1.2)
        img_aug = enhancer.enhance(factor)
    
    elif aug_type == 'compression':
        # Simulate JPEG compression artifacts
        import io
        buffer = io.BytesIO()
        quality = random.randint(60, 85)
        img_aug.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_aug = Image.open(buffer).copy()
    
    return img_aug

def create_patch_forgery(source_img, donor_img, patch_size_ratio=(0.15, 0.25)):
    """Create a forged image by copying a patch from donor to source"""
    src = source_img.copy()
    donor = donor_img.resize(src.size)
    w, h = src.size
    
    # Random patch size
    patch_w = int(w * random.uniform(*patch_size_ratio))
    patch_h = int(h * random.uniform(*patch_size_ratio))
    
    # Random position
    x_src = random.randint(0, w - patch_w)
    y_src = random.randint(0, h - patch_h)
    
    # Random donor position
    x_donor = random.randint(0, w - patch_w)
    y_donor = random.randint(0, h - patch_h)
    
    # Copy patch
    patch = donor.crop((x_donor, y_donor, x_donor + patch_w, y_donor + patch_h))
    src.paste(patch, (x_src, y_src))
    
    return src

def generate_clean_samples(num_samples=200):
    """Generate clean document samples with augmentations"""
    print(f"\nGenerating {num_samples} clean samples...")
    
    augmentation_types = ['none', 'blur', 'rotate', 'brightness', 'contrast', 'compression']
    
    for i in range(num_samples):
        # Use different seeds for variety
        seed = 1000 + i * 17  # Prime number spacing for better randomness
        
        # Create base synthetic image
        img = create_synthetic_image(width=800, height=600, seed=seed)
        
        # Apply random augmentation
        aug_type = random.choice(augmentation_types)
        img_aug = apply_augmentations(img, aug_type)
        
        # Save
        filename = f"clean_{i:04d}_seed{seed}_aug{aug_type}.jpg"
        filepath = CLEAN_DIR / filename
        img_aug.save(filepath, quality=95)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_samples} clean samples...")
    
    print(f"✓ Completed {num_samples} clean samples")

def generate_forged_samples(num_samples=200):
    """Generate forged document samples with various forgery techniques"""
    print(f"\nGenerating {num_samples} forged samples...")
    
    augmentation_types = ['none', 'blur', 'rotate', 'brightness', 'contrast', 'compression']
    patch_sizes = [(0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3)]
    
    for i in range(num_samples):
        # Use different seeds for source and donor
        seed_src = 2000 + i * 13
        seed_donor = 3000 + i * 19
        
        # Create source and donor images
        source = create_synthetic_image(width=800, height=600, seed=seed_src)
        donor = create_synthetic_image(width=800, height=600, seed=seed_donor)
        
        # Random patch size
        patch_size = random.choice(patch_sizes)
        
        # Create forgery
        forged = create_patch_forgery(source, donor, patch_size_ratio=patch_size)
        
        # Apply random augmentation to make it more realistic
        aug_type = random.choice(augmentation_types)
        forged_aug = apply_augmentations(forged, aug_type)
        
        # Save
        filename = f"forged_{i:04d}_src{seed_src}_donor{seed_donor}_aug{aug_type}.jpg"
        filepath = FORGED_DIR / filename
        forged_aug.save(filepath, quality=95)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_samples} forged samples...")
    
    print(f"✓ Completed {num_samples} forged samples")

def main():
    print("="*80)
    print("DOCUMENT FORGERY DATASET GENERATION")
    print("="*80)
    
    # Create directories
    create_directories()
    
    # Generate samples
    num_clean = 200
    num_forged = 200
    
    generate_clean_samples(num_clean)
    generate_forged_samples(num_forged)
    
    # Summary
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    
    clean_count = len(list(CLEAN_DIR.glob("*.jpg")))
    forged_count = len(list(FORGED_DIR.glob("*.jpg")))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total clean samples:  {clean_count}")
    print(f"  Total forged samples: {forged_count}")
    print(f"  Total samples:        {clean_count + forged_count}")
    print(f"  Class balance:        {clean_count/(clean_count+forged_count):.1%} clean / {forged_count/(clean_count+forged_count):.1%} forged")
    
    print(f"\n📁 Folder Structure:")
    print(f"  {DATASET_ROOT}/")
    print(f"    ├── clean/     ({clean_count} files)")
    print(f"    └── forged/    ({forged_count} files)")
    
    print("\n✅ Dataset ready for training!")

if __name__ == "__main__":
    main()
