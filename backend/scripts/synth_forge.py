# backend/scripts/synth_forge.py
from PIL import Image
import numpy as np
import random
import os

def create_synthetic_image(width=800, height=600, seed=42):
    """Create a synthetic image with random patterns"""
    np.random.seed(seed)
    # Create a random RGB image
    img_array = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Add some gradient for realism
    for i in range(height):
        img_array[i, :, :] += int((i / height) * 50)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array, 'RGB')

def create_patch_forgery(src_path, donor_path, out_path, blur_radius=8, quality=85):
    src = Image.open(src_path).convert('RGB')
    donor = Image.open(donor_path).convert('RGB').resize(src.size)
    w, h = src.size

    rw, rh = w//4, h//6
    x = random.randint(0, w - rw)
    y = random.randint(0, h - rh)
    patch = donor.crop((x, y, x+rw, y+rh))

    # optional: rotate and slightly scale patch
    angle = random.uniform(-8, 8)
    patch = patch.rotate(angle, expand=True).resize((rw, rh))

    # color match: adjust patch brightness to match source region mean
    region = src.crop((x, y, x+rw, y+rh)).convert('L')
    patch_gray = patch.convert('L')
    src_mean = np.array(region).mean()
    patch_mean = np.array(patch_gray).mean() + 1e-6
    factor = src_mean / patch_mean
    patch = ImageEnhance.Brightness(patch).enhance(factor)

    # create soft mask for blending
    mask = Image.new('L', patch.size, 255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # paste with mask (soft blend)
    src.paste(patch, (x, y), mask)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # save as jpeg to add compression artifacts (simulate real uploads)
    src.save(out_path, 'JPEG', quality=quality)
    print(f"✓ Saved forged image: {out_path}")

def demo_synthetic_forgery():
    """Demo: Create synthetic images and forge one"""
    print("Creating synthetic sample images...")
    
    # Create directories
    os.makedirs("samples", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Generate synthetic images
    clean_img = create_synthetic_image(width=800, height=600, seed=42)
    donor_img = create_synthetic_image(width=800, height=600, seed=99)
    
    # Save synthetic samples
    clean_path = "samples/clean_id.jpg"
    donor_path = "samples/donor.jpg"
    clean_img.save(clean_path)
    donor_img.save(donor_path)
    print(f"✓ Created synthetic clean image: {clean_path}")
    print(f"✓ Created synthetic donor image: {donor_path}")
    
    # Create forgery
    print("\nCreating patch forgery...")
    create_patch_forgery(clean_path, donor_path, "outputs/forged_demo.jpg")
    print("\n✓ Forgery demo complete!")

if __name__ == "__main__":
    demo_synthetic_forgery()
