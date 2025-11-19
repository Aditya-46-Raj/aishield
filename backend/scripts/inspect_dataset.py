"""
Visualize sample images from the dataset
"""
from pathlib import Path
from PIL import Image
import random

DATASET_ROOT = Path("dataset")
CLEAN_DIR = DATASET_ROOT / "clean"
FORGED_DIR = DATASET_ROOT / "forged"

print("="*80)
print("DATASET SAMPLE INSPECTION")
print("="*80)

# Get random samples
clean_samples = list(CLEAN_DIR.glob("*.jpg"))
forged_samples = list(FORGED_DIR.glob("*.jpg"))

print(f"\n📊 Dataset Overview:")
print(f"  Clean samples:  {len(clean_samples)}")
print(f"  Forged samples: {len(forged_samples)}")

# Check a few samples
print(f"\n🔍 Sample File Analysis:")

for label, samples in [("CLEAN", clean_samples[:3]), ("FORGED", forged_samples[:3])]:
    print(f"\n{label} Samples:")
    for sample_path in samples:
        img = Image.open(sample_path)
        print(f"  {sample_path.name}")
        print(f"    Size: {img.size[0]}x{img.size[1]} pixels")
        print(f"    Mode: {img.mode}")
        print(f"    File size: {sample_path.stat().st_size / 1024:.1f} KB")

print(f"\n✅ Dataset is ready for Phase 3.2 (CNN/ViT training)!")
