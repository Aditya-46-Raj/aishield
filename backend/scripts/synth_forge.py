# backend/scripts/synth_forge.py
from PIL import Image
import numpy as np
import random
import os

def create_patch_forgery(src_path, donor_path, out_path):
    src = Image.open(src_path).convert('RGB')
    donor = Image.open(donor_path).convert('RGB').resize(src.size)
    w, h = src.size
    # random rectangle
    rw, rh = w//4, h//6
    x = random.randint(0, w - rw)
    y = random.randint(0, h - rh)
    patch = donor.crop((x, y, x+rw, y+rh))
    src.paste(patch, (x, y))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    src.save(out_path)
    print("Saved forged:", out_path)

if __name__ == "__main__":
    create_patch_forgery("samples/clean_id.jpg", "samples/donor.jpg", "outputs/forged_demo.jpg")
