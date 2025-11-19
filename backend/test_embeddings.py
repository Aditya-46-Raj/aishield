# backend/test_embeddings.py
import numpy as np
from pathlib import Path
import os
import sys

# ensure current dir is backend (not strictly necessary if you `cd backend` before running)
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# import local models.py (same directory)
from models import extract_face_tensor, cosine_similarity

def read_bytes(p): return Path(p).read_bytes()

# USE local samples/ paths (samples folder is INSIDE backend/)
id_path = "samples/clean_id.jpg"
selfie_path = "samples/donor.jpg"   # change to selfie.jpg if your file is named selfie.jpg
# if you want to try other images in outputs/ use: "outputs/forged_demo.jpg"

for a, b in [(id_path, selfie_path), ("outputs/forged_demo.jpg", selfie_path)]:
    print("="*40)
    print("Testing:", a, "vs", b)
    # defensive: ensure files exist before calling embedding
    if not Path(a).exists():
        print("  ERROR: file not found:", a)
        continue
    if not Path(b).exists():
        print("  ERROR: file not found:", b)
        continue

    id_emb = extract_face_tensor(a)
    selfie_emb = extract_face_tensor(b)

    print("  id_emb shape", id_emb.shape, "norm", np.linalg.norm(id_emb))
    print("  selfie_emb shape", selfie_emb.shape, "norm", np.linalg.norm(selfie_emb))
    print("  cosine sim:", cosine_similarity(id_emb, selfie_emb))
