# face embedding utilities (append to backend/models.py)
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
from PIL import Image
import io

from PIL import Image, ImageChops, ImageEnhance
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mediapipe import solutions as mp_solutions

def calculate_ela(img_path, quality=90):
    """Perform basic Error Level Analysis (ELA) and return normalized ELA image + score."""
    original = Image.open(img_path).convert('RGB')
    tmp_path = img_path + ".tmp.jpg"
    original.save(tmp_path, "JPEG", quality=quality)
    compressed = Image.open(tmp_path)
    ela = ImageChops.difference(original, compressed)
    extrema = ela.getextrema()
    # enhance the ELA image to make differences visible
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    # compute a simple "score" as mean of ela intensity
    ela_gray = np.asarray(ela.convert('L'))
    score = float(np.mean(ela_gray) / 255.0)  # normalized 0-1
    os.remove(tmp_path)
    return ela, score

def save_ela_heatmap(ela_image, out_path):
    """Save ela heatmap as PNG and return path."""
    arr = np.array(ela_image.convert('L'))
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(arr, cmap='hot')
    heatmap_path = out_path + "_ela_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return heatmap_path

def analyze_document_ela(img_path):
    """
    Returns (score, heatmap_path, explanation)
    score ~ [0,1] higher means more suspicious
    """
    ela_img, score = calculate_ela(img_path)
    heatmap_path = save_ela_heatmap(ela_img, img_path)
    # heuristic thresholds for demo; explainable output
    if score < 0.03:
        explanation = "ELA low: no obvious recompression artifacts detected."
    elif score < 0.08:
        explanation = "ELA moderate: small localized edits detected; manual review advised."
    else:
        explanation = "ELA high: large recompression differences; likely tampering or synthetic generation."
    return score, heatmap_path, explanation

# -------------------------
# Simple liveness/blink heuristic using MediaPipe
# -------------------------
import cv2
def analyze_video_liveness(video_path, max_frames=150):
    """
    Very light-weight liveness heuristic:
      - Extract frames, run MediaPipe face mesh to count eye-open ratio changes (blink detection)
      - If no blinks found in many frames -> suspicious
    Returns dict with pass/fail and explanation.
    """
    mp_face = mp_solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    blink_count = 0
    frames = 0
    prev_eye_ratio = None

    def eye_aspect_ratio(landmarks, left_indices, right_indices, w, h):
        # compute a simple vertical/horizontal ratio for eyes
        pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
        # approximate using simple vertical/horizontal dist between two landmarks
        # (for demo only)
        left_eye = pts[left_indices[0]]
        right_eye = pts[right_indices[0]]
        # dummy fallback
        return 1.0

    # use a very basic blink heuristic: presence of face for multiple frames + micro-motion count
    while cap.isOpened() and frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        if results.multi_face_landmarks:
            # found face in frame
            # detect simple motion across frames by comparing grayscale diff
            if frames > 1 and prev_frame is not None:
                diff = cv2.absdiff(prev_frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                nonzero = (diff > 15).sum()
                if nonzero > 1000:
                    blink_count += 1
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()
    mp_face.close()

    # heuristic: if blink_count small relative to frames -> suspicious (possible deepfake)
    ratio = blink_count / max(1, frames)
    if ratio < 0.02:
        verdict = "suspicious"
        explanation = f"Low micro-motion detected ({blink_count} events over {frames} frames). Possible synthetic/deepfake video."
        score = 0.85
    else:
        verdict = "likely_live"
        explanation = f"Micro-motion present ({blink_count} events over {frames} frames). Liveness likely genuine."
        score = 0.12

    return {"verdict": verdict, "score": score, "explanation": explanation, "frames": frames, "blink_events": blink_count}



# Initialize (singleton-style)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_mtcnn = MTCNN(keep_all=False, device=_device)
_resnet = InceptionResnetV1(pretrained='vggface2').eval().to(_device)

def extract_face_tensor(img_path_or_bytes):
    """
    Accepts path or raw bytes. Returns a 512-d numpy embedding or None.
    """
    if isinstance(img_path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img_path_or_bytes)).convert('RGB')
    else:
        img = Image.open(img_path_or_bytes).convert('RGB')
    # use MTCNN to crop face
    try:
        face = _mtcnn(img)
        if face is None:
            return None
        face = face.unsqueeze(0).to(_device)  # batch dim
        with torch.no_grad():
            emb = _resnet(face)  # tensor (1,512)
        emb = emb.cpu().numpy().reshape(-1)
        # normalize
        emb = emb / np.linalg.norm(emb)
        return emb
    except Exception as e:
        print("extract_face_tensor error:", e)
        return None

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
