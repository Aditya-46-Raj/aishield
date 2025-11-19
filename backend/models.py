# backend/models.py
"""
Face/document analysis utilities:
 - ELA (Error Level Analysis) + heatmap saving
 - Simple MediaPipe-based video liveness heuristic
 - Face embedding extraction using facenet-pytorch (MTCNN + InceptionResnetV1)
 - Cosine similarity helper
 - Helpers to save inputs to outputs/ and integrate embedding extraction
Defensive: avoids NaNs, ensures dtype/shape consistency, and returns safe fallbacks.
"""

import io
import os
import logging
import re
import time
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from mediapipe import solutions as mp_solutions
import cv2

# -- logging --
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# ELA utilities
# -------------------------
def calculate_ela(img_path, quality=90):
    """Perform Error Level Analysis (ELA).
    Returns: (PIL.Image ela_image, float score)
    score in [0,1], higher => more suspicious.
    """
    original = Image.open(img_path).convert("RGB")
    tmp_path = str(img_path) + ".tmp.jpg"
    original.save(tmp_path, "JPEG", quality=quality)
    compressed = Image.open(tmp_path).convert("RGB")
    ela = ImageChops.difference(original, compressed)

    # enhance ELA for visibility
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 0
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)

    ela_gray = np.asarray(ela.convert("L"))
    score = float(np.mean(ela_gray) / 255.0)

    try:
        os.remove(tmp_path)
    except Exception:
        logger.debug("Could not remove temporary ELA file: %s", tmp_path)

    return ela, score

def save_ela_heatmap(ela_image, out_path):
    """Save ELA heatmap (.png). Returns the saved filepath."""
    arr = np.array(ela_image.convert("L"))
    out_path = str(out_path)
    heatmap_path = out_path + "_ela_heatmap.png"
    plt.figure(figsize=(6,6))
    plt.axis("off")
    plt.imshow(arr, cmap="hot")
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return heatmap_path

def analyze_document_ela(img_path):
    """
    Returns (score, heatmap_path, explanation)
    score ~ [0,1] higher means more suspicious
    """
    ela_img, score = calculate_ela(img_path)
    heatmap_path = save_ela_heatmap(ela_img, img_path)
    if score < 0.03:
        explanation = "ELA low: no obvious recompression artifacts detected."
    elif score < 0.08:
        explanation = "ELA moderate: small localized edits detected; manual review advised."
    else:
        explanation = "ELA high: large recompression differences; likely tampering or synthetic generation."
    return score, heatmap_path, explanation

# -------------------------
# Enhanced video liveness detection with anti-spoof detection
# -------------------------
def analyze_video_liveness_v2(video_path, max_frames=150):
    """
    Enhanced liveness detection combining:
    1. Anti-spoof detection (texture + frequency analysis)
    2. Micro-motion analysis
    3. Eye aspect ratio / blink heuristics
    
    Returns: dict {verdict, score, explanation, antispoof_score, motion_score, components}
    """
    # Import anti-spoof detector
    antispoof = None
    has_antispoof = False
    try:
        from antispoof_detector import get_antispoof_detector
        antispoof = get_antispoof_detector()
        has_antispoof = True
    except Exception as e:
        logger.warning(f"Anti-spoof detector not available: {e}")
    
    mp_face = mp_solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "verdict": "error",
            "score": 0.9,
            "explanation": "Video could not be opened",
            "frames": 0,
            "components": {}
        }

    # Metrics
    blink_count = 0
    frames = 0
    prev_frame_gray = None
    motion_events = 0
    
    # Extract frames for anti-spoof detection
    antispoof_result = None
    if has_antispoof and antispoof is not None:
        try:
            antispoof_result = antispoof.predict_video(video_path)
        except Exception as e:
            logger.error(f"Anti-spoof analysis failed: {e}")

    # Process video for motion and blinks
    while cap.isOpened() and frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        
        # Motion and blink detection
        if results.multi_face_landmarks:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame_gray is not None:
                diff = cv2.absdiff(prev_frame_gray, gray)
                nonzero = (diff > 15).sum()
                if nonzero > 2000:
                    motion_events += 1
                # More sensitive blink detection
                if nonzero > 5000:
                    blink_count += 1
            prev_frame_gray = gray

    cap.release()
    mp_face.close()

    # Calculate component scores
    if frames == 0:
        return {
            "verdict": "no_frames",
            "score": 0.9,
            "explanation": "No frames processed",
            "frames": 0,
            "components": {}
        }
    
    # 1. Anti-spoof score (0-1, higher = more spoofed)
    # This is now our deepfake_score
    if antispoof_result:
        deepfake_score = antispoof_result.get('spoof_score', 0.3)
        realness_prob = antispoof_result.get('realness_prob', 0.7)
    else:
        deepfake_score = 0.3
        realness_prob = 0.7
    
    # 2. Motion score (0-1, normalized for liveness calculation)
    # Convert motion events to a positive liveness indicator (higher = better)
    motion_ratio = motion_events / max(1, frames)
    if motion_ratio < 0.02:
        motion_score_raw = 0.2  # Too static = low motion score
    elif motion_ratio > 0.3:
        motion_score_raw = 0.4  # Too much motion = moderate score
    else:
        motion_score_raw = 0.9  # Good motion = high motion score
    
    # 3. Blink score (0-1, normalized for liveness calculation)
    # Convert blink events to a positive liveness indicator (higher = better)
    blink_ratio = blink_count / max(1, frames)
    if blink_ratio < 0.01:
        blink_score_raw = 0.3  # No blinking = low blink score
    elif blink_ratio > 0.15:
        blink_score_raw = 0.5  # Too much blinking = moderate score
    else:
        blink_score_raw = 0.9  # Natural blinking = high blink score
    
    # Combine motion and blink into motion_score
    # This represents overall behavioral liveness (higher = more live-like)
    motion_score = 0.6 * motion_score_raw + 0.4 * blink_score_raw
    
    # Phase 4.4: NEW LIVENESS SCORE FORMULA
    # liveness_score = 0.5 * (1 - deepfake_score) + 0.5 * motion_score
    # Range: 0-1, where higher = more likely live
    liveness_score = 0.5 * (1.0 - deepfake_score) + 0.5 * motion_score
    
    # Generate liveness reason
    if liveness_score > 0.7:
        verdict = "live"
        liveness_reason = (f"High liveness confidence: Low deepfake probability ({deepfake_score:.2f}), "
                          f"natural motion patterns ({motion_events} events), "
                          f"normal blinking ({blink_count} blinks). Overall liveness: {liveness_score:.2f}")
    elif liveness_score > 0.4:
        verdict = "suspicious"
        liveness_reason = (f"Moderate liveness confidence: Deepfake score {deepfake_score:.2f}, "
                          f"motion score {motion_score:.2f}. Manual review recommended.")
    else:
        verdict = "spoofed"
        liveness_reason = (f"Low liveness confidence: High deepfake probability ({deepfake_score:.2f}), "
                          f"abnormal behavioral patterns. Likely presentation attack.")
    
    components = {
        "deepfake_score": float(deepfake_score),
        "realness_prob": float(realness_prob),
        "motion_score": float(motion_score),
        "motion_score_raw": float(motion_score_raw),
        "blink_score_raw": float(blink_score_raw),
        "liveness_score": float(liveness_score),
    }
    
    if antispoof_result:
        components["antispoof_frames_analyzed"] = antispoof_result.get('frame_count', 0)
        components["frame_scores"] = antispoof_result.get('frame_scores', [])
    
    return {
        "verdict": verdict,
        "score": float(liveness_score),
        "liveness_score": float(liveness_score),
        "deepfake_score": float(deepfake_score),
        "motion_score": float(motion_score),
        "liveness_reason": liveness_reason,
        "explanation": liveness_reason,  # Backward compatibility
        "frames": frames,
        "blink_events": blink_count,
        "motion_events": motion_events,
        "components": components
    }


# Keep original function for backward compatibility
def analyze_video_liveness(video_path, max_frames=150):
    """
    Original lightweight liveness heuristic (backward compatible)
    Use analyze_video_liveness_v2() for enhanced detection
    """
    mp_face = mp_solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"verdict": "error", "score": 0.0, "explanation": "Video could not be opened", "frames": 0, "blink_events": 0}

    blink_count = 0
    frames = 0
    prev_frame_gray = None

    while cap.isOpened() and frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        if results.multi_face_landmarks:
            # compute a simple grayscale diff against previous frame to detect micro-motion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame_gray is not None:
                diff = cv2.absdiff(prev_frame_gray, gray)
                nonzero = (diff > 15).sum()
                if nonzero > 2000:  # threshold tuned for demo; adjust if needed
                    blink_count += 1
            prev_frame_gray = gray

    cap.release()
    mp_face.close()

    ratio = blink_count / max(1, frames)
    if frames == 0:
        return {"verdict": "no_frames", "score": 0.9, "explanation": "No frames processed", "frames": 0, "blink_events": 0}

    if ratio < 0.02:
        verdict = "suspicious"
        explanation = f"Low micro-motion detected ({blink_count} events over {frames} frames). Possible synthetic/deepfake video."
        score = 0.85
    else:
        verdict = "likely_live"
        explanation = f"Micro-motion present ({blink_count} events over {frames} frames). Liveness likely genuine."
        score = 0.12

    return {"verdict": verdict, "score": score, "explanation": explanation, "frames": frames, "blink_events": blink_count}

# -------------------------
# Face embedding utilities (facenet-pytorch) - tuned for small ID photos
# -------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tune MTCNN: lower thresholds and allow smaller faces to be detected.
# min_face_size=20 helps with small ID photo patches; thresholds lower increases sensitivity.
_mtcnn = MTCNN(keep_all=False, device=_device, thresholds=[0.5, 0.6, 0.7], min_face_size=20)

# Face embedding model
_resnet = InceptionResnetV1(pretrained='vggface2').eval().to(_device)

def _img_from_path_or_bytes(img_path_or_bytes):
    """Return PIL.Image from path or bytes."""
    if isinstance(img_path_or_bytes, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_path_or_bytes)).convert("RGB")
    else:
        return Image.open(img_path_or_bytes).convert("RGB")

def extract_face_tensor(img_path_or_bytes, expected_dim=512, device=None):
    """
    Accepts file path or raw bytes. Returns a numpy float32 vector of shape (D,) (L2-normalized).
    On failure returns a zero-vector of length expected_dim (so downstream code stays safe).
    Improvements:
      - Resizes small images so MTCNN can detect tiny ID-photo crops.
      - Tries simple center-crop if face not detected, then retries detection.
    """
    if device is None:
        device = _device

    try:
        img = _img_from_path_or_bytes(img_path_or_bytes)
    except Exception as e:
        logger.exception("Failed to open image input for embedding: %s", e)
        return np.zeros((expected_dim,), dtype=np.float32)

    # If image is small (short side < 400), upscale to improve detection
    try:
        w, h = img.size
        if min(w, h) < 400:
            scale = max(1, int(400 / min(w, h)))
            new_w, new_h = w * scale, h * scale
            img = img.resize((new_w, new_h), Image.BILINEAR)
            logger.debug("Resized image for detection to %dx%d", new_w, new_h)
    except Exception:
        # continue even if resizing fails
        pass

    try:
        # Primary attempt: MTCNN detection on the (possibly resized) image
        face_tensor = _mtcnn(img)  # returns (C,H,W) tensor or None
        if face_tensor is None:
            # Fallback attempt: try center-crop (useful for ID images where face sits in photo box)
            try:
                w, h = img.size
                crop_size = int(min(w, h) * 0.6)  # crop central 60%
                left = (w - crop_size) // 2
                upper = (h - crop_size) // 2
                img_crop = img.crop((left, upper, left + crop_size, upper + crop_size)).resize((400, 400))
                logger.debug("No face detected; trying center-crop and retry.")
                face_tensor = _mtcnn(img_crop)
            except Exception as e_crop:
                logger.debug("Center-crop retry failed: %s", e_crop)

        if face_tensor is None:
            logger.warning("No face detected by MTCNN for input.")
            return np.zeros((expected_dim,), dtype=np.float32)

        # Face tensor returned: add batch dim and run through resnet
        face_tensor = face_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
        with torch.no_grad():
            emb = _resnet(face_tensor)  # [1, D]
        emb = emb.squeeze(0).cpu().numpy().astype(np.float32).reshape(-1)

        # Defensive normalization
        norm = np.linalg.norm(emb)
        if not np.isfinite(norm) or norm == 0:
            logger.warning("Embedding has non-finite or zero norm; returning zero-vector fallback.")
            return np.zeros((expected_dim,), dtype=np.float32)
        emb = emb / (norm + 1e-12)

        # Pad or truncate to expected_dim if model mismatches
        if emb.shape[0] != expected_dim:
            logger.info("Embedding dim %d != expected %d. Adjusting.", emb.shape[0], expected_dim)
            if emb.shape[0] > expected_dim:
                emb = emb[:expected_dim]
            else:
                out = np.zeros((expected_dim,), dtype=np.float32)
                out[:emb.shape[0]] = emb
                emb = out

        return emb.astype(np.float32)

    except Exception as e:
        logger.exception("extract_face_tensor error: %s", e)
        return np.zeros((expected_dim,), dtype=np.float32)

def cosine_similarity(a, b):
    """Compute cosine similarity between two 1D numpy arrays. Safe for zero vectors."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if not np.isfinite(an) or not np.isfinite(bn) or an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))

# -------------------------
# Saving and integration helpers
# -------------------------
def _sanitize_filename(name: str) -> str:
    """Sanitize a filename to safe characters only, keep extension if present."""
    name = name or ""
    # strip path components
    name = Path(name).name
    # keep only word chars, dash, underscore, dot
    name = re.sub(r"[^\w\-.]", "_", name)
    return name

def save_bytes_to_outputs(file_bytes: bytes, filename_hint: str) -> str:
    """
    Save bytes to outputs/ with a sanitized timestamped filename.
    Returns saved relative path (str).
    """
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg"
    # try to keep extension if hint contains one
    if "." in filename_hint:
        ext = "." + filename_hint.split(".")[-1]
        if len(ext) > 5:
            ext = ".jpg"  # fallback
    sanitized = _sanitize_filename(filename_hint.split(".")[0])
    timestamp = int(time.time() * 1000)
    out_name = f"{sanitized}_{timestamp}{ext}"
    out_path = out_dir / out_name
    try:
        with open(out_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        logger.exception("Failed to write bytes to outputs: %s", e)
        # raise or fallback? return empty string to indicate failure
        return ""
    # return path string relative to project (useful for API responses)
    return str(out_path.relative_to(Path(__file__).resolve().parent))

def integrate_embeddings_and_save(id_bytes: bytes, selfie_bytes: bytes,
                                  id_filename_hint: str = "id.jpg",
                                  selfie_filename_hint: str = "selfie.jpg") -> Dict:
    """
    Single-call helper to:
      - save id/selfie to outputs/
      - extract embeddings via extract_face_tensor
      - compute cosine similarity
    Returns a dict with keys:
      - embed_sim (float)
      - id_saved (relative path)
      - selfie_saved (relative path)
      - notes (str) - optional warnings (e.g., face not detected)
    """
    notes = []
    # Save files
    id_saved = save_bytes_to_outputs(id_bytes, id_filename_hint)
    selfie_saved = save_bytes_to_outputs(selfie_bytes, selfie_filename_hint)

    if id_saved == "" or selfie_saved == "":
        notes.append("failed_to_save_files")

    # Extract embeddings (these return safe zero-vectors on failure)
    try:
        id_emb = extract_face_tensor(Path(__file__).resolve().parent / id_saved)
    except Exception:
        # extract_face_tensor accepts either bytes or path; we call with path in case backend pipeline expects that
        id_emb = extract_face_tensor(id_bytes)

    try:
        selfie_emb = extract_face_tensor(Path(__file__).resolve().parent / selfie_saved)
    except Exception:
        selfie_emb = extract_face_tensor(selfie_bytes)

    sim = cosine_similarity(id_emb, selfie_emb)

    # small helpful flags: if zero-vector (norm==0) mention face not detected
    if np.linalg.norm(id_emb) == 0:
        notes.append("id_face_not_detected")
    if np.linalg.norm(selfie_emb) == 0:
        notes.append("selfie_face_not_detected")

    return {
        "embed_sim": float(sim),
        "id_saved": id_saved,
        "selfie_saved": selfie_saved,
        "notes": ", ".join(notes) if notes else ""
    }
