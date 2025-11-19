from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import time
import joblib
import logging
import re
import uvicorn
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import shap

# local model utilities
from models import (
    analyze_document_ela, 
    analyze_video_liveness, 
    analyze_video_liveness_v2
)
# Try to import the helper; if not present we'll fall back later
try:
    from models import integrate_embeddings_and_save, extract_face_tensor, cosine_similarity
    _HAS_INTEGRATE_HELPER = True
except Exception:
    # attempt to import fallback pieces
    try:
        from models import extract_face_tensor, cosine_similarity
    except Exception:
        extract_face_tensor = None
        cosine_similarity = None
    _HAS_INTEGRATE_HELPER = False

# Import audit logger
from audit_logger import log_verification

# Import compliance configuration
from compliance_config import (
    COMPLIANCE_CONFIG,
    is_file_deletion_enabled,
    is_anonymization_enabled,
    generate_anonymous_filename,
    get_compliance_summary
)

logger = logging.getLogger("aishield")
logging_basic = logging.getLogger()
logging_basic.setLevel(logging.INFO)

app = FastAPI()

# Ensure outputs dir exists and serve it
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Utility: sanitize filenames (used when writing uploads)
def _sanitize_filename(name: str) -> str:
    name = name or "file"
    name = Path(name).name  # drop directories
    name = re.sub(r"[^\w\-.]", "_", name)
    return name

# --------------------------
# Load Document Classifier at Startup
# --------------------------
DOC_CLASSIFIER_PATH = BASE_DIR / "models" / "doc_classifier.pt"
doc_classifier_model = None
doc_transform = None

def load_document_classifier():
    """Load the trained EfficientNet-B0 document forgery classifier"""
    global doc_classifier_model, doc_transform
    
    if not DOC_CLASSIFIER_PATH.exists():
        logger.warning(f"Document classifier not found at {DOC_CLASSIFIER_PATH}")
        return
    
    try:
        # Create model architecture (must match training)
        model = models.efficientnet_b0(pretrained=False)
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
        
        # Load trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(DOC_CLASSIFIER_PATH, map_location=device)
        
        # Handle both dict format (with metadata) and direct state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        
        # Define preprocessing transform (must match training)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        doc_classifier_model = model
        doc_transform = transform
        logger.info(f"✓ Document classifier loaded from {DOC_CLASSIFIER_PATH}")
        logger.info(f"  Device: {device}")
        
    except Exception as e:
        logger.error(f"Failed to load document classifier: {e}")

def predict_document_forgery(image_path: str) -> float:
    """Run classifier inference on an image
    Returns: fraud probability [0,1] where 1 = likely forged
    """
    if doc_classifier_model is None or doc_transform is None:
        logger.warning("Document classifier not loaded, returning default score")
        return 0.5  # neutral score
    
    try:
        device = next(doc_classifier_model.parameters()).device
        img = Image.open(image_path).convert('RGB')
        img_tensor = doc_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = doc_classifier_model(img_tensor)
            prob = float(output.item())
        
        return prob
        
    except Exception as e:
        logger.error(f"Error during classifier inference: {e}")
        return 0.5  # neutral fallback

def _save_upload_bytes(content: bytes, filename_hint: str) -> str:
    """Save bytes into outputs/ and return relative path (from backend)."""
    sanitized = _sanitize_filename(filename_hint)
    timestamp = int(time.time() * 1000)
    # keep extension if present
    ext = ".jpg"
    if "." in sanitized:
        ext = "." + sanitized.split(".")[-1]
        sanitized = ".".join(sanitized.split(".")[:-1]) or sanitized.replace(ext, "")
    out_name = f"{sanitized}_{timestamp}{ext}"
    out_path = OUTPUT_DIR / out_name
    out_path.write_bytes(content)
    # return path relative to backend (so returned URL can use /outputs/<name>)
    return str(out_path.name)

def _save_upload_with_compliance(content: bytes, filename_hint: str, file_type: str, case_id: str = None) -> tuple:
    """
    Save uploaded file with compliance features (anonymization, tracking)
    
    Args:
        content: File bytes
        filename_hint: Original filename
        file_type: Type of file ("id_document", "selfie", "video")
        case_id: Case identifier for anonymization
        
    Returns:
        Tuple of (saved_path, original_path, should_delete)
    """
    # Extract extension
    ext = ".jpg"
    if "." in filename_hint:
        ext = "." + filename_hint.split(".")[-1].lower()
    
    # Generate anonymized filename if enabled
    if is_anonymization_enabled() and case_id:
        out_name = generate_anonymous_filename(file_type, case_id, ext)
    else:
        # Fall back to timestamped name
        sanitized = _sanitize_filename(filename_hint)
        timestamp = int(time.time() * 1000)
        if "." in sanitized:
            sanitized = ".".join(sanitized.split(".")[:-1])
        out_name = f"{sanitized}_{timestamp}{ext}"
    
    # Save file
    out_path = OUTPUT_DIR / out_name
    out_path.write_bytes(content)
    
    # Return path and cleanup flag
    should_delete = is_file_deletion_enabled()
    return str(out_path), str(out_path.name), should_delete

def _cleanup_files(file_paths: list):
    """
    Clean up temporary files securely
    
    Args:
        file_paths: List of file paths to delete
    """
    if not is_file_deletion_enabled():
        return
    
    for file_path in file_paths:
        try:
            path_obj = Path(file_path)
            if path_obj.exists():
                # Secure deletion: overwrite before deleting
                if COMPLIANCE_CONFIG.get("secure_file_deletion", False):
                    # Overwrite file with random data
                    file_size = path_obj.stat().st_size
                    with open(path_obj, 'wb') as f:
                        f.write(os.urandom(file_size))
                
                # Delete file
                path_obj.unlink()
                logger.info(f"🗑️ Cleaned up: {path_obj.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")

# Simple health / index
@app.get("/")
def index():
    return {
        "status": "AIShield backend",
        "outputs_dir": str(OUTPUT_DIR),
        "compliance": get_compliance_summary()
    }

@app.post("/analyze/document")
async def analyze_document(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    saved_name = _save_upload_bytes(content, file.filename)
    saved_path = OUTPUT_DIR / saved_name
    
    # Get ELA score
    ela_score, heatmap_path, ela_explanation = analyze_document_ela(str(saved_path))
    
    # Get CNN classifier probability
    clf_prob = predict_document_forgery(str(saved_path))
    
    # Combine scores: 40% ELA + 60% CNN classifier
    doc_score = 0.4 * ela_score + 0.6 * clf_prob
    
    # Build comprehensive explanation
    doc_explanation = f"ELA Analysis: {ela_explanation}\n"
    doc_explanation += f"CNN Classifier: {clf_prob:.1%} forgery probability.\n"
    
    if doc_score < 0.3:
        doc_explanation += "Overall Assessment: Document appears authentic with low tampering risk."
        risk_level = "LOW"
        verdict = "PASS"
        risk_label = "CLEAN"
    elif doc_score < 0.6:
        doc_explanation += "Overall Assessment: Moderate suspicion detected; manual review recommended."
        risk_level = "MEDIUM"
        verdict = "REVIEW"
        risk_label = "MODERATE"
    else:
        doc_explanation += "Overall Assessment: High forgery risk detected; likely tampered or synthetic."
        risk_level = "HIGH"
        verdict = "FAIL"
        risk_label = "HIGH_RISK"
    
    # Build a URL pointing to the static files
    heatmap_name = Path(heatmap_path).name if heatmap_path else None
    heatmap_url = str(request.base_url) + f"outputs/{heatmap_name}" if heatmap_name else None

    # Log to audit trail
    try:
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", None)
        
        case_data = {
            "doc_score": float(doc_score),
            "deepfake_score": 0.0,
            "liveness_score": 0.0,
            "embed_sim": 0.0,
            "fused_fraud_prob": float(doc_score),
            "risk_label": risk_label,
            "verdict": verdict,
            "confidence": 0.85,
            "heatmap_path": heatmap_path,
            "document_path": str(saved_path),
            "video_path": None,
            "selfie_path": None,
            "ip_address": client_host,
            "user_agent": user_agent,
        }
        log_path = log_verification(case_data)
        if log_path:
            logger.info(f"✅ Document analysis logged: {log_path}")
    except Exception as log_err:
        logger.error(f"Failed to log document analysis: {log_err}")

    return JSONResponse({
        "filename": file.filename,
        "ela_score": float(ela_score),
        "clf_prob": float(clf_prob),
        "doc_score": float(doc_score),
        "risk_level": risk_level,
        "heatmap": heatmap_url,
        "explanation": doc_explanation,
        "saved_path": f"outputs/{saved_name}"
    })

@app.post("/analyze/video")
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    use_enhanced: bool = True  # Use v2 by default
):
    """
    Analyze video for liveness detection.
    
    Args:
        file: Video file upload
        use_enhanced: If True, use v2 with deepfake detection (default). 
                     If False, use basic v1 detection.
    """
    content = await file.read()
    saved_name = _save_upload_bytes(content, file.filename)
    saved_path = OUTPUT_DIR / saved_name
    
    if use_enhanced:
        result = analyze_video_liveness_v2(str(saved_path))
    else:
        result = analyze_video_liveness(str(saved_path))
    
    # Log to audit trail
    try:
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", None)
        
        liveness_score = result.get("score", 0.5)
        components = result.get("components", {})
        deepfake_score = components.get("deepfake_score", 0.0)
        
        # Determine verdict based on liveness score
        if liveness_score > 0.7:
            verdict = "PASS"
            risk_label = "CLEAN"
        elif liveness_score > 0.4:
            verdict = "REVIEW"
            risk_label = "MODERATE"
        else:
            verdict = "FAIL"
            risk_label = "HIGH_RISK"
        
        case_data = {
            "doc_score": 0.0,
            "deepfake_score": float(deepfake_score),
            "liveness_score": float(liveness_score),
            "embed_sim": 0.0,
            "fused_fraud_prob": float(1.0 - liveness_score),
            "risk_label": risk_label,
            "verdict": verdict,
            "confidence": 0.80,
            "heatmap_path": None,
            "document_path": None,
            "video_path": str(saved_path),
            "selfie_path": None,
            "ip_address": client_host,
            "user_agent": user_agent,
        }
        log_path = log_verification(case_data)
        if log_path:
            logger.info(f"✅ Video liveness logged: {log_path}")
    except Exception as log_err:
        logger.error(f"Failed to log video liveness: {log_err}")
    
    return JSONResponse(result)


# Load fusion model if present
FUSION_MODEL_PATH = BASE_DIR / "models" / "fusion_lgb.joblib"
fusion_model = None
shap_explainer = None

# Load models at startup
@app.on_event("startup")
async def startup_event():
    """Load ML models at application startup"""
    global fusion_model, shap_explainer
    
    # Load fusion model
    try:
        fusion_model = joblib.load(FUSION_MODEL_PATH)
        logger.info("✓ Loaded fusion model from %s", FUSION_MODEL_PATH)
        
        # Initialize SHAP explainer for the model
        try:
            shap_explainer = shap.TreeExplainer(fusion_model)
            logger.info("✓ SHAP TreeExplainer initialized successfully")
        except Exception as shap_err:
            logger.warning("SHAP explainer initialization failed: %s", shap_err)
            shap_explainer = None
            
    except Exception as e:
        fusion_model = None
        logger.info("No fusion model loaded: %s", str(e))
    
    # Load document classifier
    load_document_classifier()

@app.post("/analyze/final")
async def analyze_final(request: Request,
                        id_file: UploadFile = File(...),
                        selfie_file: UploadFile = File(...),
                        video_file: UploadFile = File(None)):
    """
    Final analysis endpoint with compliance features:
      - Saves files with anonymized names (if enabled)
      - Runs ELA on the ID doc
      - Optionally runs video liveness if provided
      - Computes embedding similarity
      - Runs fusion (model if available, else heuristic)
      - Logs to audit trail
      - Cleans up raw uploads (if enabled)
    Returns JSON with embed_sim, fused_fraud_prob, heatmap_url, saved image paths, notes
    """
    
    # Track files for cleanup
    files_to_cleanup = []
    
    try:
        # Generate case ID first for anonymization
        from audit_logger import _get_next_case_number
        case_number = _get_next_case_number()
        case_id = f"case_{case_number:04d}"

        # 1) Save uploaded files with compliance (anonymized names if enabled)
        id_bytes = await id_file.read()
        selfie_bytes = await selfie_file.read()
        
        # Use compliance-aware file saving
        id_saved_full_path, id_saved_name, should_delete_id = _save_upload_with_compliance(
            id_bytes, id_file.filename, "id_document", case_id
        )
        selfie_saved_full_path, selfie_saved_name, should_delete_selfie = _save_upload_with_compliance(
            selfie_bytes, selfie_file.filename, "selfie", case_id
        )
        
        id_saved_path = Path(id_saved_full_path)
        selfie_saved_path = Path(selfie_saved_full_path)
        
        # Track for cleanup if configured
        if should_delete_id:
            files_to_cleanup.append(id_saved_full_path)
        if should_delete_selfie:
            files_to_cleanup.append(selfie_saved_full_path)

        notes = []

        # 2) Document Analysis - ELA + CNN Classifier
        try:
            ela_score, heatmap_path, ela_expl = analyze_document_ela(str(id_saved_path))
            clf_prob = predict_document_forgery(str(id_saved_path))
            
            # Combined document score: 40% ELA + 60% CNN
            doc_score = 0.4 * ela_score + 0.6 * clf_prob
            
            # Build comprehensive explanation
            doc_expl = f"ELA: {ela_expl}\\nCNN Classifier: {clf_prob:.1%} forgery probability.\\n"
            if doc_score < 0.3:
                doc_expl += "Overall: Document appears authentic."
            elif doc_score < 0.6:
                doc_expl += "Overall: Moderate suspicion detected."
            else:
                doc_expl += "Overall: High forgery risk detected."
                
        except Exception as e:
            logger.exception("Document analysis failed: %s", e)
            ela_score = 0.0
            clf_prob = 0.5
            doc_score = 0.0
            heatmap_path = None
            doc_expl = f"document_analysis_failed: {e}"
            notes.append("doc_analysis_failed")

        # build heatmap URL
        heatmap_url = None
        if heatmap_path:
            heatmap_url = str(request.base_url) + f"outputs/{Path(heatmap_path).name}"

        # 3) Video liveness - Use enhanced v2 with deepfake detection
        video_saved_path = None
        video_saved_full_path = None
        if video_file:
            try:
                video_bytes = await video_file.read()
                
                # Use compliance-aware saving for video
                video_saved_full_path, video_saved_name, should_delete_video = _save_upload_with_compliance(
                    video_bytes, video_file.filename, "video", case_id
                )
                video_saved_path = Path(video_saved_full_path)
                
                # Track for cleanup if configured
                if should_delete_video:
                    files_to_cleanup.append(video_saved_full_path)
                
                # Use enhanced liveness v2 with deepfake detection
                vid_result = analyze_video_liveness_v2(str(video_saved_path))
                live_score = float(vid_result.get("score", 0.5))
                live_expl = vid_result.get("explanation", "")
                
                # Extract component scores if available
                components = vid_result.get("components", {})
                deepfake_score = components.get("deepfake_score", 0.0)
                motion_score = components.get("motion_score", 0.0)
                blink_score = components.get("blink_score", 0.0)
                
            except Exception as e:
                logger.exception("Video liveness failed: %s", e)
                live_score = 0.5
                live_expl = f"video_liveness_failed: {e}"
                deepfake_score = 0.0
                motion_score = 0.0
                blink_score = 0.0
                notes.append("video_liveness_failed")
        else:
            live_score = 0.2
            live_expl = "No video provided; heuristic applied."
            deepfake_score = 0.0
            motion_score = 0.0
            blink_score = 0.0

        # 4) Embedding similarity & saving (use integrate helper if available)
        embed_sim = 0.0
        embed_notes = ""
        id_image_rel = f"outputs/{id_saved_name}"
        selfie_image_rel = f"outputs/{selfie_saved_name}"

        try:
            if _HAS_INTEGRATE_HELPER:
                # integrate_embeddings_and_save expects bytes and filename hints per design earlier
                res = integrate_embeddings_and_save(id_bytes, selfie_bytes, id_saved_name, selfie_saved_name)
                embed_sim = float(res.get("embed_sim", 0.0))
                # The helper saves files; but we already saved the uploads — unify paths
                id_image_rel = res.get("id_saved") or id_image_rel
                selfie_image_rel = res.get("selfie_saved") or selfie_image_rel
                if res.get("notes"):
                    embed_notes = res.get("notes")
                    notes.append(embed_notes)
            else:
                # fallback: directly call extract_face_tensor on bytes or path
                if extract_face_tensor is None or cosine_similarity is None:
                    logger.warning("Embedding functions not available in models.py")
                    notes.append("embedding_functions_missing")
                    embed_sim = 0.0
                else:
                    # extract_face_tensor accepts bytes or file path; try bytes first
                    emb_id = extract_face_tensor(id_bytes)
                    emb_self = extract_face_tensor(selfie_bytes)
                    embed_sim = float(cosine_similarity(emb_id, emb_self))
                    # note missing faces
                    import numpy as _np
                    if _np.linalg.norm(emb_id) == 0:
                        notes.append("id_face_not_detected")
                    if _np.linalg.norm(emb_self) == 0:
                        notes.append("selfie_face_not_detected")
        except Exception as e:
            logger.exception("Embedding integration failed: %s", e)
            notes.append("embedding_integration_failed")

        # 5) Behavior anomaly (demo placeholder)
        behavior = 0.0

        # 6) Fusion (model or heuristic)
        fused_prob = 0.0
        fusion_explanation = {}

        try:
            # Structure features in exact order: [doc_score, liveness_score, embed_sim, behavior_anomaly]
            features = [[float(doc_score), float(live_score), float(embed_sim), float(behavior)]]
            feature_names = ["doc_score", "liveness_score", "embed_sim", "behavior_anomaly"]
            
            # Log input features for debugging
            logger.info(f"Fusion input features: doc={doc_score:.4f}, live={live_score:.4f}, embed={embed_sim:.4f}, behavior={behavior:.4f}")
            
            if fusion_model is not None:
                try:
                    # Measure prediction time
                    start_time = time.time()
                    
                    # LightGBM Booster uses .predict() which returns probabilities for binary classification
                    pred = fusion_model.predict(features)
                    # pred is array-like, extract single probability value
                    if hasattr(pred, "__len__"):
                        fused_prob = float(pred[0])
                    else:
                        fused_prob = float(pred)
                    
                    # Compute SHAP values if explainer available
                    shap_dict = {}
                    if shap_explainer is not None:
                        try:
                            # Convert features to numpy array for SHAP
                            features_np = np.array(features)
                            shap_values = shap_explainer.shap_values(features_np)
                            
                            # shap_values is a numpy array of shape (1, num_features)
                            # Extract first row
                            shap_array = shap_values[0]
                            
                            # Convert to dictionary with feature names
                            shap_dict = {feature_names[i]: float(shap_array[i]) for i in range(len(feature_names))}
                            
                            # Get base value (expected value)
                            if hasattr(shap_explainer, 'expected_value'):
                                shap_dict["base_value"] = float(shap_explainer.expected_value)
                            
                        except Exception as shap_err:
                            logger.warning(f"SHAP computation failed: {shap_err}")
                            shap_dict = {"error": str(shap_err)}
                    
                    prediction_time_ms = (time.time() - start_time) * 1000
                    
                    logger.info(f"Fusion model prediction: {fused_prob:.4f} (took {prediction_time_ms:.2f}ms)")
                    fusion_explanation = {
                        "model": "LightGBM",
                        "features": features[0],
                        "shap_values": shap_dict,
                        "runtime_ms": round(prediction_time_ms, 2)
                    }

                except Exception as model_err:
                    logger.exception("Fusion model predict failed: %s", model_err)
                    # fallback to heuristic
                    fused_prob = min(
                        1.0,
                        max(0.0,
                            0.5 * float(doc_score)
                            + 0.35 * (1.0 - float(embed_sim))
                            + 0.15 * (1.0 - float(live_score))
                        )
                    )
                    fusion_explanation = {
                        "heuristic": "model predict failed, fallback heuristic used"
                    }
                    notes.append("fusion_model_predict_failed")

            else:
                # NO fusion model → pure heuristic
                fused_prob = min(
                    1.0,
                    max(0.0,
                        0.5 * float(doc_score)
                        + 0.35 * (1.0 - float(embed_sim))
                        + 0.15 * (1.0 - float(live_score))
                    )
                )
                fusion_explanation = {
                    "heuristic": "model_missing, using weighted heuristic"
                }
                logger.warning("Fusion model not loaded, using heuristic")

        except Exception as e:
            logger.exception("Fusion step failed: %s", e)
            fused_prob = 0.0
            fusion_explanation = {"msg": "fusion_failed:" + str(e)}
            notes.append("fusion_failed")


        # Final JSON response
        resp = {
            "doc_score": float(doc_score),
            "doc_explanation": doc_expl,
            "document_details": {
                "ela_score": float(ela_score),
                "clf_prob": float(clf_prob),
                "combined_score": float(doc_score)
            },
            "heatmap": heatmap_url,
            "liveness_score": float(live_score),
            "liveness_explanation": live_expl,
            "liveness_details": {
                "deepfake_score": float(deepfake_score),
                "motion_score": float(motion_score),
                "blink_score": float(blink_score),
                "combined_score": float(live_score)
            },
            "embed_sim": float(embed_sim),
            "embed_explanation": embed_notes or f"Embedding similarity (cosine): {embed_sim:.3f}",
            "id_image": id_image_rel,
            "selfie_image": selfie_image_rel,
            "behavior_anomaly": float(behavior),
            "fused_fraud_prob": float(fused_prob),
            "fusion_explanation": fusion_explanation,
            "notes": ", ".join(notes) if notes else ""
        }

        # Log verification to audit trail
        try:
            # Determine verdict based on fraud probability
            if fused_prob < 0.3:
                verdict = "PASS"
                risk_label = "CLEAN"
            elif fused_prob < 0.6:
                verdict = "REVIEW"
                risk_label = "MODERATE"
            else:
                verdict = "FAIL"
                risk_label = "HIGH_RISK"
            
            # Extract client info from request
            client_host = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", None)
            
            # Build comprehensive case data for audit log
            case_data = {
                # Required fields for audit logger (top-level for compatibility)
                "doc_score": float(doc_score),
                "deepfake_score": float(deepfake_score),
                "liveness_score": float(live_score),
                "embed_sim": float(embed_sim),
                "fused_fraud_prob": float(fused_prob),
                "risk_label": risk_label,
                "verdict": verdict,
                "confidence": 0.95,
                "heatmap_path": heatmap_path,
                "document_path": str(id_saved_path),
                "video_path": str(video_saved_path) if video_file else None,
                "selfie_path": str(selfie_saved_path),
                "ip_address": client_host,
                "user_agent": user_agent,
                "processing_time_ms": fusion_explanation.get("runtime_ms") if fusion_explanation else None,
                "frames_analyzed": 10 if video_file else 0,
            }
            
            # Log the verification with error handling
            try:
                logger.info(f"📝 Attempting to log verification to audit trail...")
                log_path = log_verification(case_data)
                if log_path:
                    logger.info(f"✅ Verification logged to: {log_path}")
                else:
                    logger.error(f"❌ Failed to create audit log - log_verification returned None")
            except Exception as log_err:
                logger.error(f"❌ Failed to log verification: {log_err}")
                logger.exception(log_err)  # Print full traceback
            
        except Exception as log_err:
            logger.error(f"❌ Audit logging section failed: {log_err}")
            logger.exception(log_err)
        
        # Cleanup raw uploaded files (compliance)
        if files_to_cleanup:
            logger.info(f"🔒 Compliance: Cleaning up {len(files_to_cleanup)} raw upload(s)")
            _cleanup_files(files_to_cleanup)

        return JSONResponse(resp)

    except Exception as e:
        # Handle any errors in the main analysis flow (outer try block from line 369)
        logger.exception(f"Error in final analysis: {e}")
        
        # Still try to cleanup files even on error
        if 'files_to_cleanup' in locals() and files_to_cleanup:
            _cleanup_files(files_to_cleanup)
        
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Analysis failed"}
        )


if __name__ == "__main__":
    # Run uvicorn when executed directly
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
