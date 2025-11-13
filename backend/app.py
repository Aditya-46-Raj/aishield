# backend/app.py (patch)
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from models import analyze_document_ela, analyze_video_liveness
import uvicorn
from pathlib import Path
import os

app = FastAPI()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve outputs folder at /outputs
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

@app.post("/analyze/document")
async def analyze_document(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    out_path = OUTPUT_DIR / file.filename
    out_path.write_bytes(content)
    score, heatmap_path, explanation = analyze_document_ela(str(out_path))

    # Build a URL that points to the static file served by FastAPI
    filename = os.path.basename(heatmap_path)
    heatmap_url = str(request.base_url) + f"outputs/{filename}"

    return JSONResponse({
        "filename": file.filename,
        "score": float(score),
        "heatmap": heatmap_url,
        "explanation": explanation
    })

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    content = await file.read()
    out_path = OUTPUT_DIR / file.filename
    out_path.write_bytes(content)
    result = analyze_video_liveness(str(out_path))
    return JSONResponse(result)


# new endpoint in backend/app.py (below other endpoints)
import joblib
from models import extract_face_tensor, cosine_similarity

# load model if exists, else None
FUSION_MODEL_PATH = "models/fusion_lgb.joblib"
try:
    fusion_model = joblib.load(FUSION_MODEL_PATH)
except Exception:
    fusion_model = None

@app.post("/analyze/final")
async def analyze_final(request: Request, id_file: UploadFile = File(...), selfie_file: UploadFile = File(...), video_file: UploadFile = File(None)):
    # save inputs
    id_bytes = await id_file.read()
    selfie_bytes = await selfie_file.read()
    id_path = OUTPUT_DIR / id_file.filename.replace(" ", "_")
    selfie_path = OUTPUT_DIR / selfie_file.filename.replace(" ", "_")
    id_path.write_bytes(id_bytes)
    selfie_path.write_bytes(selfie_bytes)

    # 1) doc ELA
    doc_score, heatmap_path, doc_expl = analyze_document_ela(str(id_path))

    # 2) video liveness (if provided)
    if video_file:
        video_bytes = await video_file.read()
        video_path = OUTPUT_DIR / video_file.filename.replace(" ", "_")
        video_path.write_bytes(video_bytes)
        vid_result = analyze_video_liveness(str(video_path))
        live_score = vid_result.get("score", 0.5)
        live_expl = vid_result.get("explanation", "")
    else:
        # fallback: treat selfie short sequence as live (or compute micro-motion later)
        live_score = 0.2
        live_expl = "No video provided; heuristic live_score applied."

    # 3) embedding similarity
    emb_id = extract_face_tensor(id_bytes)
    emb_self = extract_face_tensor(selfie_bytes)
    sim = cosine_similarity(emb_id, emb_self)
    emb_expl = f"Embedding similarity: {sim:.3f}"

    # 4) behavior anomaly heuristic (for demo we use 0)
    behavior = 0

    # 5) fusion
    features = [[doc_score, live_score, sim, behavior]]
    if fusion_model is not None:
        fused_prob = float(fusion_model.predict_proba(features)[0][1])
        # SHAP explanation (small)
        try:
            import shap
            explainer = shap.TreeExplainer(fusion_model)
            shap_vals = explainer.shap_values(features)
            # convert to human readable pairs
            feature_names = ['doc_score','liveness_score','embed_sim','behavior_anomaly']
            explanation = {feature_names[i]: float(shap_vals[1][0][i]) for i in range(len(feature_names))}
        except Exception as e:
            explanation = {"msg":"SHAP failed: "+str(e)}
    else:
        # weighted heuristic fallback
        fused_prob = min(1.0, 0.5*doc_score + 0.3*(1-sim) + 0.4*behavior + 0.2*(1-live_score))
        explanation = {"heuristic":"weights applied: doc*0.5 + (1-embed)*0.3 + behavior*0.4 + (1-live)*0.2"}

    # build response, expose heatmap URL
    heatmap_url = str(request.base_url) + f"outputs/{heatmap_path.split('/')[-1]}" if heatmap_path else None

    return JSONResponse({
        "doc_score": doc_score,
        "doc_explanation": doc_expl,
        "heatmap": heatmap_url,
        "liveness_score": live_score,
        "liveness_explanation": live_expl,
        "embed_sim": sim,
        "embed_explanation": emb_expl,
        "behavior_anomaly": behavior,
        "fused_fraud_prob": fused_prob,
        "fusion_explanation": explanation
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
