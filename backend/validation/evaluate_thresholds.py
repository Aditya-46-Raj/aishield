# backend/validation/evaluate_thresholds.py
from pathlib import Path
import sys, os
import csv
import json
import numpy as np
from sklearn import metrics

# ensure backend on path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from models import extract_face_tensor, cosine_similarity, analyze_document_ela

BACKEND = HERE
RESULTS_CSV = BACKEND / "validation_results.csv"
SUGGEST_JSON = BACKEND / "validation_threshold_suggestion.json"

# ----- EDIT THIS: add all labeled pairs you have -----
# Format: (id_rel_path, selfie_rel_path, label) where paths are relative to backend/
PAIRS = [
    ("samples/clean_id.jpg", "samples/donor.jpg", 0),
    ("outputs/forged_demo.jpg", "samples/donor.jpg", 1),
    ("samples/clean_id.jpg", "samples/selfie.jpg", 0),  # Clean ID + real selfie
    ("outputs/forged_demo.jpg", "samples/selfie.jpg", 1),  # Forged ID + real selfie
]
# ----------------------------------------------------

def resolve(p):
    p = Path(p)
    if not p.exists():
        alt = BACKEND / p
        if alt.exists():
            return str(alt)
    return str(p)

def safe_compute(idp, selfp):
    # doc score
    try:
        doc_score, heatmap, expl = analyze_document_ela(resolve(idp))
    except Exception as e:
        doc_score, heatmap, expl = 0.0, "", f"ela_failed:{e}"
    # embeddings
    id_emb = extract_face_tensor(resolve(idp))
    self_emb = extract_face_tensor(resolve(selfp))
    embed_sim = float(cosine_similarity(id_emb, self_emb))
    # heuristic fused (same as API fallback, using live_score=0.2 default)
    fused = min(1.0, max(0.0, 0.5*float(doc_score) + 0.35*(1.0 - float(embed_sim)) + 0.15*(1.0 - 0.2)))
    return {
        "doc_score": float(doc_score),
        "embed_sim": float(embed_sim),
        "fused": float(fused),
        "heatmap": heatmap,
        "doc_expl": expl
    }

def main():
    rows = []
    y_true = []
    embed_scores = []
    fused_scores = []

    for idp, selfp, label in PAIRS:
        idp_r = resolve(idp)
        selfp_r = resolve(selfp)
        if not Path(idp_r).exists():
            print("Missing:", idp_r); continue
        if not Path(selfp_r).exists():
            print("Missing:", selfp_r); continue
        res = safe_compute(idp, selfp)
        rows.append({
            "id": idp,
            "selfie": selfp,
            "label": label,
            "doc_score": res["doc_score"],
            "embed_sim": res["embed_sim"],
            "fused": res["fused"],
            "heatmap": res["heatmap"],
            "doc_expl": res["doc_expl"]
        })
        y_true.append(label)
        embed_scores.append(1.0 - res["embed_sim"])   # invert: higher -> more likely forged
        fused_scores.append(res["fused"])

    # write CSV
    keys = ["id","selfie","label","doc_score","embed_sim","fused","heatmap","doc_expl"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if len(y_true) < 2:
        print("Need at least 2 labeled pairs (both classes) to compute ROC. CSV saved at", RESULTS_CSV)
        return

    y = np.array(y_true)
    def analyze(score, name):
        fpr, tpr, thr = metrics.roc_curve(y, score)
        auc = metrics.auc(fpr, tpr)
        youden_idx = np.argmax(tpr - fpr)
        youden_thr = thr[youden_idx]
        prec, rec, pr_thr = metrics.precision_recall_curve(y, score)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
        best_f1_idx = int(np.nanargmax(f1))
        best_f1_thr = pr_thr[best_f1_idx] if best_f1_idx < len(pr_thr) else pr_thr[-1]
        print(f"{name} AUC={auc:.4f} Youden_thr={youden_thr:.4f} bestF1_thr={best_f1_thr:.4f}")
        return {"auc": float(auc), "youden_thr": float(youden_thr), "bestF1_thr": float(best_f1_thr)}

    res_embed = analyze(np.array(embed_scores), "embed_sim_inverted(1-embed)")
    res_fused = analyze(np.array(fused_scores), "fused")

    suggestions = {
        "embed_youen": res_embed["youden_thr"],
        "embed_f1": res_embed["bestF1_thr"],
        "fused_youen": res_fused["youden_thr"],
        "fused_f1": res_fused["bestF1_thr"]
    }
    with open(SUGGEST_JSON, "w", encoding="utf-8") as fo:
        json.dump(suggestions, fo, indent=2)

    print("Saved CSV:", RESULTS_CSV)
    print("Saved suggestions:", SUGGEST_JSON)

if __name__ == "__main__":
    main()
