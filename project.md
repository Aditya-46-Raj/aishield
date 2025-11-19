# **AIShield — Multi-Modal KYC Fraud Detection Engine**

### *Explainable multi‑modal defense against deepfakes and forged identity documents.*

**Team 0AI – Aditya Raj**
**GitHub:** [https://github.com/Aditya-46-Raj/aishield](https://github.com/Aditya-46-Raj/aishield)

---

# **1. Executive Summary**

AI-driven identity fraud—synthetic IDs, face swaps, and deepfake videos—has rapidly outpaced traditional KYC verification systems. AIShield provides a practical, deployable, and explainable solution designed to make digital onboarding secure, transparent, and fraud-resistant.

AIShield performs multi-modal analysis across three core channels—**document forensics**, **biometric liveness/deepfake detection**, and **face embedding verification**—and fuses these signals via a machine-learned LightGBM risk model with **SHAP explainability**. It exposes clear, auditable reasoning behind each fraud decision, making it suitable for regulated environments.

This prototype includes an end-to-end implementation: a **Streamlit UI**, **FastAPI backend**, **document CNN**, **ELA heatmaps**, **anti-spoof liveness model**, **fusion engine**, **admin dashboard**, and a **full audit trail system** storing per-case evidence and explanations. The architecture is modular, fast (<2s for full case analysis), and built for future expansion.

AIShield demonstrates how explainable, multi-signal fraud defense can improve KYC processes while remaining transparent and regulator-friendly.

---

# **2. Problem Context**

Online KYC has made banking accessible, but it has also opened the door to new forms of fraud—AI-generated IDs, deepfake selfies, replay attacks, and subtle document manipulations. Traditional systems rely heavily on OCR and simple face matching, leaving them vulnerable to:

* **Document tampering** (text edits, face swaps, region-level manipulations)
* **AI-generated synthetic documents**
* **Deepfake videos used in liveness checks**
* **Printed-photo and screen-replay spoofs**
* **Lack of explainability**, making manual review difficult

Regulatory bodies now expect transparent, justifiable KYC decisions. AIShield addresses all major gaps with a **transparent, explainable, multi-modal defense system**, capable of identifying synthetic fraud attempts while offering low-friction onboarding.

---

# **3. Solution Overview**

AIShield creates an intelligent, layered KYC risk engine that analyzes:

### **1. Document Forensics (ELA + CNN)**

* ELA exposes recompression artifacts indicating tampering.
* A fine-tuned EfficientNet-B0 classifier flags forged textures.
* Combined `doc_score` improves accuracy.
* Heatmap visualizations support manual review.

### **2. Liveness & Anti-Spoofing**

* Frame-level motion and blink analysis.
* Lightweight anti-spoof model using texture + FFT cues.
* Aggregates into `deepfake_score` and `liveness_score`.

### **3. Face Embedding Matching**

* High-quality face embeddings ensure ID-selfie consistency.
* Outputs `embed_similarity`.

### **4. Fusion Model (LightGBM + SHAP)**

* Multi-modal features → final `fused_fraud_prob`.
* SHAP explanations clarify decisions.

### **5. Admin Dashboard & Audit Logs**

* Timestamped JSON logs
* Case reviews with heatmaps & SHAP
* Evidence export & filters

Together, these modules generate a robust, explainable fraud decision.

---

# **4. System Architecture**

```
 ┌───────────────┐      ┌──────────────────┐
 │   Streamlit    │ ---> │   FastAPI API    │
 └───────┬────────┘      └──────┬───────────┘
         │                       │
         ▼                       ▼
 ┌───────────────┐      ┌────────────────────┐
 │ Document       │      │ Liveness/Anti-     │
 │ Forensics      │      │ Spoof Analysis     │
 │ (ELA + CNN)    │      └────────────────────┘
 └──────┬────────┘                │
        │                         │
        ▼                         ▼
 ┌───────────────────┐     ┌───────────────────┐
 │ Face Embedding    │     │ Behavioral Signals │
 │ Matching          │     └──────┬────────────┘
 └────────┬──────────┘            │
          ▼                       ▼
      ┌─────────────────────────────────┐
      │     Fusion Model (LightGBM)     │
      └─────────────────┬──────────────┘
                        ▼
           ┌────────────────────────┐
           │ Final Decision + SHAP  │
           └────────────────────────┘
```

---

# **5. Document Forensics Engine**

### **5.1 Error Level Analysis (ELA)**

* Detects tampered regions via recompression artifacts.
* Produces heatmaps for transparency.
* Generates `ela_score`.

### **5.2 CNN-based Classifier**

* Trained EfficientNet-B0 on clean vs forged dataset.
* Learns texture inconsistencies & forged patches.
* Outputs `clf_prob`.

### **5.3 Combined Document Score**

```
 doc_score = 0.4 × ela_score + 0.6 × clf_prob
```

---

# **6. Liveness & Anti-Spoofing Engine**

### **6.1 Motion/Blink Analysis**

* MediaPipe-based landmark tracking.
* Natural head pose & blink detection.
* Provides `motion_score`.

### **6.2 Anti-Spoof Model**

* Lightweight heuristics using texture and FFT.
* Per-frame spoof scores aggregated.
* Produces `deepfake_score`.

### **6.3 Liveness Score**

```
 liveness_score = 0.5 × (1 - deepfake_score) + 0.5 × motion_score
```

---

# **7. Face Embedding Matching**

* InceptionResnetV1 computes 512-dimensional embeddings.
* Cosine similarity between ID-face & selfie yields `embed_similarity`.
* Provides biometric grounding.

---

# **8. Fusion Model & Explainability**

### **Features**

* doc_score
* liveness_score
* embed_similarity
* behavior_score

### **LightGBM Fusion Engine**

* Trained on synthetic multi-modal dataset.
* Outputs `fused_fraud_prob` (0–1).

### **Explainability with SHAP**

* SHAP values expose feature contributions.
* Visualized as bar charts on admin panel.
* Essential for regulatory compliance.

---

# **9. Audit Trail System**

AIShield uses **file-based JSON logs**, ideal for hackathon and demo reliability.

Each case logs:

* Scores (doc, liveness, embedding, fusion)
* Heatmap paths
* SHAP explanations
* Model versions
* Case timestamp

Logs are:

* Anonymized
* Easy to export
* Automatically cleaned using retention policy

---

# **10. Admin Dashboard**

Admin View includes:

* Case list with timestamps
* Filters: risk level, date
* Expandable evidence for each case
* Heatmaps, SHAP charts
* Downloadable logs

Designed for real-world reviewer workflows.

---

# **11. Security, Privacy & Compliance**

### **Implemented:**

* Filename anonymization
* Automatic raw file deletion
* Evidence retention policy
* Explainability for all decisions
* Local encrypted storage (optional)

### **Planned for production:**

* Optional Postgres/MongoDB integration
* Role-based access control
* Cloud-deployment hardening

---

# **12. Scalability & Future Work**

* On-device preprocessing to reduce PII footprint
* Larger document-forgery dataset
* Anti-deepfake model upgrade
* Federated learning support
* Integration with AML/PEP systems

---

# **13. Impact & Business Value**

AIShield:

* Reduces fraud during onboarding
* Reduces manual verification load
* Increases trust and safety
* Supports regulatory justification
* Enables transparent case review

---

# **14. Demo Flow**

1. Upload forged ID → ELA heatmap highlights forgery → high doc_score
2. Upload replay/deepfake video → high deepfake_score → low liveness
3. Upload clean ID & real selfie → PASS scenario
4. Open Admin Dashboard → review case evidence

---

# **15. Tech Stack**

* Streamlit (UI)
* FastAPI (Backend)
* PyTorch (CNN, embeddings)
* LightGBM (Fusion model)
* MediaPipe (landmarks)
* scikit-learn (anomaly detection)
* JSON logs (audit)

---

# **16. Repository & Execution**

GitHub: [https://github.com/Aditya-46-Raj/aishield](https://github.com/Aditya-46-Raj/aishield)

**Run steps**:

* `uvicorn app:app --reload` (backend)
* `streamlit run app.py` (frontend)

---

# **17. Conclusion**

AIShield demonstrates how layered, explainable AI can harden KYC onboarding against advanced document and deepfake fraud. It is transparent, deployable, auditable, and designed with modular expansion in mind. With multi-modal fusion, explainability, and a reviewer-friendly admin panel, AIShield stands as a practically valuable and technologically robust fraud-prevention solution.
