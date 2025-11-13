import streamlit as st
import requests
from PIL import Image
import io
import os

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")  # if using docker-compose; otherwise http://localhost:8000

st.set_page_config(page_title="AIShield Demo", layout="centered")
st.title("AIShield — KYC Fraud Demo")

st.header("1) Document Authenticity Check")
doc_file = st.file_uploader("Upload ID image (jpg/png)", type=["jpg","jpeg","png"])
if doc_file:
    files = {"file": (doc_file.name, doc_file.getvalue(), "image/jpeg")}
    with st.spinner("Analyzing document..."):
        resp = requests.post(f"{BACKEND}/analyze/document", files=files, timeout=120)
    if resp.ok:
        data = resp.json()
        st.subheader("Result")
        st.write(f"Score (0-1): **{data['score']:.3f}**")
        st.write(data['explanation'])
        if data.get("heatmap"):
            # fetch heatmap - backend serves static? we returned path
            heatmap_url = f"{BACKEND}/outputs/{doc_file.name}_ela_heatmap.png"
            # if backend can't serve static files, display by requesting from file system via separate endpoint.
            st.image(data['heatmap'], caption="ELA Heatmap", use_container_width=True)

st.markdown("---")
st.header("2) Liveness / Video Check (short video)")
video_file = st.file_uploader("Upload short selfie video (mp4, max 10s)", type=["mp4","mov"])
if video_file:
    files = {"file": (video_file.name, video_file.getvalue(), "video/mp4")}
    with st.spinner("Analyzing video..."):
        resp = requests.post(f"{BACKEND}/analyze/video", files=files, timeout=180)
    if resp.ok:
        data = resp.json()
        st.subheader("Result")
        st.write(f"Verdict: **{data['verdict']}**")
        st.write(f"Score (0-1): **{data['score']:.3f}**")
        st.write(data['explanation'])


st.markdown("---")
st.header("3) Combined Final Analysis (ID + Selfie [+Video])")
id_file2 = st.file_uploader("Upload ID image (final)", type=["jpg","jpeg","png"], key="id_final")
selfie_file2 = st.file_uploader("Upload selfie image (final)", type=["jpg","jpeg","png"], key="selfie_final")
video_file2 = st.file_uploader("Upload optional short video (mp4)", type=["mp4"], key="video_final")
if st.button("Run Final Analysis"):
    if not id_file2 or not selfie_file2:
        st.error("Upload both ID and selfie first.")
    else:
        files = {
            "id_file": (id_file2.name, id_file2.getvalue(), "image/jpeg"),
            "selfie_file": (selfie_file2.name, selfie_file2.getvalue(), "image/jpeg")
        }
        if video_file2:
            files["video_file"] = (video_file2.name, video_file2.getvalue(), "video/mp4")
        with st.spinner("Running fusion analysis..."):
            resp = requests.post(f"{BACKEND}/analyze/final", files=files, timeout=240)
        if resp.ok:
            d = resp.json()
            st.subheader("Final Fraud Probability")
            st.metric("Fraud Score (0-1)", f"{d['fused_fraud_prob']:.3f}")
            st.write("Reasons / Explanations:")
            st.json({
                "doc": d.get("doc_explanation"),
                "liveness": d.get("liveness_explanation"),
                "embed": d.get("embed_explanation"),
                "fusion": d.get("fusion_explanation")
            })
            if d.get("heatmap"):
                try:
                    img_resp = requests.get(d["heatmap"])
                    img = Image.open(io.BytesIO(img_resp.content))
                    st.image(img, caption="ELA Heatmap", use_container_width=True)
                except:
                    pass
        else:
            st.error(f"Error from backend: {resp.status_code}")
