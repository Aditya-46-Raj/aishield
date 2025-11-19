import streamlit as st
import requests
from PIL import Image
import io
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import time

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="AIShield Demo", layout="wide")

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")  # if using docker-compose; otherwise http://localhost:8000

# Path to backend logs directory - handle both development and production paths
LOGS_DIR = Path(__file__).parent.parent / "backend" / "logs"
# Fallback: if running from frontend directory, try relative path
if not LOGS_DIR.exists():
    LOGS_DIR = Path("../backend/logs").resolve()
# Fallback: if that doesn't work, try absolute path from current working directory
if not LOGS_DIR.exists():
    LOGS_DIR = Path(os.getcwd()) / "backend" / "logs"

# ============================================================================
# PHASE 5.4: COLOR PALETTE & STYLING
# ============================================================================
COLORS = {
    "clean": "#28A745",      # Green - Passed
    "suspicious": "#FFC107", # Yellow - Review Required
    "fraudulent": "#DC3545", # Red - High Risk
    "neutral": "#6C757D",    # Gray
    "info": "#17A2B8"        # Blue
}

# Custom CSS for enhanced UI
st.markdown(f"""
<style>
    /* Badge styling */
    .badge {{
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.9em;
        font-weight: 700;
        line-height: 1;
        color: #fff;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin: 0.2em;
    }}
    .badge-clean {{
        background-color: {COLORS['clean']};
    }}
    .badge-suspicious {{
        background-color: {COLORS['suspicious']};
        color: #212529;
    }}
    .badge-fraudulent {{
        background-color: {COLORS['fraudulent']};
    }}
    
    /* Timeline styling */
    .timeline {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border-left: 4px solid {COLORS['info']};
    }}
    .timeline-step {{
        text-align: center;
        flex: 1;
        position: relative;
    }}
    .timeline-step-active {{
        color: {COLORS['info']};
        font-weight: bold;
    }}
    .timeline-step-complete {{
        color: {COLORS['clean']};
        font-weight: bold;
    }}
    .timeline-arrow {{
        color: {COLORS['neutral']};
        font-size: 1.5em;
    }}
    
    /* Explanation banners */
    .explanation-banner {{
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }}
    .explanation-passed {{
        background-color: #d4edda;
        border-color: {COLORS['clean']};
        color: #155724;
    }}
    .explanation-flagged {{
        background-color: #f8d7da;
        border-color: {COLORS['fraudulent']};
        color: #721c24;
    }}
    .explanation-review {{
        background-color: #fff3cd;
        border-color: {COLORS['suspicious']};
        color: #856404;
    }}
    
    /* Score cards */
    .score-card {{
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }}
    .score-card-clean {{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid {COLORS['clean']};
    }}
    .score-card-suspicious {{
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid {COLORS['suspicious']};
    }}
    .score-card-fraudulent {{
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid {COLORS['fraudulent']};
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS FOR PHASE 5.4
# ============================================================================

def render_badge(label: str, badge_type: str = "clean"):
    """Render a colored badge"""
    badge_class = f"badge badge-{badge_type}"
    return f'<span class="{badge_class}">{label}</span>'

def render_timeline(current_step: str = "upload"):
    """Render processing timeline"""
    steps = [
        ("upload", "📤 Upload"),
        ("document", "📄 Document"),
        ("liveness", "🎭 Liveness"),
        ("embeddings", "👤 Face Match"),
        ("fusion", "🔄 Fusion"),
        ("result", "✅ Result")
    ]
    
    step_order = {s[0]: i for i, s in enumerate(steps)}
    current_idx = step_order.get(current_step, 0)
    
    timeline_html = '<div class="timeline">'
    for idx, (step_key, step_label) in enumerate(steps):
        if idx == current_idx:
            step_class = "timeline-step timeline-step-active"
        elif idx < current_idx:
            step_class = "timeline-step timeline-step-complete"
        else:
            step_class = "timeline-step"
        
        timeline_html += f'<div class="{step_class}">{step_label}</div>'
        
        # Add arrow between steps
        if idx < len(steps) - 1:
            timeline_html += '<div class="timeline-arrow">→</div>'
    
    timeline_html += '</div>'
    return timeline_html

def render_explanation_banner(verdict: str, fraud_prob: float, reasons: list):
    """Render explanation banner based on verdict"""
    if verdict == "PASS" or fraud_prob < 0.3:
        banner_class = "explanation-passed"
        icon = "✅"
        title = "Why This Passed"
        summary = f"Low fraud probability ({fraud_prob:.1%}). All security checks indicate authentic verification."
    elif verdict == "FAIL" or fraud_prob > 0.7:
        banner_class = "explanation-flagged"
        icon = "🚨"
        title = "Why This Was Flagged"
        summary = f"High fraud probability ({fraud_prob:.1%}). Multiple security indicators suggest fraudulent activity."
    else:
        banner_class = "explanation-review"
        icon = "⚠️"
        title = "Why Manual Review Required"
        summary = f"Moderate fraud probability ({fraud_prob:.1%}). Some indicators require human verification."
    
    banner_html = f"""
    <div class="explanation-banner {banner_class}">
        <h3>{icon} {title}</h3>
        <p><strong>{summary}</strong></p>
        <ul>
    """
    
    for reason in reasons:
        banner_html += f"<li>{reason}</li>"
    
    banner_html += """
        </ul>
    </div>
    """
    
    return banner_html

def get_risk_badge(score: float, reverse: bool = False):
    """Get badge HTML for a risk score"""
    # reverse=True for scores where higher is better (e.g., liveness)
    if reverse:
        if score > 0.7:
            return render_badge("✅ Passed", "clean")
        elif score > 0.4:
            return render_badge("⚠️ Review", "suspicious")
        else:
            return render_badge("🚨 High Risk", "fraudulent")
    else:
        if score < 0.3:
            return render_badge("✅ Passed", "clean")
        elif score < 0.6:
            return render_badge("⚠️ Review", "suspicious")
        else:
            return render_badge("🚨 High Risk", "fraudulent")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🛡️ AIShield")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["👤 User Mode", "🔐 Admin Mode"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.info("""
**User Mode**: Run KYC verification  
**Admin Mode**: Review audit logs & evidence
""")

# ============================================================================
# ADMIN MODE
# ============================================================================
if app_mode == "🔐 Admin Mode":
    st.title("🔐 Admin Review Console")
    st.markdown("### 📊 KYC Verification Audit Logs")
    
    # Debug: Show resolved logs directory path
    with st.expander("🔧 Debug Info - Logs Directory Path"):
        st.code(f"Logs Directory: {LOGS_DIR}")
        st.code(f"Exists: {LOGS_DIR.exists()}")
        st.code(f"Absolute Path: {LOGS_DIR.resolve()}")
        if LOGS_DIR.exists():
            st.code(f"Files in directory: {list(LOGS_DIR.iterdir())}")
    
    # Check if logs directory exists
    if not LOGS_DIR.exists():
        st.warning(f"📁 Logs directory not found: `{LOGS_DIR}`")
        st.info("Run some verifications in User Mode first to generate audit logs.")
        st.stop()
    
    # Load all log files (exclude .case_counter file)
    all_json_files = list(LOGS_DIR.glob("*.json"))
    log_files = sorted([f for f in all_json_files if not f.name.startswith('.')], reverse=True)  # Most recent first, exclude hidden files
    
    if not log_files:
        st.info("📭 No audit logs found yet. Run verifications in User Mode to generate logs.")
        st.stop()
    
    # ========================================================================
    # FILTERS
    # ========================================================================
    st.markdown("---")
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])
    
    with col_filter1:
        # Date filter
        date_filter = st.selectbox(
            "📅 Filter by Date:",
            ["All Dates", "Today", "Last 7 Days", "Last 30 Days"],
            index=0
        )
    
    with col_filter2:
        # Risk filter
        risk_filter = st.selectbox(
            "⚠️ Filter by Risk Level:",
            ["All Levels", "CLEAN", "MODERATE", "HIGH_RISK"],
            index=0
        )
    
    with col_filter3:
        # Verdict filter
        verdict_filter = st.selectbox(
            "⚖️ Verdict:",
            ["All", "PASS", "REVIEW", "FAIL"],
            index=0
        )
    
    st.markdown("---")
    
    # ========================================================================
    # LOAD AND FILTER LOGS
    # ========================================================================
    logs_data = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                log_data['_filename'] = log_file.name
                log_data['_filepath'] = str(log_file)
                logs_data.append(log_data)
        except Exception as e:
            st.warning(f"Failed to load {log_file.name}: {e}")
    
    # Apply filters
    filtered_logs = []
    for log in logs_data:
        # Date filter
        if date_filter != "All Dates":
            log_date = datetime.fromisoformat(log.get('timestamp', ''))
            now = datetime.now()
            delta_days = (now - log_date).days
            
            if date_filter == "Today" and delta_days > 0:
                continue
            elif date_filter == "Last 7 Days" and delta_days > 7:
                continue
            elif date_filter == "Last 30 Days" and delta_days > 30:
                continue
        
        # Risk filter
        if risk_filter != "All Levels":
            risk_label = log.get('risk_assessment', {}).get('risk_label', '')
            if risk_label != risk_filter:
                continue
        
        # Verdict filter
        if verdict_filter != "All":
            verdict = log.get('risk_assessment', {}).get('verdict', '')
            if verdict != verdict_filter:
                continue
        
        filtered_logs.append(log)
    
    # ========================================================================
    # DISPLAY SUMMARY STATISTICS
    # ========================================================================
    st.markdown(f"### 📈 Summary ({len(filtered_logs)} cases)")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        total_cases = len(filtered_logs)
        st.metric("📋 Total Cases", total_cases)
    
    with col_stat2:
        pass_count = sum(1 for log in filtered_logs if log.get('risk_assessment', {}).get('verdict') == 'PASS')
        st.metric("✅ PASS", pass_count)
    
    with col_stat3:
        review_count = sum(1 for log in filtered_logs if log.get('risk_assessment', {}).get('verdict') == 'REVIEW')
        st.metric("⚠️ REVIEW", review_count)
    
    with col_stat4:
        fail_count = sum(1 for log in filtered_logs if log.get('risk_assessment', {}).get('verdict') == 'FAIL')
        st.metric("🚨 FAIL", fail_count)
    
    # Average fraud probability
    if filtered_logs:
        avg_fraud = sum(log.get('scores', {}).get('fused_fraud_probability', 0) for log in filtered_logs) / len(filtered_logs)
        st.info(f"📊 **Average Fraud Probability:** {avg_fraud:.1%}")
    
    st.markdown("---")
    
    # ========================================================================
    # CASE LISTING
    # ========================================================================
    st.markdown("### 📋 Case List")
    
    if not filtered_logs:
        st.warning("No cases match the selected filters.")
    else:
        for idx, log in enumerate(filtered_logs):
            case_id = log.get('case_id', 'Unknown')
            case_number = log.get('case_number', 0)
            timestamp = log.get('timestamp', 'Unknown')
            
            # Scores
            scores = log.get('scores', {})
            fraud_prob = scores.get('fused_fraud_probability', 0)
            liveness_score = scores.get('liveness_score', 0)
            doc_score = scores.get('document_score', 0)
            embed_sim = scores.get('embedding_similarity', 0)
            deepfake_score = scores.get('deepfake_score', 0)
            
            # Risk assessment
            risk_assessment = log.get('risk_assessment', {})
            verdict = risk_assessment.get('verdict', 'UNKNOWN')
            risk_label = risk_assessment.get('risk_label', 'UNKNOWN')
            
            # Phase 5.4: Enhanced color-coded verdict with badges
            if verdict == 'PASS':
                verdict_emoji = "✅"
                verdict_color = "green"
                verdict_badge = render_badge("✅ PASSED", "clean")
            elif verdict == 'REVIEW':
                verdict_emoji = "⚠️"
                verdict_color = "orange"
                verdict_badge = render_badge("⚠️ REVIEW REQUIRED", "suspicious")
            elif verdict == 'FAIL':
                verdict_emoji = "🚨"
                verdict_color = "red"
                verdict_badge = render_badge("🚨 HIGH RISK", "fraudulent")
            else:
                verdict_emoji = "❓"
                verdict_color = "gray"
                verdict_badge = render_badge("❓ UNKNOWN", "neutral")
            
            # Summary row (collapsible)
            with st.expander(
                f"{verdict_emoji} **Case #{case_number}** | FRAUD: {fraud_prob:.2%} | Liveness: {liveness_score:.2f} | Document: {doc_score:.2f} | 📅 {timestamp}",
                expanded=False
            ):
                # ============================================================
                # CASE DETAILS
                # ============================================================
                st.markdown(f"#### {verdict_emoji} Case Details: `{case_id}`")
                
                # Main metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🎯 Fraud Prob", f"{fraud_prob:.1%}")
                with col2:
                    st.metric("📄 Document", f"{doc_score:.3f}")
                with col3:
                    st.metric("🎭 Liveness", f"{liveness_score:.3f}")
                with col4:
                    st.metric("👤 Face Match", f"{embed_sim:.3f}")
                with col5:
                    st.metric("🤖 Deepfake", f"{deepfake_score:.3f}")
                
                # Phase 5.4: Enhanced risk assessment with badges
                st.markdown("---")
                col_risk1, col_risk2, col_risk3 = st.columns(3)
                with col_risk1:
                    st.markdown(f"**Verdict:** {verdict_badge}", unsafe_allow_html=True)
                with col_risk2:
                    # Risk label badge
                    if risk_label == "CLEAN":
                        risk_badge = render_badge(f"🟢 {risk_label}", "clean")
                    elif risk_label == "MODERATE":
                        risk_badge = render_badge(f"⚠️ {risk_label}", "suspicious")
                    else:
                        risk_badge = render_badge(f"🚨 {risk_label}", "fraudulent")
                    st.markdown(f"**Risk Label:** {risk_badge}", unsafe_allow_html=True)
                with col_risk3:
                    confidence = risk_assessment.get('confidence', 0) or 0.0
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    st.progress(confidence)
                
                # Evidence paths
                st.markdown("---")
                st.markdown("#### 📁 Evidence Files")
                evidence = log.get('evidence', {})
                col_ev1, col_ev2, col_ev3 = st.columns(3)
                with col_ev1:
                    doc_path = evidence.get('document_path', 'N/A')
                    st.caption(f"📄 Document: `{Path(doc_path).name if doc_path else 'N/A'}`")
                with col_ev2:
                    video_path = evidence.get('video_path', 'N/A')
                    st.caption(f"🎥 Video: `{Path(video_path).name if video_path else 'N/A'}`")
                with col_ev3:
                    selfie_path = evidence.get('selfie_path', 'N/A')
                    st.caption(f"🤳 Selfie: `{Path(selfie_path).name if selfie_path else 'N/A'}`")
                
                # Display ELA Heatmap
                heatmap_path = evidence.get('heatmap_path')
                if heatmap_path and Path(heatmap_path).exists():
                    st.markdown("---")
                    st.markdown("#### 🔥 ELA Heatmap (Document Tampering Detection)")
                    try:
                        heatmap_img = Image.open(heatmap_path)
                        st.image(heatmap_img, caption="Error Level Analysis Heatmap", use_column_width=True)
                    except Exception as e:
                        st.warning(f"Could not load heatmap: {e}")
                
                # Component details
                st.markdown("---")
                st.markdown("#### 🔍 Component Analysis")
                
                components = log.get('components', {})
                
                # Document analysis
                doc_analysis = components.get('document_analysis', {})
                if doc_analysis:
                    st.markdown("**📄 Document Forensics Details:**")
                    st.json(doc_analysis)
                    st.markdown("")
                
                # Liveness details
                liveness_details = components.get('liveness_details', {})
                if liveness_details:
                    st.markdown("**🎭 Liveness Detection Details:**")
                    st.json(liveness_details)
                    st.markdown("")
                
                # Face matching
                face_matching = components.get('face_matching', {})
                if face_matching:
                    st.markdown("**👤 Face Matching Details:**")
                    st.json(face_matching)
                    st.markdown("")
                
                # SHAP Explanation
                fusion_expl = components.get('fusion_explanation', {})
                if fusion_expl:
                    st.markdown("---")
                    st.markdown("#### 🧠 SHAP Explainability")
                    
                    shap_values = fusion_expl.get('shap_values', {})
                    if shap_values and "error" not in shap_values:
                        # Create SHAP bar chart
                        shap_features = {k: v for k, v in shap_values.items() if k != "base_value"}
                        
                        if shap_features:
                            # Create DataFrame
                            df_shap = pd.DataFrame({
                                'Feature': list(shap_features.keys()),
                                'SHAP Value': list(shap_features.values())
                            })
                            df_shap = df_shap.sort_values('SHAP Value', ascending=True)
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(8, 3))
                            colors = ['red' if x > 0 else 'green' for x in df_shap['SHAP Value']]
                            ax.barh(df_shap['Feature'], df_shap['SHAP Value'], color=colors, alpha=0.7)
                            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                            ax.set_xlabel('SHAP Value (Impact on Fraud Probability)')
                            ax.set_title('Feature Contributions')
                            ax.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Show raw SHAP values
                        st.markdown("**📊 Raw SHAP Values:**")
                        st.json(shap_values)
                    else:
                        st.info("SHAP values not available")
                
                # Performance metrics
                st.markdown("---")
                st.markdown("#### ⚡ Performance Metrics")
                performance = log.get('performance', {})
                col_perf1, col_perf2, col_perf3 = st.columns(3)
                with col_perf1:
                    proc_time = performance.get('processing_time_ms', 0)
                    st.metric("⏱️ Processing Time", f"{proc_time:.2f}ms" if proc_time else "N/A")
                with col_perf2:
                    frames = performance.get('frames_analyzed', 0)
                    st.metric("🎞️ Frames Analyzed", frames)
                with col_perf3:
                    latency = performance.get('detection_latency_ms')
                    st.metric("⚡ Detection Latency", f"{latency:.2f}ms" if latency else "N/A")
                
                # System info
                st.markdown("---")
                st.markdown("#### 🖥️ System Information")
                system = log.get('system', {})
                context = log.get('context', {})
                
                col_sys1, col_sys2, col_sys3 = st.columns(3)
                with col_sys1:
                    st.caption(f"**Version:** {system.get('version', 'N/A')}")
                with col_sys2:
                    st.caption(f"**Environment:** {system.get('environment', 'N/A')}")
                with col_sys3:
                    st.caption(f"**IP:** {context.get('ip_address', 'N/A')}")
                
                # Download button
                st.markdown("---")
                st.markdown("#### 📥 Download Evidence")
                
                # Create download button for JSON log
                json_str = json.dumps(log, indent=2)
                st.download_button(
                    label="⬇️ Download JSON Log",
                    data=json_str,
                    file_name=log.get('_filename', 'case_log.json'),
                    mime="application/json",
                    key=f"download_{case_id}"
                )

# ============================================================================
# USER MODE
# ============================================================================
else:
    st.title("🛡️ AIShield — Enterprise KYC Verification")
    st.markdown("### Secure Identity Verification with AI-Powered Fraud Detection")
    
    # Show initial timeline
    st.markdown(render_timeline("upload"), unsafe_allow_html=True)
    st.markdown("---")
    
    st.header("1️⃣ Document Authenticity Check")
doc_file = st.file_uploader("Upload ID image (jpg/png)", type=["jpg","jpeg","png"])
if doc_file:
    # Update timeline
    st.markdown(render_timeline("document"), unsafe_allow_html=True)
    
    files = {"file": (doc_file.name, doc_file.getvalue(), "image/jpeg")}
    
    # Phase 5.4: Micro-loading indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Uploading document...")
    progress_bar.progress(20)
    time.sleep(0.3)
    
    status_text.text("🔍 Analyzing Error Level Analysis (ELA)...")
    progress_bar.progress(40)
    
    with st.spinner("🤖 Running CNN forgery detection..."):
        resp = requests.post(f"{BACKEND}/analyze/document", files=files, timeout=120)
    
    progress_bar.progress(80)
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(100)
    time.sleep(0.5)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if resp.ok:
        data = resp.json()
        
        # Main Document Forensics Section with Phase 5.4 enhancements
        st.subheader("📄 Document Forensics Report")
        
        # Risk Level Banner with color-coded badges
        doc_score = data.get('doc_score', data.get('score', 0))
        risk_level = data.get('risk_level', 'UNKNOWN')
        
        # Phase 5.4: Enhanced risk display with badges
        if risk_level == 'LOW':
            st.markdown(f"""
            <div class="score-card score-card-clean">
                <h2>{render_badge("✅ CLEAN - Document Authentic", "clean")}</h2>
                <h3>Combined Score: {doc_score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)
        elif risk_level == 'MEDIUM':
            st.markdown(f"""
            <div class="score-card score-card-suspicious">
                <h2>{render_badge("⚠️ SUSPICIOUS - Review Recommended", "suspicious")}</h2>
                <h3>Combined Score: {doc_score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)
        elif risk_level == 'HIGH':
            st.markdown(f"""
            <div class="score-card score-card-fraudulent">
                <h2>{render_badge("🚨 LIKELY FORGED - High Risk", "fraudulent")}</h2>
                <h3>Combined Score: {doc_score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Score: {doc_score:.3f}")
        
        # Detailed Scores in Columns with badges
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ela_score = data.get('ela_score', data.get('score', 0))
            st.metric("🔍 ELA Score", f"{ela_score:.3f}", 
                     help="Error Level Analysis - detects recompression artifacts")
            st.markdown(get_risk_badge(ela_score), unsafe_allow_html=True)
        
        with col2:
            clf_prob = data.get('clf_prob', 0)
            st.metric("🤖 CNN Classifier", f"{clf_prob:.1%}", 
                     help="Deep learning forgery probability")
            # Add progress bar for CNN probability
            st.progress(clf_prob)
            st.markdown(get_risk_badge(clf_prob), unsafe_allow_html=True)
        
        with col3:
            st.metric("📊 Combined Score", f"{doc_score:.3f}",
                     help="Weighted combination: 40% ELA + 60% CNN")
            st.markdown(get_risk_badge(doc_score), unsafe_allow_html=True)
        
        # Phase 5.4: Explanation banner
        explanation_text = data.get('explanation', 'No explanation available')
        st.markdown("### 📝 Analysis Details")
        
        # Determine verdict from risk level
        if risk_level == 'LOW':
            reasons = [
                f"ELA score is low ({ela_score:.3f}) - minimal compression artifacts detected",
                f"CNN classifier confidence of forgery is low ({clf_prob:.1%})",
                "Document appears to have consistent compression throughout"
            ]
            st.markdown(render_explanation_banner("PASS", doc_score, reasons), unsafe_allow_html=True)
        elif risk_level == 'MEDIUM':
            reasons = [
                f"ELA score shows moderate artifacts ({ela_score:.3f})",
                f"CNN classifier indicates possible tampering ({clf_prob:.1%})",
                "Some regions show inconsistent compression levels",
                "Recommend manual verification by compliance officer"
            ]
            st.markdown(render_explanation_banner("REVIEW", doc_score, reasons), unsafe_allow_html=True)
        else:
            reasons = [
                f"High ELA score ({ela_score:.3f}) - significant compression inconsistencies",
                f"CNN classifier high confidence of forgery ({clf_prob:.1%})",
                "Multiple regions show evidence of digital manipulation",
                "Document likely edited or completely fabricated"
            ]
            st.markdown(render_explanation_banner("FAIL", doc_score, reasons), unsafe_allow_html=True)
        
        st.info(f"**Technical Details:** {explanation_text}")
        
        # ELA Heatmap
        if data.get("heatmap"):
            st.markdown("### 🔥 ELA Heatmap (Tampering Visualization)")
            st.image(data['heatmap'], caption="Regions with different compression levels (potential tampering)", 
                    use_container_width=True)
            st.caption("💡 **How to read:** Bright areas indicate regions with different compression levels than the rest of the image, suggesting potential tampering.")

st.markdown("---")
st.header("2️⃣ Liveness / Video Check")
st.caption("Upload a short selfie video to verify you're a real person")
    
video_file = st.file_uploader("Upload short selfie video (mp4, max 10s)", type=["mp4","mov"])
if video_file:
    # Update timeline
    st.markdown(render_timeline("liveness"), unsafe_allow_html=True)
    
    files = {"file": (video_file.name, video_file.getvalue(), "video/mp4")}
    
    # Phase 5.4: Micro-loading indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Uploading video...")
    progress_bar.progress(15)
    time.sleep(0.3)
    
    status_text.text("🎬 Extracting frames...")
    progress_bar.progress(30)
    time.sleep(0.3)
    
    status_text.text("🤖 Running anti-spoofing detection...")
    progress_bar.progress(50)
    
    with st.spinner("🎭 Analyzing liveness indicators..."):
        resp = requests.post(f"{BACKEND}/analyze/video", files=files, timeout=180)
    
    progress_bar.progress(90)
    status_text.text("✅ Liveness analysis complete!")
    progress_bar.progress(100)
    time.sleep(0.5)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if resp.ok:
        data = resp.json()
        st.subheader("🎭 Liveness Detection Result")
        
        # Phase 5.4: Enhanced verdict display with badges
        verdict = data.get('verdict', 'UNKNOWN')
        score = data.get('score', 0)
        
        if verdict == "REAL" or score > 0.7:
            st.markdown(f"""
            <div class="score-card score-card-clean">
                <h2>{render_badge("✅ REAL - Authentic Liveness", "clean")}</h2>
                <h3>Liveness Score: {score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)
            reasons = [
                "Natural facial movements detected",
                "No presentation attack indicators found",
                "Video shows genuine human characteristics"
            ]
            st.markdown(render_explanation_banner("PASS", 1.0 - score, reasons), unsafe_allow_html=True)
        elif verdict == "SPOOF" or score < 0.3:
            st.markdown(f"""
            <div class="score-card score-card-fraudulent">
                <h2>{render_badge("🚨 SPOOF - Presentation Attack Detected", "fraudulent")}</h2>
                <h3>Liveness Score: {score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)
            reasons = [
                "Video shows characteristics of presentation attack",
                "Lack of natural facial movements",
                "Possible screen replay or photo-based spoof"
            ]
            st.markdown(render_explanation_banner("FAIL", score, reasons), unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="score-card score-card-suspicious">
                <h2>{render_badge("⚠️ UNCERTAIN - Manual Review Required", "suspicious")}</h2>
                <h3>Liveness Score: {score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)
            reasons = [
                "Inconclusive liveness indicators",
                "Video quality may be affecting detection",
                "Recommend repeating verification or manual review"
            ]
            st.markdown(render_explanation_banner("REVIEW", 0.5, reasons), unsafe_allow_html=True)
        
        st.info(f"**Technical Details:** {data.get('explanation', 'N/A')}")


st.markdown("---")
st.header("3️⃣ Combined Final Analysis")
st.caption("🎯 Complete KYC verification with multi-modal fraud detection")
    
id_file2 = st.file_uploader("Upload ID image (final)", type=["jpg","jpeg","png"], key="id_final")
selfie_file2 = st.file_uploader("Upload selfie image (final)", type=["jpg","jpeg","png"], key="selfie_final")
video_file2 = st.file_uploader("Upload optional short video (mp4)", type=["mp4"], key="video_final")

if st.button("🚀 Run Final Analysis", type="primary"):
    if not id_file2 or not selfie_file2:
        st.error("❌ Upload both ID and selfie first.")
    else:
        # Phase 5.4: Show complete timeline
        st.markdown(render_timeline("fusion"), unsafe_allow_html=True)
        
        files = {
            "id_file": (id_file2.name, id_file2.getvalue(), "image/jpeg"),
            "selfie_file": (selfie_file2.name, selfie_file2.getvalue(), "image/jpeg")
        }
        if video_file2:
            files["video_file"] = (video_file2.name, video_file2.getvalue(), "video/mp4")
        
        # Phase 5.4: Detailed micro-loading with single timeline
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📤 Uploading files...")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        status_text.text("📄 Analyzing document forensics...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        status_text.text("🎭 Running liveness detection...")
        progress_bar.progress(45)
        time.sleep(0.5)
        
        status_text.text("👤 Matching face embeddings...")
        progress_bar.progress(65)
        time.sleep(0.5)
        
        status_text.text("🔄 Fusion model scoring...")
        progress_bar.progress(80)
        
        with st.spinner("🧠 Generating SHAP explanations..."):
            resp = requests.post(f"{BACKEND}/analyze/final", files=files, timeout=240)
        
        progress_bar.progress(95)
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if resp.ok:
            d = resp.json()
            
            # Phase 5.4: Enhanced main fraud score display with badge
            st.markdown("---")
            st.subheader("🎯 Final Fraud Probability")
            fraud_score = d['fused_fraud_prob']
            
            # Color-coded metric based on risk level with enhanced styling
            if fraud_score < 0.3:
                st.markdown(f"""
                <div class="score-card score-card-clean">
                    <h1>{render_badge("✅ LOW RISK - Verification Passed", "clean")}</h1>
                    <h2>Fraud Probability: {fraud_score:.1%}</h2>
                    <p>Identity verification successful. All security checks passed.</p>
                </div>
                """, unsafe_allow_html=True)
                overall_verdict = "PASS"
            elif fraud_score < 0.6:
                st.markdown(f"""
                <div class="score-card score-card-suspicious">
                    <h1>{render_badge("⚠️ MEDIUM RISK - Manual Review Required", "suspicious")}</h1>
                    <h2>Fraud Probability: {fraud_score:.1%}</h2>
                    <p>Some indicators require human verification. Proceed with caution.</p>
                </div>
                """, unsafe_allow_html=True)
                overall_verdict = "REVIEW"
            else:
                st.markdown(f"""
                <div class="score-card score-card-fraudulent">
                    <h1>{render_badge("🚨 HIGH RISK - Verification Failed", "fraudulent")}</h1>
                    <h2>Fraud Probability: {fraud_score:.1%}</h2>
                    <p>Multiple fraud indicators detected. Identity verification failed.</p>
                </div>
                """, unsafe_allow_html=True)
                overall_verdict = "FAIL"
            
            # Display individual component scores with badges
            st.markdown("---")
            st.markdown("### 📊 Component Scores")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                doc_score = d.get('doc_score', 0)
                st.metric("📄 Document", f"{doc_score:.3f}")
                st.markdown(get_risk_badge(doc_score), unsafe_allow_html=True)
                st.progress(doc_score)
            
            with col2:
                live_score = d.get('liveness_score', 0)
                st.metric("🎭 Liveness", f"{live_score:.3f}")
                st.markdown(get_risk_badge(live_score, reverse=True), unsafe_allow_html=True)
                st.progress(live_score)
            
            with col3:
                embed_sim = d.get('embed_sim', 0)
                st.metric("👤 Face Match", f"{embed_sim:.3f}")
                st.markdown(get_risk_badge(1.0 - embed_sim), unsafe_allow_html=True)
                st.progress(embed_sim)
            
            with col4:
                behavior = d.get('behavior_anomaly', 0)
                st.metric("⚡ Behavior", f"{behavior:.3f}")
                st.markdown(get_risk_badge(behavior), unsafe_allow_html=True)
                st.progress(behavior)
            
            # Phase 5.4: Comprehensive explanation banner
            st.markdown("---")
            reasons = []
            
            # Build reasons list
            if doc_score > 0.6:
                reasons.append(f"🚨 Document shows high forgery probability ({doc_score:.1%})")
            elif doc_score > 0.3:
                reasons.append(f"⚠️ Document shows moderate tampering indicators ({doc_score:.1%})")
            else:
                reasons.append(f"✅ Document appears authentic ({doc_score:.1%} forgery probability)")
            
            if live_score < 0.4:
                reasons.append(f"🚨 Low liveness score ({live_score:.3f}) - possible presentation attack")
            elif live_score < 0.7:
                reasons.append(f"⚠️ Moderate liveness indicators ({live_score:.3f})")
            else:
                reasons.append(f"✅ Strong liveness indicators ({live_score:.3f})")
            
            if embed_sim < 0.5:
                reasons.append(f"🚨 Face mismatch detected - ID photo doesn't match selfie ({embed_sim:.3f} similarity)")
            elif embed_sim < 0.7:
                reasons.append(f"⚠️ Face similarity is marginal ({embed_sim:.3f})")
            else:
                reasons.append(f"✅ Face match confirmed ({embed_sim:.3f} similarity)")
            
            if behavior > 0.6:
                reasons.append(f"🚨 Behavioral anomalies detected ({behavior:.3f})")
            elif behavior > 0.3:
                reasons.append(f"⚠️ Some behavioral inconsistencies ({behavior:.3f})")
            else:
                reasons.append(f"✅ Normal behavioral patterns ({behavior:.3f})")
            
            st.markdown(render_explanation_banner(overall_verdict, fraud_score, reasons), unsafe_allow_html=True)
            
            # Enhanced Liveness Details (Phase 4.4)
            liveness_details = d.get("liveness_details", {})
            if liveness_details and any(liveness_details.values()):
                with st.expander("🎭 Enhanced Liveness Detection Details (Phase 4.4)", expanded=True):
                    st.markdown("**Multi-Modal Liveness Analysis:**")
                    
                    # Main liveness score display
                    liveness_score = liveness_details.get("liveness_score", 0)
                    st.markdown(f"### 🎯 Overall Liveness Score: {liveness_score:.3f}")
                    st.progress(min(1.0, liveness_score))
                    
                    if liveness_score > 0.7:
                        st.success("✅ HIGH LIVENESS CONFIDENCE - Likely authentic")
                    elif liveness_score > 0.4:
                        st.warning("⚠️ MODERATE LIVENESS - Manual review recommended")
                    else:
                        st.error("🚨 LOW LIVENESS - Likely presentation attack")
                    
                    st.markdown("---")
                    st.markdown("**Component Breakdown:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        df_score = liveness_details.get("deepfake_score", 0)
                        st.metric("🤖 Deepfake Score", f"{df_score:.3f}")
                        st.progress(min(1.0, df_score))
                        st.caption("Higher = More likely fake/spoofed")
                    with col2:
                        motion_score = liveness_details.get("motion_score", 0)
                        st.metric("💨 Motion Score", f"{motion_score:.3f}")
                        st.progress(min(1.0, motion_score))
                        st.caption("Higher = More natural behavior")
                    
                    st.markdown("---")
                    st.markdown("**Formula:** `liveness_score = 0.5 × (1 - deepfake_score) + 0.5 × motion_score`")
                    
                    # Show liveness reason
                    liveness_reason = liveness_details.get("liveness_reason", "")
                    if liveness_reason:
                        st.info(f"**Explanation:** {liveness_reason}")
                    
                    # Additional metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        blink_events = liveness_details.get("blink_events", 0)
                        st.caption(f"👁️ Blink events: {blink_events}")
                    with col_b:
                        motion_events = liveness_details.get("motion_events", 0)
                        st.caption(f"🏃 Motion events: {motion_events}")
            
            
            # SHAP Explainability Section
            st.markdown("---")
            st.subheader("🔍 SHAP Explainability - Feature Contributions")
            
            fusion_expl = d.get("fusion_explanation", {})
            shap_values = fusion_expl.get("shap_values", {})
            
            if shap_values and "error" not in shap_values:
                # Display SHAP values as JSON
                st.json(shap_values)
                
                # Create SHAP bar chart (excluding base_value)
                shap_features = {k: v for k, v in shap_values.items() if k != "base_value"}
                
                if shap_features:
                    # Create DataFrame for plotting
                    df_shap = pd.DataFrame({
                        'Feature': list(shap_features.keys()),
                        'SHAP Value': list(shap_features.values())
                    })
                    df_shap = df_shap.sort_values('SHAP Value', ascending=True)
                    
                    # Create horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['red' if x > 0 else 'green' for x in df_shap['SHAP Value']]
                    ax.barh(df_shap['Feature'], df_shap['SHAP Value'], color=colors, alpha=0.7)
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                    ax.set_xlabel('SHAP Value (Impact on Fraud Probability)', fontsize=10)
                    ax.set_title('Feature Contributions to Fraud Detection', fontsize=12, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Add value labels
                    for i, v in enumerate(df_shap['SHAP Value']):
                        ax.text(v, i, f' {v:.3f}', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Explanation
                    st.info("""
                    **How to read this chart:**
                    - 🔴 Red bars (positive) → Increase fraud probability
                    - 🟢 Green bars (negative) → Decrease fraud probability
                    - Longer bars = stronger impact on the prediction
                    """)
            else:
                st.warning("SHAP values not available for this prediction")
            
            # Model runtime
            runtime_ms = fusion_expl.get("runtime_ms", 0)
            st.caption(f"⏱️ Model inference time: {runtime_ms:.2f}ms")
            
            # Detailed explanations
            st.markdown("---")
            st.subheader("📋 Detailed Explanations")
            
            # Document Forensics Breakdown
            with st.expander("📄 Document Forensics Details", expanded=False):
                st.write("**Analysis:**", d.get("doc_explanation", "N/A"))
                doc_data = d.get('document_details', {})
                if doc_data:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("ELA Score", f"{doc_data.get('ela_score', 0):.3f}")
                    with col_b:
                        st.metric("CNN Forgery Prob", f"{doc_data.get('clf_prob', 0):.1%}")
            
            st.write("**Liveness Check:**", d.get("liveness_explanation", "N/A"))
            st.write("**Face Embedding:**", d.get("embed_explanation", "N/A"))
            
            # Display heatmap if available
            if d.get("heatmap"):
                st.markdown("---")
                st.subheader("🔥 ELA Heatmap (Document Tampering Detection)")
                try:
                    img_resp = requests.get(d["heatmap"])
                    img = Image.open(io.BytesIO(img_resp.content))
                    st.image(img, caption="Error Level Analysis Heatmap", use_container_width=True)
                except:
                    st.warning("Could not load heatmap image")
        else:
            st.error(f"Error from backend: {resp.status_code}")
