"""
Simulate Streamlit output for documentation
"""
import json

# Load test results
with open("outputs/test_clean_sample.json", "r") as f:
    clean_data = json.load(f)

with open("outputs/test_forged_sample.json", "r") as f:
    forged_data = json.load(f)

print("="*80)
print("STREAMLIT INTERFACE SIMULATION")
print("="*80)

def display_result(data, label):
    print(f"\n{'='*80}")
    print(f"{label.upper()}")
    print(f"{'='*80}\n")
    
    fraud_score = data['fused_fraud_prob']
    
    # Risk level
    if fraud_score < 0.3:
        risk = "✅ LOW RISK"
    elif fraud_score < 0.6:
        risk = "⚠️ MEDIUM RISK"
    else:
        risk = "🚨 HIGH RISK"
    
    print(f"🎯 Final Fraud Probability: {risk} - {fraud_score:.1%}\n")
    
    # Component scores
    print("Component Scores:")
    print(f"  Document Score:    {data.get('doc_score', 0):.3f}")
    print(f"  Liveness Score:    {data.get('liveness_score', 0):.3f}")
    print(f"  Face Similarity:   {data.get('embed_sim', 0):.3f}\n")
    
    # SHAP values
    print("🔍 SHAP Explainability - Feature Contributions:")
    shap_values = data.get('fusion_explanation', {}).get('shap_values', {})
    
    if shap_values and 'error' not in shap_values:
        print(json.dumps(shap_values, indent=2))
        
        # Feature importance summary
        print("\n📊 Feature Impact Summary:")
        shap_features = {k: v for k, v in shap_values.items() if k != "base_value"}
        sorted_features = sorted(shap_features.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feat, val in sorted_features:
            direction = "🔴 Increases" if val > 0 else "🟢 Decreases"
            print(f"  {feat:20s}: {val:+.4f}  {direction} fraud probability")
    
    # Runtime
    runtime = data.get('fusion_explanation', {}).get('runtime_ms', 0)
    print(f"\n⏱️ Model inference time: {runtime:.2f}ms")
    
    # Explanations
    print(f"\n📋 Detailed Explanations:")
    print(f"  Document: {data.get('doc_explanation', 'N/A')}")
    print(f"  Liveness: {data.get('liveness_explanation', 'N/A')}")
    print(f"  Face:     {data.get('embed_explanation', 'N/A')}")
    
    if data.get('notes'):
        print(f"\n⚠️ Notes: {data['notes']}")

# Display both cases
display_result(clean_data, "Clean Sample (clean_id.jpg + selfie.jpg)")
display_result(forged_data, "Forged Sample (forged_demo.jpg + selfie.jpg)")

print("\n" + "="*80)
print("SHAP BAR CHART DESCRIPTION")
print("="*80)
print("""
The Streamlit interface displays a horizontal bar chart showing:
- Feature names on Y-axis (doc_score, liveness_score, embed_sim, behavior_anomaly)
- SHAP values on X-axis (contribution to fraud probability)
- Red bars (positive values) → Increase fraud probability
- Green bars (negative values) → Decrease fraud probability
- Vertical line at 0 separates positive/negative contributions
- Longer bars indicate stronger impact on the final prediction
""")
