# Test SHAP values format
import joblib
import numpy as np
import shap

# Load model
model = joblib.load("models/fusion_lgb.joblib")
explainer = shap.TreeExplainer(model)

# Test features - must be numpy array!
features = np.array([[0.005851703956850734, 0.2, 0.14355364441871643, 0.0]])

# Get SHAP values
shap_values = explainer.shap_values(features)

print("Type of shap_values:", type(shap_values))
print("Is list?", isinstance(shap_values, list))
if isinstance(shap_values, list):
    print("Length:", len(shap_values))
    for i, sv in enumerate(shap_values):
        print(f"  shap_values[{i}] type:", type(sv), "shape:", sv.shape if hasattr(sv, 'shape') else 'N/A')
else:
    print("Shape:", shap_values.shape if hasattr(shap_values, 'shape') else 'N/A')

print("\nExpected value:", explainer.expected_value)
print("Type:", type(explainer.expected_value))

# Try to extract
if isinstance(shap_values, list):
    print("\nUsing shap_values[0]:")
    print(shap_values[0])
