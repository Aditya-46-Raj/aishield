"""
Test the integrated document classifier
"""
import requests

# Test document analysis endpoint
print("Testing /analyze/document endpoint...")
url = "http://127.0.0.1:8000/analyze/document"

with open("samples/clean_id.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("\nResponse:")
    print(f"  ELA Score: {data.get('ela_score', 'N/A'):.4f}")
    print(f"  CNN Classifier Prob: {data.get('clf_prob', 'N/A'):.4f}")
    print(f"  Combined Doc Score: {data.get('doc_score', 'N/A'):.4f}")
    print(f"  Risk Level: {data.get('risk_level', 'N/A')}")
    print(f"\n  Explanation:")
    print(f"    {data.get('explanation', 'N/A')}")
else:
    print(f"Error: {response.text}")
