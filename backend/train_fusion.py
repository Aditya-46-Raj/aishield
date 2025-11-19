# backend/train_fusion.py
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

try:
    import lightgbm as lgb
except ImportError:
    print("Error: lightgbm not installed. Run: pip install lightgbm")
    exit(1)

# Synthetic dataset generator
def make_synthetic_data(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    
    # Generate clean samples (70% of data)
    n_clean = int(n * 0.7)
    n_fraud = n - n_clean
    
    # Clean samples: low doc_score, high liveness, high embed_sim, no behavior anomaly
    doc_clean = np.clip(rng.normal(0.02, 0.015, n_clean), 0, 0.1)
    live_clean = np.clip(rng.normal(0.85, 0.1, n_clean), 0, 1)
    embed_clean = np.clip(rng.normal(0.85, 0.12, n_clean), 0.3, 1)
    behavior_clean = rng.binomial(1, 0.02, n_clean)
    
    # Fraud samples: high doc_score OR low embed_sim OR behavior anomaly
    doc_fraud = np.clip(rng.normal(0.12, 0.05, n_fraud), 0, 1)
    live_fraud = np.clip(rng.normal(0.3, 0.2, n_fraud), 0, 1)
    embed_fraud = np.clip(rng.normal(0.25, 0.2, n_fraud), 0, 0.7)
    behavior_fraud = rng.binomial(1, 0.3, n_fraud)
    
    # Combine
    doc = np.concatenate([doc_clean, doc_fraud])
    live = np.concatenate([live_clean, live_fraud])
    embed = np.concatenate([embed_clean, embed_fraud])
    behavior = np.concatenate([behavior_clean, behavior_fraud])
    y = np.concatenate([np.zeros(n_clean), np.ones(n_fraud)])
    
    # Shuffle
    idx = rng.permutation(n)
    
    df = pd.DataFrame({
        'doc_score': doc[idx],
        'liveness_score': live[idx],
        'embed_sim': embed[idx],
        'behavior_anomaly': behavior[idx],
        'label': y[idx]
    })
    return df

def train_and_save(path="models/fusion_lgb.joblib"):
    print("Generating synthetic training data...")
    df = make_synthetic_data(2000)
    X = df[['doc_score','liveness_score','embed_sim','behavior_anomaly']]
    y = df['label']
    
    print(f"Dataset shape: {X.shape}, Fraud rate: {y.mean():.2%}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective':'binary',
        'metric':'auc',
        'verbosity': -1,
        'boosting_type':'gbdt',
        'num_leaves':31,
        'learning_rate':0.05,
    }
    
    print("Training LightGBM model...")
    bst = lgb.train(
        params, 
        train_data, 
        num_boost_round=200, 
        valid_sets=[val_data], 
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )
    
    # evaluate
    preds = bst.predict(X_val)
    auc_score = roc_auc_score(y_val, preds)
    print(f"Validation AUC: {auc_score:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(bst, path)
    print(f"✓ Model saved to {path}")
    
    return bst, auc_score

if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()