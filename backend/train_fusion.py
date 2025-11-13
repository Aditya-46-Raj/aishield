# backend/train_fusion.py
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Synthetic dataset generator
def make_synthetic_data(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    # features: doc_score (ELA), liveness_score, embed_sim, behavior_anomaly (0/1)
    doc = np.clip(rng.normal(0.05, 0.03, n), 0, 1)
    live = np.clip(rng.normal(0.15, 0.1, n), 0, 1)
    embed = np.clip(rng.normal(0.8, 0.15, n), 0, 1)
    behavior = rng.binomial(1, 0.05, n)
    # label: 1=fraud, 0=clean (simulate higher doc + low embed + behavior anomaly -> fraud)
    prob = 0.2*doc + 0.5*(1-embed) + 0.6*behavior + 0.1*(1-live)
    prob = np.clip(prob, 0, 1)
    y = rng.binomial(1, prob)
    df = pd.DataFrame({
        'doc_score': doc,
        'liveness_score': live,
        'embed_sim': embed,
        'behavior_anomaly': behavior,
        'label': y
    })
    return df

def train_and_save(path="models/fusion_lgb.joblib"):
    df = make_synthetic_data(2000)
    X = df[['doc_score','liveness_score','embed_sim','behavior_anomaly']]
    y = df['label']
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
    bst = lgb.train(params, train_data, num_boost_round=200, valid_sets=[val_data], early_stopping_rounds=20, verbose_eval=False)
    # evaluate
    preds = bst.predict(X_val)
    print("Val AUC:", roc_auc_score(y_val, preds))
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(bst, path)
    print("Saved model to", path)

if __name__ == "__main__":
    train_and_save()
