"""
Trains a gradient boosting (XGBoost) classifier that predicts the
probability a shot attempt is made, given its distance, angle, defender
distance, shot clock, and time remaining -- i.e. a "shot quality" score.

Usage:
    python train.py --data ../data/processed/features.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS

MODEL_PATH = "../models/shot_quality_model.json"
METRICS_PATH = "../models/metrics.json"


def train(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["made"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = {
        "val_accuracy": accuracy_score(y_val, val_preds),
        "val_auc": roc_auc_score(y_val, val_probs),
        "val_log_loss": log_loss(y_val, val_probs),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "base_rate": float(y.mean()),
    }

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.astype(float)))
    metrics["feature_importances"] = {k: round(v, 4) for k, v in importances.items()}

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train the shot quality model")
    parser.add_argument("--data", default="../data/processed/features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model, metrics = train(df)

    model.save_model(MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Validation accuracy: {metrics['val_accuracy']:.1%}")
    print(f"Validation AUC:      {metrics['val_auc']:.3f}")
    print(f"Validation log loss: {metrics['val_log_loss']:.3f}")
    print("Feature importances:")
    for feat, imp in sorted(metrics["feature_importances"].items(), key=lambda kv: -kv[1]):
        print(f"  {feat:<20} {imp:.3f}")


if __name__ == "__main__":
    main()
