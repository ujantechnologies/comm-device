"""Offline retraining script for the head-pose expression classifier.

Run after accumulating feedback corrections in the SQLite database:

    python scripts/train_classifier.py

The script loads the feature vectors (stored as float32 BLOBs by data_store.py)
and trains an SVC with RBF kernel. The resulting classifier.pkl is picked up
automatically on the next app restart.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from comm_device.config import load_config
from comm_device.data_store import DataStore

_MIN_SAMPLES = 10


def main() -> int:
    cfg = load_config()
    store = DataStore(cfg.db_path)
    X, y = store.load_features_and_labels()

    if len(X) < _MIN_SAMPLES:
        print(f"Need at least {_MIN_SAMPLES} labeled samples; only {len(X)} found.")
        print("Keep using the device and pressing the feedback buttons to build up training data.")
        return 1

    print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(unique) > 1 else None
    )

    model: Pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
    ])
    model.fit(X_train, y_train)

    print("\nTest set report:")
    print(classification_report(y_test, model.predict(X_test)))

    out = Path(cfg.classifier_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"Saved classifier to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
