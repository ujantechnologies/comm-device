import sqlite3
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DB_PATH = Path("artifacts/comm_device.sqlite3")
MODEL_PATH = Path("models/classifier.pkl")


def load_rows() -> tuple[np.ndarray, np.ndarray]:
    if not DB_PATH.exists():
        return np.empty((0, 2)), np.empty((0,))

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT frame_id, confidence, COALESCE(corrected_label, predicted_label)
            FROM predictions
            """
        ).fetchall()

    if not rows:
        return np.empty((0, 2)), np.empty((0,))

    feats = np.array([[float(r[0]), float(r[1])] for r in rows], dtype=np.float32)
    labels = np.array([str(r[2]) for r in rows])
    return feats, labels


def main() -> int:
    x, y = load_rows()
    if len(x) < 10:
        print("Not enough labeled samples yet. Need at least 10.")
        return 1

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True)),
    ])
    model.fit(x, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved classifier to {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
