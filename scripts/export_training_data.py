import csv
import sqlite3
from pathlib import Path

DB_PATH = Path("artifacts/comm_device.sqlite3")
OUT_PATH = Path("artifacts/predictions.csv")


def main() -> int:
    if not DB_PATH.exists():
        print("Database not found")
        return 1

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, created_at, frame_id, predicted_label, confidence, corrected_label FROM predictions"
        ).fetchall()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "frame_id", "predicted_label", "confidence", "corrected_label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
