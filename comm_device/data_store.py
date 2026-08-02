from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from .expression import ExpressionLabel

logger = logging.getLogger(__name__)


class DataStore:
    """SQLite-backed store for predictions, feedback corrections, and LLM responses.

    The `features` column stores raw numpy float32 arrays as BLOBs so the
    offline train_classifier.py script can reconstruct feature vectors without
    re-running MediaPipe inference.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._shared_conn: Optional[sqlite3.Connection] = None
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            # Keep a single shared connection for in-memory databases so all
            # calls see the same schema and data.
            self._shared_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._shared_conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at     TEXT    DEFAULT CURRENT_TIMESTAMP,
                    frame_id       INTEGER NOT NULL,
                    predicted_label TEXT   NOT NULL,
                    confidence     REAL    NOT NULL,
                    corrected_label TEXT,
                    features       BLOB
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_responses (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at     TEXT    DEFAULT CURRENT_TIMESTAMP,
                    trigger_label  TEXT    NOT NULL,
                    response_text  TEXT    NOT NULL,
                    rating         INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_videos (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at        TEXT    DEFAULT CURRENT_TIMESTAMP,
                    intent            TEXT    NOT NULL,
                    question_text     TEXT    NOT NULL,
                    file_path         TEXT    NOT NULL,
                    frame_count       INTEGER NOT NULL,
                    duration_seconds  REAL    NOT NULL
                )
                """
            )

    def save_prediction(
        self,
        frame_id: int,
        label: ExpressionLabel,
        confidence: float,
        corrected_label: Optional[ExpressionLabel] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        feat_blob = features.astype(np.float32).tobytes() if features is not None else None
        corrected = corrected_label.value if corrected_label is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (frame_id, predicted_label, confidence, corrected_label, features)
                VALUES (?, ?, ?, ?, ?)
                """,
                (frame_id, label.value, confidence, corrected, feat_blob),
            )

    def save_response(
        self, trigger_label: ExpressionLabel, response_text: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO llm_responses (trigger_label, response_text) VALUES (?, ?)",
                (trigger_label.value, response_text),
            )

    def load_features_and_labels(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) from labeled rows for offline retraining."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT features, COALESCE(corrected_label, predicted_label)
                FROM predictions
                WHERE features IS NOT NULL
                """
            ).fetchall()

        if not rows:
            return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=str)

        xs = [np.frombuffer(r[0], dtype=np.float32) for r in rows]
        ys = [str(r[1]) for r in rows]
        return np.stack(xs), np.array(ys)

    def save_training_video(
        self,
        intent: str,
        question_text: str,
        file_path: str,
        frame_count: int,
        duration_seconds: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_videos
                    (intent, question_text, file_path, frame_count, duration_seconds)
                VALUES (?, ?, ?, ?, ?)
                """,
                (intent, question_text, file_path, int(frame_count), float(duration_seconds)),
            )

    def get_latest_training_video(self) -> Optional[dict[str, object]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, created_at, intent, question_text, file_path, frame_count, duration_seconds
                FROM training_videos
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        return {
            "id": int(row[0]),
            "created_at": str(row[1]),
            "intent": str(row[2]),
            "question_text": str(row[3]),
            "file_path": str(row[4]),
            "frame_count": int(row[5]),
            "duration_seconds": float(row[6]),
        }

    def delete_training_video(self, video_id: int) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM training_videos WHERE id = ?",
                (int(video_id),),
            ).fetchone()
            if row is None:
                return None
            file_path = str(row[0])
            conn.execute("DELETE FROM training_videos WHERE id = ?", (int(video_id),))
            return file_path

    def list_training_videos(self) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, intent, question_text, file_path, frame_count, duration_seconds
                FROM training_videos
                ORDER BY id DESC
                """
            ).fetchall()

        return [
            {
                "id": int(row[0]),
                "created_at": str(row[1]),
                "intent": str(row[2]),
                "question_text": str(row[3]),
                "file_path": str(row[4]),
                "frame_count": int(row[5]),
                "duration_seconds": float(row[6]),
            }
            for row in rows
        ]

    def delete_all_training_videos(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT file_path FROM training_videos").fetchall()
            conn.execute("DELETE FROM training_videos")
        return [str(row[0]) for row in rows]
