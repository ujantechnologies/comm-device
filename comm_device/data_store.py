import sqlite3
from pathlib import Path
from .expression import ExpressionLabel


class DataStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    frame_id INTEGER NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    corrected_label TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    trigger_label TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    rating INTEGER
                )
                """
            )

    def save_prediction(self, frame_id: int, label: ExpressionLabel, confidence: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions (frame_id, predicted_label, confidence) VALUES (?, ?, ?)",
                (frame_id, label.value, confidence),
            )

    def save_response(self, trigger_label: ExpressionLabel, response_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO llm_responses (trigger_label, response_text) VALUES (?, ?)",
                (trigger_label.value, response_text),
            )
