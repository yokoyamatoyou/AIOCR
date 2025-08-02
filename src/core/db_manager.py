from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


class DBManager:
    """Simple SQLite wrapper for OCR results."""

    def __init__(self, db_path: str = "database/ocr_results.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def initialize(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_jobs (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                roi_name TEXT NOT NULL,
                text_mini TEXT,
                text_nano TEXT,
                final_text TEXT,
                confidence_score REAL,
                status TEXT,
                corrected_by_user INTEGER DEFAULT 0,
                FOREIGN KEY(job_id) REFERENCES ocr_jobs(job_id)
            )
            """
        )
        self.conn.commit()

    def create_job(self, template_name: str, created_at: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO ocr_jobs (template_name, created_at) VALUES (?, ?)",
            (template_name, created_at),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_result(
        self,
        job_id: int,
        image_name: str,
        roi_name: str,
        text_mini: str | None = None,
        text_nano: str | None = None,
        final_text: str | None = None,
        confidence_score: float | None = None,
        status: str | None = None,
        corrected_by_user: bool = False,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO ocr_results (
                job_id, image_name, roi_name, text_mini, text_nano,
                final_text, confidence_score, status, corrected_by_user
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                image_name,
                roi_name,
                text_mini,
                text_nano,
                final_text,
                confidence_score,
                status,
                int(corrected_by_user),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def fetch_results(self, job_id: int) -> Iterable[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM ocr_results WHERE job_id = ?", (job_id,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def update_result(self, result_id: int, new_text: str, status: str = "confirmed") -> None:
        """Update the text of a result and mark it as corrected.

        Parameters
        ----------
        result_id:
            Primary key of the ``ocr_results`` row to update.
        new_text:
            Replacement text supplied by the human reviewer.
        status:
            Optional new status for the result.  Defaults to ``"confirmed"``
            to indicate that the human reviewer has verified the value.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE ocr_results
            SET final_text = ?, corrected_by_user = 1, status = ?
            WHERE result_id = ?
            """,
            (new_text, status, result_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
