"""Kho SQLite tối thiểu cho sensor event, risk prediction và technician review."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteRiskEventStore:
    """Lưu risk event bền vững qua restart; không retrain trực tiếp từ review."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured_path = db_path or os.getenv(
            "RISK_EVENT_DB_PATH", "data/runtime/risk_events.sqlite3"
        )
        self.db_path = str(configured_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sensor_events (
                    event_id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    shift TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS risk_predictions (
                    event_id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    reliability_status TEXT NOT NULL,
                    action TEXT NOT NULL,
                    queue_eligible INTEGER NOT NULL,
                    observed_conditions_json TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES sensor_events(event_id)
                );
                CREATE TABLE IF NOT EXISTS maintenance_reviews (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    technician_action TEXT,
                    confirmed_issue INTEGER,
                    failure_mode TEXT,
                    notes TEXT,
                    reviewed_at TEXT,
                    FOREIGN KEY(event_id) REFERENCES sensor_events(event_id)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(sensor_events)"
                ).fetchall()
            }
            if "shift" not in columns:
                connection.execute("ALTER TABLE sensor_events ADD COLUMN shift TEXT")

    def record_event(self, event: dict[str, Any], payload: dict[str, Any]) -> None:
        """Ghi sensor event và risk prediction cùng một transaction."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sensor_events(event_id, asset_id, event_time, shift, payload_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    asset_id = excluded.asset_id,
                    event_time = excluded.event_time,
                    shift = excluded.shift,
                    payload_json = excluded.payload_json
                """,
                (
                    event["event_id"],
                    event["asset_id"],
                    event["event_time"],
                    event.get("shift"),
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
            connection.execute(
                """
                INSERT INTO risk_predictions(
                    event_id, model_version, policy_version, risk_score,
                    reliability_status, action, queue_eligible, observed_conditions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    model_version = excluded.model_version,
                    policy_version = excluded.policy_version,
                    risk_score = excluded.risk_score,
                    reliability_status = excluded.reliability_status,
                    action = excluded.action,
                    queue_eligible = excluded.queue_eligible,
                    observed_conditions_json = excluded.observed_conditions_json
                """,
                (
                    event["event_id"],
                    event["model_version"],
                    event["policy_version"],
                    event["risk_score"],
                    event["reliability_status"],
                    event["action"],
                    int(event["queue_eligible"]),
                    json.dumps(
                        event.get("observed_conditions", []), ensure_ascii=False
                    ),
                ),
            )

    def list_risk_events(self) -> list[dict[str, Any]]:
        """Đọc các risk event để dựng queue sau khi API restart."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.event_id, s.asset_id, s.event_time, s.shift,
                       r.model_version, r.policy_version, r.risk_score,
                       r.reliability_status, r.action, r.queue_eligible,
                       r.observed_conditions_json
                FROM sensor_events s
                JOIN risk_predictions r ON r.event_id = s.event_id
                ORDER BY s.event_time ASC
                """
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "asset_id": row["asset_id"],
                "event_time": row["event_time"],
                "shift": row["shift"],
                "model_version": row["model_version"],
                "policy_version": row["policy_version"],
                "risk_score": row["risk_score"],
                "reliability_status": row["reliability_status"],
                "action": row["action"],
                "queue_eligible": bool(row["queue_eligible"]),
                "observed_conditions": json.loads(row["observed_conditions_json"]),
            }
            for row in rows
        ]

    def record_review(
        self,
        event_id: str,
        asset_id: str,
        technician_action: str,
        confirmed_issue: bool,
        failure_mode: str | None = None,
        notes: str | None = None,
        reviewed_at: str | None = None,
    ) -> None:
        """Lưu kết quả kỹ thuật viên; dữ liệu này chỉ phục vụ QA và retrain offline."""
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sensor_events WHERE event_id = ? AND asset_id = ?",
                (event_id, asset_id),
            ).fetchone()
            if exists is None:
                raise ValueError("Không tìm thấy sensor event tương ứng với review.")
            connection.execute(
                """
                INSERT INTO maintenance_reviews(
                    event_id, asset_id, technician_action, confirmed_issue,
                    failure_mode, notes, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    asset_id,
                    technician_action,
                    int(confirmed_issue),
                    failure_mode,
                    notes,
                    reviewed_at,
                ),
            )
