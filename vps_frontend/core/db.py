"""
SupoClip — SQLite Task Queue Manager
Thread-safe database layer for managing video processing tasks.
"""

import sqlite3
import threading
import json
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "supoclip_tasks.db"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    UPLOADING = "uploading"
    TRANSCRIPTION = "transcription"
    AI_CURATING = "ai_curating"
    RUNPOD_RENDERING = "runpod_rendering"
    COMPLETED = "completed"
    FAILED = "failed"


STATUS_EMOJI = {
    TaskStatus.QUEUED: "🔄 Queued",
    TaskStatus.UPLOADING: "☁️ Uploading",
    TaskStatus.TRANSCRIPTION: "📝 Transcription",
    TaskStatus.AI_CURATING: "🧠 AI Curating",
    TaskStatus.RUNPOD_RENDERING: "🎬 RunPod Rendering",
    TaskStatus.COMPLETED: "✅ Completed",
    TaskStatus.FAILED: "❌ Failed",
}

STATUS_COLOR = {
    TaskStatus.QUEUED: "gray",
    TaskStatus.UPLOADING: "blue",
    TaskStatus.TRANSCRIPTION: "orange",
    TaskStatus.AI_CURATING: "purple",
    TaskStatus.RUNPOD_RENDERING: "blue",
    TaskStatus.COMPLETED: "green",
    TaskStatus.FAILED: "red",
}

_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database schema (idempotent)."""
    with _lock:
        conn = get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                file_size   INTEGER DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'queued',
                video_url   TEXT,
                runpod_job_id TEXT,
                ai_json     TEXT,
                output_urls TEXT,
                error_log   TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)
        """)
        conn.commit()
        conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def add_task(task_id: str, filename: str, file_size: int = 0) -> None:
    """Add a new task to the queue."""
    now = datetime.utcnow().isoformat()
    with _lock:
        conn = get_connection()
        conn.execute(
            """INSERT OR IGNORE INTO tasks
               (id, filename, file_size, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (task_id, filename, file_size, TaskStatus.QUEUED, now, now),
        )
        conn.commit()
        conn.close()


def update_status(
    task_id: str,
    status: TaskStatus,
    *,
    video_url: str = None,
    runpod_job_id: str = None,
    ai_json: list = None,
    output_urls: list = None,
    error_log: str = None,
) -> None:
    """Update a task's status and optional fields."""
    now = datetime.utcnow().isoformat()
    fields = ["status = ?", "updated_at = ?"]
    values = [status, now]

    if video_url is not None:
        fields.append("video_url = ?")
        values.append(video_url)
    if runpod_job_id is not None:
        fields.append("runpod_job_id = ?")
        values.append(runpod_job_id)
    if ai_json is not None:
        fields.append("ai_json = ?")
        values.append(json.dumps(ai_json, ensure_ascii=False))
    if output_urls is not None:
        fields.append("output_urls = ?")
        values.append(json.dumps(output_urls))
    if error_log is not None:
        fields.append("error_log = ?")
        values.append(error_log)

    values.append(task_id)
    sql = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?"

    with _lock:
        conn = get_connection()
        conn.execute(sql, values)
        conn.commit()
        conn.close()


def log_error(task_id: str, error: str) -> None:
    """Mark task as failed and save error message."""
    update_status(task_id, TaskStatus.FAILED, error_log=error)


def get_all_tasks() -> list[dict]:
    """Return all tasks ordered by created_at descending."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM tasks ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        if d.get("ai_json"):
            try:
                d["ai_json"] = json.loads(d["ai_json"])
            except Exception:
                pass
        if d.get("output_urls"):
            try:
                d["output_urls"] = json.loads(d["output_urls"])
            except Exception:
                pass
        result.append(d)
    return result


def get_task(task_id: str) -> dict | None:
    """Return a single task by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    if d.get("ai_json"):
        try:
            d["ai_json"] = json.loads(d["ai_json"])
        except Exception:
            pass
    return d


def delete_task(task_id: str) -> None:
    """Remove a task from the queue."""
    with _lock:
        conn = get_connection()
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        conn.close()


def clear_completed() -> int:
    """Remove all completed tasks. Returns count deleted."""
    with _lock:
        conn = get_connection()
        cur = conn.execute(
            "DELETE FROM tasks WHERE status IN ('completed', 'failed')"
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
    return count
