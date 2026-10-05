import os
import sqlite3
import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                study_hours REAL NOT NULL,
                attendance REAL NOT NULL,
                assignment_score REAL NOT NULL,
                previous_marks REAL NOT NULL,
                mock_test_score REAL NOT NULL,
                predicted_marks REAL NOT NULL,
                min_range REAL NOT NULL,
                max_range REAL NOT NULL,
                performance_level TEXT NOT NULL
            );
        """)
        conn.commit()

def save_prediction(record: dict) -> int:
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO prediction_history (
                timestamp, study_hours, attendance, assignment_score,
                previous_marks, mock_test_score, predicted_marks,
                min_range, max_range, performance_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            record["study_hours"],
            record["attendance"],
            record["assignment_score"],
            record["previous_marks"],
            record["mock_test_score"],
            record["predicted_marks"],
            record["prediction_range"]["min"],
            record["prediction_range"]["max"],
            record["performance_level"]
        ))
        conn.commit()
        return cursor.lastrowid

def get_recent_predictions(limit: int = 50) -> list[dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

def clear_predictions() -> int:
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM prediction_history")
        conn.commit()
        return cursor.rowcount
