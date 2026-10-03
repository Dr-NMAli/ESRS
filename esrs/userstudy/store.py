"""SQLite storage for participant responses."""
from __future__ import annotations
import os, sqlite3, time, uuid, json


SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    pid          TEXT PRIMARY KEY,
    condition    TEXT NOT NULL,
    started_at   REAL NOT NULL,
    finished_at  REAL,
    age          INTEGER,
    gender       TEXT,
    country      TEXT,
    shopping_freq TEXT,
    sustainability_attitude INTEGER,
    device       TEXT
);
CREATE TABLE IF NOT EXISTS impressions (
    pid          TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    position     INTEGER,
    was_green    INTEGER,
    explanation_type TEXT,
    PRIMARY KEY (pid, task_id, item_id)
);
CREATE TABLE IF NOT EXISTS clicks (
    pid          TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    clicked_at   REAL,
    PRIMARY KEY (pid, task_id, item_id)
);
CREATE TABLE IF NOT EXISTS ratings (
    pid          TEXT NOT NULL,
    metric       TEXT NOT NULL,      -- PI / Trust / Transparency /
                                     -- Persuasiveness / Specificity /
                                     -- Coherence / Fidelity
    value        INTEGER NOT NULL,   -- 1..5
    PRIMARY KEY (pid, metric)
);
"""


class Store:
    def __init__(self, path="esrs_userstudy.sqlite"):
        self.path = path
        new = not os.path.exists(path)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        if new:
            print(f"[store] created {path}")

    # ---- participants ----
    def create_participant(self, condition):
        pid = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO participants(pid, condition, started_at) VALUES (?,?,?)",
            (pid, condition, time.time()))
        self.conn.commit()
        return pid

    def get_condition(self, pid):
        row = self.conn.execute(
            "SELECT condition FROM participants WHERE pid=?", (pid,)).fetchone()
        return row[0] if row else None

    def finalize_participant(self, pid, demographics):
        self.conn.execute(
            "UPDATE participants SET finished_at=?, age=?, gender=?, country=?, "
            "shopping_freq=?, sustainability_attitude=?, device=? WHERE pid=?",
            (time.time(),
             demographics.get("age"),
             demographics.get("gender"),
             demographics.get("country"),
             demographics.get("shopping_freq"),
             demographics.get("sustainability_attitude"),
             demographics.get("device"),
             pid))
        self.conn.commit()

    # ---- impressions ----
    def record_impression(self, pid, task_id, item):
        self.conn.execute(
            "INSERT OR REPLACE INTO impressions"
            "(pid, task_id, item_id, position, was_green, explanation_type) "
            "VALUES (?,?,?,?,?,?)",
            (pid, task_id, item["item_id"], item.get("position", 0),
             int(item["eco"] > 0.7), item.get("explanation_type")))
        self.conn.commit()

    # ---- clicks ----
    def record_click(self, pid, task_id, item_id):
        self.conn.execute(
            "INSERT OR IGNORE INTO clicks(pid, task_id, item_id, clicked_at) "
            "VALUES (?,?,?,?)",
            (pid, task_id, item_id, time.time()))
        self.conn.commit()

    # ---- ratings ----
    def record_rating(self, pid, metric, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO ratings(pid, metric, value) VALUES (?,?,?)",
            (pid, metric, int(value)))
        self.conn.commit()

    # ---- dump ----
    def as_dataframes(self):
        import pandas as pd
        parts = {t: pd.read_sql_query(f"SELECT * FROM {t}", self.conn)
                 for t in ("participants", "impressions", "clicks", "ratings")}
        return parts