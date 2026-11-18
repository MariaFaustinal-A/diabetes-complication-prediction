"""Local SQLite prediction history (history.db next to this file)."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

DB = Path(__file__).with_name("history.db")


def _q(sql, args=(), fetch=None):
    with closing(sqlite3.connect(DB)) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, args)
        out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else cur.lastrowid
        c.commit()
        return out


def init():
    _q("""CREATE TABLE IF NOT EXISTS predictions(
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, disease TEXT, patient TEXT,
          probability REAL, band TEXT, label TEXT, inputs TEXT)""")


def save(disease, patient, probability, band, label, inputs):
    return _q("INSERT INTO predictions(created_at,disease,patient,probability,band,label,inputs) VALUES(?,?,?,?,?,?,?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M"), disease, patient or "", probability, band, label,
               json.dumps(inputs)))


def all_rows(disease=None, patient=None):
    sql, args = "SELECT * FROM predictions WHERE 1=1", []
    if disease:
        sql += " AND disease=?"; args.append(disease)
    if patient:
        sql += " AND patient LIKE ?"; args.append(f"%{patient}%")
    return _q(sql + " ORDER BY id DESC", tuple(args), "all")


def get(pid):
    return _q("SELECT * FROM predictions WHERE id=?", (pid,), "one")


def delete(pid):
    _q("DELETE FROM predictions WHERE id=?", (pid,))


def clear():
    _q("DELETE FROM predictions")
