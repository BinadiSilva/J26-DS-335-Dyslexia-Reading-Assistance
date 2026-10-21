"""SQLite persistence with parameterized queries and immutable assessment history."""

from contextlib import contextmanager
from datetime import date
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from werkzeug.security import generate_password_hash, check_password_hash

from .schema import day, number, text, validate_session

DDL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS learners (
    id TEXT PRIMARY KEY, alias TEXT NOT NULL, age INTEGER NOT NULL, grade INTEGER NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('synthetic','collected')),
    target_speed REAL NOT NULL, target_accuracy REAL NOT NULL, target_fluency REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('teacher','parent')),
    learner_id TEXT REFERENCES learners(id),
    CHECK(role='teacher' OR learner_id IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY, learner_id TEXT NOT NULL REFERENCES learners(id),
    assessed_on TEXT NOT NULL, payload TEXT NOT NULL,
    UNIQUE(learner_id, assessed_on)
);
CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY, learner_id TEXT NOT NULL REFERENCES learners(id),
    started_on TEXT NOT NULL, baseline_session_id INTEGER NOT NULL REFERENCES sessions(id),
    plan TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
    outcome_session_id INTEGER REFERENCES sessions(id), adherence REAL, outcome_notes TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_plan ON interventions(learner_id) WHERE status='active';
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, learner_id TEXT NOT NULL REFERENCES learners(id),
    author TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, body TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript(DDL)
        self.path.chmod(0o600)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_learner(self, data, learner_id=None):
        source = data.get("source")
        if source not in {"synthetic", "collected"}:
            raise ValueError("Choose synthetic or collected data provenance.")
        learner = {
            "id": learner_id or uuid4().hex[:12],
            "alias": text(data.get("alias"), "Learner alias", 80),
            "age": number(data.get("age"), "Age", 4, 18, True),
            "grade": number(data.get("grade"), "Grade", 1, 13, True),
            "source": source,
            "target_speed": number(data.get("target_speed"), "Speed goal", 1, 300),
            "target_accuracy": number(data.get("target_accuracy"), "Accuracy goal", 1, 100),
            "target_fluency": number(data.get("target_fluency"), "Fluency goal", 1, 100),
        }
        with self.connection() as db:
            db.execute("INSERT INTO learners VALUES (:id,:alias,:age,:grade,:source,:target_speed,:target_accuracy,:target_fluency)", learner)
        return learner

    def learners(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM learners ORDER BY alias")]

    def learner(self, learner_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM learners WHERE id=?", (learner_id,)).fetchone()
        if row is None:
            raise LookupError("Learner not found.")
        return dict(row)

    def create_user(self, username, password, role, learner_id=None):
        username = text(username, "Username", 80)
        if not isinstance(password, str) or len(password) < 12:
            raise ValueError("Use a password with at least 12 characters.")
        if role not in {"teacher", "parent"} or (role == "parent" and not learner_id):
            raise ValueError("Parents must be assigned to a learner.")
        with self.connection() as db:
            db.execute("INSERT INTO users VALUES (?,?,?,?)", (username, generate_password_hash(password), role, learner_id))

    def user(self, username):
        with self.connection() as db:
            row = db.execute("SELECT username,role,learner_id FROM users WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None

    def authenticate(self, username, password):
        with self.connection() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            return self.user(username)
        return None

    def add_session(self, learner_id, data):
        self.learner(learner_id)
        validated = validate_session(data)
        with self.connection() as db:
            try:
                result = db.execute("INSERT INTO sessions(learner_id,assessed_on,payload) VALUES (?,?,?)",
                                    (learner_id, validated["assessed_on"], json.dumps(validated)))
            except sqlite3.IntegrityError:
                raise ValueError("This learner already has an assessment on that date.") from None
        return result.lastrowid

    def sessions(self, learner_id):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM sessions WHERE learner_id=? ORDER BY assessed_on,id", (learner_id,)).fetchall()
        return [{**json.loads(row["payload"]), "id": row["id"], "learner_id": learner_id} for row in rows]

    def start_intervention(self, learner_id, plan):
        """Save an educator-approved plan and the exact baseline used to create it."""
        sessions = self.sessions(learner_id)
        if not sessions:
            raise ValueError("Record an assessment before starting an intervention.")
        approved = dict(plan)
        approved["activity"] = text(plan.get("activity"), "Activity", 1000)
        approved["minutes"] = number(plan.get("minutes"), "Session minutes", 1, 60, True)
        approved["frequency"] = number(plan.get("frequency"), "Sessions per week", 1, 7, True)
        approved["review_days"] = number(plan.get("review_days"), "Review days", 1, 90, True)
        if plan.get("target_skill") not in {"accuracy", "speed", "fluency"}:
            raise ValueError("Select accuracy, speed or fluency as the target skill.")
        started_on = day(plan.get("started_on"))
        if not sessions[-1]["assessed_on"] <= started_on <= date.today().isoformat():
            raise ValueError("The start date must be on/after the latest assessment and not in the future.")
        with self.connection() as db:
            try:
                result = db.execute("INSERT INTO interventions(learner_id,started_on,baseline_session_id,plan) VALUES (?,?,?,?)",
                                    (learner_id, started_on, sessions[-1]["id"], json.dumps(approved)))
            except sqlite3.IntegrityError:
                raise ValueError("Review and complete the active intervention before starting another.") from None
        return result.lastrowid

    def complete_intervention(self, learner_id, intervention_id, outcome_id, adherence, notes):
        adherence = number(adherence, "Adherence percent", 0, 100)
        with self.connection() as db:
            plan = db.execute("SELECT * FROM interventions WHERE id=? AND learner_id=?", (intervention_id, learner_id)).fetchone()
            outcome = db.execute("SELECT * FROM sessions WHERE id=? AND learner_id=?", (outcome_id, learner_id)).fetchone()
            if not plan or plan["status"] != "active":
                raise ValueError("Active intervention not found for this learner.")
            if not outcome or outcome["assessed_on"] <= plan["started_on"]:
                raise ValueError("Choose this learner's assessment after the intervention started.")
            db.execute("UPDATE interventions SET status='completed',outcome_session_id=?,adherence=?,outcome_notes=? WHERE id=?",
                       (outcome_id, adherence, str(notes)[:2000], intervention_id))

    def interventions(self, learner_id):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM interventions WHERE learner_id=? ORDER BY started_on,id", (learner_id,)).fetchall()
        return [{**dict(row), "plan": json.loads(row["plan"])} for row in rows]

    def add_note(self, learner_id, author, body):
        self.learner(learner_id)
        with self.connection() as db:
            db.execute("INSERT INTO notes(learner_id,author,body) VALUES (?,?,?)", (learner_id, author, text(body, "Note", 2000)))

    def notes(self, learner_id):
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM notes WHERE learner_id=? ORDER BY id DESC", (learner_id,))]
