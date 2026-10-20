"""Pure-Python data helpers for the demo frontend (no Streamlit imports, easy to test)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

NAMES = [
    "Ava", "Leo", "Mila", "Noah", "Zoe", "Eli", "Sora", "Theo", "Rina", "Kai", "Maya", "Omar",
    "Isla", "Ravi", "Luna", "Finn", "Anya", "Jude", "Nila", "Ezra", "Tara", "Milo", "Aria", "Sam",
]
SKILLS = ["phonological", "decoding", "fluency", "pronunciation"]
STRONG = 75
WEAK = 60


def load_sessions(log_path: str = "outputs/session_log.jsonl",
                  fallback: str = "outputs/session_outputs.json") -> list[dict]:
    p = Path(log_path)
    if p.exists() and p.stat().st_size > 0:
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    f = Path(fallback)
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    return []


def load_demo_children(path: str = "frontend/demo_children.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def group_by_child(sessions: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in sessions:
        out.setdefault(str(s.get("child_id")), []).append(s)
    for rows in out.values():
        rows.sort(key=lambda r: (str(r.get("timestamp", "")), str(r.get("session_id", ""))))
    return dict(sorted(out.items()))


def _num(child_id: str) -> int:
    m = re.search(r"(\d+)", child_id)
    return int(m.group(1)) if m else 1


DEMO_NAMES = {1: "Lara", 2: "Krish", 3: "Harsha"}


def display_id(raw_id: str) -> str:
    """Render all user-facing IDs in a clear, non-conflicting scheme:

    - child ids: demo_child_01 -> C001
    - session ids: demo_child_01-S02 -> S02
    - other ids stay unchanged
    """
    raw = str(raw_id)
    child = re.fullmatch(r"demo_child_(\d+)", raw)
    if child:
        return f"C{int(child.group(1)):03d}"

    session = re.fullmatch(r"demo_child_(\d+)-S(\d+)", raw)
    if session:
        return f"S{int(session.group(2)):02d}"

    return raw


def display_name(child_id: str) -> str:
    """demo_child_01 -> 'Lara', demo_child_02 -> 'Krish', demo_child_03 -> 'Harsha'; others get a friendly first name."""
    m = re.fullmatch(r"demo_child_(\d+)", str(child_id))
    if m and int(m.group(1)) in DEMO_NAMES:
        return DEMO_NAMES[int(m.group(1))]
    n = _num(str(child_id))
    return NAMES[(n - 1) % len(NAMES)]


def child_info(child_id: str, demo: dict, sessions: list[dict] | None = None) -> dict:
    d = demo.get(child_id, {})
    n = _num(child_id)
    summary = (sessions[-1].get("input_summary") if sessions else None) or {}
    # target group: ages 8-10, grades 3-5
    age = d.get("age") or summary.get("age") or (8 + n % 3)
    grade = d.get("grade") or summary.get("grade") or (int(age) - 5)
    return {
        "child_id": child_id,
        "display_id": display_id(child_id),
        "name": d.get("first_name") or display_name(child_id),
        "age": age,
        "grade": grade,
    }


# ---- support intensity: the only wording Component 2 shows (Component 03 owns recommendations) ----
SUPPORT_LABELS = {"low": "Low support required", "moderate": "Medium support required",
                  "medium": "Medium support required", "high": "High support required"}
SUPPORT_KIND = {"Low support required": "green", "Medium support required": "amber", "High support required": "red"}


def support_label(intensity: str) -> str:
    return SUPPORT_LABELS.get(str(intensity).strip().lower(), "Medium support required")


def session_number(sessions: list[dict], session_id: str) -> int:
    for i, s in enumerate(sessions, 1):
        if s.get("session_id") == session_id:
            return i
    return len(sessions)


def session_title(sessions: list[dict], session: dict) -> str:
    return f"Session {session_number(sessions, session.get('session_id')):02d} · {session.get('timestamp', '')}"


def certainty(session: dict) -> tuple[str, float]:
    """Plain-language certainty = the less certain of the two classifiers (weak skill, support level).
    The difficulty score is a regression, so it has no probability of its own."""
    c = min(float(session.get("weak_skill_confidence", 0.0)), float(session.get("intervention_confidence", 0.0)))
    return ("High" if c >= 0.8 else "Medium" if c >= 0.6 else "Low"), c


def weak_skills_for_display(session: dict) -> tuple[str | None, list[str]]:
    """(model-predicted main area, other skills under the weakness threshold)."""
    main = str(session.get("weak_skill") or "").lower() or None
    others = [w for w in (session.get("cognitive_profile", {}).get("weaknesses") or []) if w != main]
    return main, others


def trend_label(scores: list[float]) -> str:
    """Same rule as pipeline.longitudinal_summary: change over the last 3 sessions, +-0.05."""
    recent = scores[-3:]
    if len(recent) < 2:
        return "stable"
    change = recent[-1] - recent[0]
    if change <= -0.05:
        return "improving"
    if change >= 0.05:
        return "worsening"
    return "stable"


def summarize_child(sessions: list[dict]) -> dict:
    scores = [float(s.get("difficulty_score", 0.0)) for s in sessions]
    counts = Counter(str(s.get("weak_skill", "unknown")) for s in sessions)
    return {
        "n_sessions": len(sessions),
        "scores": scores,
        "trend": trend_label(scores),
        "delta_vs_previous": (scores[-1] - scores[-2]) if len(scores) >= 2 else 0.0,
        "most_frequent_weak_skill": counts.most_common(1)[0][0] if counts else None,
    }


def build_overview(sessions: list[dict], demo: dict) -> list[dict]:
    rows = []
    for cid, rs in group_by_child(sessions).items():
        info = child_info(cid, demo, rs)
        latest = rs[-1]
        summ = summarize_child(rs)
        rows.append({
            **info,
            "latest": latest,
            "n_sessions": summ["n_sessions"],
            "trend": summ["trend"],
            "review": bool(latest.get("intervention", {}).get("requires_professional_review")),
        })
    return rows


def strengths_for_kid(session: dict) -> tuple[list[str], bool]:
    """Strengths above the threshold; if none, the best skill relative to the others.
    Returns (skills, is_relative)."""
    prof = session.get("cognitive_profile", {})
    strengths = list(prof.get("strengths") or [])
    if strengths:
        return strengths[:2], False
    scores = prof.get("subskill_scores") or {}
    if not scores:
        return ["fluency"], True
    best = max(scores, key=lambda k: float(scores[k]))
    return [best], True


def stars_from_index(index: float) -> int:
    return int(min(5, max(1, round(float(index) / 20))))


def prosody_index(session: dict) -> float:
    p = session.get("prosody_engagement_index", {})
    v = p.get("prosody_engagement_index") if isinstance(p, dict) else None
    return float(v) if v is not None else 0.0
