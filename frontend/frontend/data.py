"""Streamlit-cached data access. All real logic lives in logic.py (pure Python)."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from . import logic

NOTES_PATH = Path("outputs/teacher_notes.json")


@st.cache_resource
def load_model_artifacts():
    from backend.app.services.reading_difficulty.pipeline import load_artifacts
    return load_artifacts("outputs/artifacts.joblib")


def run_new_session(raw_row: dict) -> dict:
    from backend.app.services.reading_difficulty.pipeline import predict_session
    return predict_session(load_model_artifacts(), raw_row)


@st.cache_data
def get_sessions() -> list[dict]:
    return logic.load_sessions()


@st.cache_data
def get_children() -> list[dict]:
    """One row per child: info + latest session + trend + review flag."""
    return logic.build_overview(get_sessions(), logic.load_demo_children())


@st.cache_data
def get_child_sessions(child_id: str) -> list[dict]:
    return logic.group_by_child(get_sessions()).get(str(child_id), [])


def get_child_info(child_id: str) -> dict:
    return logic.child_info(str(child_id), logic.load_demo_children(), get_child_sessions(str(child_id)))


def get_summary(child_id: str) -> dict:
    return logic.summarize_child(get_child_sessions(child_id))


def load_teacher_notes() -> dict:
    if not NOTES_PATH.exists():
        return {}
    try:
        return json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_teacher_notes(notes: dict) -> None:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(json.dumps(notes, indent=2), encoding="utf-8")
