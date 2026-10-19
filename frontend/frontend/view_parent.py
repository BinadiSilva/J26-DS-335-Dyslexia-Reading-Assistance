"""Child / parent view: a short listening animation, then ONE page with the complete cognitive profile."""
from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

from . import data, kid_html, logic, parent_html
from .theme import PARENT_CSS

HERO_H = 300


def render_parent_view() -> None:
    st.markdown(f"<style>{PARENT_CSS}</style>", unsafe_allow_html=True)
    ss = st.session_state
    cid = ss.get("child_id")
    sessions = data.get_child_sessions(cid) if cid else []
    if not sessions:
        st.warning("No sessions found for this learner. Run run_demo_payloads.py first.")
        return
    info = data.get_child_info(cid)

    # short "Lumi is listening" animation once per login
    if not ss.get("p_seen_listen"):
        components.html(kid_html.listening_page(info["name"]), height=520)
        time.sleep(2.2)
        ss.p_seen_listen = True
        st.rerun()

    # latest session by default; the session history is optional
    ids = [s["session_id"] for s in sessions]
    if len(ids) > 1:
        sid = st.selectbox("Session", ids, index=len(ids) - 1, key=f"p_session_{cid}",
                           format_func=lambda k: (lambda s: logic.session_title(sessions, s) if s else k)(next((s for s in sessions if s["session_id"] == k), None)))
    else:
        sid = ids[0]
    session = next(s for s in sessions if s["session_id"] == sid)
    line = logic.session_title(sessions, session)

    components.html(kid_html.profile_hero(info["name"], line, session.get("difficulty_level", "Moderate")), height=HERO_H)
    st.markdown(parent_html.profile_body(info, session, line), unsafe_allow_html=True)

    if st.button("Open Learning Plan  ▶", key="p_plan"):
        ss.p_plan_open = not ss.get("p_plan_open", False)
    if ss.get("p_plan_open"):
        st.info(f"The learning plan for {info['name']} is created by the Recommendations part of the system "
                f"(Component 03), using the assessment shown above. It is not part of this page.")
    st.caption("Decision support only; not a diagnosis. Prototype built on synthetic demonstration data.")
