"""Teacher / specialist view: students -> assessment sessions -> cognitive profile of that session,
then a placeholder for Component 04's learning-progress prediction."""
from __future__ import annotations

import streamlit as st

from . import data, logic
from . import teacher_html as T
from .theme import TEACHER_CSS


def _md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


def render_teacher_view() -> None:
    _md(f"<style>{TEACHER_CSS}</style>")
    rows = data.get_children()
    if not rows:
        st.warning("No sessions found. Run run_demo_payloads.py first.")
        return

    _md('<div class="t-title">Teacher dashboard</div>'
        '<div class="t-sub">Students, their assessment sessions and each session\'s cognitive profile. Decision support only; not a diagnosis.</div>')
    _md(T.stat_strip([
        ("Students", str(len(rows)), "with assessments"),
        ("Sessions", str(sum(r["n_sessions"] for r in rows)), "assessed in total"),
        ("Review needed", str(sum(r["review"] for r in rows)), "latest session flagged"),
        ("Improving", str(sum(r["trend"] == "improving" for r in rows)), "difficulty trending down"),
    ]))

    # 1) students
    st.markdown("#### 1 · Students")
    ids = [r["child_id"] for r in rows]
    names = {r["child_id"]: f'{r["name"]}  ({logic.display_id(r["child_id"])})' for r in rows}
    cid = st.selectbox("Select a student", ids, format_func=lambda k: names.get(k, k), key="t_child")
    _md(T.class_table(rows, cid))

    # 2) sessions of that student
    sessions = data.get_child_sessions(cid)
    info = data.get_child_info(cid)
    summ = data.get_summary(cid)
    st.markdown(f"#### 2 · Assessment sessions for {info['name']}")
    sids = [s["session_id"] for s in sessions]
    sid = st.radio("Select a session", sids, index=len(sids) - 1, horizontal=True, key=f"t_sess_{cid}",
                   format_func=lambda k: (lambda s: logic.session_title(sessions, s) if s else k)(next((s for s in sessions if s["session_id"] == k), None)))
    s = next(x for x in sessions if x["session_id"] == sid)
    line = logic.session_title(sessions, s)
    _md(T.session_table(sessions, sid))

    # 3) cognitive profile of that session
    st.markdown(f"#### 3 · Cognitive profile · {info['name']} · {line}")
    iv = s.get("intervention", {})
    _md(T.child_header(info, summ, bool(iv.get("requires_professional_review"))))
    if iv.get("requires_professional_review"):
        _md(T.banner("red", "Professional review suggested", iv.get("review_reasons", [])))
    _md(T.kpi_cards(s))

    tabs = st.tabs(["Cognitive profile", "Why this prediction", "Progress over time", "Prosody", "Notes"])

    with tabs[0]:
        prof = s.get("cognitive_profile", {})
        agree = prof.get("model_rule_agreement")
        _md('<div class="grid2"><div class="t-card"><h4>Sub-skill scores</h4>' + T.skill_bars(prof.get("subskill_scores", {})) + '</div>'
            '<div class="t-card"><h4>Strengths and weaknesses</h4>'
            '<div class="note" style="margin-top:.2rem">Strengths</div>' + T.chips(prof.get("strengths", []), "green", "None above the strength threshold") +
            '<div class="note" style="margin-top:.8rem">Weaknesses</div>' + T.chips(prof.get("weaknesses", []), "red", "None below the weakness threshold") +
            '<div class="note" style="margin-top:.8rem">Model focus area vs lowest rule-based score</div>' +
            f'<div style="margin-top:.3rem"><b>{str(prof.get("primary_focus_area", "")).title()}</b> vs <b>{str(prof.get("rule_based_weakest_area", "")).title()}</b> ' +
            (T.pill("Agree", "green") if agree else T.pill("Different", "amber")) + '</div>'
            '<p class="note">These are two different methods (a learned model and fixed thresholds), so they can differ.</p></div></div>')

    with tabs[1]:
        shap = s.get("shap_top_features", {})
        specs = [("weak_skill", "Weak skill", "Model log-odds: positive values push towards the predicted skill."),
                 ("intervention_intensity", "Support intensity", "Model log-odds: positive values push towards the predicted level."),
                 ("difficulty_score", "Difficulty score", "Change in the predicted difficulty score (small numbers, different scale).")]
        for key, title, note in specs:
            _md(f'<div class="t-card"><h4>{title}</h4>' + T.signed_bars_svg([(a, b) for a, b in shap.get(key, [])]) +
                f'<div class="note">{note} Each chart has its own scale; do not compare sizes across charts.</div></div>')

    with tabs[2]:
        labels = [f"S{i + 1}" for i in range(len(sessions))]
        _md('<div class="grid2"><div class="t-card"><h4>Difficulty score over time</h4>' +
            T.line_chart_svg([float(x.get("difficulty_score", 0)) for x in sessions], labels,
                             bands=[(0, .26, "#e3f6ec", "Low"), (.26, .5, "#fff1d6", "Moderate"), (.5, 1, "#fde4e4", "High")]) + '</div>'
            '<div class="t-card"><h4>Reading expression over time</h4>' +
            T.line_chart_svg([logic.prosody_index(x) for x in sessions], labels, ymin=0, ymax=100, fmt="{:.0f}", color="#6b5bd6") + '</div></div>')
        delta = summ["delta_vs_previous"]
        _md(f'<div class="t-card"><h4>Trend across sessions</h4>{T.trend_pill(summ["trend"])} &nbsp; Change since previous session: <b>{delta:+.3f}</b> '
            f'&nbsp; Most frequent weak skill: <b>{str(summ["most_frequent_weak_skill"]).title()}</b>'
            '<h4 style="margin-top:1rem">Weak skill per session</h4>' + T.session_strip(sessions, "weak_skill") +
            '<h4 style="margin-top:1rem">Support intensity per session</h4>' + T.session_strip(sessions, "intervention_intensity") + '</div>')

    with tabs[3]:
        pi = s.get("prosody_engagement_index", {})
        _md('<div class="t-card"><h4>Reading expression (prosody engagement index)</h4>'
            f'<p style="font-size:1.6rem;font-weight:900;margin:.2rem 0">{pi.get("prosody_engagement_index", "n/a")}<span class="note"> / 100</span></p>' +
            T.component_bars(pi.get("components", {})) +
            f'<p class="note">{pi.get("note", "Rule-based indicator; not clinically validated.")}</p></div>')

    with tabs[4]:
        notes = data.load_teacher_notes()
        text = st.text_area(f"Professional notes for {info['name']}", value=notes.get(cid, ""), key=f"t_notes_{cid}", height=110)
        if st.button("Save notes", key=f"t_save_{cid}"):
            notes[cid] = text
            data.save_teacher_notes(notes)
            st.success("Notes saved.")

    # 4) what happens next in the full system
    _md(T.component4_box(info["name"], line))
    st.caption("Decision support only; not a diagnosis. Prototype built on synthetic demonstration data.")
