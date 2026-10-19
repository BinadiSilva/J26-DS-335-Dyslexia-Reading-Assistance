"""One-page cognitive profile for the child / parent view (pure Python, no Streamlit).

Shows assessment output only. Support intensity is a single label; recommendations
belong to Component 03 and are deliberately not shown here.
"""
from __future__ import annotations

import html

from . import copy, logic

esc = html.escape
SKILLS = ["phonological", "decoding", "fluency", "pronunciation"]


def compact(s: str) -> str:
    return " ".join(s.split())


def _card(title: str, body: str, cls: str = "") -> str:
    return f'<div class="gu-card {cls}"><h4>{esc(title)}</h4>{body}</div>'


def difficulty_card(s: dict) -> str:
    """Level only. Scores, percentages and certainty are teacher-only."""
    level = str(s.get("difficulty_level", "Moderate")).title()
    kind = {"Low": "g", "Moderate": "a", "High": "r"}.get(level, "a")
    track = "".join(f'<span class="step {k if l == level else ""}">{l}</span>'
                    for l, k in (("Low", "g"), ("Moderate", "a"), ("High", "r")))
    return _card("Overall reading difficulty",
                 f'<div class="big-row"><span class="lvl {kind}">{esc(level)}</span></div>'
                 f'<div class="steps">{track}</div>'
                 f'<p>{esc(copy.parent_level_message(level))}</p>')


def support_card(s: dict) -> str:
    label = logic.support_label(s.get("intervention_intensity", ""))
    kind = {"green": "g", "amber": "a", "red": "r"}[logic.SUPPORT_KIND[label]]
    return _card("Support needed", f'<div class="support {kind}">{esc(label)}</div>'
                 '<div class="gu-note">How much extra reading support this assessment points to. '
                 'Ideas for practice come from a different part of the system.</div>')


def strengths_card(s: dict) -> str:
    skills, relative = logic.strengths_for_kid(s)
    title = "Best skill in this session" if relative else "Strengths"
    items = "".join(f'<li>{copy.skill_emoji(k)} <b>{esc(copy.skill_nick(k))}</b>: {esc(copy.skill_blurb(k))}</li>' for k in skills)
    note = ('<div class="gu-note">No skill reached the strength level in this session; this is the strongest one.</div>'
            if relative else "")
    return _card(title, f"<ul>{items}</ul>{note}")


def weak_card(s: dict) -> str:
    main, others = logic.weak_skills_for_display(s)
    parts = []
    if main:
        parts.append(f'<div class="main-skill">{copy.skill_emoji(main)} <b>{esc(copy.skill_nick(main))}</b>'
                     f'<span>{esc(copy.skill_blurb(main))}</span></div>')
    if others:
        chips = "".join(f'<span class="chip-w">{copy.skill_emoji(k)} {esc(copy.skill_nick(k))}</span>' for k in others)
        parts.append(f'<div class="gu-note" style="margin-top:.5rem">Also below the target level</div><div class="chips2">{chips}</div>')
    if not parts:
        parts.append('<p>No weak skills were found in this session.</p>')
    return _card("Skills to build", "".join(parts))


def profile_card(s: dict) -> str:
    scores = s.get("cognitive_profile", {}).get("subskill_scores", {})
    rows = []
    for k in SKILLS:
        v = max(0.0, min(100.0, float(scores.get(k, 0))))
        kind = "g" if v >= 75 else ("a" if v >= 60 else "r")
        rows.append(f'<div class="sk"><div class="skt"><span>{copy.skill_emoji(k)} {esc(copy.skill_nick(k))}</span><span>{ {"g": "Strength", "a": "Developing", "r": "Skill to build"}[kind] }</span></div>'
                    f'<div class="skb"><div class="skf {kind}" style="width:{v}%"></div></div></div>')
    stars_n = logic.stars_from_index(logic.prosody_index(s))
    stars = "★" * stars_n + "☆" * (5 - stars_n)
    return _card("Cognitive profile",
                 "".join(rows) + '<div class="gu-note">Longer bars mean a stronger skill. Bars in green are strengths; red bars are skills to build.</div>'
                 f'<div class="expr"><span>Reading expression</span><b>{stars}</b></div>'
                 '<div class="gu-note">A gentle indicator of expression and rhythm. Indicative only; not clinically validated.</div>')


def why_card(s: dict) -> str:
    shap = s.get("shap_top_features", {})
    a = "".join(f"<li>{esc(p)}</li>" for p in copy.explanation_phrases(shap, "difficulty_score"))
    b = "".join(f"<li>{esc(p)}</li>" for p in copy.explanation_phrases(shap, "weak_skill"))
    return _card("Why the Lumi-IT says this?",
                 f"<p><b>The difficulty result</b> was mostly influenced by:</p><ul>{a}</ul>"
                 f"<p><b>The skill to build</b> was mostly influenced by:</p><ul>{b}</ul>"
                 '<div class="gu-note">These are the patterns that stood out in this session. They describe what was noticed, not a cause.</div>')


def profile_body(info: dict, s: dict, session_line: str) -> str:
    review = ""
    if s.get("intervention", {}).get("requires_professional_review"):
        review = '<div class="review">We suggest sharing this profile with your child\'s specialist.</div>'
    head = (f'<div class="idrow"><div><b>{esc(info["name"])}</b> <span>ID {esc(logic.display_id(info["child_id"]))}</span></div>'
            f'<div><span>Age {esc(str(info["age"]))} · Grade {esc(str(info["grade"]))}</span></div>'
            f'<div><b>{esc(session_line)}</b></div></div>')
    grid = (f'<div class="pgrid">{difficulty_card(s)}{support_card(s)}'
            f'{strengths_card(s)}{weak_card(s)}{profile_card(s)}{why_card(s)}</div>')
    return compact(head + review + grid)
