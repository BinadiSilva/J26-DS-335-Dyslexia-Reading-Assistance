"""Dashboard checks. Part A is pure Python (fast). Part B drives the real Streamlit app with AppTest."""
import re
import sys

sys.path.insert(0, ".")

from frontend import copy, kid_html, logic
from frontend import teacher_html as T

BANNED = re.compile(r"\b(confidence|shap|log-odds|xgboost|ebm|random ?forest)\b", re.I)


def visible_text(doc: str) -> str:
    doc = re.sub(r"<style.*?</style>", " ", doc, flags=re.S)
    doc = re.sub(r"<[^>]+>", " ", doc)
    return " ".join(doc.split())


def test_pure():
    from frontend import parent_html
    sessions = logic.load_sessions()
    assert sessions, "no sessions found: run run_demo_payloads.py"
    labels = {"Low support required", "Medium support required", "High support required"}
    for s in sessions:
        for target, feats in s["shap_top_features"].items():
            for name, _v in feats:
                phrase = copy.parent_feature_phrase(name)
                assert phrase.strip() and "_" not in phrase, name
        info = logic.child_info(s["child_id"], {}, [s])
        line = logic.session_title([s], s)

        # --- child / parent one-page profile -----------------------------------------
        page = parent_html.profile_body(info, s, line)
        txt = visible_text(page)
        assert not BANNED.search(txt), (s["session_id"], BANNED.search(txt).group(0))
        assert "_" not in txt, s["session_id"]
        assert "demo_child" not in txt and "Child A" not in txt
        # required sections are all on the one page
        for needle in ["Overall reading difficulty", "Support needed", "Skills to build",
                       "Cognitive profile", "Why the Lumi-IT says this?", logic.display_id(info["child_id"]), line]:
            assert needle in txt, (s["session_id"], needle)
        assert "Strengths" in txt or "Best skill in this session" in txt
        # exactly one support label; no recommendations or activities anywhere
        assert sum(txt.count(l) for l in labels) == 1, (s["session_id"], txt)
        for act in s.get("intervention", {}).get("recommended_activities", []):
            assert act not in txt
        assert "Next" not in txt and "practice plan" not in txt.lower()
        # numbers and certainty are teacher-only
        assert "%" not in txt and "Score" not in txt and "certain" not in txt.lower(), (s["session_id"], txt)
        assert not re.search(r"\b0\.\d\d\b", visible_text(parent_html.difficulty_card(s)))
        assert 8 <= info["age"] <= 10 and 3 <= info["grade"] <= 5
        hero = visible_text(kid_html.profile_hero(info["name"], line, s["difficulty_level"]))
        assert not BANNED.search(hero)
        assert logic.support_label(s["intervention_intensity"]) in labels

        # --- teacher fragments -------------------------------------------------------
        html_kpi = T.kpi_cards(s)
        assert "%" in html_kpi and "Support intensity" in html_kpi
        assert logic.support_label(s["intervention_intensity"]) in html_kpi
        flags = s["low_confidence_flags"]
        if flags["weak_skill"] or flags["intervention_intensity"]:
            assert "Low confidence" in html_kpi
        T.skill_bars(s["cognitive_profile"]["subskill_scores"])
        for key in ("difficulty_score", "weak_skill", "intervention_intensity"):
            assert "<img" in T.signed_bars_svg(s["shap_top_features"][key])

    # --- overview, per-student session lists, Component 04 box ------------------------
    ov = logic.build_overview(sessions, {})
    assert ov and all(r["name"] for r in ov)
    by = logic.group_by_child(sessions)
    for r in ov:
        sess = by[r["child_id"]]
        tbl = visible_text(T.session_table(sess, sess[-1]["session_id"]))
        assert all(logic.display_id(s_["session_id"]) in tbl for s_ in sess)
        assert not any(a in tbl for s_ in sess for a in s_.get("intervention", {}).get("recommended_activities", []))
        assert logic.session_title(sess, sess[-1]) .startswith(f"Session {len(sess):02d}")
    # background details (age, grade, gender, passage) never appear as parent-facing reasons
    ph = copy.explanation_phrases({"weak_skill": [["age", .9], ["grade", .8], ["gender_Male", .7], ["word_error_rate", .5]]}, "weak_skill")
    assert ph == [copy.FEATURE_PHRASES["word_error_rate"]], ph
    c4 = visible_text(T.component4_box("Lara", "Session 01 · 2026-10-05"))
    assert "Component 04" in c4 and "Pending integration" in c4 and "No Component 04 results" in c4
    ct = visible_text(T.class_table(ov))
    assert "demo_child" not in ct and "C001" in ct
    # demo children have real first names and C00x ids; sessions use Sxx to avoid prefix clashes
    assert [logic.display_name(f"demo_child_0{i}") for i in (1, 2, 3)] == ["Lara", "Krish", "Harsha"]
    assert logic.display_id("demo_child_01") == "C001" and logic.display_id("demo_child_02-S03") == "S03"
    assert "demo_child" not in visible_text(T.session_table(by["demo_child_01"], None)) if "demo_child_01" in by else True
    assert logic.support_label("Moderate") == "Medium support required"
    cert, c = logic.certainty({"weak_skill_confidence": 0.9, "intervention_confidence": 0.55})
    assert cert == "Low" and abs(c - 0.55) < 1e-9
    print("Part A (pure) passed:", len(sessions), "sessions,", len(ov), "children")


def test_app():
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:  # pragma: no cover
        print("Part B skipped (AppTest unavailable):", e)
        return

    def all_text(at):
        parts = [getattr(e, "value", "") for e in at.markdown] + [getattr(e, "value", "") for e in at.caption] + \
                [getattr(e, "value", "") for e in at.info]
        return " ".join(p for p in parts if p)

    # teacher: students -> sessions -> profile -> Component 04 box
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label.startswith("Log in as teacher")).click()
    at.run()
    assert not at.exception, at.exception
    txt = visible_text(all_text(at))
    for needle in ["Teacher dashboard", "Assessment sessions", "Cognitive profile", "Learning Progress Prediction", "Pending integration"]:
        assert needle in txt, needle

    # child A, then child B: each sees their own profile on one page
    seen = {}
    for idx in (0, 1):
        at = AppTest.from_file("frontend/app.py", default_timeout=60)
        at.run()
        sel = next(s for s in at.selectbox if s.key == "login_child")
        label = sel.options[idx]                      # e.g. "Lara  (C001)"
        sel.select_index(idx)
        at.run()
        next(b for b in at.button if b.label.startswith("Log in as child")).click()
        at.run()
        assert not at.exception, at.exception
        txt = visible_text(all_text(at))
        assert "Overall reading difficulty" in txt and "Support needed" in txt and "Skills to build" in txt
        assert not BANNED.search(txt), BANNED.search(txt).group(0)
        assert not any(b.label.strip() in ("Next", "Next ▶") for b in at.button)
        seen[idx] = label
        assert label.split("  (")[0] in txt, (label, txt[:300])
        assert "demo_child" not in txt and "Open Learning Plan" in " ".join(b.label for b in at.button)
        assert any(b.label.startswith("← Back") for b in at.button)
    assert seen[0] != seen[1]
    print("Part B (Streamlit app) passed")


if __name__ == "__main__":
    test_pure()
    test_app()
    print("Dashboard checks passed.")
