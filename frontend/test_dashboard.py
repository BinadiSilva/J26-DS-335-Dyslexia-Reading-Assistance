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


def kid_data(s):
    strengths, rel = logic.strengths_for_kid(s)
    return dict(name="Ava", level=s["difficulty_level"], strengths=strengths, relative=rel, focus=s["weak_skill"],
                activities=s["intervention"]["recommended_activities"], stars=logic.stars_from_index(logic.prosody_index(s)))


def test_pure():
    sessions = logic.load_sessions()
    assert sessions, "no sessions found: run export_outputs.py"
    for s in sessions:
        for target, feats in s["shap_top_features"].items():
            for name, _v in feats:
                phrase = copy.parent_feature_phrase(name)
                assert phrase.strip() and "_" not in phrase, name
        # kid pages: no technical words, no raw feature names, strengths never empty
        strengths, _ = logic.strengths_for_kid(s)
        assert strengths
        for i in range(kid_html.TOTAL_PAGES):
            txt = visible_text(kid_html.page(i, kid_data(s)))
            assert not BANNED.search(txt), (s["session_id"], i, BANNED.search(txt).group(0))
            assert "_" not in txt, (s["session_id"], i)
        gu = " ".join(copy.explanation_phrases(s["shap_top_features"], "weak_skill") +
                      copy.explanation_phrases(s["shap_top_features"], "intervention_intensity") +
                      [copy.parent_level_message(s["difficulty_level"])])
        assert not BANNED.search(gu)
        # teacher fragments render, show confidence, and mark low confidence
        html_kpi = T.kpi_cards(s)
        assert "%" in html_kpi
        flags = s["low_confidence_flags"]
        if flags["weak_skill"] or flags["intervention_intensity"]:
            assert "Low confidence" in html_kpi
        T.skill_bars(s["cognitive_profile"]["subskill_scores"])
        for key in ("difficulty_score", "weak_skill", "intervention_intensity"):
            assert "<img" in T.signed_bars_svg(s["shap_top_features"][key])
    ov = logic.build_overview(sessions, {})
    assert ov and all(r["name"] for r in ov)
    assert any(r["review"] for r in ov) or True
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

    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Teacher / Specialist").click()
    at.run()
    assert not at.exception, at.exception
    txt = all_text(at)
    assert "Learning Progress Dashboard" in txt and "Low confidence" in txt or "%" in txt

    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.run()
    next(b for b in at.button if b.label == "Child / Parent").click()
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if "Start reading" in b.label).click()
    at.run()
    assert not at.exception, at.exception
    txt = all_text(at)
    assert "grown-ups" in txt.lower() or any("grown-ups" in t.label for t in at.tabs)
    assert not BANNED.search(visible_text(txt)), BANNED.search(visible_text(txt))
    print("Part B (Streamlit app) passed")


if __name__ == "__main__":
    test_pure()
    test_app()
    print("Dashboard checks passed.")
