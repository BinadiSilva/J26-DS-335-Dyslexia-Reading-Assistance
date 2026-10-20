"""Baselines vs trained models on the SAME learner-level hold-out split (proposal section 3.3 / SO3 / Table 5).

Run from the repo root (inside Docker):
    python evaluate_baselines.py
    python evaluate_baselines.py --raw datasets/comp02_raw_reading_sessions.csv --artifacts outputs/artifacts.joblib

What it reports (hold-out learners only, never seen in training):
  A) difficulty_score  : RMSE / MAE / R2 for the baselines and (if artifacts exist) the trained model
  B) difficulty level  : accuracy, macro precision / recall / F1, confusion matrix (Low / Moderate / High)
  C) weak_skill        : majority baseline, rule-based weakest sub-skill, and the trained model
  D) intervention      : majority baseline, conventional-measures tree, and the trained model
Baselines:
  - Majority class / mean predictor  (no information)
  - Conventional measures only       (accuracy, reading speed, fluency: what a teacher reads off a session)
  - Rule-based weakest sub-skill     (profiling.subskill_scores; for weak_skill only)
Writes outputs/evaluation_report.md so the numbers can go straight into the report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score)
from sklearn.tree import DecisionTreeClassifier

from backend.app.services.reading_difficulty.features import (
    apply_grade_norms, clean, engineer_features, fit_grade_norms, load_raw,
    stratified_learner_level_split,
)
from backend.app.services.reading_difficulty.profiling import subskill_scores

LEVELS = ["Low", "Moderate", "High"]
SKILLS = ["phonological", "decoding", "fluency", "pronunciation"]
LOW_CUT, HIGH_CUT = 0.26, 0.50          # same thresholds as pipeline._difficulty_level_from_score
SEED = 42


def level_from_score(x: float) -> str:
    return "Low" if x < LOW_CUT else ("Moderate" if x < HIGH_CUT else "High")


def conventional(df: pd.DataFrame) -> pd.DataFrame:
    """What a teacher can read off one session without any ML: accuracy, pace, fluency."""
    return pd.DataFrame({
        "accuracy": df["words_correct"] / df["words_total"],
        "reading_speed_wpm": df["reading_speed_wpm"],
        "fluency_score": df["fluency_score"],
    })


def reg_metrics(y, p) -> dict:
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p))}


def cls_metrics(y, p, labels) -> dict:
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    return {"Accuracy": float(accuracy_score(y, p)), "Precision (macro)": float(pr),
            "Recall (macro)": float(rc), "F1 (macro)": float(f1)}


def align(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0
    return out[cols]


def md_table(rows: list[dict], first: str = "Model") -> str:
    if not rows:
        return "_none_\n"
    cols = list(rows[0].keys())
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def md_confusion(cm, labels, title) -> str:
    head = "| actual \\ predicted | " + " | ".join(labels) + " |\n|" + "---|" * (len(labels) + 1) + "\n"
    body = "".join(f"| {l} | " + " | ".join(str(int(v)) for v in cm[i]) + " |\n" for i, l in enumerate(labels))
    return f"**{title}**\n\n{head}{body}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/comp02_raw_reading_sessions.csv")
    ap.add_argument("--artifacts", default="outputs/artifacts.joblib")
    ap.add_argument("--out", default="outputs/evaluation_report.md")
    a = ap.parse_args()

    df = engineer_features(clean(load_raw(a.raw)))
    train, test = stratified_learner_level_split(df, stratify_col="intervention_intensity")
    norms = fit_grade_norms(train)
    train, test = apply_grade_norms(train, norms), apply_grade_norms(test, norms)
    assert not set(train["child_id"]) & set(test["child_id"]), "learner leakage between train and test"
    n_children = test["child_id"].nunique()
    print(f"hold-out: {len(test)} sessions from {n_children} learners (train: {len(train)} sessions, "
          f"{train['child_id'].nunique()} learners)")

    Xc_tr, Xc_te = conventional(train), conventional(test)
    y_tr, y_te = train["difficulty_score"], test["difficulty_score"]
    lvl_tr, lvl_te = train["difficulty_level"], test["difficulty_level"]
    report = [f"# Evaluation report (hold-out learners only)\n",
              f"Hold-out: **{len(test)} sessions from {n_children} learners** (never used in training). "
              f"With so few learners these figures are indicative, not precise. Data are synthetic: this validates "
              f"the pipeline, not real-world accuracy.\n"]

    # ---------------- helpers for trained-model predictions ----------------
    def trained_pred_cls(target):
        t = art[target]
        raw = t["model"].predict(align(test, t["feature_columns"]))
        return [t["class_labels"][int(v)] if not isinstance(v, str) else v for v in raw]

    # ---------------- A/B: difficulty score and the derived Low/Moderate/High level ----------------
    reg_rows, lvl_rows, cms = [], [], {}

    def add_reg(name, pred):
        pred = np.asarray(pred, dtype=float)
        reg_rows.append({"Model": name, **reg_metrics(y_te, pred)})
        plev = [level_from_score(v) for v in pred]
        lvl_rows.append({"Model": name, **cls_metrics(lvl_te, plev, LEVELS)})
        cms[name] = confusion_matrix(lvl_te, plev, labels=LEVELS)

    add_reg("Baseline: mean score", DummyRegressor().fit(Xc_tr, y_tr).predict(Xc_te))
    add_reg("Baseline: conventional measures (linear)", LinearRegression().fit(Xc_tr, y_tr).predict(Xc_te))

    art = None
    if Path(a.artifacts).exists():
        import joblib
        art = joblib.load(a.artifacts)
        t = art["difficulty_score"]
        add_reg(f"Trained model: {t['best_model_name']}", t["model"].predict(align(test, t["feature_columns"])))
    else:
        print(f"(no {a.artifacts}: trained-model rows skipped; run training first)")

    # A constant predictor gives the same result whether it predicts the mean score or the majority level,
    # so only one row is shown (no duplicate rows).
    lvl_rows = [r for r in lvl_rows if r["Model"] != "Baseline: mean score"]
    cms.pop("Baseline: mean score", None)
    maj = DummyClassifier(strategy="most_frequent").fit(Xc_tr, lvl_tr)
    lvl_rows.insert(0, {"Model": "Baseline: majority level", **cls_metrics(lvl_te, maj.predict(Xc_te), LEVELS)})
    cms["Baseline: majority level"] = confusion_matrix(lvl_te, maj.predict(Xc_te), labels=LEVELS)
    tree = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(Xc_tr, lvl_tr)
    lvl_rows.insert(1, {"Model": "Baseline: conventional measures (depth-2 tree)", **cls_metrics(lvl_te, tree.predict(Xc_te), LEVELS)})
    cms["Baseline: conventional measures (depth-2 tree)"] = confusion_matrix(lvl_te, tree.predict(Xc_te), labels=LEVELS)

    report += ["## A. Difficulty score (regression)\n", md_table(reg_rows),
               "## B. Difficulty level (Low / Moderate / High)\n", md_table(lvl_rows)]
    for name, cm in cms.items():
        report.append(md_confusion(cm, LEVELS, f"Difficulty level, confusion matrix: {name}"))

    # ---------------- C: weak skill ----------------
    ws_rows = []
    y_ws = test["weak_skill"]
    ws_maj = DummyClassifier(strategy="most_frequent").fit(Xc_tr, train["weak_skill"])
    ws_rows.append({"Model": "Baseline: majority skill", **cls_metrics(y_ws, ws_maj.predict(Xc_te), SKILLS)})
    rule = [min((s := subskill_scores(r)), key=s.get) for _, r in test.iterrows()]
    ws_rows.append({"Model": "Baseline: rule-based weakest sub-skill", **cls_metrics(y_ws, rule, SKILLS)})
    ws_cms = {"Baseline: rule-based weakest sub-skill": confusion_matrix(y_ws, rule, labels=SKILLS)}
    if art is not None:
        t = art["weak_skill"]
        pred_ws = trained_pred_cls("weak_skill")
        ws_rows.append({"Model": f"Trained model: {t['best_model_name']}", **cls_metrics(y_ws, pred_ws, SKILLS)})
        ws_cms[f"Trained model: {t['best_model_name']}"] = confusion_matrix(y_ws, pred_ws, labels=SKILLS)
    report += ["## C. Weak skill (4 classes)\n", md_table(ws_rows)]
    for name, cm in ws_cms.items():
        report.append(md_confusion(cm, SKILLS, f"Weak skill, confusion matrix: {name}"))

    # ---------------- D: intervention intensity ----------------
    iv_rows = []
    y_iv = test["intervention_intensity"]
    iv_rows.append({"Model": "Baseline: majority level", **cls_metrics(y_iv, maj.predict(Xc_te), LEVELS)})
    iv_tree = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(Xc_tr, train["intervention_intensity"])
    iv_rows.append({"Model": "Baseline: conventional measures (depth-2 tree)", **cls_metrics(y_iv, iv_tree.predict(Xc_te), LEVELS)})
    report += ["## D. Intervention intensity (3 classes)\n"]
    pred_iv = None
    if art is not None:
        pred_iv = trained_pred_cls("intervention_intensity")
        iv_rows.append({"Model": f"Trained model: {art['intervention_intensity']['best_model_name']}",
                        **cls_metrics(y_iv, pred_iv, LEVELS)})
    report.append(md_table(iv_rows))
    if pred_iv is not None:
        report.append(md_confusion(confusion_matrix(y_iv, pred_iv, labels=LEVELS), LEVELS,
                                   "Intervention intensity, confusion matrix: trained model"))
    report.append("_Note: the intervention-intensity label is generated as a separate support decision (reading "
                  "severity, how concentrated the weakness is, and a child-level support need), not copied from "
                  "difficulty level. Its accuracy reflects this synthetic support policy._\n")

    # ---------------- E: professional-review flag audit (uses the real predict_session output) ----------------
    if art is not None:
        from backend.app.services.reading_difficulty.pipeline import predict_session
        raw_all = clean(load_raw(a.raw)).set_index("session_id", drop=False)
        recs = []
        for i, sid in enumerate(test["session_id"], 1):
            out = predict_session(art, raw_all.loc[sid].to_dict())
            reasons = out["intervention"].get("review_reasons", []) or []
            recs.append({
                "session_id": sid,
                "flag": bool(out["intervention"]["requires_professional_review"]),
                "flag_non_auto": any("high intensity" not in r for r in reasons),
                "pred_level": out["difficulty_level"], "pred_int": out["intervention_intensity"],
                "pred_ws": out["weak_skill"], "reasons": reasons,
            })
            if i % 20 == 0:
                print(f"  audited {i}/{len(test)} sessions")
        au = pd.DataFrame(recs).merge(
            test[["session_id", "difficulty_level", "intervention_intensity", "weak_skill"]], on="session_id")
        au["lvl_err"] = au["pred_level"] != au["difficulty_level"]
        au["int_err"] = au["pred_int"] != au["intervention_intensity"]
        au["ws_err"] = au["pred_ws"] != au["weak_skill"]
        au["any_err"] = au[["lvl_err", "int_err", "ws_err"]].any(axis=1)

        def rate(mask, col):
            n = int(mask.sum())
            return (f"{int(au.loc[mask, col].sum())}/{n} ({au.loc[mask, col].mean():.0%})" if n else "n/a")

        report += ["## E. Professional-review flag audit\n",
                   "The flag is raised automatically whenever predicted intensity is High, so a wrong call that "
                   "ends in High is flagged by rule, which says nothing about the flag's usefulness. The "
                   "'non-automatic' column therefore ignores the high-intensity reason and counts only the other "
                   "triggers (low confidence, several weak areas, level/intensity disagreement).\n",
                   f"- Sessions flagged overall: **{int(au.flag.sum())}/{len(au)} ({au.flag.mean():.0%})**; "
                   f"non-automatic flags: **{int(au.flag_non_auto.sum())}/{len(au)} ({au.flag_non_auto.mean():.0%})**.\n"]
        rows = []
        for label, mask in [("Sessions with no error (level, intensity and weak skill all correct)", ~au.any_err),
                            ("Sessions with at least one wrong prediction", au.any_err),
                            ("Difficulty level wrong", au.lvl_err),
                            ("Intensity wrong", au.int_err),
                            ("Weak skill wrong", au.ws_err)]:
            rows.append({"Group": label, "Sessions": int(mask.sum()),
                         "Flagged (any reason)": rate(mask, "flag"),
                         "Flagged (non-automatic)": rate(mask, "flag_non_auto")})
        report.append(md_table(rows))
        sev = (au.difficulty_level == "High") & (au.pred_level != "High")
        esc = (au.intervention_intensity == "Moderate") & (au.pred_int == "High")
        esc_l = (au.difficulty_level == "Moderate") & (au.pred_level == "High")
        report += ["### Costly errors\n",
                   f"- Difficulty level: truly High predicted below High: **{int(sev.sum())} of "
                   f"{int((au.difficulty_level == 'High').sum())}**; flagged {rate(sev, 'flag')} "
                   f"(non-automatic {rate(sev, 'flag_non_auto')}).",
                   f"- Difficulty level: truly Moderate predicted High: **{int(esc_l.sum())} of "
                   f"{int((au.difficulty_level == 'Moderate').sum())}**.",
                   f"- Intensity: truly Moderate escalated to High: **{int(esc.sum())} of "
                   f"{int((au.intervention_intensity == 'Moderate').sum())}**; flagged {rate(esc, 'flag')} "
                   f"(always, by the high-intensity rule; non-automatic {rate(esc, 'flag_non_auto')}).\n"]
        if len(au) < 100:
            report.append("_Small hold-out: treat these rates as indicative only._\n")

    text = "\n".join(report)
    print("\n" + text)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
