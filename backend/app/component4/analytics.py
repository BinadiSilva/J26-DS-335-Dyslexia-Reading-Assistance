"""Explainable monitoring, proposed support intensity and observed outcomes.

Thresholds and schedules below are versioned demonstration rules for educator
review, not validated norms or estimates of causal intervention benefit.
"""

from datetime import date, timedelta
import numpy as np

from .schema import TARGETS

RULE_VERSION = "pp1-rules-1"
ACTIVITIES = {
    "accuracy": "Practise educator-selected word patterns, then read a short passage with corrective feedback.",
    "speed": "Use repeated supported reading of a familiar, appropriately levelled passage; prioritise accuracy.",
    "fluency": "Use modelled and paired reading, practising phrasing and expression in a short passage.",
}


def monitor(sessions):
    recent = sorted(sessions, key=lambda s: s["assessed_on"])[-4:]
    result = {"level": "Insufficient data", "slopes_per_week": {}, "recent_decline": False,
              "rule_version": RULE_VERSION, "reasons": [], "comparable": False}
    if len(recent) < 3:
        result["reasons"] = ["Record at least three comparable assessments to screen a trend."]
        return result
    if len({s["difficulty_level"] for s in recent}) != 1:
        result["reasons"] = ["Passage difficulty changed; collect comparable assessments before interpreting a trend."]
        return result
    days = np.array([(date.fromisoformat(s["assessed_on"]) - date.fromisoformat(recent[0]["assessed_on"])).days for s in recent])
    if np.any(np.diff(days) <= 0):
        raise ValueError("Assessment dates must be distinct.")
    if days[-1] < 14:
        result["reasons"] = ["Observe at least 14 days before screening stagnation."]
        return result
    slopes = {k: float(np.polyfit(days / 7, [s[k] for s in recent], 1)[0]) for k in TARGETS}
    result["slopes_per_week"] = slopes
    result["comparable"] = True
    # WER is reported separately: it overlaps accuracy and is not double-counted.
    low_growth = sum(slopes[k] <= cutoff for k, cutoff in zip(TARGETS[:3], (0.5, 0.3, 0.5)))
    decline = any(recent[-1][k] - recent[-2][k] < -cutoff for k, cutoff in zip(TARGETS[:3], (2, 2, 2)))
    ceiling = recent[-1]["reading_accuracy"] >= 98 and recent[-1]["fluency_score"] >= 98
    level = "High" if low_growth >= 2 else "Medium" if low_growth == 1 else "Low"
    if decline:
        level = "High"
    if ceiling and not decline:
        level = "Review ceiling"
    result.update(level=level, recent_decline=decline)
    result["reasons"] = [f"{k.replace('_', ' ')}: {v:+.2f} per week." for k, v in slopes.items()]
    if decline:
        result["reasons"].append("The latest assessment declined; earlier gains do not cancel this signal.")
    if ceiling:
        result["reasons"].append("Accuracy and fluency are near their scale ceiling; low growth may be expected.")
    return result


def intervention_outcomes(sessions, interventions):
    by_id = {s["id"]: s for s in sessions}
    results = []
    for intervention in interventions:
        item = dict(intervention)
        before = by_id.get(item["baseline_session_id"])
        after = by_id.get(item["outcome_session_id"])
        item.update(changes=None, comparable=False, outcome="Awaiting follow-up")
        if before and after:
            item["changes"] = {k: round(after[k] - before[k], 3) for k in TARGETS}
            item["comparable"] = before["difficulty_level"] == after["difficulty_level"]
            target = {"speed": "reading_speed_wpm", "accuracy": "reading_accuracy", "fluency": "fluency_score"}[item["plan"]["target_skill"]]
            change = item["changes"][target]
            item["outcome"] = (
                "Different difficulty; review comparability" if not item["comparable"] else
                "Observed improvement" if change > 0.5 else "Observed decline" if change < -0.5 else "Little observed change"
            )
        item["interpretation"] = "Before/after association only; does not establish that this intervention caused the change."
        results.append(item)
    return results


def propose_plan(learner, sessions, risk, forecast, outcomes):
    if not sessions:
        return None
    latest = sessions[-1]
    goals = {"speed": learner["target_speed"], "accuracy": learner["target_accuracy"], "fluency": learner["target_fluency"]}
    observed = {"speed": latest["reading_speed_wpm"], "accuracy": latest["reading_accuracy"], "fluency": latest["fluency_score"]}
    deficits = {key: max(0, (goal - observed[key]) / goal) for key, goal in goals.items()}
    target = max(deficits, key=deficits.get)
    if not any(deficits.values()):
        target = "fluency"
    reasons = [f"Compared with educator-entered goals: {target} has the largest relative shortfall ({100 * deficits[target]:.1f}%)."]
    if not any(deficits.values()):
        reasons = ["All educator-entered goals are met; review goals and maintain suitable practice."]
    level = "High" if risk["level"] == "High" else "Medium" if risk["level"] == "Medium" else "Low"
    # Only a supported forecast can increase priority; history always remains visible.
    if forecast.get("status") == "available":
        predicted = forecast["values"]
        if predicted["fluency_score"] < latest["fluency_score"] - 2:
            level = "High" if level == "High" else "Medium"
            reasons.append("The experimental one-week forecast suggests lower fluency; corroborate with further observations.")
    if risk["level"] in {"Insufficient data", "Review ceiling"}:
        level = "Review"
        reasons.extend(risk["reasons"][:1])
    activity = ACTIVITIES[target]
    previous = [i for i in outcomes if i["status"] == "completed" and i["plan"]["target_skill"] == target]
    if previous:
        last = previous[-1]
        reasons.append(f"Previous {target} intervention: {last['outcome']}; recorded adherence {last['adherence']:g}%.")
        if last["adherence"] < 60:
            activity = "Agree a manageable practice routine and discuss barriers with the educator. " + activity
        elif last["comparable"] and last["outcome"] != "Observed improvement":
            activity = "Educator review: change the passage or teaching approach before repeating the previous plan. " + activity
    minutes, frequency = {"High": (15, 4), "Medium": (10, 3), "Low": (10, 2), "Review": (10, 2)}[level]
    return {
        "target_skill": target, "priority": level, "activity": activity,
        "minutes": minutes, "frequency": frequency, "review_days": 14,
        "rationale": reasons, "rule_version": RULE_VERSION, "requires_educator_approval": True,
        "intensity_basis": "Provisional rule-based schedule; educator must confirm or edit before saving.",
        "expected_improvement": None,
    }


def learner_report(store, learner_id, model=None):
    learner = store.learner(learner_id)
    sessions = store.sessions(learner_id)
    risk = monitor(sessions)
    forecast = model.forecast(sessions, learner["source"]) if model else {
        "status": "unavailable", "reason": "Train or load a session model to enable forecasts."
    }
    outcomes = intervention_outcomes(sessions, store.interventions(learner_id))
    active = next((i for i in outcomes if i["status"] == "active"), None)
    if active:
        active["review_on"] = (date.fromisoformat(active["started_on"]) + timedelta(days=active["plan"]["review_days"])).isoformat()
    return {
        "learner": learner, "sessions": sessions, "risk": risk, "forecast": forecast,
        "proposed_plan": propose_plan(learner, sessions, risk, forecast, outcomes),
        "interventions": outcomes, "active_intervention": active, "notes": store.notes(learner_id),
        "notice": "Synthetic demonstration" if learner["source"] == "synthetic" else "Collected observations; research predictions and educational suggestions require review.",
    }
