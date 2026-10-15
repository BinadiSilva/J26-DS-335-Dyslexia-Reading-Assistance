"""
profiling.py - rule-based cognitive profile + prosody engagement index.

- cognitive_profile: rule-based strengths/weaknesses from 4 sub-skill scores.
  primary_focus_area is the ML-predicted weak_skill (Model 2); the rule-based
  weakest area is reported alongside it with an agreement flag.
- prosody_engagement_index: 0-100, rule-based, calibrated on the train cohort
  via prosody_calibration.json. NOT clinically validated; indicative only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

STRENGTH_THRESHOLD = 75
WEAKNESS_THRESHOLD = 60
CALIBRATION_PATH = Path(__file__).with_name("prosody_calibration.json")
PROSODY_DISCLAIMER = "Rule-based indicator; not clinically validated."

# Fallback only if the calibration file has not been generated yet.
_FALLBACK = {
    "pitch_variability":   {"p5": 0.0,  "p95": 0.22, "direction": -1, "degenerate": False},
    "energy_mean":         {"p5": 0.30, "p95": 0.85, "direction": +1, "degenerate": False},
    "energy_variability":  {"p5": 0.0,  "p95": 0.42, "direction": -1, "degenerate": False},
    "pause_pattern_score": {"p5": 0.0,  "p95": 15.0, "direction": -1, "degenerate": False},
}


def load_calibration(path: Path | str = CALIBRATION_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text())["features"]
    except (OSError, KeyError, ValueError):
        return _FALLBACK


def _clip100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def subskill_scores(row) -> dict:
    return {
        "phonological": _clip100(100 - 4 * (row["hesitation_count"] + row["repetition_count"])),
        "decoding":     _clip100(100 - 4 * (row["omission_count"] + row["substitution_count"])),
        "fluency":      _clip100(row["fluency_score"]),
        "pronunciation": _clip100(row["pronunciation_accuracy"]),
    }


def classify_strengths_weaknesses(scores: dict) -> tuple[list, list]:
    strengths = [k for k, v in scores.items() if v >= STRENGTH_THRESHOLD]
    weaknesses = [k for k, v in scores.items() if v < WEAKNESS_THRESHOLD]
    return strengths, weaknesses


def build_cognitive_profile(row, predicted_weak_skill: str,
                            predicted_difficulty_level: str) -> dict:
    scores = subskill_scores(row)
    strengths, weaknesses = classify_strengths_weaknesses(scores)
    weakest = min(scores, key=scores.get)
    return {
        "subskill_scores": scores,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "difficulty_level": predicted_difficulty_level,
        "primary_focus_area": predicted_weak_skill,
        "rule_based_weakest_area": weakest,
        "model_rule_agreement": predicted_weak_skill == weakest,
    }


def prosody_engagement_index(row, calibration: dict | None = None) -> dict:
    cal = calibration or load_calibration()
    parts = {}
    for col, c in cal.items():
        if c.get("degenerate") or col not in row:
            continue
        span = c["p95"] - c["p5"]
        s = float(np.clip((row[col] - c["p5"]) / span, 0, 1))
        parts[col] = 100 * (s if c["direction"] > 0 else 1 - s)
    if not parts:
        return {"prosody_engagement_index": None, "components": {}, "note": PROSODY_DISCLAIMER}
    return {
        "prosody_engagement_index": round(float(np.mean(list(parts.values()))), 1),
        "components": {k: round(v, 1) for k, v in parts.items()},
        "note": PROSODY_DISCLAIMER,
    }
