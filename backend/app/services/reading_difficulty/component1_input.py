"""Component 1 -> Component 2 input contract: validation + processing.

One code path is used by the dashboard, the demo script and (optionally) an API
endpoint, so a synthetic payload goes through exactly the route a real Component 1
payload will take: validate -> predict_session -> upsert into the session log.

Pure Python on purpose: validation imports nothing heavy and is easy to test.
Adjust the ranges in FIELD_RULES if your dataset uses different scales.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

# name: (kind, min, max)   kind: "str" | "int" | "float"
FIELD_RULES: dict[str, tuple[str, float | None, float | None]] = {
    # identifiers & session metadata
    "child_id": ("str", None, None),
    "session_id": ("str", None, None),
    "gender": ("str", None, None),
    "passage_id": ("str", None, None),
    "passage_level": ("str", None, None),
    "age": ("int", 8, 10),
    "grade": ("int", 3, 5),
    # speech & reading features
    "reading_duration_sec": ("float", 0.001, 3600),
    "words_total": ("int", 1, 5000),
    "words_correct": ("int", 0, 5000),
    "words_incorrect": ("int", 0, 5000),
    "reading_speed_wpm": ("float", 0, 400),
    "pronunciation_accuracy": ("float", 0, 100),
    "fluency_score": ("float", 0, 100),
    "word_error_rate": ("float", 0, 2),
    "hesitation_count": ("int", 0, 500),
    "repetition_count": ("int", 0, 500),
    "omission_count": ("int", 0, 500),
    "substitution_count": ("int", 0, 500),
    "insertion_count": ("int", 0, 500),
    "pitch_mean": ("float", 0, 1000),
    "energy_mean": ("float", 0, 1e6),
    "avg_pause_duration": ("float", 0, 60),
}

# The raw per-frame / per-pause sample lists are required Component-1 inputs for
# the prosody variability features that Component 2 derives.
SAMPLE_FIELDS = ["pitch_samples", "energy_samples", "pause_durations"]
REQUIRED_FIELDS = list(FIELD_RULES) + ["session_date", *SAMPLE_FIELDS]
OPTIONAL_FIELDS = ["transcribed_text"]

# Derived by Component 2 itself; if a sender includes them they are ignored.
DERIVED_FIELDS = [
    "error_pattern_features", "normalized_speed", "normalized_fluency",
    "normalized_accuracy", "pitch_variability", "energy_variability",
    "pause_pattern_score",
]
OUTPUT_FIELDS = [
    "difficulty_score", "difficulty_level", "weak_skill", "intervention_intensity",
]


class Component1ValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _to_number(name: str, v: Any, kind: str) -> float | int:
    if isinstance(v, bool):
        raise ValueError(f"{name} must be a number")
    if isinstance(v, str):
        v = float(v.strip())
    if not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number")
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number")
    if kind == "int":
        if float(v) != int(v):
            raise ValueError(f"{name} must be a whole number")
        return int(v)
    return float(v)


def _samples_to_string(name: str, v: Any) -> str:
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(";") if p.strip()]
    elif isinstance(v, (list, tuple)):
        parts = list(v)
    else:
        raise ValueError(f"{name} must be a list of numbers or a ';'-separated string")
    nums = [_to_number(name, p, "float") for p in parts]
    if any(n < 0 for n in nums):
        raise ValueError(f"{name} must not contain negative values")
    return ";".join(repr(float(n)) for n in nums)


def validate_component1_payload(payload: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    record: dict[str, Any] = {}

    if not isinstance(payload, dict):
        return ValidationResult(False, ["payload must be a JSON object"])

    for name in REQUIRED_FIELDS:
        if _is_missing(payload.get(name)):
            errors.append(f"missing required field: {name}")

    for name, (kind, lo, hi) in FIELD_RULES.items():
        v = payload.get(name)
        if _is_missing(v):
            continue
        try:
            if kind == "str":
                record[name] = str(v).strip()
            else:
                num = _to_number(name, v, kind)
                if lo is not None and num < lo:
                    raise ValueError(f"{name}={num} is below the allowed minimum {lo}")
                if hi is not None and num > hi:
                    raise ValueError(f"{name}={num} is above the allowed maximum {hi}")
                record[name] = num
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))

    sd = payload.get("session_date")
    if not _is_missing(sd):
        try:
            parsed = datetime.strptime(str(sd)[:10], "%Y-%m-%d").date()
            if parsed > date.today():
                errors.append(f"session_date {parsed} is in the future")
            record["session_date"] = parsed.isoformat()
        except ValueError:
            errors.append("session_date must be formatted YYYY-MM-DD")

    for name in SAMPLE_FIELDS:
        v = payload.get(name)
        if _is_missing(v) or (isinstance(v, (list, tuple)) and not v):
            continue
        try:
            record[name] = _samples_to_string(name, v)
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))

    if not _is_missing(payload.get("transcribed_text")):
        record["transcribed_text"] = str(payload["transcribed_text"])

    # cross-field consistency
    try:
        total, ok_, bad = record["words_total"], record["words_correct"], record["words_incorrect"]
        if abs((ok_ + bad) - total) > 1:
            errors.append(
                f"words_correct + words_incorrect ({ok_ + bad}) does not match words_total ({total})"
            )
    except KeyError:
        pass

    ignored = [k for k in payload
               if k not in REQUIRED_FIELDS and k not in OPTIONAL_FIELDS]
    sent_outputs = [k for k in ignored if k in OUTPUT_FIELDS]
    sent_derived = [k for k in ignored if k in DERIVED_FIELDS]
    if sent_outputs:
        warnings.append(f"ignored output fields sent in input: {', '.join(sent_outputs)}")
    if sent_derived:
        warnings.append(f"ignored derived fields (Component 2 computes these): {', '.join(sent_derived)}")
    other = [k for k in ignored if k not in OUTPUT_FIELDS and k not in DERIVED_FIELDS]
    if other:
        warnings.append(f"ignored unknown fields: {', '.join(other)}")

    return ValidationResult(not errors, errors, warnings, record if not errors else {})


def upsert_session_log(result: dict[str, Any], path: str | Path = "outputs/session_log.jsonl") -> None:
    """Insert the session, replacing any earlier row with the same child_id + session_id.

    This makes repeated demo runs idempotent (no duplicate sessions in the history).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    key = (str(result.get("child_id")), str(result.get("session_id")))
    kept: list[str] = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (str(row.get("child_id")), str(row.get("session_id"))) != key:
                kept.append(line)
    kept.append(json.dumps(result, default=str))
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    os.replace(tmp, p)


def assess_component1_payload(
    payload: dict[str, Any],
    artifacts: dict[str, Any],
    log_path: str | Path | None = "outputs/session_log.jsonl",
) -> dict[str, Any]:
    """Validate a Component 1 payload, run the Component 2 pipeline, optionally log it."""
    check = validate_component1_payload(payload)
    if not check.ok:
        raise Component1ValidationError(check.errors)

    from backend.app.services.reading_difficulty.pipeline import predict_session

    result = predict_session(artifacts, check.record)
    result["source"] = "component1_payload"
    result["input_warnings"] = check.warnings
    result["input_summary"] = {k: check.record[k] for k in ("age", "grade", "passage_id", "passage_level") if k in check.record}
    if log_path:
        upsert_session_log(result, log_path)
    return result
