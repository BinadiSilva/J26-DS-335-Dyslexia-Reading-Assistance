"""Build synthetic Component 1 payloads for the demo (run once; output is committed).

Rows come from the HELD-OUT test learners (same split as training), so the model has
never seen them. Targets and Component-2-derived columns are stripped, leaving only
the agreed Component 1 contract fields. Dates are set to recent past days.

    python make_demo_payloads.py
"""
import json
from pathlib import Path

import pandas as pd

from backend.app.services.reading_difficulty.component1_input import (
    FIELD_RULES, SAMPLE_FIELDS, validate_component1_payload,
)
from backend.app.services.reading_difficulty.features import (
    clean, engineer_features, load_raw, stratified_learner_level_split,
)

RAW_CSV = "datasets/comp02_raw_reading_sessions.csv"
OUT_DIR = Path("demo_payloads")
# demo child -> (internal id, session dates)
PLAN = [
    ("demo_child_01", ["2026-10-05", "2026-10-12"]),
    ("demo_child_02", ["2026-10-06", "2026-10-13"]),
    ("demo_child_03", ["2026-10-09"]),
]


def main() -> None:
    raw = clean(load_raw(RAW_CSV))
    eng = engineer_features(raw)
    _, test_df = stratified_learner_level_split(eng, stratify_col="intervention_intensity")
    test_ids = set(test_df["child_id"])
    pool = raw[raw["child_id"].isin(test_ids)].copy()
    pool = pool.sort_values(["child_id", "session_date"])

    # prefer learners with several sessions; spread them over different intensity levels
    counts = pool.groupby("child_id").size()
    multi = list(counts[counts >= 2].index)
    first = pool.groupby("child_id").first()["intervention_intensity"]
    chosen: list[str] = []
    for level in ["High", "Moderate", "Low"]:
        for cid in multi:
            if cid not in chosen and str(first[cid]) == level:
                chosen.append(cid)
                break
    for cid in multi + list(counts.index):
        if len(chosen) >= len(PLAN):
            break
        if cid not in chosen:
            chosen.append(cid)
    chosen = chosen[: len(PLAN)]

    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("*.json"):
        old.unlink()

    contract = list(FIELD_RULES) + ["session_date", "transcribed_text"] + SAMPLE_FIELDS
    n = 0
    for (demo_id, dates), src in zip(PLAN, chosen):
        rows = pool[pool["child_id"] == src].head(len(dates))
        for k, (_, row) in enumerate(rows.iterrows()):
            payload = {c: row[c] for c in contract if c in row.index and pd.notna(row[c])}
            for c in SAMPLE_FIELDS:
                if c in payload and isinstance(payload[c], str):
                    payload[c] = [float(x) for x in payload[c].split(";") if x.strip()]
            payload["child_id"] = demo_id
            payload["session_id"] = f"{demo_id}-S{k + 1:02d}"
            payload["session_date"] = dates[k]
            payload = {k2: (v.item() if hasattr(v, "item") else v) for k2, v in payload.items()}
            check = validate_component1_payload(payload)
            if not check.ok:
                raise SystemExit(f"{payload['session_id']} failed validation: {check.errors}")
            (OUT_DIR / f"{payload['session_id']}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
            n += 1
    print(f"wrote {n} synthetic Component 1 payloads to {OUT_DIR}/")
    if any(len(pool[pool['child_id'] == c]) < len(d) for (_, d), c in zip(PLAN, chosen)):
        print("note: some learners had fewer sessions than planned; those children have fewer sessions")


if __name__ == "__main__":
    main()
