"""Single-session inference must give the SAME result as the batch/training path.

predict_session() sees one row at a time. If categorical encoding or normalisation
behaves differently for one row than for the full dataset, predictions silently
differ. This test compares both routes on a handful of real rows.

    python test_parity.py
"""
import numpy as np

from backend.app.services.reading_difficulty.features import (
    apply_grade_norms, clean, engineer_features, load_raw,
)
from backend.app.services.reading_difficulty.pipeline import (
    _align_to_feature_columns, load_artifacts, predict_session,
)

RAW_CSV = "datasets/comp02_raw_reading_sessions.csv"


def main() -> None:
    art = load_artifacts("outputs/artifacts.joblib")
    raw = clean(load_raw(RAW_CSV))
    batch = apply_grade_norms(engineer_features(raw), art["norms"])

    target = art["difficulty_score"]
    idxs = list(raw.index[:: max(1, len(raw) // 8)])[:8]
    worst = 0.0
    for i in idxs:
        X = _align_to_feature_columns(batch.loc[[i]], target["feature_columns"])
        expected = float(target["model"].predict(X)[0])
        got = predict_session(art, raw.loc[i].to_dict())["difficulty_score"]
        worst = max(worst, abs(expected - got))
        print(f"row {i}: batch={expected:.4f} single={got:.4f}")
    assert worst < 1e-6, f"single-row path differs from batch path (max diff {worst:.4f})"
    print("parity OK")


if __name__ == "__main__":
    main()
