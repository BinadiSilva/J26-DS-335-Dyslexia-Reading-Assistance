"""
prosody_audit.py - calibrate the prosody_engagement_index on the TRAIN cohort.

Run from repo root (inside the Docker container):
    python backend/app/services/reading_difficulty/prosody_audit.py \
        --raw datasets/comp02_raw_reading_sessions.csv

It rebuilds engineered features from the raw CSV (no processed CSV needed),
prints distributions, and writes prosody_calibration.json (robust p5/p95
bounds). Direction is THEORY-DRIVEN (see PROSODY_FEATURES), not learned from
difficulty_score, to avoid circularity with the model target.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

# +1: higher value = more engaged/steady ; -1: higher value = less steady
PROSODY_FEATURES = {
    "pitch_variability": -1,
    "energy_mean": +1,
    "energy_variability": -1,
    "pause_pattern_score": -1,
}
DEFAULT_OUT = str(Path(__file__).with_name("prosody_calibration.json"))


def load_engineered(raw_path: str) -> pd.DataFrame:
    # imported lazily so the script also works on an already-engineered CSV
    from features import load_raw, clean, engineer_features
    return engineer_features(clean(load_raw(raw_path)))


def text_hist(x: pd.Series, bins: int = 10, width: int = 30) -> list[str]:
    counts, edges = np.histogram(x.dropna(), bins=bins)
    peak = max(counts.max(), 1)
    return [f"  [{edges[i]:9.3f},{edges[i+1]:9.3f}) {'#' * int(width * c / peak):<{width}} {c}"
            for i, c in enumerate(counts)]


def build_calibration(df: pd.DataFrame, target: str | None) -> dict:
    cal = {}
    for c, direction in PROSODY_FEATURES.items():
        if c not in df.columns:
            raise SystemExit(f"Missing prosody column: {c}")
        x = df[c].dropna()
        lo, hi = float(x.quantile(0.05)), float(x.quantile(0.95))
        rho = None
        if target and target in df.columns:
            r = df[[c, target]].dropna().corr(method="spearman").iloc[0, 1]
            rho = None if np.isnan(r) else float(r)
        cal[c] = {
            "p5": lo, "p95": hi, "median": float(x.median()),
            "direction": direction,
            "spearman_vs_target_info_only": rho,
            "degenerate": bool(hi - lo < 1e-9),
            "skew": float(x.skew()),
        }
    return cal


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw reading-sessions CSV")
    ap.add_argument("--target", default="difficulty_score")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    df = load_engineered(a.raw)
    print(f"Rows: {len(df)}\n")
    for c in PROSODY_FEATURES:
        x = df[c]
        print(f"== {c} (direction {PROSODY_FEATURES[c]:+d}) ==")
        print(f"  min={x.min():.3f} p5={x.quantile(.05):.3f} med={x.median():.3f} "
              f"p95={x.quantile(.95):.3f} max={x.max():.3f} skew={x.skew():.2f} "
              f"unique={x.nunique()}")
        print("\n".join(text_hist(x)), "\n")
    print("Spearman between prosody features:")
    print(df[list(PROSODY_FEATURES)].corr(method="spearman").round(2).to_string(), "\n")

    cal = build_calibration(df, a.target)
    for c, v in cal.items():
        if v["degenerate"]:
            print(f"WARNING: {c} is degenerate (p5 == p95); index will ignore it.")
        r = v["spearman_vs_target_info_only"]
        print(f"  {c:22s} rho_vs_{a.target} = {'n/a' if r is None else f'{r:+.3f}'} (info only)")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"features": cal}, indent=2))
    print(f"\nWrote calibration -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
