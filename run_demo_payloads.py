"""Send every payload in demo_payloads/ through validation + the Component 2 pipeline.

Repeatable: results are upserted by (child_id, session_id), so running it twice
does not duplicate sessions.

    python run_demo_payloads.py                       # writes outputs/session_log.jsonl
    python run_demo_payloads.py --log outputs/other.jsonl
"""
import argparse
import json
from pathlib import Path

from backend.app.services.reading_difficulty.component1_input import (
    Component1ValidationError, assess_component1_payload,
)
from backend.app.services.reading_difficulty.pipeline import load_artifacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="demo_payloads")
    ap.add_argument("--log", default="outputs/session_log.jsonl")
    ap.add_argument("--artifacts", default="outputs/artifacts.joblib")
    args = ap.parse_args()

    artifacts = load_artifacts(args.artifacts)
    files = sorted(Path(args.dir).glob("*.json"))
    if not files:
        raise SystemExit(f"no payloads in {args.dir}/ (run make_demo_payloads.py first)")

    print(f"{'session':<22}{'date':<12}{'score':>7}  {'level':<9}{'weak skill':<14}{'intensity':<10}{'review'}")
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        try:
            r = assess_component1_payload(payload, artifacts, log_path=args.log)
        except Component1ValidationError as e:
            print(f"{f.name}: REJECTED -> {e.errors}")
            continue
        rev = r.get("intervention", {}).get("requires_professional_review", False)
        print(f"{r['session_id']:<22}{r['timestamp']:<12}{r['difficulty_score']:>7.2f}  "
              f"{r['difficulty_level']:<9}{r['weak_skill']:<14}{r['intervention_intensity']:<10}{rev}")
    print(f"\nlogged to {args.log}")


if __name__ == "__main__":
    main()
