import json
from pathlib import Path

from backend.app.services.reading_difficulty.features import load_raw
from backend.app.services.reading_difficulty.pipeline import (
    append_session_log,
    load_artifacts,
    load_history,
    longitudinal_summary,
    predict_session,
)

raw_df = load_raw("datasets/comp02_raw_reading_sessions.csv")
artifacts = load_artifacts("outputs/artifacts.joblib")

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
log_path = out_dir / "session_log.jsonl"
if log_path.exists():
    log_path.unlink()

results = []
for i in range(len(raw_df)):
    row = predict_session(artifacts, raw_df.iloc[[i]])
    results.append(row)
    append_session_log(row, path=str(log_path))

(out_dir / "session_outputs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

first_child = results[0]["child_id"] if results else ""
if first_child:
    summary = longitudinal_summary(load_history(first_child, str(log_path)))
    (out_dir / f"longitudinal_{first_child}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"Wrote {len(results)} sessions -> outputs/session_outputs.json")
if first_child:
    print(f"Wrote longitudinal summary for {first_child} -> outputs/longitudinal_{first_child}.json")