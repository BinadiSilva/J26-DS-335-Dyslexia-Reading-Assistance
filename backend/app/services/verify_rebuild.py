import json, os
import numpy as np
import pandas as pd

CSV = "datasets/comp02_raw_reading_sessions.csv"
df = pd.read_csv(CSV)
print("rows", len(df), "| children", df.child_id.nunique())
print("age", df.age.min(), "-", df.age.max(), "| grade", df.grade.min(), "-", df.grade.max())
print(pd.crosstab(df.difficulty_level, df.intervention_intensity))
lvl = np.where(df.difficulty_score < 0.26, "Low", np.where(df.difficulty_score < 0.50, "Moderate", "High"))
print("difficulty_level follows 0.26/0.50 thresholds:", float((lvl == df.difficulty_level).mean()))
print("support == difficulty:", float((df.intervention_intensity == df.difficulty_level).mean()))

rows = [json.loads(l) for l in open("outputs/session_log.jsonl") if l.strip()]
print("\nsession_log rows:", len(rows), "| children:", len({r['child_id'] for r in rows}))
for r in rows[:12]:
    print(r["session_id"], r["timestamp"], r["difficulty_level"], r["intervention_intensity"],
          r.get("level_consistency", {}).get("consistent"), r.get("input_summary"))

def m(p): return os.path.getmtime(p)
cal = "backend/app/services/reading_difficulty/prosody_calibration.json"
print("\ncalibration newer than CSV:", m(cal) > m(CSV), "| artifacts newer than CSV:", m("outputs/artifacts.joblib") > m(CSV))