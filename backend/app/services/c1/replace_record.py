
import json
import random
from pathlib import Path

# Original 1,600 metadata records
original_path = Path(
    r"F:\Research\MPS_Raw_Dataset\data.json"
)

# Selected 20 records
selected_path = Path(
    "datasets/c1/mps/selected_20.json"
)

with open(original_path, encoding="utf-8") as f:
    all_records = json.load(f)

with open(selected_path, encoding="utf-8") as f:
    selected = json.load(f)

# Problematic recording
old_id = "3a34g_EN-OL-RC-234_2"

old_record = next(
    r for r in selected if r["audioID"] == old_id
)

# Do not select an existing speaker
used_speakers = {
    r["metaData"]["speakerID"]
    for r in selected
}

# Find new Grade 3 candidates
# Prefer the same passage as the old recording
candidates = [
    r for r in all_records
    if r["metaData"]["grade"] == "3"
    and r["metaData"]["speakerID"] not in used_speakers
    and r["metaData"]["storyID"]
        == old_record["metaData"]["storyID"]
    and r["metaData"]["paragraphID"]
        == old_record["metaData"]["paragraphID"]
    and r.get("audioPath")
    and r.get("manualTranscript")
    and r.get("textAlignment")
]

if not candidates:
    raise ValueError("No suitable replacement candidate found.")

random.seed(42)
new_record = random.choice(candidates)

# Replace only the first problematic record
updated = [
    new_record if r["audioID"] == old_id else r
    for r in selected
]

# Save separately until we verify the audio
output_path = selected_path.with_name(
    "selected_20_updated.json"
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(updated, f, indent=2, ensure_ascii=False)

print("Old recording:", old_id)
print("New recording:", new_record["audioID"])
print("Audio path:", new_record["audioPath"])
print("Total records:", len(updated))
print("Saved to:", output_path)
