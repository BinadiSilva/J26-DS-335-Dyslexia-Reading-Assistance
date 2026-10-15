
import json
import random
from pathlib import Path

# Location of your downloaded original metadata
DATA_PATH = Path(r"F:\Research\MPS_Raw_Dataset\data.json")

# Where the selected metadata will be saved
OUTPUT_PATH = Path(
    "datasets/c1/mps/selected_20.json"
)

# Load JSON
with open(DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

print("JSON root type:", type(data).__name__)

# Handle a JSON list or dictionary
if isinstance(data, list):
    records = data
elif isinstance(data, dict):
    if "data" in data and isinstance(data["data"], list):
        records = data["data"]
    else:
        records = list(data.values())
else:
    raise ValueError("Unexpected JSON structure")

print("Total records:", len(records))

# Examine the first record before filtering
print("\nFirst record preview:")
print(json.dumps(records[0], indent=2, ensure_ascii=False)[:2500])

# Grade may be inside metaData
def get_grade(record):
    metadata = record.get("metaData", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return record.get("grade", metadata.get("grade"))

grade3_records = [
    record
    for record in records
    if isinstance(record, dict)
    and str(get_grade(record)).strip() == "3"
]

print("\nGrade 3 recordings:", len(grade3_records))

if len(grade3_records) < 20:
    print("Not enough Grade 3 records. Check the metadata structure.")
else:
    random.seed(42)
    selected = random.sample(grade3_records, 20)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, ensure_ascii=False)

    print("\nSuccessfully selected 20 records!")
    print("Saved to:", OUTPUT_PATH)
