
import json
from pathlib import Path
from collections import Counter


DATASET_PATH = Path(
    "datasets/c1/mps/selected_20.json"
)

AUDIO_ID = "3b30f_EN-OL-RC-234_2"


def main():
    # Load dataset metadata
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    # Find the selected recording
    record = next(
        (
            item for item in records
            if item["audioID"] == AUDIO_ID
        ),
        None
    )

    if record is None:
        raise ValueError(
            "Recording not found in selected_20.json"
        )

    print("\n=== RECORDING INFORMATION ===")
    print("Audio ID:", record["audioID"])
    print("Grade:", record["metaData"]["grade"])

    print("\n=== REFERENCE PASSAGE ===")
    print(record["promptText"])

    print("\n=== MANUAL TRANSCRIPT ===")
    print(record["manualTranscript"])

    print("\n=== READING ERROR COUNTS ===")

    counts = Counter(
        item["miscueLabel"]
        for item in record["textAlignment"]
    )

    print("Correct:", counts.get("c", 0))
    print("Substitutions:", counts.get("s", 0))
    print("Deletions:", counts.get("d", 0))
    print("Insertions:", counts.get("i", 0))

    print("\n=== NON-CORRECT ALIGNMENTS ===")

    for item in record["textAlignment"]:
        if item["miscueLabel"] != "c":
            print(
                f'Expected: {item["promptTextWord"]} | '
                f'Annotated: {item["manualTranscriptWord"]} | '
                f'Label: {item["miscueLabel"]}'
            )


if __name__ == "__main__":
    main()
