
# Import libraries for generating and saving synthetic reading data
import csv
import random
from pathlib import Path

# Use a fixed seed so the dataset can be reproduced
random.seed(42)

# Locate the project root and output CSV file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "component4" / "reading_progress.csv"

# Define the CSV columns required by Component 4
FIELDS = [
    "child_id",
    "session_number",
    "reading_speed_wpm",
    "reading_accuracy",
    "fluency_score",
    "word_error_rate",
    "progress_score",
]

# Store the generated records
records = []

# Generate five sessions for each of twenty fictional learners
for child_number in range(1, 21):
    # Assign a unique ID to each fictional learner
    child_id = f"C{child_number:03d}"

    # Give each learner a different starting ability
    speed = random.randint(20, 45)
    accuracy = random.randint(40, 65)
    fluency = random.randint(25, 50)

    # Generate five reading sessions for each learner
    for session in range(1, 6):

        # Simulate gradual improvement with some natural variation
        if session > 1:
            speed += random.randint(0, 5)
            accuracy += random.randint(0, 5)
            fluency += random.randint(0, 5)

        # Keep reading measurements within sensible limits
        speed = min(speed, 100)
        accuracy = min(accuracy, 100)
        fluency = min(fluency, 100)

        # Simulate word errors decreasing as reading accuracy improves
        error_rate = max(0, 100 - accuracy + random.randint(-3, 3))

        # Add small variation to simulate differences in measured progress
        measurement_variation = random.uniform(-3, 3)

        # Calculate the progress score from the reading measurements
        progress = (
            0.30 * speed
            + 0.40 * accuracy
            + 0.30 * fluency
            + measurement_variation
        )

        # Keep the progress score within the range of 0 to 100
        progress = max(0, min(100, progress))

        # Add the current reading session to the dataset
        records.append({
            "child_id": child_id,
            "session_number": session,
            "reading_speed_wpm": speed,
            "reading_accuracy": accuracy,
            "fluency_score": fluency,
            "word_error_rate": error_rate,
            "progress_score": round(progress, 2),
        })

# Create the output directory if it does not already exist
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save the synthetic dataset to the CSV file
with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
    # Define the CSV writer using the required column names
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)

    # Write the column headings
    writer.writeheader()

    # Write all generated reading-session records
    writer.writerows(records)

# Display a summary to verify the generated dataset
print("Synthetic reading dataset generated successfully.")
print(f"Output file: {OUTPUT_PATH}")
print(f"Total records: {len(records)}")
print(f"Total learners: {len(set(row['child_id'] for row in records))}")
print("Sessions per learner: 5")
print("Note: These are synthetic records, not real children's data.")
