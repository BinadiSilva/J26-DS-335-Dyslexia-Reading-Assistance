
# Import pandas for handling reading-session data
import pandas as pd

# Import the functions developed for Component 4
from data_preparation import load_reading_data
from progress_prediction import train_progress_model
from stagnation_detection import detect_stagnation
from intervention_planning import generate_intervention_plans


def generate_progress_report():
    """Generate a report with predictions, risk levels, and intervention plans."""

    # Step 1: Load the prepared historical reading data
    reading_data = load_reading_data()

    # Step 2: Train and evaluate both prediction models
    # Each dictionary entry contains the model and its evaluation metrics
    models = train_progress_model()

    # Step 3: Extract the actual trained Linear Regression model
    model = models["Linear Regression"]["model"]

    # Step 4: Define the reading measurements required by the model
    features = [
        "reading_speed_wpm",
        "reading_accuracy",
        "fluency_score",
        "word_error_rate",
    ]

    # Step 5: Predict progress scores for the available reading sessions
    reading_data = reading_data.copy()
    reading_data["predicted_progress_score"] = model.predict(
        reading_data[features]
    ).round(2)

    # Step 6: Find the most recent session for each child
    latest_sessions = (
        reading_data.sort_values("session_number")
        .groupby("child_id")
        .tail(1)
        .reset_index(drop=True)
    )

    # Step 7: Detect learning stagnation using the historical dataset
    stagnation_results = detect_stagnation()

    # Step 8: Generate intervention recommendations
    intervention_results = generate_intervention_plans()

    # Step 9: Add stagnation-risk information to the latest sessions
    report = latest_sessions.merge(
        stagnation_results,
        on="child_id",
        how="left",
        suffixes=("", "_risk"),
    )

    # Step 10: Add the recommended intervention plans
    report = report.merge(
        intervention_results,
        on="child_id",
        how="left",
        suffixes=("", "_intervention"),
    )

    # Step 11: Display the completed report
    print("\nLearning Progress Report")
    print("=" * 80)
    print(report.to_string(index=False))

    # Return the report for future dashboard integration
    return report


# Run the report generator when this file is executed directly
if __name__ == "__main__":
    generate_progress_report()
