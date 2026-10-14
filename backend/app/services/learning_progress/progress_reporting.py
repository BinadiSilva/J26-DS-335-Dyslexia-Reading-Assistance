
# Import pandas for handling reading-session data
import pandas as pd

# Import the functions developed for Component 4
from data_preparation import load_reading_data
from progress_prediction import train_progress_model
from stagnation_detection import detect_stagnation
from intervention_planning import generate_intervention_plans


def generate_progress_report():
    """Generate a clean report of reading progress and support plans."""

    # Step 1: Load historical reading-session data
    reading_data = load_reading_data()

    # Step 2: Train and compare the prediction models
    models = train_progress_model()

    # Step 3: Select the trained Linear Regression model
    model = models["Linear Regression"]["model"]

    # Step 4: Define the input features used by the model
    features = [
        "reading_speed_wpm",
        "reading_accuracy",
        "fluency_score",
        "word_error_rate",
    ]

    # Step 5: Predict progress scores for each reading session
    reading_data = reading_data.copy()
    reading_data["predicted_progress_score"] = model.predict(
        reading_data[features]
    ).round(2)

    # Step 6: Find the latest session for each child
    latest_sessions = (
        reading_data.sort_values("session_number")
        .groupby("child_id")
        .tail(1)
        .reset_index(drop=True)
    )

    # Step 7: Get learning stagnation-risk information
    stagnation_results = detect_stagnation()

    # Step 8: Generate personalized intervention plans
    intervention_results = generate_intervention_plans()

    # Step 9: Keep only the required stagnation fields
    risk_columns = [
        "child_id",
        "stagnation_risk",
        "explanation",
    ]
    stagnation_summary = stagnation_results[risk_columns].copy()

    # Step 10: Keep only the required intervention fields
    intervention_columns = [
        "child_id",
        "recommended_activities",
        "frequency",
        "session_duration",
    ]
    intervention_summary = intervention_results[intervention_columns].copy()

    # Step 11: Combine the latest sessions with risk information
    report = latest_sessions.merge(
        stagnation_summary,
        on="child_id",
        how="left",
    )

    # Step 12: Add the intervention recommendations
    report = report.merge(
        intervention_summary,
        on="child_id",
        how="left",
    )

    # Step 13: Rename columns to make the report easier to understand
    report = report.rename(
        columns={
            "progress_score": "latest_progress_score",
        }
    )

    # Step 14: Select and order the final report columns
    final_columns = [
        "child_id",
        "latest_progress_score",
        "predicted_progress_score",
        "stagnation_risk",
        "explanation",
        "recommended_activities",
        "frequency",
        "session_duration",
    ]

    report = report[final_columns]

    # Step 15: Display the clean progress report
    print("\nLearning Progress Report")
    print("=" * 80)
    print(report.to_string(index=False))

    # Return the report for future dashboard integration
    return report


# Run the report generator when this file is executed directly
if __name__ == "__main__":
    generate_progress_report()
