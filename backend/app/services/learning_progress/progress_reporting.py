
"""Generate the combined LIBS reading-progress report for Component 4."""

# Support both package imports and direct script execution.
try:
    from .data_preparation import load_libs_reading_data
    from .progress_prediction import (
        train_libs_progress_model,
        forecast_future_progress,
    )
    from .stagnation_detection import detect_stagnation
    from .intervention_planning import generate_intervention_plans
except ImportError:
    from data_preparation import load_libs_reading_data
    from progress_prediction import (
        train_libs_progress_model,
        forecast_future_progress,
    )
    from stagnation_detection import detect_stagnation
    from intervention_planning import generate_intervention_plans


PARTICIPANT_COLUMN = "UnlinkedParticipantNumber"


def generate_progress_report():
    """Combine observed scores, forecasts, risk screening, and support plans."""

    # Load the validated longitudinal reading dataset.
    reading_data = load_libs_reading_data()

    # Select the latest observed assessment for each participant.
    latest_assessments = (
        reading_data.sort_values(
            [PARTICIPANT_COLUMN, "time_point"]
        )
        .groupby(PARTICIPANT_COLUMN, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # Rename the observed score columns before merging to prevent duplicates.
    latest_assessments = latest_assessments[
        [PARTICIPANT_COLUMN, "reading_score", "time_point"]
    ].rename(
        columns={
            "reading_score": "latest_reading_score",
            "time_point": "latest_time_point",
        }
    )

    # Train and evaluate models using participant-level data splits.
    model_results = train_libs_progress_model(save_results=False)
    best_model_name = model_results["best_model_name"]

    # Estimate a future score for each participant with sufficient history.
    forecasts = forecast_future_progress(model_results)

    # Run risk screening once and reuse the results for intervention planning.
    stagnation_results = detect_stagnation()
    intervention_results = generate_intervention_plans(
        risk_data=stagnation_results
    )

    # Keep only the columns required in the final report.
    risk_summary = stagnation_results[
        [
            PARTICIPANT_COLUMN,
            "stagnation_risk",
            "explanation",
            "average_recent_change",
            "number_of_observed_assessments",
        ]
    ]

    intervention_summary = intervention_results[
        [
            PARTICIPANT_COLUMN,
            "recommended_activities",
            "frequency",
            "session_duration",
            "recommendation_basis",
        ]
    ]

    # Merge forecasts with latest assessments, risk categories, and plans.
    # The forecast table does not contain latest_reading_score or latest_time_point,
    # so the observed assessment columns remain unchanged.
    report = latest_assessments.merge(
        forecasts[
            [
                PARTICIPANT_COLUMN,
                "forecasted_next_score",
                "forecasted_score_change",
                "prediction_model",
            ]
        ],
        on=PARTICIPANT_COLUMN,
        how="left",
    )

    report = report.merge(
        risk_summary,
        on=PARTICIPANT_COLUMN,
        how="left",
    )

    report = report.merge(
        intervention_summary,
        on=PARTICIPANT_COLUMN,
        how="left",
    )

    # Mark whether a participant has a usable forecast.
    report["has_future_forecast"] = (
        report["forecasted_next_score"].notna()
    )

    # Keep a consistent model label for report interpretation.
    report["prediction_model"] = report[
        "prediction_model"
    ].fillna(best_model_name)

    # Define the report's column order.
    final_columns = [
        PARTICIPANT_COLUMN,
        "latest_reading_score",
        "latest_time_point",
        "forecasted_next_score",
        "forecasted_score_change",
        "has_future_forecast",
        "prediction_model",
        "average_recent_change",
        "number_of_observed_assessments",
        "stagnation_risk",
        "explanation",
        "recommended_activities",
        "frequency",
        "session_duration",
        "recommendation_basis",
    ]

    report = (
        report[final_columns]
        .sort_values(PARTICIPANT_COLUMN)
        .reset_index(drop=True)
    )

    # Print a short summary and sample rows.
    print("\nCombined LIBS Learning Progress Report")
    print("=" * 58)
    print(f"Participants in report: {len(report)}")
    print(f"Selected model: {best_model_name}")
    print(
        "Participants with a forecast: "
        f"{report['has_future_forecast'].sum()}"
    )
    print(
        "Participants without a forecast: "
        f"{(~report['has_future_forecast']).sum()}"
    )

    print("\nFirst five report rows:")
    print(report.head().to_string(index=False))

    # Clearly state the current research limitations.
    print(
        "\nLimitations:\n"
        "- Forecasts estimate a next score; they do not specify a future date "
        "or guarantee improvement.\n"
        "- Forecasts beyond the dataset's observed assessment range are "
        "experimental and are not validated future-horizon predictions.\n"
        "- Model performance is evaluated separately on held-out participants.\n"
        "- Stagnation categories use preliminary rules and are not clinically "
        "validated.\n"
        "- Intervention plans are preliminary suggestions requiring educator "
        "review; their effectiveness has not been established.\n"
        "- LIBS data do not directly measure dyslexia intervention outcomes "
        "or establish validity for children diagnosed with dyslexia."
    )

    return report


if __name__ == "__main__":
    # Generate and display the combined report.
    generate_progress_report()
