
"""Generate a combined LIBS reading-progress report for Component 4."""

# Support running this file directly or importing it as a package.
try:
    from .data_preparation import load_libs_reading_data
    from .progress_prediction import (
        prepare_libs_prediction_data,
        train_libs_progress_model,
    )
    from .stagnation_detection import detect_stagnation
    from .intervention_planning import generate_intervention_plans
except ImportError:
    from data_preparation import load_libs_reading_data
    from progress_prediction import (
        prepare_libs_prediction_data,
        train_libs_progress_model,
    )
    from stagnation_detection import detect_stagnation
    from intervention_planning import generate_intervention_plans


PARTICIPANT_COLUMN = "UnlinkedParticipantNumber"


def generate_progress_report():
    """Combine latest observed scores, historical estimates, screening, and plans."""

    # Load the LIBS assessment data.
    reading_data = load_libs_reading_data()

    # Select the latest available assessment for each participant.
    latest_assessments = (
        reading_data.sort_values(
            [PARTICIPANT_COLUMN, "time_point"]
        )
        .groupby(PARTICIPANT_COLUMN, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    latest_assessments = latest_assessments[
        [PARTICIPANT_COLUMN, "reading_score", "time_point"]
    ].rename(
        columns={
            "reading_score": "latest_reading_score",
            "time_point": "latest_time_point",
        }
    )

    # Train and evaluate models using validation-based model selection.
    # The prediction service selects a model using validation MAE and
    # reports final performance separately on held-out test participants.
    model_results = train_libs_progress_model(save_results=False)

    best_model_name = model_results["best_model_name"]
    best_model = model_results["results"][best_model_name]["model"]

    # Prepare historical examples with a known next observed score.
    prediction_data = prepare_libs_prediction_data()

    # Keep the latest eligible historical example for each participant.
    latest_examples = (
        prediction_data.sort_values(
            [PARTICIPANT_COLUMN, "time_point"]
        )
        .groupby(PARTICIPANT_COLUMN, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    features = model_results["features"]

    # Estimate the next observed score for these historical examples.
    # These estimates are not forecasts beyond the latest assessment.
    if best_model is not None and not latest_examples.empty:
        latest_examples["historical_next_score_estimate"] = (
            best_model.predict(latest_examples[features]).round(2)
        )
    else:
        latest_examples["historical_next_score_estimate"] = None

    # Keep the historical target time point for interpretation.
    historical_predictions = latest_examples[
        [
            PARTICIPANT_COLUMN,
            "historical_next_score_estimate",
            "target_time_point",
        ]
    ].rename(
        columns={
            "target_time_point": "historical_target_time_point",
        }
    )

    # Run stagnation screening once and reuse its output for support planning.
    stagnation_results = detect_stagnation()

    intervention_results = generate_intervention_plans(
        risk_data=stagnation_results
    )

    # Select the required screening fields.
    risk_summary = stagnation_results[
        [
            PARTICIPANT_COLUMN,
            "stagnation_risk",
            "explanation",
            "average_recent_change",
            "number_of_observed_assessments",
        ]
    ]

    # Select the required intervention-planning fields.
    intervention_summary = intervention_results[
        [
            PARTICIPANT_COLUMN,
            "recommended_activities",
            "frequency",
            "session_duration",
            "recommendation_basis",
        ]
    ]

    # Combine latest assessments, screening results, support plans,
    # and historical model estimates by participant identifier.
    report = latest_assessments.merge(
        risk_summary,
        on=PARTICIPANT_COLUMN,
        how="left",
    )

    report = report.merge(
        intervention_summary,
        on=PARTICIPANT_COLUMN,
        how="left",
    )

    report = report.merge(
        historical_predictions,
        on=PARTICIPANT_COLUMN,
        how="left",
    )

    # Mark whether a historical estimate is available for each participant.
    # This flag does not indicate a future prediction after the latest score.
    report["has_historical_estimate"] = (
        report["historical_next_score_estimate"].notna()
    )

    # Record which model generated the historical estimates.
    report["prediction_model"] = best_model_name

    # Arrange the final report columns in a consistent order.
    final_columns = [
        PARTICIPANT_COLUMN,
        "latest_reading_score",
        "latest_time_point",
        "historical_next_score_estimate",
        "historical_target_time_point",
        "has_historical_estimate",
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

    # Print a summary so the results can be checked from the terminal.
    print("\nCombined LIBS Learning Progress Report")
    print("=" * 58)
    print(f"Participants in report: {len(report)}")
    print(f"Model used for historical estimates: {best_model_name}")
    print(
        "Participants with a historical next-score estimate: "
        f"{report['has_historical_estimate'].sum()}"
    )

    print("\nFirst five report rows:")
    print(report.head().to_string(index=False))

    # Explain the scope and limitations of the report accurately.
    print(
        "\nLimitations:\n"
        "- Historical estimates refer to next observed scores in historical "
        "assessment examples; they are not forecasts beyond the latest "
        "assessment.\n"
        "- Model selection uses validation MAE; final model metrics are "
        "evaluated on held-out test participants.\n"
        "- The report does not currently provide a validated future forecast "
        "after each participant's latest assessment.\n"
        "- Stagnation categories use preliminary rules and are not clinically "
        "validated.\n"
        "- Support plans are preliminary suggestions requiring educator "
        "review; intervention effectiveness has not been established.\n"
        "- The LIBS assessment data do not directly measure the effectiveness "
        "of dyslexia interventions."
    )

    return report


if __name__ == "__main__":
    # Generate and print the combined Component 4 report.
    generate_progress_report()
