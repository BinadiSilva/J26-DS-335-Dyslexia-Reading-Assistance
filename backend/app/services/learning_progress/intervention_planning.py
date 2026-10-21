
"""Generate preliminary reading-support plans from LIBS score trends."""

import pandas as pd
import numpy as np

try:
    from .stagnation_detection import detect_stagnation
except ImportError:
    from stagnation_detection import detect_stagnation


PARTICIPANT_COLUMN = "UnlinkedParticipantNumber"


def generate_intervention_plans(risk_data=None, forecast_data=None):
    """Create support suggestions from existing stagnation results.

    Args:
        risk_data: Optional DataFrame returned by detect_stagnation().
            Passing it avoids running stagnation detection a second time.
        forecast_data: Optional next-score forecasts. A negative estimate
            at an observed training time point can prompt review; unsupported
            estimates cannot change the plan. No causal benefit is inferred.

    Returns:
        A DataFrame containing preliminary support suggestions.
    """

    # Run stagnation detection only when results were not supplied.
    if risk_data is None:
        risk_data = detect_stagnation()

    if not isinstance(risk_data, pd.DataFrame):
        raise TypeError("risk_data must be a pandas DataFrame.")

    required_columns = {
        PARTICIPANT_COLUMN,
        "latest_reading_score",
        "number_of_observed_assessments",
        "average_recent_change",
        "stagnation_risk",
    }

    missing_columns = required_columns - set(risk_data.columns)
    if missing_columns:
        raise ValueError(
            f"Stagnation results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if risk_data[PARTICIPANT_COLUMN].isna().any() or risk_data[PARTICIPANT_COLUMN].duplicated().any():
        raise ValueError("Stagnation results require one non-missing ID per participant.")
    if not risk_data["stagnation_risk"].isin(["High", "Medium", "Low", "Insufficient Data"]).all():
        raise ValueError("Unknown stagnation_risk category.")
    scores = pd.to_numeric(risk_data["latest_reading_score"], errors="coerce")
    counts = pd.to_numeric(risk_data["number_of_observed_assessments"], errors="coerce")
    if not scores.between(0, 90).all() or not (counts.between(1, 4) & counts.mod(1).eq(0)).all():
        raise ValueError("Plans require valid BAS scores and assessment counts (1 to 4).")
    for column in ["average_recent_change", "latest_score_change"]:
        if column in risk_data:
            values = pd.to_numeric(risk_data[column], errors="coerce")
            invalid = risk_data[column].notna() & (~np.isfinite(values) | values.abs().gt(90))
            if invalid.any():
                raise ValueError(f"Invalid score changes in {column}.")
            risk_data = risk_data.copy()
            risk_data[column] = values

    forecasts = {}
    if forecast_data is not None:
        if not isinstance(forecast_data, pd.DataFrame):
            raise TypeError("forecast_data must be a pandas DataFrame.")
        forecast_columns = {PARTICIPANT_COLUMN, "forecast_status", "forecasted_score_change"}
        if not forecast_columns.issubset(forecast_data.columns):
            raise ValueError("Forecast data are missing required columns.")
        if forecast_data[PARTICIPANT_COLUMN].isna().any() or forecast_data[PARTICIPANT_COLUMN].duplicated().any():
            raise ValueError("Forecasts require one non-missing ID per participant.")
        forecasts = forecast_data.set_index(PARTICIPANT_COLUMN).to_dict("index")

    plans = []

    for _, participant in risk_data.iterrows():
        participant_id = participant[PARTICIPANT_COLUMN]
        risk = participant["stagnation_risk"]
        latest_change = participant.get("latest_score_change", float("nan"))
        recent_decline = pd.notna(latest_change) and latest_change < 0
        evidence = [f"Historical screening category: {risk}."]
        average = participant["average_recent_change"]
        if pd.notna(average):
            evidence.append(f"Average observed change: {average:+.2f} score points.")
        if pd.notna(latest_change):
            evidence.append(f"Latest observed change: {latest_change:+.2f} score points.")

        forecast = forecasts.get(participant_id, {})
        forecast_change = forecast.get("forecasted_score_change")
        status = forecast.get("forecast_status")
        known_time = isinstance(status, str) and status == "observed_training_time_point"
        finite_change = isinstance(forecast_change, (int, float, np.number)) and np.isfinite(forecast_change)
        forecast_review = known_time and finite_change and forecast_change < 0
        if known_time and finite_change:
            forecast_use = "Experimental review context"
            evidence.append(
                f"Experimental predicted change: {forecast_change:+.2f} points; "
                "future accuracy and uncertainty for this estimate are not established."
            )
        else:
            forecast_use = "Not used: unavailable or unsupported forecast"

        if risk == "Insufficient Data":
            priority = "Collect evidence"
        elif recent_decline or risk in {"High", "Medium"} or forecast_review:
            priority = "Educator review"
        else:
            priority = "Continue and monitor"

        if risk == "High":
            activities = [
                "Review recent reading assessments with an educator",
                "Provide supported reading practice suited to the learner",
                "Reassess progress after further guided practice",
            ]
            frequency = "Discuss an appropriate schedule with an educator"
            duration = "Set by the educator"

        elif risk == "Medium":
            activities = [
                "Continue supported reading practice",
                "Review reading progress with an educator",
                "Collect further assessment evidence",
            ]
            frequency = "Discuss an appropriate schedule with an educator"
            duration = "Set by the educator"

        elif risk == "Low":
            activities = [
                "Continue suitable reading practice",
                "Monitor progress at the next assessment",
            ]
            frequency = "Follow the existing learning plan"
            duration = "Follow the existing learning plan"

        else:
            activities = [
                "Collect additional reading-assessment data",
                "Review the learner's needs with an educator",
                "Avoid assigning a trend-based risk category until "
                "enough observations are available",
            ]
            frequency = "Review after additional assessments"
            duration = "Set by the educator"

        # A recent decline needs attention even when the long-term average
        # remains positive. A positive prediction must never cancel this.
        if recent_decline:
            activities.insert(0, "Review the latest observed decline and assessment conditions with an educator")
        if forecast_review:
            activities.append("Compare the experimental predicted decline with further observations before changing support")
        if risk != "Insufficient Data":
            activities.append(
                "Use an educator-selected passage for supported oral reading with constructive feedback"
            )
            activities.append(
                "Record recurring word-reading errors; select word practice only after educator assessment"
            )
        evidence.append(
            "BAS totals alone do not identify decoding, fluency, or comprehension difficulties."
        )

        plans.append(
            {
                PARTICIPANT_COLUMN: participant_id,
                "latest_reading_score": participant["latest_reading_score"],
                "number_of_observed_assessments": participant[
                    "number_of_observed_assessments"
                ],
                "average_recent_change": participant[
                    "average_recent_change"
                ],
                "stagnation_risk": risk,
                "recommended_activities": "; ".join(activities),
                "frequency": frequency,
                "session_duration": duration,
                "recommendation_basis": (
                    " ".join(evidence) + " Preliminary educational suggestion; "
                    "not a clinically validated intervention."
                ),
                "support_priority": priority,
                "forecast_use": forecast_use,
                "review_required": True,
                "review_trigger": "Review after further comparable assessments or a newly observed decline; educator sets timing.",
            }
        )

    # Preserve the expected report columns even when no plans are generated.
    plans_df = pd.DataFrame(
        plans,
        columns=[
            PARTICIPANT_COLUMN,
            "latest_reading_score",
            "number_of_observed_assessments",
            "average_recent_change",
            "stagnation_risk",
            "recommended_activities",
            "frequency",
            "session_duration",
            "recommendation_basis",
            "support_priority",
            "forecast_use",
            "review_required",
            "review_trigger",
        ],
    )


    print("\nPreliminary LIBS Reading-Support Plans")
    print("=" * 48)
    print(f"Plans generated: {len(plans_df)}")

    if not plans_df.empty:
        print("\nPlans by risk category:")
        print(plans_df["stagnation_risk"].value_counts().to_string())

        print("\nFirst five plans:")
        print(plans_df.head().to_string(index=False))

    print(
        "\nNote: Recommendations require educator review and are not "
        "evidence of intervention effectiveness."
    )

    return plans_df


if __name__ == "__main__":
    generate_intervention_plans()
