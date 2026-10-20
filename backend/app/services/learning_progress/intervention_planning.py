
"""Generate preliminary reading-support plans from LIBS score trends."""

import pandas as pd

try:
    from .stagnation_detection import detect_stagnation
except ImportError:
    from stagnation_detection import detect_stagnation


PARTICIPANT_COLUMN = "UnlinkedParticipantNumber"


def generate_intervention_plans(risk_data=None):
    """Create support suggestions from existing stagnation results.

    Args:
        risk_data: Optional DataFrame returned by detect_stagnation().
            Passing it avoids running stagnation detection a second time.

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

    plans = []

    for _, participant in risk_data.iterrows():
        participant_id = participant[PARTICIPANT_COLUMN]
        risk = participant["stagnation_risk"]

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
                    "Preliminary rule-based suggestion from LIBS score "
                    "trends; not a clinically validated intervention."
                ),
            }
        )

    plans_df = pd.DataFrame(plans)

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
