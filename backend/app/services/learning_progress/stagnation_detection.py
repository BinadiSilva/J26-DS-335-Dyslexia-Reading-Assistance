
"""Detect possible reading-progress stagnation using LIBS assessments."""

import pandas as pd

# Support direct execution and package-based imports.
try:
    from .data_preparation import load_libs_reading_data
except ImportError:
    from data_preparation import load_libs_reading_data


PARTICIPANT_COLUMN = "UnlinkedParticipantNumber"

# These are initial development thresholds, not clinically validated cutoffs.
HIGH_RISK_THRESHOLD = 0
MEDIUM_RISK_THRESHOLD = 2
RECENT_CHANGES_WINDOW = 3
# Approximate study ages from VariableDescriptions_LIBS_July2024.txt.
# These are not each participant's measured age or exact assessment dates.
STUDY_AGES = {1: 4, 2: 5, 3: 6, 4: 9}
RESULT_COLUMNS = [
    PARTICIPANT_COLUMN, "latest_reading_score", "latest_time_point",
    "number_of_observed_assessments", "average_recent_change",
    "latest_score_change", "latest_change_direction", "recent_decline",
    "latest_study_year_gap", "recent_change_per_study_year",
    "has_missing_time_points", "stagnation_risk", "explanation",
]


def detect_stagnation():
    """Summarize recent reading-score changes for each LIBS participant.

    The risk categories are preliminary rule-based indicators. They must
    not be interpreted as clinical diagnoses or validated dyslexia risks.
    """

    # Load the prepared LIBS assessment data.
    df = load_libs_reading_data()

    # Sort each participant's observations in assessment order.
    df = df.sort_values(
        [PARTICIPANT_COLUMN, "time_point"]
    ).copy()

    results = []

    # Evaluate each participant independently.
    for participant_id, participant_data in df.groupby(
        PARTICIPANT_COLUMN,
        sort=True,
    ):
        scores = participant_data["reading_score"].tolist()
        time_points = participant_data["time_point"].tolist()

        # Calculate changes between consecutive observed assessments.
        # Missing intermediate time points are not treated as zero scores.
        changes = [
            scores[index] - scores[index - 1]
            for index in range(1, len(scores))
        ]

        # Keep the most recent changes for a simple trend summary.
        recent_changes = changes[-RECENT_CHANGES_WINDOW:]
        latest_change = changes[-1] if changes else None
        direction = (
            "Insufficient Data" if latest_change is None else
            "Decline" if latest_change < 0 else
            "Unchanged" if latest_change == 0 else "Increase"
        )
        gaps = [
            STUDY_AGES[time_points[i]] - STUDY_AGES[time_points[i - 1]]
            for i in range(1, len(time_points))
        ]
        recent_gaps = gaps[-RECENT_CHANGES_WINDOW:]
        # Report an elapsed-time-adjusted description without inventing a
        # validated annual-rate cutoff or changing the legacy risk categories.
        annual_change = (
            sum(recent_changes) / sum(recent_gaps) if recent_gaps else None
        )

        if len(recent_changes) < 2:
            average_change = None
            risk_level = "Insufficient Data"
            explanation = (
                "At least three observed assessments are needed "
                "for this preliminary trend rule."
            )
        else:
            average_change = sum(recent_changes) / len(recent_changes)

            if average_change <= HIGH_RISK_THRESHOLD:
                risk_level = "High"
                explanation = (
                    "The average change across recent observed "
                    "assessments is zero or negative."
                )
            elif average_change < MEDIUM_RISK_THRESHOLD:
                risk_level = "Medium"
                explanation = (
                    "The average score increase across recent "
                    "observed assessments is small."
                )
            else:
                risk_level = "Low"
                explanation = (
                    "The average score has increased across recent "
                    "observed assessments."
                )

        if latest_change is not None:
            explanation += (
                f" Latest observed change: {latest_change:+.2f} points over "
                f"approximately {gaps[-1]} study year(s)."
            )
        if direction == "Decline":
            explanation += " The latest score declined; review this even if the overall average is positive."
        if scores[-1] == 90:
            explanation += " The latest score is at the scale maximum; limited headroom can affect apparent progress."

        results.append(
            {
                "UnlinkedParticipantNumber": participant_id,
                "latest_reading_score": scores[-1],
                "latest_time_point": time_points[-1],
                "number_of_observed_assessments": len(scores),
                "average_recent_change": (
                    round(average_change, 2)
                    if average_change is not None
                    else None
                ),
                "stagnation_risk": risk_level,
                "explanation": explanation,
                "latest_score_change": latest_change,
                "latest_change_direction": direction,
                "recent_decline": direction == "Decline",
                "latest_study_year_gap": gaps[-1] if gaps else None,
                "recent_change_per_study_year": round(annual_change, 3) if annual_change is not None else None,
                "has_missing_time_points": any(
                    time_points[i] - time_points[i - 1] > 1
                    for i in range(1, len(time_points))
                ),
            }
        )

    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)

    print("\nLIBS Reading-Progress Stagnation Screening")
    print("=" * 52)
    print(f"Participants assessed: {len(results_df)}")

    if not results_df.empty:
        print("\nRisk-category counts:")
        print(
            results_df["stagnation_risk"]
            .value_counts()
            .to_string()
        )

        print("\nFirst five results:")
        print(results_df.head().to_string(index=False))

    print(
        "\nNote: Risk categories use preliminary rules and are "
        "not clinically validated."
    )

    return results_df


if __name__ == "__main__":
    detect_stagnation()
