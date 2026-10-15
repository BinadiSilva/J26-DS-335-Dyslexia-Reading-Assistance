
# Import pandas to organise and analyse reading-progress data
import pandas as pd

# Import the function that loads and prepares our dataset
from data_preparation import load_reading_data


def detect_stagnation():
    """Identify children whose reading progress may be stagnating."""

    # Load the historical reading data
    df = load_reading_data()

    # Sort each child's sessions in chronological session order
    df = df.sort_values(["child_id", "session_number"])

    # Store the results for each child
    results = []

    # Analyse each child separately
    for child_id, child_data in df.groupby("child_id"):

        # Get progress scores in session order
        scores = child_data["progress_score"].tolist()

        # Calculate progress changes between consecutive sessions
        changes = [
            scores[i] - scores[i - 1]
            for i in range(1, len(scores))
        ]

        # Use the latest three changes when available
        recent_changes = changes[-3:]

        # Calculate average recent progress
        if recent_changes:
            average_change = sum(recent_changes) / len(recent_changes)
        else:
            average_change = 0

        # Assign a simple risk level based on recent progress
        if len(recent_changes) < 2:
            risk_level = "Insufficient Data"
            explanation = "More reading sessions are needed."
        elif average_change <= 0:
            risk_level = "High"
            explanation = "Progress has stopped or decreased."
        elif average_change < 2:
            risk_level = "Medium"
            explanation = "Progress is increasing slowly."
        else:
            risk_level = "Low"
            explanation = "Progress is increasing steadily."

        # Save the result for this child
        results.append({
            "child_id": child_id,
            "latest_progress_score": scores[-1],
            "average_recent_change": round(average_change, 2),
            "stagnation_risk": risk_level,
            "explanation": explanation
        })

    # Convert the results into a table
    results_df = pd.DataFrame(results)

    # Display the stagnation-risk results
    print("\nLearning Stagnation Risk Detection")
    print("=" * 40)
    print(results_df.to_string(index=False))

    # Return the results so other components can use them
    return results_df


# Run the analysis when this file is executed directly
if __name__ == "__main__":
    detect_stagnation()
