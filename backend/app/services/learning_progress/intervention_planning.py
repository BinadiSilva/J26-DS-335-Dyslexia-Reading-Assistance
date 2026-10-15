
# Import pandas to organise the reading data and intervention results
import pandas as pd

# Import the function that detects learning stagnation risk
from stagnation_detection import detect_stagnation

# Import the function that loads the reading dataset
from data_preparation import load_reading_data


def generate_intervention_plans():
    """Create initial reading-support plans based on progress and risk."""

    # Load the children's historical reading data
    reading_data = load_reading_data()

    # Get the latest stagnation-risk results for each child
    risk_data = detect_stagnation()

    # Store the generated intervention plans
    plans = []

    # Analyse each child separately
    for _, child in risk_data.iterrows():

        # Find the child's latest reading session
        child_sessions = reading_data[
            reading_data["child_id"] == child["child_id"]
        ].sort_values("session_number")

        # Read the latest performance measurements
        latest_session = child_sessions.iloc[-1]
        accuracy = latest_session["reading_accuracy"]
        fluency = latest_session["fluency_score"]
        error_rate = latest_session["word_error_rate"]

        # Get the child's detected stagnation risk
        risk = child["stagnation_risk"]

        # Start with general reading-support activities
        activities = []
        frequency = "3 sessions per week"
        duration = "15 minutes per session"

        # Recommend more support for children with high risk
        if risk == "High":
            activities = [
                "One-to-one guided reading",
                "Practise difficult words",
                "Review reading progress with a teacher"
            ]
            frequency = "4 sessions per week"
            duration = "15 minutes per session"

        # Provide additional practice for medium risk
        elif risk == "Medium":
            activities = [
                "Guided reading practice",
                "Word recognition exercises",
                "Short reading fluency activities"
            ]

        # Maintain practice for children with low risk
        elif risk == "Low":
            activities = [
                "Continue personalised reading practice",
                "Read short, age-appropriate passages",
                "Monitor weekly reading progress"
            ]

        # Avoid assigning a risk-based plan when there is insufficient data
        else:
            activities = [
                "Collect additional reading-session data",
                "Discuss observations with the teacher"
            ]
            frequency = "Review after additional sessions"
            duration = "Based on teacher guidance"

        # Add activities related to the child's latest reading difficulties
        if accuracy < 70:
            activities.append("Practise reading accuracy")

        if fluency < 50:
            activities.append("Practise reading fluency")

        if error_rate > 20:
            activities.append("Practise recognising and correcting word errors")

        # Save the child's personalised plan
        plans.append({
            "child_id": child["child_id"],
            "stagnation_risk": risk,
            "reading_accuracy": accuracy,
            "fluency_score": fluency,
            "word_error_rate": error_rate,
            "recommended_activities": "; ".join(activities),
            "frequency": frequency,
            "session_duration": duration
        })

    # Convert the plans into a table
    plans_df = pd.DataFrame(plans)

    # Display the generated plans
    print("\nAI-driven Intervention Planning")
    print("=" * 40)
    print(plans_df.to_string(index=False))

    # Return the plans for future dashboard integration
    return plans_df


# Run the intervention planner when this file is executed directly
if __name__ == "__main__":
    generate_intervention_plans()
