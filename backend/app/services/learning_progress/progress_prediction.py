
# Import the baseline and comparison machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Import evaluation metrics to measure prediction errors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import a function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import NumPy for calculating the square root
import numpy as np

# Import the function that loads and prepares our reading dataset
from data_preparation import load_reading_data


def train_progress_model():
    """Train and compare two models for predicting reading progress."""

    # Load the cleaned reading-progress dataset
    df = load_reading_data()

    # Select reading-performance features used for prediction
    features = [
        "reading_speed_wpm",
        "reading_accuracy",
        "fluency_score",
        "word_error_rate",
    ]

    # Use the progress score as the value we want to predict
    X = df[features]
    y = df["progress_score"]

    # Split the dataset so we can evaluate predictions on unseen records
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define both models so their results can be compared
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
    }

    # Train and evaluate each model using the same data split
    results = {}

    for model_name, model in models.items():
        # Learn patterns from the training records
        model.fit(X_train, y_train)

        # Predict progress scores for the testing records
        predictions = model.predict(X_test)

        # Calculate metrics that describe prediction performance
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Store the trained model and its evaluation results
        results[model_name] = {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        # Display the results for this model
        print(f"\n{model_name}")
        print("-" * len(model_name))
        print(f"Training records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")

    # Return the results so other parts of the system can use them
    return results


# Run the model comparison when this file is executed directly
if __name__ == "__main__":
    train_progress_model()
