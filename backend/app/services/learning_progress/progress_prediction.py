
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

from data_preparation import load_reading_data


def train_progress_model():
    df = load_reading_data()

    # Features used to predict reading progress
    features = [
        "reading_speed_wpm",
        "reading_accuracy",
        "fluency_score",
        "word_error_rate",
    ]

    X = df[features]
    y = df["progress_score"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simple baseline model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nReading Progress Prediction Model")
    print("---------------------------------")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    return model


if __name__ == "__main__":
    train_progress_model()
