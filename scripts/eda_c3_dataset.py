
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "c3"
    / "processed"
    / "C3_Learner_Recommendation_Processed_Dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "c3" / "eda_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load processed dataset
# --------------------------------------------------
df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("C3 EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nNumerical summary:")
print(df.describe().round(2).to_string())

# --------------------------------------------------
# 3. Save summary reports
# --------------------------------------------------
df.describe(include="all").T.to_csv(
    OUTPUT_DIR / "dataset_summary.csv"
)

df.isnull().sum().to_csv(
    OUTPUT_DIR / "missing_values_summary.csv",
    header=["missing_count"]
)

# --------------------------------------------------
# 4. Numerical feature distributions
# --------------------------------------------------
numerical_columns = [
    "age",
    "grade",
    "preferred_duration_min",
    "reading_score",
    "reading_speed_wpm",
    "pronunciation_accuracy",
    "fluency_score",
    "difficulty_score",
    "previous_score",
    "activity_completion_rate",
    "recommended_duration_min",
]

for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=20, edgecolor="black")
    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{column}_distribution.png",
        dpi=150
    )
    plt.close()

# --------------------------------------------------
# 5. Categorical feature distributions
# --------------------------------------------------
categorical_columns = [
    "preferred_topic",
    "preferred_activity",
    "difficulty_level",
    "weak_skill",
    "previous_activity",
    "passage_topic",
    "passage_difficulty",
    "recommended_activity",
    "recommendation_level",
]

for column in categorical_columns:
    counts = df[column].value_counts().sort_values()

    plt.figure(figsize=(9, 5))
    counts.plot(kind="barh", edgecolor="black")
    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel("Number of records")
    plt.ylabel(column.replace("_", " ").title())
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{column}_distribution.png",
        dpi=150
    )
    plt.close()

# --------------------------------------------------
# 6. Correlation analysis
# --------------------------------------------------
correlation_columns = [
    "age",
    "grade",
    "preferred_duration_min",
    "reading_score",
    "reading_speed_wpm",
    "pronunciation_accuracy",
    "fluency_score",
    "difficulty_score",
    "previous_score",
    "activity_completion_rate",
    "recommended_duration_min",
]

correlation_matrix = df[correlation_columns].corr()

correlation_matrix.to_csv(
    OUTPUT_DIR / "numeric_correlation_matrix.csv"
)

plt.figure(figsize=(11, 8))
image = plt.imshow(correlation_matrix, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(image, label="Correlation coefficient")

plt.xticks(
    range(len(correlation_columns)),
    correlation_columns,
    rotation=75,
    ha="right"
)
plt.yticks(range(len(correlation_columns)), correlation_columns)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "numeric_correlation_matrix.png",
    dpi=150
)
plt.close()

# --------------------------------------------------
# 7. Explore recommendation labels
# --------------------------------------------------
print("\nRecommended activity counts:")
print(df["recommended_activity"].value_counts())

print("\nRecommendation level counts:")
print(df["recommendation_level"].value_counts())

print("\nMean learner measures by recommended activity:")
print(
    df.groupby("recommended_activity")[
        [
            "reading_score",
            "reading_speed_wpm",
            "pronunciation_accuracy",
            "fluency_score",
            "previous_score",
        ]
    ].mean().round(2).to_string()
)

# Save group analysis
df.groupby("recommended_activity")[
    [
        "reading_score",
        "reading_speed_wpm",
        "pronunciation_accuracy",
        "fluency_score",
        "previous_score",
    ]
].mean().round(2).to_csv(
    OUTPUT_DIR / "learner_measures_by_recommendation.csv"
)

print("\nEDA completed successfully.")
print("Results saved to:")
print(OUTPUT_DIR)
