# Import libraries for inspecting the LIBS dataset
from pathlib import Path
import pandas as pd

# Locate the project root and the real Excel dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
file_path = PROJECT_ROOT / "LIBS_DataSet_UnLinked_July2024.xlsx"

# Load the first worksheet if the dataset exists
if file_path.exists():
    df = pd.read_excel(file_path, engine="openpyxl")

    # Display the dataset size
    print("DATASET SUMMARY")
    print("----------------")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # List all column names clearly
    print("\nCOLUMN NAMES")
    for column in df.columns:
        print("-", column)

    # Show missing-value counts only for columns with missing values
    print("\nMISSING VALUES")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())

    # Display the first five participant records
    print("\nFIRST FIVE RECORDS")
    print(df.head().to_string(index=False))
else:
    # Explain where the file should be located
    print("Dataset not found:", file_path)