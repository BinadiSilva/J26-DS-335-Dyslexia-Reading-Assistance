
# Inspect workbook sheets and the first participant's original scores
import pandas as pd
from pathlib import Path

# Locate the original LIBS workbook
workbook_path = Path("LIBS_DataSet_UnLinked_July2024.xlsx")

# Load the workbook's sheet names
excel_file = pd.ExcelFile(workbook_path, engine="openpyxl")
print("Workbook sheets:")
print(excel_file.sheet_names)

# Inspect the original score columns
df = pd.read_excel(workbook_path, engine="openpyxl")
score_columns = [
    "UnlinkedParticipantNumber",
    "T1BAS_word_reading",
    "T2BAS_word_reading",
    "T3BAS_word_reading",
    "T4BAS_word_reading",
]

# Display the first ten participants' original values
print("\nOriginal workbook values:")
print(df[score_columns].head(10).to_string(index=False))

# Display any available workbook sheets with README in the name
for sheet in excel_file.sheet_names:
    if "readme" in sheet.lower():
        print(f"\nREADME sheet: {sheet}")
        print(pd.read_excel(workbook_path, sheet_name=sheet, header=None).head(20).to_string(index=False, header=False))
