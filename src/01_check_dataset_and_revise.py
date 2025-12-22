import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_mel_01.xlsx")

df = pd.read_excel(DATA_PATH)

print("=== DATASET OVERVIEW ===")
print("Shape (rows, columns):", df.shape)

print("\n=== COLUMN NAMES ===")
for col in df.columns:
    print(col)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== BASIC STATISTICS ===")
print(df.describe(include="all"))

# Label analysis
label_cols = [c for c in df.columns if "label" in c.lower() or "class" in c.lower()]
print("\nPossible label columns:", label_cols)

if label_cols:
    label_col = label_cols[0]
    print("\nLabel distribution:")
    print(df[label_col].value_counts().sort_index())
