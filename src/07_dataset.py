import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_mel_01.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

# --- LOAD DATA ---
df = pd.read_excel(DATA_PATH)

print("Dataset loaded:", df.shape)

# --- FEATURES & LABELS ---
feature_cols = [c for c in df.columns if c.startswith("feature_")]
X = df[feature_cols].values
y = df["target_label"].values

print("Feature matrix X:", X.shape)
print("Label vector y:", y.shape)

# --- TRAIN / TEST SPLIT (STRATIFIED) ---
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train set:", X_train.shape, y_train.shape)
print("Test set :", X_test.shape, y_test.shape)

# --- SAVE ---
np.savez(
    os.path.join(OUTPUT_DIR, "dataset_split.npz"),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

print("Train/test dataset saved to data/dataset_split.npz")
