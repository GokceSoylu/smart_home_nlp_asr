import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_split.npz")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- LOAD DATA ---
data = np.load(DATA_PATH)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Dataset loaded.")

# --- TRAIN BEST MODEL (MLP) ---
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=50,
        random_state=42
    ))
])

print("Training MLP model...")
model.fit(X_train, y_train)

# --- PREDICT ---
y_pred = model.predict(X_test)

# --- CONFUSION MATRIX ---
cm = confusion_matrix(y_test, y_pred)

# --- PLOT ---
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    cmap="Blues",
    square=True,
    cbar=True,
    xticklabels=False,
    yticklabels=False
)

plt.title("Confusion Matrix – MLP Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

output_path = os.path.join(RESULTS_DIR, "confusion_matrix_mlp.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print("Confusion matrix saved to:", output_path)
