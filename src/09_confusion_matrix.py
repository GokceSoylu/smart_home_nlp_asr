import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

# Load dataset
data = np.load("../data/dataset_split.npz")

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

print("Dataset loaded.")

# Train MLP model
print("Training MLP model...")
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=50, random_state=42)
mlp.fit(X_train, y_train)

# Predictions
y_pred = mlp.predict(X_test)

# ---- NEW PART ----
# Select top 10 most frequent labels
unique_labels, counts = np.unique(y_test, return_counts=True)
top_labels = unique_labels[np.argsort(counts)[-10:]]

# Filter data
mask = np.isin(y_test, top_labels)
y_test_filtered = y_test[mask]
y_pred_filtered = y_pred[mask]

# Confusion matrix
cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=top_labels)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=top_labels,
    yticklabels=top_labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – MLP (Top 10 Classes)")
plt.tight_layout()

# Save
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/confusion_matrix_mlp.png", dpi=300)

plt.close()

print("Confusion matrix saved to results/confusion_matrix_mlp.png")
