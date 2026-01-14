import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/roof_dataset.csv"
MODEL_OUT = "models/roof_model.joblib"
METRICS_OUT = "results/roof_metrics.txt"

def main():
    df = pd.read_csv(DATA_PATH)

    # record_id ve y_true ayrılır
    y = df["y_true"].astype(int)
    X = df.drop(columns=["record_id", "y_true"])

    # Küçük veri olabileceği için stratify güvenli olmayabilir.
    # Ama veri yeterliyse stratify ekleyebilirsin.
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


    # Model: Logistic Regression + StandardScaler
    # (TF-IDF kolonları da numeric olduğu için scaler ile sorunsuz çalışır)
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=1000))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    joblib.dump(clf, MODEL_OUT)

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        f.write("ROOF MODEL RESULTS\n")
        f.write("===================\n")
        f.write(f"Accuracy   : {acc:.4f}\n")
        f.write(f"Macro F1   : {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")

    print("Roof model trained and saved.")
    print(f"Model  : {MODEL_OUT}")
    print(f"Metrics: {METRICS_OUT}")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

if __name__ == "__main__":
    main()
