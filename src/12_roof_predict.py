import sys
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

LINE_RE = re.compile(r"PRED=(\d+)\s*\|\s*p=([0-9]*\.?[0-9]+)")

MODEL_PATH = "models/roof_model.joblib"
ROOF_DATASET_PATH = "data/roof_dataset.csv"

def parse_log_file(path: str):
    preds = []
    probs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            preds.append(int(m.group(1)))
            probs.append(float(m.group(2)))
    return preds, probs

def compute_stats(preds, probs):
    non_silence_idx = [i for i, pr in enumerate(preds) if pr != 0]
    non_silence_count = len(non_silence_idx)
    total = len(preds)
    non_silence_ratio = non_silence_count / total if total > 0 else 0.0

    if non_silence_count > 0:
        ps = [probs[i] for i in non_silence_idx]
        mean_p_non_silence = float(np.mean(ps))
        max_p_non_silence = float(np.max(ps))
    else:
        mean_p_non_silence = 0.0
        max_p_non_silence = 0.0

    cmd_counts = {}
    for pr in preds:
        if pr == 0:
            continue
        cmd_counts[pr] = cmd_counts.get(pr, 0) + 1

    if non_silence_count > 0 and cmd_counts:
        top_cmd = max(cmd_counts, key=cmd_counts.get)
        top_command_ratio = cmd_counts[top_cmd] / non_silence_count
    else:
        top_command_ratio = 0.0

    transition_count = 0
    for i in range(1, total):
        if preds[i] != preds[i-1]:
            transition_count += 1

    return {
        "non_silence_count": non_silence_count,
        "non_silence_ratio": non_silence_ratio,
        "mean_p_non_silence": mean_p_non_silence,
        "max_p_non_silence": max_p_non_silence,
        "top_command_ratio": top_command_ratio,
        "transition_count": transition_count,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/12_roof_predict.py logs/<file>.txt")
        sys.exit(1)

    log_path = sys.argv[1]

    # 1) Modeli yükle
    clf = joblib.load(MODEL_PATH)

    # 2) TF-IDF vocab'ı için roof_dataset.csv'yi kullanarak aynı vectorizer'ı üret
    df = pd.read_csv(ROOF_DATASET_PATH)

    # dataset'teki TF-IDF kolonlarını bul
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    if not tfidf_cols:
        raise RuntimeError("No TF-IDF columns found in roof_dataset.csv")

    # TF-IDF vocab'ı yeniden kurmak için dataset'teki dokümanları tekrar üretmemiz gerekirdi.
    # Ama biz pratik çözüm yapıyoruz:
    # - roof_dataset.csv içindeki tfidf_* kolon isimlerinden tokenları çıkarıyoruz
    # - vectorizer'ı bu token listesi ile sabitliyoruz (feature order aynı olur)
    tokens = [c.replace("tfidf_", "") for c in tfidf_cols]
    vectorizer = TfidfVectorizer(vocabulary=tokens)

    # 3) Log’u parse et
    preds, probs = parse_log_file(log_path)
    if len(preds) == 0:
        raise RuntimeError(f"No valid lines parsed from log: {log_path}")

    # 4) Stat feature’ları çıkar
    stats = compute_stats(preds, probs)

    # 5) TF-IDF dokümanı oluştur
    doc = " ".join([f"label_{p}" for p in preds])
    X_tfidf = vectorizer.fit_transform([doc]).toarray()[0]  # vocabulary fixed

    # 6) Tek satırlık feature vector oluştur (roof_dataset kolon sırasıyla aynı olacak)
    # roof_dataset.csv'de record_id ve y_true hariç kolonlar:
    feature_cols = [c for c in df.columns if c not in ("record_id", "y_true")]

    # stats + tfidf kolonlarını birleştir
    row = {}
    row.update(stats)
    for i, col in enumerate(tfidf_cols):
        row[col] = float(X_tfidf[i])

    X_row = pd.DataFrame([row])

    # kolonları roof_dataset ile aynı sıraya sok
    X_row = X_row.reindex(columns=feature_cols, fill_value=0.0)

    # 7) Tahmin
    pred_label = int(clf.predict(X_row)[0])

    print(f"LOG: {log_path}")
    print(f"ROOF PREDICTED FINAL LABEL: {pred_label}")

if __name__ == "__main__":
    main()
