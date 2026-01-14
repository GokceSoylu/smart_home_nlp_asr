import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

LOGS_DIR = "logs"
OUT_CSV = "data/roof_dataset.csv"


# Log line örneği:
# t=0.80 | PRED=1 | p=0.88
LINE_RE = re.compile(r"PRED=(\d+)\s*\|\s*p=([0-9]*\.?[0-9]+)")

def parse_log_file(path: str):
    preds = []
    probs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            pred = int(m.group(1))
            p = float(m.group(2))
            preds.append(pred)
            probs.append(p)

    return preds, probs

def compute_stats(preds, probs):
    # 0 = silence varsayımı
    non_silence_idx = [i for i, pr in enumerate(preds) if pr != 0]
    non_silence_count = len(non_silence_idx)
    total = len(preds)
    non_silence_ratio = non_silence_count / total if total > 0 else 0.0

    if non_silence_count > 0:
        ps = [probs[i] for i in non_silence_idx]
        mean_p_non_silence = sum(ps) / len(ps)
        max_p_non_silence = max(ps)
    else:
        mean_p_non_silence = 0.0
        max_p_non_silence = 0.0

    # top_command_ratio: en sık görülen komut / non-silence
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

    # transition_count: sınıf değişimi sayısı
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
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    documents = []
    record_ids = []

    # logs içindeki .txt dosyalarını oku
    for fname in sorted(os.listdir(LOGS_DIR)):
        if not fname.endswith(".txt"):
            continue

        # y_true: dosya adından çekiyoruz: rec01__y3.txt -> 3
        y_true = 0
        m = re.search(r"__y(\d+)\.txt$", fname)
        if m:
            y_true = int(m.group(1))

        path = os.path.join(LOGS_DIR, fname)
        preds, probs = parse_log_file(path)

        stats = compute_stats(preds, probs)

        # TF-IDF için: predicted id’leri "tokene" çeviriyoruz
        # örn: 0 -> label_0, 3 -> label_3 ...
        doc = " ".join([f"label_{p}" for p in preds])
        documents.append(doc)
        record_ids.append(fname.replace(".txt", ""))

        row = {
            "record_id": fname.replace(".txt", ""),
            "y_true": y_true,
            **stats,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No log files found in {LOGS_DIR}")

    # TF-IDF çıkar
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(documents)
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=[f"tfidf_{c}" for c in tfidf.get_feature_names_out()])

    base_df = pd.DataFrame(rows)
    out_df = pd.concat([base_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Roof dataset created: {OUT_CSV}")
    print("Shape:", out_df.shape)

if __name__ == "__main__":
    main()
