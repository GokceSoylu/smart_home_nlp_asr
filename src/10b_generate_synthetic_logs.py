import os
import random

OUT_DIR = "logs"

# 0 = silence, diğerleri command sınıfları
COMMAND_LABELS = [1, 2, 3, 4, 5]  # istersen artırabilirsin
N_LOGS_PER_CLASS = 5              # toplam ~ (len(COMMAND_LABELS)+1)*N
LINE_COUNT_RANGE = (40, 80)       # her kayıtta kaç satır olsun
SILENCE_RATIO_RANGE = (0.3, 0.6)  # sessizlik oranı

random.seed(42)

def make_log(label: int, idx: int):
    n_lines = random.randint(*LINE_COUNT_RANGE)
    silence_ratio = random.uniform(*SILENCE_RATIO_RANGE)

    preds = []
    probs = []

    for i in range(n_lines):
        if random.random() < silence_ratio:
            pred = 0
            p = random.uniform(0.85, 0.99)
        else:
            # çoğunluk doğru sınıf, bazen karışsın
            if random.random() < 0.85:
                pred = label
            else:
                pred = random.choice([c for c in COMMAND_LABELS if c != label])
            p = random.uniform(0.65, 0.95)

        preds.append(pred)
        probs.append(p)

    # transition biraz gerçekçi olsun: ardışık aynı sınıf blokları
    # basitçe listeyi olduğu gibi bırakıyoruz, zaten karışım var.

    fname = f"syn_{label}_{idx:02d}__y{label}.txt"
    path = os.path.join(OUT_DIR, fname)

    with open(path, "w", encoding="utf-8") as f:
        t = 0.0
        for pr, p in zip(preds, probs):
            f.write(f"t={t:.2f} | PRED={pr} | p={p:.2f}\n")
            t += 0.10

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # silence-only örnekleri de ekleyelim (UNKNOWN gibi düşünebilirsin)
    for i in range(N_LOGS_PER_CLASS):
        # y0: tamamen sessizlik/boş kayıt senaryosu
        n_lines = random.randint(*LINE_COUNT_RANGE)
        path = os.path.join(OUT_DIR, f"syn_silence_{i:02d}__y0.txt")
        with open(path, "w", encoding="utf-8") as f:
            t = 0.0
            for _ in range(n_lines):
                p = random.uniform(0.90, 0.99)
                f.write(f"t={t:.2f} | PRED=0 | p={p:.2f}\n")
                t += 0.10

    # her komut için log üret
    for lab in COMMAND_LABELS:
        for i in range(N_LOGS_PER_CLASS):
            make_log(lab, i)

    print("Synthetic logs created in logs/")

if __name__ == "__main__":
    main()
