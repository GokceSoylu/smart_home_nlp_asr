# Smart Home Voice Command Recognition  
### CSE 431 – Natural Language Processing with Machine Learning  
**Term Project – Phase 2**

---

## 1. Project Overview

This project focuses on speech-based smart home command recognition using signal-processed acoustic features. Unlike text-based NLP systems, this phase directly classifies spoken commands using Mel-based feature representations extracted from voice recordings.

Multiple machine learning and deep learning models are trained and evaluated under severe class imbalance conditions.

---

## 2. Dataset

- File: `dataset_mel_01.xlsx`
- Total samples: 27,471
- Feature dimension: 480 (Mel-based acoustic features)
- Number of classes: 69
- Label column: `target_label`

Each row represents a fixed-length audio segment (window) obtained from speech signals.

---

## 3. Project Structure


smart_home_asr_project2/
├── data/
│ ├── dataset_mel_01.xlsx
│ └── dataset_split.npz
├── src/
│ ├── 01_check_dataset_and_revise.py
│ ├── 07_dataset.py
│ ├── 08_train_test.py
│ └── 09_confusion_matrix.py
├── results/
│ ├── model_comparison_results.csv
│ └── confusion_matrix_mlp.png
├── README_EN.md
└── README_TR.md


---

## 4. Execution Order

Run the scripts in the following order:

```bash
cd src
python 01_check_dataset_and_revise.py
python 07_dataset.py
python 08_train_test.py
python 09_confusion_matrix.py
```

## 5. Models Implemented

- Decision Tree (class-weight balanced)
- Random Forest
- Linear Support Vector Machine (SVM)
- Multi-Layer Perceptron (ANN)

## 6. Evaluation Metrics

Due to severe class imbalance, the following macro-averaged metrics are used:

- Precision
-Recall
- F1-score

The best-performing model is selected based on Macro F1-score.

## 7. Results
The Multi-Layer Perceptron (MLP) achieved the best overall performance.

Detailed numerical results:

results/model_comparison_results.csv

Confusion matrix of the best model:

results/confusion_matrix_mlp.png

## 8. Requirements

- Python 3.10
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
## 9. Author

Gökçe Soylu :)

Aydın Adnan Menderes University

Department of Computer Engineering