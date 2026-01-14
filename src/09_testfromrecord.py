# -*- coding: utf-8 -*-

"""
Realtime Model Test (Tutorial Style)

This script demonstrates a clean, tutorial-oriented workflow for:
1) Selecting a pretrained model bundle (joblib) by language and window length
2) Loading a WAV file
3) Running sliding-window inference during playback
4) (Optional) Producing a simple "ground truth" label from the folder name

Design choices for a tutorial feel:
- Explanations are provided only via triple-quoted strings (no synthetic separators).
- Console messages are short, explanatory, and C1 English.
- The code avoids decorative ASCII blocks such as "=====".

Prerequisites (typical):
pip install numpy librosa sounddevice scipy scikit-learn joblib

Notes:
- Window length must match the training window of the chosen model.
- Hop controls how frequently predictions are emitted during playback.
"""

import os
import glob
import time
import numpy as np
import librosa
#import sounddevice as sd
import joblib
from scipy.signal import butter, filtfilt


"""
User settings
- WAV_PATH: the sample audio to test
- MODELS_SEARCH_DIR: where trained *.joblib bundles live (recursively)
- LANGUAGE: "TR" or "EN"
- REQUEST_WINDOW_SEC: 1.0 / 1.5 / 2.0 (must match the model you pick)
- REQUEST_FEATURE: optionally constrain selection to "MEL" or "MFCC"
- REQUEST_HOP_SEC: how often to predict (e.g., 0.1 seconds)
"""
WAV_PATH = r"Noisy_16kHz_DynamicPadded_Segments\1\TR_201805074_001_1.wav"
MODELS_SEARCH_DIR = r"reports"

LANGUAGE = "TR"
REQUEST_WINDOW_SEC = 1.5
REQUEST_FEATURE = None  # None, "MEL", "MFCC"
REQUEST_HOP_SEC = 0.1

"""
Output controls
- PRINT_EVERY_N: print every N-th prediction (useful if hop is small)
- PRINT_LOW_PROB: whether to print low-confidence predictions
- MIN_PROB_FOR_COMMAND: a threshold used only if PRINT_LOW_PROB is False
"""
PRINT_EVERY_N = 1
PRINT_LOW_PROB = True
MIN_PROB_FOR_COMMAND = 0.0

"""
Optional "ground truth" printing (tutorial/demo use)
- If TRUE_LABEL_FROM_FOLDER=True and the parent folder name is a digit, that digit is used as the true label.
- If your dataset uses padded silence at the beginning and end, the helper below can label those regions as 0.
"""
PRINT_GT = True
TRUE_LABEL_FROM_FOLDER = True
TRUE_LABEL_OVERRIDE = None

PAD_SIL_SEC = 1.5
MID_MIN_COVER_RATIO = 0.5

"""
Bandpass filter parameters often used in audio command pipelines
If your training pipeline applied a similar preprocessing step, keep these consistent.
"""
BP_FMIN = 100.0
BP_FMAX = 6000.0
BP_ORDER = 6


def butter_bandpass_filter(y, sr, fmin=BP_FMIN, fmax=BP_FMAX, order=BP_ORDER):
    """
    Apply a Butterworth bandpass filter.
    This is a common preprocessing step to reduce low-frequency rumble and very high-frequency noise.
    """
    nyq = 0.5 * sr
    low = fmin / nyq
    high = fmax / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y).astype(np.float32)


def window_iter_full(y, win_len, hop_len):
    """
    Yield (start_index, end_index, window_samples) over a signal y.

    If the last window exceeds the signal length, we zero-pad it.
    This keeps the feature vector length stable and prevents shape mismatch at the tail.
    """
    L = len(y)
    start = 0
    while start < L:
        end = start + win_len
        if end <= L:
            w = y[start:end]
        else:
            w = np.zeros(win_len, dtype=np.float32)
            seg = y[start:]
            w[:len(seg)] = seg
        yield start, end, w
        start += hop_len


def overlap_ratio(a_start, a_end, b_start, b_end):
    """
    Compute inter(a,b) / len(a). This is used to decide whether a window overlaps the "middle" region enough.
    """
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    inter = max(0.0, right - left)
    return inter / max(1e-9, (a_end - a_start))


def gt_label_for_window(start_sec, end_sec, file_duration_sec, true_class_label):
    """
    A simple labeling rule for padded segments:

    - We define a middle region that excludes PAD_SIL_SEC seconds at both ends.
    - If a prediction window overlaps the middle region enough, we label it as the (true) command.
    - Otherwise, we label it as 0 (silence/background).
    """
    mid_start = PAD_SIL_SEC
    mid_end = max(PAD_SIL_SEC, file_duration_sec - PAD_SIL_SEC)
    r_mid = overlap_ratio(start_sec, end_sec, mid_start, mid_end)
    return int(true_class_label) if r_mid >= MID_MIN_COVER_RATIO else 0


def mel_features_from_window(w, sr, n_fft, hop_len_feat, n_mels, fmin, fmax):
    """
    Extract a flattened MEL-spectrogram vector (in dB).
    Flattening keeps the feature interface simple for classic ML pipelines.
    """
    S = librosa.feature.melspectrogram(
        y=w,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_len_feat,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        center=False
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.flatten().astype(np.float32)


def mfcc_features_from_window(w, sr, n_fft, hop_len_feat, n_mfcc):
    """
    Extract a flattened MFCC vector.
    MFCCs are a classic baseline representation for speech/command recognition tasks.
    """
    mfcc = librosa.feature.mfcc(
        y=w,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_len_feat,
        center=False
    )
    return mfcc.flatten().astype(np.float32)


def infer_true_label_from_path(wav_path):
    """
    If the parent folder is a digit (e.g., .../12/file.wav), treat it as the class label.
    This matches a common dataset layout for command classification.
    """
    parent = os.path.basename(os.path.dirname(wav_path))
    return int(parent) if parent.isdigit() else None


def pick_model_path(models_root, lang, window_sec, feature=None):
    """
    Select the newest joblib bundle matching the requested language and window length.

    Expected filename conventions (example):
    - contains 'best_TR_' or 'best_EN_'
    - contains '_win1.5_'
    - optionally contains 'feat-MEL' or 'feat-MFCC'
    """
    win_tag = f"_win{window_sec:.1f}_"
    lang_tag = f"best_{lang}_"
    feature_tag = f"feat-{feature}" if feature is not None else None

    candidates = []
    for p in glob.glob(os.path.join(models_root, "**", "*.joblib"), recursive=True):
        bn = os.path.basename(p)
        if (lang_tag in bn) and (win_tag in bn):
            if feature_tag is not None and (feature_tag not in bn):
                continue
            candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def load_bundle(model_path):
    """
    Load a joblib bundle and enforce that it includes a 'pipeline'.
    This reduces confusion when the wrong file is accidentally selected.
    """
    b = joblib.load(model_path)
    if "pipeline" not in b:
        raise ValueError(
            "The selected joblib file does not look like a model bundle.\n"
            "It should contain a 'pipeline' object used for prediction."
        )
    return b


if __name__ == "__main__":
    """
    Main tutorial flow:
    - Validate inputs
    - Select and load a model bundle
    - Load audio and run inference windows during playback
    """

    if not os.path.exists(WAV_PATH):
        raise FileNotFoundError(f"WAV file not found: {WAV_PATH}")

    model_path = pick_model_path(MODELS_SEARCH_DIR, LANGUAGE, REQUEST_WINDOW_SEC, REQUEST_FEATURE)
    if model_path is None:
        raise FileNotFoundError(
            "No matching model bundle was found.\n"
            f"Search directory: {MODELS_SEARCH_DIR}\n"
            f"Requested: language={LANGUAGE}, window={REQUEST_WINDOW_SEC}, feature={REQUEST_FEATURE}\n"
            "Tip: ensure your model filename includes: 'best_{LANG}' and '_win{X.X}_'."
        )

    bundle = load_bundle(model_path)
    pipeline = bundle["pipeline"]
    feature_type = str(bundle.get("feature_type", "MEL")).upper()
    label_map = bundle.get("label_map", {})
    sr = int(bundle.get("sr", 16000))

    """
    Window length must match training.
    We read window_sec from the bundle to avoid accidental mismatches.
    """
    window_sec = float(bundle.get("window_sec", REQUEST_WINDOW_SEC))
    hop_sec = float(REQUEST_HOP_SEC)
    if hop_sec <= 0:
        raise ValueError("REQUEST_HOP_SEC must be > 0.")

    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    """
    Feature parameters are typically stored in the bundle.
    We include robust fallbacks to keep the demo runnable.
    """
    n_fft = int(bundle.get("n_fft", 1024))
    hop_len_feat = int(bundle.get("hop_len_feat", n_fft // 2))
    n_mels = int(bundle.get("n_mels", 20))
    n_mfcc = int(bundle.get("n_mfcc", 20))
    fmin_mel = int(bundle.get("fmin_mel", 100))
    fmax_mel = int(bundle.get("fmax_mel", 6000))

    """
    Ground truth selection for demo printing
    """
    true_label = None
    if PRINT_GT:
        if TRUE_LABEL_FROM_FOLDER:
            true_label = infer_true_label_from_path(WAV_PATH)
            if true_label is None:
                raise ValueError(
                    "Ground-truth inference from folder name failed.\n"
                    "Either rename the parent folder to a digit class label, or set:\n"
                    "TRUE_LABEL_FROM_FOLDER=False and TRUE_LABEL_OVERRIDE=<int>."
                )
        else:
            if TRUE_LABEL_OVERRIDE is None:
                raise ValueError(
                    "TRUE_LABEL_FROM_FOLDER is False but TRUE_LABEL_OVERRIDE is not provided.\n"
                    "Please set TRUE_LABEL_OVERRIDE (e.g., 12)."
                )
            true_label = int(TRUE_LABEL_OVERRIDE)

    print(f"Model selected: {model_path}")
    print(f"Model metadata: language={bundle.get('language','?')}, feature={feature_type}, window={window_sec}s, hop={hop_sec}s, sr={sr}")

    if PRINT_GT:
        print(f"Demo ground truth: true_label={true_label} (padded-silence rule: PAD_SIL_SEC={PAD_SIL_SEC}s)")

    """
    Load the WAV file
    """
    y, _ = librosa.load(WAV_PATH, sr=sr)
    dur = len(y) / sr
    print(f"Audio loaded: {WAV_PATH}")
    print(f"Duration: {dur:.2f}s. Predictions will be emitted about every {hop_sec:.2f}s.")

    """
    Apply filtering if used in training
    """
    y_f = butter_bandpass_filter(y.astype(np.float32), sr)

    """
    Start playback and run inference aligned with playback time.
    This makes the test feel "realtime" rather than batch-only.
    """
    sd.stop()
    sd.play(y, sr, blocking=False)
    t0 = time.time()

    """
    First-window sanity check: if feature length is wrong, we want a clear error early.
    """
    first_check_done = False

    step = 0
    for start, end, w in window_iter_full(y_f, win_len, hop_len):
        step += 1

        center_time = ((start + end) / 2) / sr
        while time.time() - t0 < center_time:
            time.sleep(0.001)

        if feature_type == "MEL":
            fv = mel_features_from_window(w, sr, n_fft, hop_len_feat, n_mels, fmin_mel, fmax_mel)
        else:
            fv = mfcc_features_from_window(w, sr, n_fft, hop_len_feat, n_mfcc)

        X_win = fv.reshape(1, -1)

        if not first_check_done:
            try:
                _ = pipeline.predict(X_win)
            except Exception as ex:
                raise RuntimeError(
                    "Input dimensionality mismatch.\n"
                    "This usually means the selected model was trained with a different window length or feature configuration.\n"
                    f"Selected model: {model_path}\n"
                    f"Using: window_sec={window_sec}, feature_vector_len={fv.size}\n"
                    f"Original error: {ex}"
                )
            first_check_done = True

        """
        Predict and, if supported, compute the confidence from predict_proba.
        """
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_win)[0]
            pred = int(np.argmax(proba))
            p = float(np.max(proba))
        else:
            pred = int(pipeline.predict(X_win)[0])
            p = 1.0

        if (step % PRINT_EVERY_N) != 0:
            continue
        if (not PRINT_LOW_PROB) and (p < MIN_PROB_FOR_COMMAND):
            continue

        pred_text = label_map.get(pred, f"class_{pred}")

        if PRINT_GT:
            start_sec = start / sr
            end_sec = end / sr
            gt = gt_label_for_window(start_sec, end_sec, dur, true_label)
            gt_text = label_map.get(gt, f"class_{gt}")
            ok = "match" if pred == gt else "mismatch"

            print(f"t={center_time:5.2f}s | step={step:4d} | GT={gt:2d} ({gt_text}) | PRED={pred:2d} ({pred_text}) | p={p:.2f} | {ok}")
        else:
            print(f"t={center_time:5.2f}s | step={step:4d} | PRED={pred:2d} ({pred_text}) | p={p:.2f}")

    sd.wait()
    print("Finished. Realtime inference demo is complete.")
