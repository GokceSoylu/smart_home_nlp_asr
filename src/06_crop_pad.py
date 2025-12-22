# - Read paired WAV + CSV files from DATA_FOLDER
# - Resample audio to 16 kHz
# - Add light white noise (augmentation)
# - Build a per-sample label array at 16 kHz (0 = silence, >=1 = command classes)
# - Create a dynamic silence pad from naturally silent regions
# - For each labeled command segment (label >= 1): crop, pad (silence + segment + silence), and save to OUTPUT_FOLDER/<label>

import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import warnings
import random

# Suppress non-critical warnings (you can comment this out if you want to see everything)
warnings.filterwarnings(action='ignore', category=UserWarning)

# Project Parameters (edit as needed) 
DATA_FOLDER   = 'VC_dataset_TR'                      # Folder containing paired *.wav and *.csv
OUTPUT_FOLDER = 'Noisy_16kHz_Padded_Segments' # Where processed segments will be saved
TARGET_SR     = 16000                                 # Target sample rate (Hz)

# Silence padding params
SILENCE_PAD_SEC   = 1.5   # Amount of silence to add before and after each segment (seconds)
SILENCE_CHUNK_SEC = 0.1   # Base chunk length (seconds) drawn from natural silence to build the pad

# Synthetic noise params
APPLY_NOISE     = True
NOISE_STD_RANGE = (0.001, 0.005)  # Standard deviation range for white noise amplitude

#Discovery & Output prep 
print(f"Scanning '{DATA_FOLDER}' for paired WAV + CSV files...")
csv_files = glob.glob(os.path.join(DATA_FOLDER, '*.csv'))

if len(csv_files) == 0:
    raise SystemExit(f"No CSV files found in '{DATA_FOLDER}'. Check your path and try again.")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Created output folder: {OUTPUT_FOLDER}")

total_segments_saved = 0

# Main processing loop over files 
for csv_path in csv_files:
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    wav_path  = os.path.join(DATA_FOLDER, base_name + '.wav')

    print("\n" + "="*72)
    print(f"Processing pair: {base_name}.wav  <->  {base_name}.csv")

    if not os.path.exists(wav_path):
        print(f"[WARN] WAV not found, skipping: {wav_path}")
        continue

    #Load audio + labels 
    try:
        audio_orig, sr_orig = sf.read(wav_path)     # audio_orig: float32 or float64, shape (n,) or (n, ch)
        if audio_orig.ndim > 1:
            # Convert to mono by averaging channels
            audio_orig = np.mean(audio_orig, axis=1)
        df = pd.read_csv(csv_path)                  # Expect columns: start_sample, end_sample, label
    except Exception as e:
        print(f"[ERROR] Failed to read inputs for '{base_name}': {e}")
        continue

    print(f"- Original SR: {sr_orig} Hz | Samples: {len(audio_orig)}")
    print(f"- CSV rows: {len(df)} (expected columns: start_sample, end_sample, label)")

    #Resample to 16 kHz 
    audio_16khz = librosa.resample(y=audio_orig, orig_sr=sr_orig, target_sr=TARGET_SR)
    print(f"- Resampled to {TARGET_SR} Hz | Samples: {len(audio_16khz)}")

    #Add synthetic white noise to the WHOLE track (if enabled) 
    if APPLY_NOISE:
        noise_std = random.uniform(NOISE_STD_RANGE[0], NOISE_STD_RANGE[1])
        noise     = np.random.normal(0.0, noise_std, size=audio_16khz.shape).astype(audio_16khz.dtype)
        audio_processed = audio_16khz + noise
        audio_processed = np.clip(audio_processed, -1.0, 1.0)
        print(f"- Noise added with std≈{noise_std:.6f} (clipped to [-1, 1])")
    else:
        audio_processed = audio_16khz.copy()
        print("- Noise addition disabled (APPLY_NOISE=False)")

    #Build per-sample labels aligned to 16 kHz timeline 
    labels_per_sample_16khz = np.zeros(len(audio_processed), dtype=np.int32)
    scale_factor = TARGET_SR / float(sr_orig)  # convert original sample indices to 16 kHz indices

    # Fill labels_per_sample_16khz with command labels (>=1) and 0 for unlabelled/silence
    valid_rows = 0
    for idx, row in df.iterrows():
        try:
            start_16 = int(row['start_sample'] * scale_factor)
            end_16   = int(row['end_sample']   * scale_factor)
            label    = int(row['label'])

            # Clip to audio bounds
            start_16 = max(0, min(start_16, len(audio_processed)))
            end_16   = max(0, min(end_16,   len(audio_processed)))

            if start_16 < end_16:
                labels_per_sample_16khz[start_16:end_16] = label
                valid_rows += 1
        except Exception as e:
            print(f"  [WARN] Skipping row {idx} due to parse error: {e}")

    print(f"- Label array built at 16 kHz | shape={labels_per_sample_16khz.shape} | valid rows={valid_rows}")

    #Build dynamic silence pad from naturally silent parts (label == 0) 
    pad_samples   = int(SILENCE_PAD_SEC   * TARGET_SR)
    chunk_samples = int(SILENCE_CHUNK_SEC * TARGET_SR)

    silent_indices = np.where(labels_per_sample_16khz == 0)[0]

    if len(silent_indices) < chunk_samples:
        # Not enough natural silence; fall back to zeros
        dynamic_pad = np.zeros(pad_samples, dtype=np.float32)
        used_natural_silence = False
        print(f"- Not enough natural silence. Using zero-pad of {SILENCE_PAD_SEC:.2f}s.")
    else:
        # Collect the silent samples and stitch random chunks until we reach pad length
        silent_audio_available = audio_processed[silent_indices]
        num_chunks_needed = int(np.ceil(pad_samples / chunk_samples))
        max_start_index   = max(0, len(silent_audio_available) - chunk_samples)

        pad_chunks = []
        for _ in range(num_chunks_needed):
            start_idx = random.randint(0, max_start_index)
            chunk = silent_audio_available[start_idx:start_idx + chunk_samples]
            pad_chunks.append(chunk)

        dynamic_pad = np.concatenate(pad_chunks)[:pad_samples].astype(np.float32)
        used_natural_silence = True
        print(f"- Built dynamic pad from natural silence | length={len(dynamic_pad)} samples "
              f"({SILENCE_PAD_SEC:.2f}s), chunk={SILENCE_CHUNK_SEC:.2f}s")

    #Iterate over labeled command segments, crop + pad + save 
    file_segment_count = 0

    for i, row in df.iterrows():
        try:
            label = int(row['label'])
        except Exception:
            continue

        # Only process command classes (>=1). Skip class 0.
        if label == 0:
            continue

        start_16 = int(row['start_sample'] * scale_factor)
        end_16   = int(row['end_sample']   * scale_factor)

        start_16 = max(0, min(start_16, len(audio_processed)))
        end_16   = max(0, min(end_16,   len(audio_processed)))
        if start_16 >= end_16:
            continue

        segment = audio_processed[start_16:end_16].astype(np.float32)

        # Compose: [pad] + segment + [pad]
        padded_segment = np.concatenate((dynamic_pad, segment, dynamic_pad))

        # Prepare output path by label
        label_folder = os.path.join(OUTPUT_FOLDER, str(label))
        os.makedirs(label_folder, exist_ok=True)

        out_name = f"{base_name}_{i+1:03d}_{label}.wav"
        out_path = os.path.join(label_folder, out_name)

        try:
            sf.write(out_path, padded_segment, TARGET_SR, format='WAV')
            file_segment_count += 1
        except Exception as e:
            print(f"  [ERROR] Could not write segment (label={label}) to '{out_path}': {e}")

    total_segments_saved += file_segment_count
    print(f"- Saved {file_segment_count} padded segment(s) for '{base_name}'. "
          f"{'Natural pad used' if used_natural_silence else 'Zero pad used'}.")

print("\n" + "-"*72)
print("DONE.")
print(f"Total saved segments: {total_segments_saved}")
print(f"Output root: {OUTPUT_FOLDER}")
