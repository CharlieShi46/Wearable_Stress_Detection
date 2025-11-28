"""
preprocessing.py

Processing steps for WESAD (pkl-based):
- resampling (to 32 Hz)
- filtering
- normalization
- trimming bad segments
- output clean multimodal signals
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample
from pathlib import Path

# --------------------------------------------------
# 1. Filters
# --------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut/nyq, highcut/nyq
    return butter(order, [low, high], btype='band')

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, cutoff/nyq, btype='low')

def apply_filter(sig, fs, low=None, high=None, type="band"):
    if sig is None:
        return None
    if np.isnan(sig).all():
        return sig

    if type == "band":
        b, a = butter_bandpass(low, high, fs)
    elif type == "low":
        b, a = butter_lowpass(high, fs)
    else:
        return sig

    return filtfilt(b, a, sig)


# --------------------------------------------------
# 2. Resampling
# --------------------------------------------------

def resample_signal(signal, orig_fs, target_fs=32):
    """
    Resample 1D or 2D signal to target sampling rate.
    """
    if signal.ndim == 1:
        n_samples = int(len(signal) * target_fs / orig_fs)
        return resample(signal, n_samples)

    elif signal.ndim == 2:
        n_samples = int(signal.shape[0] * target_fs / orig_fs)
        out = []
        for ch in range(signal.shape[1]):
            out.append(resample(signal[:, ch], n_samples))
        return np.stack(out, axis=1)

    return signal


# --------------------------------------------------
# 3. Normalization
# --------------------------------------------------

def zscore(sig):
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-6)


def normalize_subject(data_dict):
    """
    Normalize all signals except labels.
    """
    out = {}
    for k, v in data_dict.items():
        if k == "label":
            out[k] = v
        else:
            if v.ndim == 1:
                out[k] = zscore(v)
            else:
                out[k] = np.stack([zscore(v[:, i]) for i in range(v.shape[1])], axis=1)
    return out


# --------------------------------------------------
# 4. Master preprocessing
# --------------------------------------------------

def preprocess_wesad_subject(subject_dict, orig_fs_chest=700, orig_fs_wrist=32, target_fs=32):
    """
    subject_dict = output from .pkl
    Returns cleaned, resampled, normalized data.
    """

    processed = {}

    # -------------------------
    # Chest signals
    # -------------------------
    chest_signals = {
        "chest_ecg": ("band", 0.5, 40),
        "chest_eda": ("low", None, 1),
        "chest_resp": ("band", 0.1, 1),
        "chest_emg": ("band", 20, 200),
        "chest_temp": ("low", None, 0.5),
        "chest_acc": ("none", None, None)
    }

    for name, (ftype, low, high) in chest_signals.items():
        if name not in subject_dict:
            continue

        sig = subject_dict[name]

        # 1) original sampling is 700Hz for chest
        sig_rs = resample_signal(sig, orig_fs_chest, target_fs)

        # 2) filtering
        if ftype == "band":
            sig_f = apply_filter(sig_rs, target_fs, low, high, "band")
        elif ftype == "low":
            sig_f = apply_filter(sig_rs, target_fs, None, high, "low")
        else:
            sig_f = sig_rs

        processed[name] = sig_f

    # -------------------------
    # Wrist signals (already 32 Hz)
    # -------------------------
    wrist_signals = ["wrist_acc", "wrist_bvp", "wrist_eda", "wrist_temp"]

    for name in wrist_signals:
        if name not in subject_dict:
            continue

        sig = subject_dict[name]

        if sig.ndim == 1:       # BVP, EDA, Temp
            sig_rs = sig
        else:                   # Accelerometer 3D
            sig_rs = sig

        # filtering for wrist BVP/EDA
        if "eda" in name:
            sig_f = apply_filter(sig_rs, target_fs, None, 1, "low")
        elif "bvp" in name:
            sig_f = apply_filter(sig_rs, target_fs, 0.5, 8, "band")
        else:
            sig_f = sig_rs

        processed[name] = sig_f

    # -------------------------
    # Labels
    # -------------------------
    processed["label"] = subject_dict["label"]

    # -------------------------
    # Normalize
    # -------------------------
    processed = normalize_subject(processed)

    return processed


# --------------------------------------------------
# 5. Dataset-level Preprocessing
# --------------------------------------------------

def preprocess_wesad_dataset(raw_dict, target_fs=32):
    """
    raw_dict = output of load_wesad_pkl_dataset()
    Returns: { subject_id : cleaned_data }
    """
    out = {}
    for sid, data in raw_dict.items():
        print(f"[Preprocess] Processing {sid} ...")
        out[sid] = preprocess_wesad_subject(data, target_fs=target_fs)

    return out