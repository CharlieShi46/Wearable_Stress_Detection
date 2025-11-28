"""
windows.py

Sliding-window segmentation for wearable time-series data.

Input:
    processed_subjects = {
        "S2": {
            "ecg": np.ndarray,
            "eda": np.ndarray,
            "acc": np.ndarray,
            ...
            "label": np.ndarray
        },
        ...
    }

Output:
    A list/dict of windowed samples:
        {
            "X": np.array of shape (num_windows, num_channels, window_length),
            "y": labels per window,
            "subject_ids": per-window subject mapping
        }

Supports:
- variable window size (seconds)
- stride (seconds)
- multi-channel concatenation
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


def create_windows_for_subject(subject_data,
                               subject_id,
                               sampling_rate,
                               window_size_sec,
                               stride_sec):
    """
    Segment one subject's signals into sliding windows.
    """

    # Determine window length in samples
    win_len = int(window_size_sec * sampling_rate)
    stride = int(stride_sec * sampling_rate)

    # Identify channels (exclude label)
    channels = [k for k in subject_data.keys() if k != "label"]
    n_channels = len(channels)

    # Convert channel signals to same-length arrays
    min_len = min(len(subject_data[ch]) for ch in channels)
    for ch in channels:
        subject_data[ch] = subject_data[ch][:min_len]

    # Labels (per-sample)
    labels = subject_data["label"][:min_len]

    # Prepare storage
    X_list = []
    y_list = []
    sid_list = []

    start = 0
    while start + win_len <= min_len:
        end = start + win_len

        # Build window (channels x time)
        window = np.zeros((n_channels, win_len))
        for idx, channel in enumerate(channels):
            window[idx] = subject_data[channel][start:end]

        # Window label = majority label
        window_label = np.bincount(labels[start:end]).argmax()

        X_list.append(window)
        y_list.append(window_label)
        sid_list.append(subject_id)

        start += stride

    return X_list, y_list, sid_list


def create_windows(processed_subjects,
                   sampling_rate,
                   window_size_sec,
                   stride_sec):
    """
    Segment all subjects into windows.
    """

    X_all = []
    y_all = []
    subject_all = []

    for subject_id, subject_data in tqdm(processed_subjects.items(), desc="Windowing"):
        X_list, y_list, sid_list = create_windows_for_subject(
            subject_data,
            subject_id,
            sampling_rate,
            window_size_sec,
            stride_sec
        )
        X_all.extend(X_list)
        y_all.extend(y_list)
        subject_all.extend(sid_list)

    X_all = np.array(X_all)         # shape: (N, C, T)
    y_all = np.array(y_all)
    subject_all = np.array(subject_all)

    return {
        "X": X_all,
        "y": y_all,
        "subject_ids": subject_all
    }