"""
data_loading.py

This module loads raw data for:
- WESAD: Stress detection dataset
- PAMAP2: Activity recognition dataset

Output format:
{
    "subject_id": {
        "signal_name": numpy_array,   # shape = (time,)
        ...
        "label": numpy_array          # optional, depending on dataset
    }
}
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# ===========================
#   WESAD LOADER
# ===========================
import pickle

def load_wesad_pkl(pkl_path: str):
    """
    Load WESAD synchronized .pkl file.
    This file contains:
    - chest signals (ECG, EDA, RESP, EMG, ACC, TEMP)
    - wrist signals (ACC, BVP, EDA, TEMP)
    - labels
    Already synchronized and aligned.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    subject_id = data["subject"]

    # Flatten structure into consistent dict
    flat = {}

    # Chest signals
    for k, v in data["signal"]["chest"].items():
        flat[f"chest_{k.lower()}"] = np.array(v)

    # Wrist signals
    for k, v in data["signal"]["wrist"].items():
        flat[f"wrist_{k.lower()}"] = np.array(v)

    # Labels
    flat["label"] = np.array(data["label"]).astype(int)

    return subject_id, flat


def load_wesad_pkl_dataset(path: str):
    """
    Load all WESAD .pkl subjects from the dataset folder.
    """
    dataset_path = Path(path)
    subjects = {}

    for folder in dataset_path.iterdir():
        if folder.is_dir() and folder.name.startswith("S"):

            pkl_file = folder / f"{folder.name}.pkl"
            if pkl_file.exists():
                sid, data = load_wesad_pkl(pkl_file)
                subjects[sid] = data
            else:
                print(f"[WESAD] Skipping {folder} (no pkl file)")

    print(f"[WESAD] Loaded {len(subjects)} subjects from PKL")
    return subjects
    
def load_wesad_subject(subject_path: str):
    """
    Load one subject folder from WESAD dataset.
        ACC.csv
        BVP.csv
        EDA.csv
        ECG.csv
        RESP.csv
        TEMP.csv
        LABELS.csv
    """

    subject_data = {}
    subject_path = Path(subject_path)

    if not subject_path.exists():
        raise FileNotFoundError(f"[WESAD] Subject path not found: {subject_path}")

    # Iterate through CSV files
    for file in subject_path.iterdir():
        if file.suffix == ".csv":
            name = file.stem.lower()  # e.g., 'acc', 'eda', 'labels'
            df = pd.read_csv(file, header=None)
            arr = df.values.squeeze()

            # Labels are typically categorical
            if "label" in name:
                subject_data["label"] = arr.astype(int)
            else:
                subject_data[name] = arr.astype(float)

    print(f"[WESAD] Loaded subject from {subject_path}")
    return subject_data


def load_wesad_dataset(path: str):
    """
    Load all WESAD subjects under raw directory.
    """
    dataset_path = Path(path)
    subjects = {}

    for folder in dataset_path.iterdir():
        if folder.is_dir() and folder.name.startswith("S"):
            subject_id = folder.name
            subjects[subject_id] = load_wesad_subject(folder)

    print(f"[WESAD] Loaded {len(subjects)} subjects.")
    return subjects


# ===========================
#   PAMAP2 LOADER
# ===========================

def load_pamap2_file(file_path: str):
    """
    Load one PAMAP2 data file (.dat).
    Each file corresponds to one subject performing multiple activities.

    PAMAP2 format (columns):
        0: timestamp
        1: activity_label
        2: heart_rate
        3-5: IMU 1: acc (x,y,z)
        6-17: IMU 1: gyro/mag
        ...
        (multiple sensors)

    We only load subset needed for ML:
        - activity labels
        - heart rate
        - accelerometer (x,y,z) from IMU1
    """

    df = pd.read_csv(
        file_path,
        sep=" ",
        header=None,
        comment="#",
        low_memory=False
    ).dropna(axis=1, how="all")

    arr = df.values
    data = {
        "activity": arr[:, 1].astype(int),
        "heart_rate": arr[:, 2].astype(float),
        "acc_x": arr[:, 3].astype(float),
        "acc_y": arr[:, 4].astype(float),
        "acc_z": arr[:, 5].astype(float),
    }

    print(f"[PAMAP2] Loaded file: {file_path}")
    return data


def load_pamap2_dataset(path: str):
    """
    Load all PAMAP2 subjects.
    Raw folder contains multiple .dat files:
        subject101.dat
        subject102.dat
        ...

    Output:
    {
        "subject101": { "activity": ..., "acc_x": ... },
        ...
    }
    """
    dataset_path = Path(path)
    subjects = {}

    for file in dataset_path.iterdir():
        if file.suffix == ".dat":
            subject_id = file.stem
            subjects[subject_id] = load_pamap2_file(file)

    print(f"[PAMAP2] Loaded {len(subjects)} subjects.")
    return subjects


# ===========================
#   UNIFIED LOADER
# ===========================

def load_dataset(dataset_name: str, path: str):
    """
    Unified loader for both datasets.
    """
    if dataset_name.lower() == "wesad":
        return load_wesad_dataset(path)

    elif dataset_name.lower() == "pamap2":
        return load_pamap2_dataset(path)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")