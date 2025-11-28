"""
features.py

Extract handcrafted behavioral features:
- HRV (SDNN, RMSSD, pNN50, mean HR)
- EDA (tonic mean, number of peaks, mean peak amplitude)
- Respiration (breathing rate & variability)
- Accelerometer (mean, std, energy, SMA)

Input:
    window (channels × time)
    channel_names = list of channel strings

Output:
    feature dict {feature_name: value}
"""

import numpy as np
import neurokit2 as nk


# ===========================================================
# HRV FROM ECG OR BVP (PPG)
# ===========================================================
def extract_hrv(signal, sampling_rate):
    try:
        # Clean ECG/BVP
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
        # R-peaks detection
        peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        rpeaks = np.where(peaks["ECG_R_Peaks"] == 1)[0]

        if len(rpeaks) < 2:
            return {"hr_mean": np.nan, "sdnn": np.nan,
                    "rmssd": np.nan, "pnn50": np.nan}

        # HRV metrics
        hrv = nk.hrv_time(peaks, sampling_rate=sampling_rate)

        return {
            "hr_mean": hrv.get("HRV_MeanNN", np.nan),
            "sdnn": hrv.get("HRV_SDNN", np.nan),
            "rmssd": hrv.get("HRV_RMSSD", np.nan),
            "pnn50": hrv.get("HRV_pNN50", np.nan)
        }
    except:
        return {"hr_mean": np.nan, "sdnn": np.nan,
                "rmssd": np.nan, "pnn50": np.nan}


# ===========================================================
# EDA FEATURES
# ===========================================================
def extract_eda(eda_signal, sampling_rate):
    try:
        eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate)
        eda_df = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)

        tonic = np.mean(eda_df["EDA_Tonic"])
        peaks = eda_df["SCR_Peaks"].values
        n_peaks = np.sum(peaks)
        amp = np.mean(eda_df["SCR_Amplitude"].values)

        return {
            "eda_tonic_mean": tonic,
            "eda_n_peaks": n_peaks,
            "eda_peak_amp_mean": amp
        }
    except:
        return {"eda_tonic_mean": np.nan,
                "eda_n_peaks": np.nan,
                "eda_peak_amp_mean": np.nan}


# ===========================================================
# RESPIRATION FEATURES
# ===========================================================
def extract_resp(resp_signal, sampling_rate):
    try:
        peaks, _ = nk.rsp_peaks(resp_signal, sampling_rate=sampling_rate)
        peak_locs = np.where(peaks["RSP_Peaks"] == 1)[0]

        if len(peak_locs) < 2:
            return {"resp_rate": np.nan,
                    "resp_variability": np.nan}

        resp_rate = nk.rsp_rate(peaks, sampling_rate=sampling_rate)
        variability = np.std(np.diff(peak_locs))

        return {
            "resp_rate": np.mean(resp_rate),
            "resp_variability": variability
        }
    except:
        return {"resp_rate": np.nan,
                "resp_variability": np.nan}


# ===========================================================
# ACC FEATURES
# ===========================================================
def extract_acc(acc_x, acc_y, acc_z):
    mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    return {
        "acc_mean": np.mean(mag),
        "acc_std": np.std(mag),
        "acc_energy": np.sum(mag ** 2),
        "acc_sma": np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z))
    }


# ===========================================================
# MAIN FEATURE EXTRACTOR
# ===========================================================
def extract_features_from_window(window, channels, sampling_rate):
    features = {}

    # Build dict: channel_name -> signal
    data = {ch: window[i] for i, ch in enumerate(channels)}

    # HRV from ECG or BVP
    if "ecg" in data:
        features.update(extract_hrv(data["ecg"], sampling_rate))
    elif "bvp" in data:
        features.update(extract_hrv(data["bvp"], sampling_rate))

    # EDA
    if "eda" in data:
        features.update(extract_eda(data["eda"], sampling_rate))

    # Resp
    if "resp" in data:
        features.update(extract_resp(data["resp"], sampling_rate))

    # ACC
    acc_channels = [k for k in data.keys() if "acc" in k]
    if len(acc_channels) >= 3:
        features.update(
            extract_acc(
                data.get("acc_x", data[acc_channels[0]]),
                data.get("acc_y", data[acc_channels[1]]),
                data.get("acc_z", data[acc_channels[2]]),
            )
        )

    return features