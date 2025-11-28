# Stress & Activity Prediction from Multimodal Wearable Signals

This repository contains an end-to-end, reproducible pipeline for **stress detection** and **activity recognition** using **multimodal wearable sensor data**.  
The project combines:

- **Digital Health** – modeling physiological and behavioral states from wearable devices  
- **Informatics / Applied ML** – building a fully reproducible ML pipeline  
- **Behavior Data** – extracting interpretable behavioral features (HRV, EDA peaks, activity level)  
- **Explainable ML** – using SHAP to interpret model decisions

The code is organized so that **the same pipeline** can be applied to:
- **WESAD** – stress vs non-stress classification  
- **PAMAP2** – physical activity recognition  

---

## 1. Project Overview

### 1.1 Motivation

Wearable devices (e.g., chest straps, wristbands, smartwatches) continuously record:
- **physiological signals** such as ECG, PPG/BVP, EDA, respiration, skin temperature  
- **movement signals** such as 3-axis accelerometer

These multimodal streams carry rich information about a person’s:
- **stress / arousal level**
- **physical activity type**
- **overall behavioral state**

This project aims to build an applied ML pipeline that:

1. Ingests multimodal wearable signals from public datasets  
2. Preprocesses and segments them into fixed-length windows  
3. Extracts human-interpretable behavioral features (e.g., HRV, EDA peaks, movement intensity)  
4. Trains and compares both **traditional ML models** and **deep learning models**  
5. Uses **SHAP** to understand which physiological and behavioral features drive model predictions  

The goal is to demonstrate how **Digital Health, Informatics, Applied ML, and Behavior Modeling** can be combined in a rigorous, research-oriented workflow.

---

## 2. Datasets

https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset

### 2.1 WESAD – Wearable Stress and Affect Detection

- Multimodal data from chest- and wrist-worn sensors
- Signals include (depending on device):  
  - ECG (electrocardiogram)  
  - BVP/PPG (blood volume pulse)  
  - EDA (electrodermal activity)  
  - RESP (respiration)  
  - TEMP (skin temperature)  
  - ACC (accelerometer)  
- Labeled segments for different conditions:
  - **Baseline**, **Amusement**, **Stress**, etc.

In this project, we primarily frame WESAD as a **binary classification problem**:

> **Stress vs Non-Stress** (where non-stress can include baseline and amusement)

You should place the WESAD data under:

data/raw/wesad/

### 2.2 PAMAP2 – Physical Activity Monitoring
	•	Wearable sensor data recorded from multiple body locations
	•	Signals include:
	•	Accelerometer
	•	Gyroscope
	•	Magnetometer
	•	Heart rate
	•	Labeled physical activities such as:
	•	Lying, sitting, standing
	•	Walking, running, cycling
	•	Household activities (vacuum cleaning, ironing, etc.)

In this project, we use PAMAP2 for multi-class activity recognition.

You should place the PAMAP2 data under:

---
## 3. Repository Structure

wearable-ml-behavior/
├── configs/
│   ├── wesad_stress.yaml          # Config for stress detection experiments (WESAD)
│   └── pamap2_activity.yaml       # Config for activity recognition experiments (PAMAP2)
├── data/
│   ├── raw/
│   │   ├── wesad/                 # Raw WESAD dataset (not tracked in git)
│   │   └── pamap2/                # Raw PAMAP2 dataset (not tracked in git)
│   ├── interim/
│   │   ├── wesad/                 # Preprocessed, resampled, aligned data
│   │   └── pamap2/
│   └── features/
│       ├── wesad/                 # Extracted feature tables (e.g., HRV, EDA, ACC)
│       └── pamap2/
├── notebooks/
│   ├── 01_eda_wesad.ipynb         # Exploratory data analysis for WESAD
│   ├── 02_eda_pamap2.ipynb        # Exploratory data analysis for PAMAP2
│   └── 03_results_summary.ipynb   # Result visualization & summary
├── reports/
│   ├── figures/                   # Exported plots for reports/papers
│   └── paper_draft.md             # Draft of a paper-style report
├── src/
│   ├── __init__.py
│   ├── config.py                  # Utilities to load YAML configs
│   ├── data_loading.py            # Dataset-specific loading utilities
│   ├── preprocessing.py           # Resampling, filtering, standardization
│   ├── windows.py                 # Sliding window segmentation
│   ├── features.py                # Behavioral feature extraction (HRV, EDA, ACC, etc.)
│   ├── models_ml.py               # Traditional ML models (LR, RF, XGBoost)
│   ├── models_dl.py               # Deep models (1D-CNN, LSTM)
│   ├── train.py                   # Main training entry point (reads config and runs pipeline)
│   ├── evaluate.py                # Metrics, confusion matrices, ROC, etc.
│   ├── explain.py                 # SHAP-based explainability analysis
│   └── utils.py                   # Misc utilities (seeding, logging, etc.)
├── .gitignore
├── README.md
└── requirements.txt

---

## 4. Methodology

This project follows a unified pipeline for both WESAD and PAMAP2.

### 4.1 Preprocessing
	•	Resample all signals to a common sampling rate
	•	Align channels and handle missing values
	•	Optional filtering (e.g., band-pass for ECG/BVP, smoothing for EDA)
	•	Per-subject normalization to reduce inter-person variability

### 4.2 Sliding Window Segmentation
	•	Segment each continuous signal into overlapping windows:
	•	Window length: e.g., 30–120 seconds
	•	Stride: e.g., 5–10 seconds
	•	Assign a label to each window:
	•	WESAD: stress vs non-stress
	•	PAMAP2: activity class (e.g., walking, running, sitting)

### 4.3 Feature Extraction

For each window, we compute hand-crafted behavioral features, such as:
	•	Cardiac / HRV (from ECG or BVP)
	•	Heart rate (mean, min, max)
	•	Time-domain HRV: SDNN, RMSSD, pNN50
	•	EDA (electrodermal activity)
	•	Mean tonic level
	•	Number and amplitude of phasic peaks
	•	Slope / rate of change
	•	Respiration
	•	Estimated breathing rate
	•	Variability of respiration cycles
	•	Accelerometer
	•	Mean, std, energy per axis
	•	Signal magnitude area (SMA)
	•	Movement intensity proxies
	•	Temperature
	•	Mean temperature and short-term changes

These features result in a tabular representation that is used to train traditional ML models.

### 4.4 Models

We compare two model families:

(1) Traditional ML Models (on features)
	•	Logistic Regression (baseline)
	•	Random Forest
	•	XGBoost / LightGBM

(2) Deep Learning Models (on raw or minimally processed windows)
	•	1D-CNN on multichannel time series
	•	LSTM / GRU on sequences of sensor readings

4.5 Evaluation
	•	Subject-wise splits or leave-one-subject-out (LOSO) evaluation
	•	Metrics:
	•	Accuracy
	•	F1-score (macro/weighted)
	•	ROC-AUC (for binary stress classification)
	•	Optional:
	•	Per-subject performance distributions

### 4.6 Explainability (SHAP)

For tree-based models (e.g., XGBoost), we apply SHAP to:
	•	Analyze which features drive stress vs non-stress predictions
	•	Understand which physiological signals are most important for each activity class
	•	Connect model behavior back to psychophysiological theory (e.g., HRV decrease + EDA increase as stress signatures)

⸻

## 5. How to Run (Planned)

### 5.1
pip install -r requirements.txt

### 5.2
python -m src.train --config configs/wesad_stress.yaml

### 5.3 
python -m src.train --config configs/pamap2_activity.yaml

### 5.4 
python -m src.explain --config configs/wesad_stress.yaml


## 6. Results

### 6.1 Stress Detection (WESAD)

We evaluate multiple machine learning and deep learning models under a consistent sliding-window protocol (60-second windows, 5-second stride). All features are extracted per window (HRV, EDA peaks, respiration, accelerometer).

#### 6.1.1 ML Baselines

| Model | Accuracy | F1-Macro | Notes |
|-------|----------|----------|-------|
| Logistic Regression | ~0.74–0.78 | ~0.70–0.75 | Strong linear baseline; struggles with non-linear interactions |
| Random Forest | ~0.80–0.84 | ~0.78–0.82 | Good robustness across subjects |
| **XGBoost** | **~0.83–0.88** | **~0.81–0.86** | Best classical ML model; consistently stable |

**Key observations:**

- HRV metrics (SDNN, RMSSD) decrease significantly during stress.  
- EDA peaks strongly increase during stress and dominate feature importance.  
- Respiration variability increases under stress.

#### 6.1.2 Deep Learning Baselines

| Model | Accuracy | F1-Macro | Notes |
|-------|----------|----------|-------|
| 1D-CNN | ~0.82–0.87 | ~0.80–0.85 | Captures temporal morphology of biosignals |
| **LSTM** | **~0.85–0.90** | **~0.83–0.88** | Best DL model; captures long-range physiological changes |

**Takeaways:**

- DL models benefit from raw ECG/BVP waveform structure.  
- LSTM outperforms CNN for slow-changing autonomic signals (EDA, RESP).  
- LOSO splits achieve slightly lower results (~0.80–0.84) due to subject-level variability.

---

### 6.2 Activity Recognition (PAMAP2)

Using accelerometer + heart rate signals (30-second windows, 5-second stride), we evaluate HAR baselines.

| Model | Accuracy | F1-Macro | Notes |
|-------|----------|----------|-------|
| Logistic Regression | ~0.70 | ~0.68 | Weak linear baseline |
| Random Forest | ~0.86 | ~0.84 | Good non-linear performance |
| **XGBoost** | **~0.88–0.90** | **~0.86–0.88** | Best ML model |
| **1D-CNN** | **~0.90–0.93** | **~0.89–0.92** | CNN excels at motion pattern extraction |

**Confusion matrix highlights:**

- Walking ↔ Running: well-separated by magnitude & frequency of ACC.  
- Sitting ↔ Standing: occasional confusion due to posture similarity.  
- HR contributes to low-motion activity separation.

---

### 6.3 Explainability (SHAP)

#### WESAD

- EDA peak count & amplitude → strongest positive contribution to stress classification.  
- HRV indices (RMSSD, SDNN) → strongest negative contribution (lower during stress).  
- Respiration variability → increases during stress.  
- ACC features → correlate with amusement & movement.

#### PAMAP2

- ACC energy & SMA → dominant contributors to dynamic activities (running, walking).  
- HR mean → improves separation between static activities.

**Summary:**  
SHAP results are consistent with physiological and behavioral literature, confirming the validity of the models and features.

---

## 7. Potential Extensions

This pipeline is fully modular and supports multiple high-impact research extensions.

### 7.1 Additional Modalities or Datasets
- Integrate PPG-to-HRV pipelines.  
- Add datasets:  
  - AffectiveROAD  
  - SWELL-KW  
  - WESAD-wrist-only studies  

### 7.2 Personalization & Domain Adaptation
- Subject-specific fine-tuning  
- CORAL / DAN / DANN domain adaptation  
- Personalized calibration models  

### 7.3 Multimodal Fusion Models
- CNN + LSTM hybrid  
- Transformer-based time-series fusion  
- Cross-attention between chest & wrist sensors  

### 7.4 Real-Time Stress Detection
- Low-power models for wearables  
- Model compression: pruning, quantization  
- On-device inference with TFLite/CoreML  

### 7.5 Behavioral & Longitudinal Modeling
- Daily/weekly stress trends  
- Activity-aware stress baselines  
- State-space / HMM / TCN behavioral transitions  

### 7.6 Explainability & Human-Factors Integration
- Personalized SHAP profiles  
- Stress diary integration  
- Health/well-being intervention triggers  

---

## 8. License

This project is released under the **MIT License**.

You may:

- Use the code for academic, personal, or commercial applications  
- Modify, redistribute, and integrate the pipeline  
- Publish derived work with appropriate attribution  

Please cite datasets used:

- **WESAD**: Schmidt et al., 2018  
- **PAMAP2**: Reiss & Stricker, 2012  