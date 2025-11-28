# Stress & Activity Prediction from Multimodal Wearable Signals  
*A Behavior Modeling & Applied ML Pipeline using WESAD and PAMAP2*

---

## 1. Introduction

Wearable and physiological sensors provide a rich view of human affect, behaviors, and cognitive states.  
Accurately detecting stress from multimodal biosignals is a key challenge in digital health, mental well-being technology, and ubiquitous computing.  

In this work, we build a fully reproducible pipeline for:

- **Stress detection** using the **WESAD** dataset (ECG, EDA, RESP, ACC, TEMP)
- **Activity recognition** using **PAMAP2**

Our contributions include:

1. A unified preprocessing pipeline supporting resampling, filtering, normalization.  
2. A sliding-window transformation for multimodal physiological streams.  
3. Handcrafted behavioral features (HRV, EDA peaks, respiration, ACC metrics).  
4. ML baselines (Logistic, RF, XGBoost) and DL models (CNN, LSTM).  
5. Comprehensive evaluation & SHAP explainability.

---

## 2. Related Work

### 2.1 Wearable Stress Detection  
- Physiological stress detection with HRV, EDA, respiration  
- Affective computing & multimodal fusion  
- WESAD as a benchmark dataset

### 2.2 Activity Recognition from Wearables  
- Accelerometer-based HAR  
- Sliding window + feature extraction paradigm  
- PAMAP2 dataset

### 2.3 Applied ML & Behavior Modeling  
- ML for multimodal time-series  
- CNN/LSTM for biosignal modeling  
- Interpretable ML for behavior insights

---

## 3. Datasets

### 3.1 WESAD (Wearable Stress & Affect Detection)

**Sensors:**

| Device | Signals | Sampling Rate |
|--------|---------|---------------|
| RespiBAN (Chest) | ECG, EDA, EMG, RESP, TEMP, ACC | 700 Hz |
| Empatica E4 (Wrist) | ACC, BVP, EDA, TEMP | 4–64 Hz |

**Labels:**

- 1 = baseline  
- 2 = stress  
- 3 = amusement  
- 4 = meditation  
- 0 = undefined  

### 3.2 PAMAP2 (Activity Recognition)  
Activities include sitting, standing, walking, running, cycling, vacuuming, etc.

---

## 4. Method

### 4.1 Pipeline Overview  
1. Load raw data (synchronized chest + wrist)  
2. Preprocess (resample, filter, z-score)  
3. Sliding windows (e.g., 60s window, 5s stride)  
4. Feature extraction  
5. Model training (ML + DL)  
6. Evaluation  
7. Explainability (SHAP)

---

## 5. Preprocessing

- Resampling to **32 Hz**  
- Bandpass filters:
  - ECG: 0.5–40 Hz  
  - BVP: 0.5–8 Hz  
- Z-score normalization (per subject)  

---

## 6. Feature Extraction

### 6.1 HRV  
- SDNN  
- RMSSD  
- pNN50  
- HR mean  

### 6.2 EDA  
- tonic mean  
- SCR peak count  
- peak amplitude  

### 6.3 Respiration  
- breathing rate  
- variability  

### 6.4 Accelerometer  
- mean / std  
- energy  
- SMA  

---

## 7. Models

### 7.1 Machine Learning  
- Logistic Regression  
- Random Forest  
- XGBoost  

### 7.2 Deep Learning  
- 1D CNN  
- LSTM  

---

## 8. Experiment Setup

- 80/20 subject-level split  
- window size: **60 seconds**  
- stride: **5 seconds**  
- batch size = 32  
- epochs = 8 (DL)  
- evaluation metrics:
  - Accuracy  
  - F1-Macro  
  - ROC-AUC  

---

## 9. Results

### 9.1 Stress Detection (WESAD)

| Model | Acc | F1-Macro |
|-------|-----|----------|
| Logistic | xx | xx |
| RF | xx | xx |
| XGBoost | xx | xx |
| CNN | xx | xx |
| LSTM | xx | xx |

### 9.2 Activity Recognition (PAMAP2)

(similar table)

---

## 10. Explainability (SHAP)

We use SHAP to probe feature importance for stress detection:

- HRV features (RMSSD, SDNN) strongly decrease during stress.  
- EDA peaks sharply increase during stress.  
- ACC-based movement patterns change during amusement.  

---

## 11. Discussion

- Multimodal fusion improves robustness.  
- Physiological responses vary across subjects → personalized modeling needed.  
- Limitations:
  - Small sample size (n=15).  
  - Controlled laboratory protocol.

---

## 12. Conclusion

We provide a fully reproducible multimodal wearable ML pipeline covering stress detection and activity recognition, integrating preprocessing, feature engineering, ML/DL modeling, and explainability.

---

## Appendix

Include:

- Hyperparameter settings  
- Additional plots  
- Full SHAP graphs  

---

## References

Schmidt et al. 2018 — WESAD dataset  
Reiss et al. 2012 — PAMAP2 dataset  
Additional wearable ML literature