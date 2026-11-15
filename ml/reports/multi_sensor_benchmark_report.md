# Multi-Sensor Independent Model Benchmark & Cross-Validation Report
**Project:** RockWatch (SIH25071) — AI-Based Rockfall Prediction & Early Warning System
**Evaluation Standard:** 5-Fold Stratified Cross-Validation with Gradient Boosting Algorithms (CatBoost, LightGBM, XGBoost)
**Date:** November 2026

---
## Executive Summary
In accordance with open-pit mining safety standards and the user directive to avoid monolithic synthetic models, RockWatch employs a **decoupled multi-sensor late-fusion architecture**. Dedicated gradient boosting models are trained independently on physical domain datasets for each monitoring modality:

1. **Environmental Weather & Infiltration Stream**: Trained on 2,555 real-world rockfall/landslide precipitation events from the Smart India Hackathon repository, quantifying pore-water pressure and hydraulic infiltration triggers.
2. **Geotechnical Slope Stability & Kinematics Stream**: Grounded in Bishop/Janbu limit equilibrium mechanics and Hoek-Bray rock mass discontinuity formulations with realistic field radar creep velocities, joint cohesion scatter, and blast vibration noise.
3. **Computer Vision Pit Detector**: Algorithmic and pre-trained pit surveillance detector (`ml/vision_detector.py`) tracking optical velocity, bounding boxes, and detachment scarps.

All models avoid artificial 100% data leakage while delivering high-safety predictive performance:
- **Weather Stream Champion**: **CatBoost (iter=200, depth=5, lr=0.06)** (ROC-AUC: **0.763**, F1: **0.929**, Recall: **0.982** at threshold 0.45)
- **Geotechnical Stream Champion**: **CatBoost (iter=200, depth=5, lr=0.06)** (ROC-AUC: **0.900**, F1: **0.792**, Recall: **0.741** at threshold 0.35)

---
## 1. Environmental Weather & Infiltration Stream (5-Fold Stratified CV)

### Dataset Specification
- **Source**: Real SIH Open-Pit Rockfall Weather & Hazard Corpus (`featured_weather_data.csv` + `enhanced_landslide_dataset.csv`)
- **Sample Count**: 2,555 records
- **Class Balance**: 86.1% precipitation-triggered slope events vs 13.9% other triggers
- **Features Used**: `precipitation_sum_last_3_days`, `precipitation_sum_last_7_days`, `precipitation_sum_last_10_days`, `soil_moisture_0_to_7cm_mean_last_5_days`, `rainfall_trend_last_5_days`, `days_with_rain_gt_20mm_last_10_days`, `consecutive_rainy_days`, `temperature_2m_max_in_last_10_days`, `temperature_2m_min_in_last_10_days`

### 5-Fold Cross-Validation Comparison
| Model Candidate | Mean ROC-AUC | PR-AUC | Optimal Cutoff | Precision | Recall | F1-Score | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (iter=200, depth=5, lr=0.06)** *(Champion)* | 0.7631 | 0.9457 | 0.45 | 0.8824 | 0.9818 | **0.9295** | 0.5841 |
| **CatBoost (iter=150, depth=4, lr=0.08)** | 0.7615 | 0.9410 | 0.45 | 0.8735 | 0.9886 | **0.9275** | 0.5494 |
| **LightGBM (n_est=200, depth=5, lr=0.05)** | 0.7399 | 0.9350 | 0.45 | 0.8833 | 0.9805 | **0.9294** | 0.5877 |
| **LightGBM (n_est=150, depth=4, lr=0.08)** | 0.7364 | 0.9345 | 0.35 | 0.8741 | 0.9877 | **0.9275** | 0.5518 |
| **XGBoost (n_est=200, depth=5, lr=0.05)** | 0.7405 | 0.9362 | 0.45 | 0.8821 | 0.9791 | **0.9281** | 0.5828 |
| **XGBoost (n_est=150, depth=4, lr=0.08)** | 0.7386 | 0.9357 | 0.4 | 0.8797 | 0.9832 | **0.9286** | 0.5735 |

### Feature Importance Analysis (Champion Model)
- **`rainfall_trend_last_5_days`**: 16.71%
- **`soil_moisture_0_to_7cm_mean_last_5_days`**: 16.06%
- **`precipitation_sum_last_3_days`**: 13.13%
- **`temperature_2m_max_in_last_10_days`**: 11.28%
- **`temperature_2m_min_in_last_10_days`**: 11.25%
- **`precipitation_sum_last_10_days`**: 10.36%
- **`precipitation_sum_last_7_days`**: 9.89%
- **`consecutive_rainy_days`**: 6.18%
- **`days_with_rain_gt_20mm_last_10_days`**: 5.14%

---
## 2. Geotechnical Slope Stability & Kinematics Stream (5-Fold Stratified CV)

### Dataset Specification
- **Formulation**: Bishop/Janbu Limit Equilibrium of Slices + Hoek-Bray rock mass shear strength
- **Sample Count**: 3,500 bench segments
- **Class Balance**: 17.3% unstable (FS < 1.0), 82.7% stable (FS >= 1.0, mean FS = 1.367)
- **Features Used**: `unit_weight_kn_m3`, `cohesion_kpa`, `friction_angle_deg`, `slope_angle_deg`, `slope_height_m`, `pore_water_ratio_ru`, `radar_velocity_mm_hr`

### 5-Fold Cross-Validation Comparison
| Model Candidate | Mean ROC-AUC | PR-AUC | Optimal Cutoff | Precision | Recall | F1-Score | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (iter=200, depth=5, lr=0.06)** *(Champion)* | 0.9003 | 0.8372 | 0.35 | 0.8517 | 0.7405 | **0.7922** | 0.8568 |
| **CatBoost (iter=150, depth=4, lr=0.08)** | 0.9029 | 0.8368 | 0.3 | 0.8225 | 0.7504 | **0.7848** | 0.8583 |
| **LightGBM (n_est=200, depth=5, lr=0.05)** | 0.8952 | 0.8272 | 0.35 | 0.8433 | 0.7207 | **0.7772** | 0.8463 |
| **LightGBM (n_est=150, depth=4, lr=0.08)** | 0.8954 | 0.8269 | 0.4 | 0.8566 | 0.7207 | **0.7828** | 0.8477 |
| **XGBoost (n_est=200, depth=5, lr=0.05)** | 0.8953 | 0.8292 | 0.3 | 0.8149 | 0.7421 | **0.7768** | 0.8535 |
| **XGBoost (n_est=150, depth=4, lr=0.08)** | 0.8948 | 0.8280 | 0.35 | 0.8372 | 0.7223 | **0.7755** | 0.8465 |

### Feature Importance Analysis (Champion Model)
- **`radar_velocity_mm_hr`**: 30.83%
- **`cohesion_kpa`**: 19.04%
- **`slope_height_m`**: 13.97%
- **`slope_angle_deg`**: 11.11%
- **`pore_water_ratio_ru`**: 9.70%
- **`friction_angle_deg`**: 7.79%
- **`unit_weight_kn_m3`**: 7.57%

---
## 3. Late-Fusion Architecture (Option B) Verification

In production operation, single sensor streams feed into `ml/meta_fusion.py`:
- **Single-Sensor Dominance Override**: If any single sensor spikes >= 0.80 (e.g. radar velocity creep > 15 mm/hr or visual rock detachment), it directly triggers high alert without dilution.
- **Dynamic Weight Redistribution**: If a sector lacks radar or cameras are fogged over, the weights redistribute across active modalities and inflate the uncertainty margin.
- **Inviolable Red TARP Safety Floor**: The 15 mm/hr Red TARP threshold is hard-coded and cannot be suppressed by any model.

---
*Report generated automatically by `ml/train_independent_models.py`.*
