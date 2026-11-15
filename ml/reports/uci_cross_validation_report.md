# Real Mining Telemetry: Cross-Validation & Hyperparameter Tuning Report

**Dataset:** UCI Machine Learning Repository — Seismic Bumps Dataset (Upper Silesian Coal Basin)  
**Total Records:** 2,584 real mining shifts | **Hazardous Precursor Events:** 170 (6.58% class imbalance)  
**Validation Methodology:** Stratified 5-Fold Cross-Validation (`GridSearchCV`) with 80% Train / 20% Test Hold-Out Split.

---

## 1. Cross-Validation & Test Benchmark Results

| Model Algorithm | Hyperparameter Search Grid | Test Accuracy | Test Precision | Test Recall (Safety Priority) | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **RandomForestClassifier (Balanced) (Champion)** | `n_estimators`: [50, 100, 150]<br>`max_depth`: [5, 10, None]<br>`min_samples_split`: [2, 5, 10]<br>`class_weight`: 'balanced' | **0.8549** | **0.2405** | **0.5588** | **0.3363** | **0.7646** |
| **GradientBoostingClassifier (Runner-Up)** | `n_estimators`: [50, 100]<br>`learning_rate`: [0.03, 0.1]<br>`max_depth`: [3, 5] | **0.9342** | **0.0000** | **0.0000** | **0.0000** | **0.7282** |

---

## 2. Real-World Confusion Matrix (20% Hold-Out Test Set: 517 Shifts)

### Champion: RandomForestClassifier (Balanced)
* **True Negatives (Correctly identified quiet shifts):** `423`
* **False Positives (Nuisance alarms on heavy drilling/traffic):** `60`
* **False Negatives (Missed bumps):** `15`
* **True Positives (Correctly caught impending rock failures):** `19`

> **Geotechnical Analysis:**  
> On real mining data, achieving ~100% is impossible and fraudulent. The balanced Random Forest model catches **55.9% of true hazardous bumps** on the real test split, with a ROC-AUC of **0.7646**, demonstrating genuine discriminative power across noisy mine acoustics.
