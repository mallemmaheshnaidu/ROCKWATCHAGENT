# Multi-Sensor Base Model Cross-Validation & Hyperparameter Report

| Sensor Modality | Champion Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Seismic** | GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Displacement** | GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Environmental** | GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Confusion Matrices on Test Split (20% Hold-Out)

### Seismic Model
- True Negatives: `540` | False Positives: `0`
- False Negatives: `0` | True Positives: `60`

### Displacement Model
- True Negatives: `532` | False Positives: `0`
- False Negatives: `0` | True Positives: `68`

### Environmental Model
- True Negatives: `537` | False Positives: `0`
- False Negatives: `0` | True Positives: `63`

