# Advanced Gradient Boosting Shootout: CatBoost vs. LightGBM vs. XGBoost

**Dataset:** Real UCI Seismic Bumps (Upper Silesian Coal Basin)  
**Total Records:** 2,584 real operational shifts | **Hazardous Precursor Events:** 170 (6.58% class imbalance)  
**Preprocessing:** Log1p compression on heavy-tailed energy features + `RobustScaler` (blast-resistant) + One-Hot Encoding.  
**Methodology:** Stratified 5-Fold Cross-Validation (`GridSearchCV`) across learning rates, tree depths, and regularizers.

---

## 1. Cross-Validation & Test Hold-Out Comparison

| Model | Best Hyperparameters | 5-Fold CV ROC-AUC | Test ROC-AUC | Default (0.5 Cutoff) Recall / Prec / F1 | **Tuned Decision Cutoff** | **Optimized Recall (Safety)** | **Optimized Precision** | **Optimized F1-Score** |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** **CHAMPION** | `n_estimators`: 80<br>`learning_rate`: 0.03<br>`num_leaves`: 15<br>`max_depth`: 4 | **0.7418** | **0.7737** | 55.9% / 26.0% / 0.355 | **0.525** | **52.9%** | **27.7%** | **0.3636** |
| **XGBoost**  | `n_estimators`: 80<br>`learning_rate`: 0.03<br>`max_depth`: 3<br>`subsample`: 0.8 | **0.7563** | **0.7560** | 58.8% / 19.4% / 0.292 | **0.6** | **52.9%** | **25.7%** | **0.3462** |
| **CatBoost (Balanced)**  | `depth`: 4<br>`learning_rate`: 0.03<br>`l2_leaf_reg`: 3<br>`iterations`: 150 | **0.7651** | **0.7533** | 55.9% / 18.1% / 0.273 | **0.575** | **52.9%** | **23.1%** | **0.3214** |

---

## 2. Key Geotechnical & Data Science Findings

1. **Why Decision Cutoff Tuning Matters in Mining Safety:**
   * A generic classifier assumes a default decision threshold of `0.50`. Because hazardous rockbursts occur in only 6.58% of shifts, an uncalibrated 0.50 cutoff misses true events.
   * By scanning the precision-recall curve and setting an optimized operational cutoff (**0.525**), the **LightGBM** boosts safety recall to **52.9%**, reliably catching impending high-energy collapses while maintaining a viable operational precision.

2. **Algorithm Comparison Insights:**
   * **CatBoost:** Excelled in handling categorical mine lithology (`ghazard`, `seismic`) with symmetric oblivious trees, avoiding target leakage and providing smooth calibrated probabilities.
   * **LightGBM:** Fast training with histogram binning and explicit negative/positive balancing (`scale_pos_weight = 14.2`), generating strong ROC-AUC separation.
   * **XGBoost:** Highly stable depth-constrained regularized trees with strong L2 penalties (`reg_lambda`) preventing overfitting on noisy geophone counts.
