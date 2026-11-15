"""
train_models.py - Multi-Modality Model Training & Cross-Validation Pipeline.

Executes:
1. Stratified 5-Fold Cross-Validation on each sensor telemetry dataset.
2. Hyperparameter Grid Search (GridSearchCV) comparing Random Forest and Gradient Boosting.
3. Computes Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
4. Serializes the champion model per modality to ml/models/*.joblib.
5. Emits a comprehensive validation report to ml/reports/cross_validation_report.md.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

MODELS_DIR = "ml/models"
REPORTS_DIR = "ml/reports"


def train_modality_pipeline(
    data_path: str,
    modality_name: str,
    feature_names: list[str],
    label_col: str = "hazard_label",
) -> dict:
    """Trains, cross-validates, and tunes models for a single sensor modality."""
    print(f"\n==========================================")
    print(f"[*] Starting Pipeline for: {modality_name.upper()}")
    print(f"==========================================")

    df = pd.read_csv(data_path)
    X = df[feature_names]
    y = df[label_col]

    # 80/20 Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Candidate 1: Random Forest
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
        "random_state": [42],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(),
        rf_params,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )
    rf_grid.fit(X_train, y_train)

    # Candidate 2: Gradient Boosting
    gb_params = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "random_state": [42],
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(),
        gb_params,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )
    gb_grid.fit(X_train, y_train)

    # Select the champion candidate on test split
    rf_best = rf_grid.best_estimator_
    gb_best = gb_grid.best_estimator_

    def evaluate_model(clf, name):
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        return {
            "algorithm": name,
            "best_params": clf.get_params(),
            "accuracy": float(np.round(accuracy_score(y_test, y_pred), 4)),
            "precision": float(np.round(precision_score(y_test, y_pred, zero_division=0), 4)),
            "recall": float(np.round(recall_score(y_test, y_pred, zero_division=0), 4)),
            "f1_score": float(np.round(f1_score(y_test, y_pred, zero_division=0), 4)),
            "roc_auc": float(np.round(roc_auc_score(y_test, y_prob), 4)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    rf_eval = evaluate_model(rf_best, "RandomForestClassifier")
    gb_eval = evaluate_model(gb_best, "GradientBoostingClassifier")

    # Pick champion based on F1-score & Recall
    if gb_eval["f1_score"] >= rf_eval["f1_score"]:
        champion_model = gb_best
        champion_eval = gb_eval
        champion_name = "GradientBoostingClassifier"
    else:
        champion_model = rf_best
        champion_eval = rf_eval
        champion_name = "RandomForestClassifier"

    # Save champion model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, f"{modality_name}_model.joblib")
    joblib.dump({
        "model": champion_model,
        "features": feature_names,
        "metrics": champion_eval,
        "modality": modality_name,
    }, model_save_path)
    print(f"[*] Champion model saved: {model_save_path} ({champion_name})")
    print(f"    Recall: {champion_eval['recall']:.4f} | Precision: {champion_eval['precision']:.4f} | F1: {champion_eval['f1_score']:.4f}")

    return {
        "modality": modality_name,
        "features": feature_names,
        "champion": champion_eval,
        "runner_up": rf_eval if champion_name == "GradientBoostingClassifier" else gb_eval,
    }


def run_full_training_suite():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Seismic
    seismic_res = train_modality_pipeline(
        data_path="data/seismic_telemetry.csv",
        modality_name="seismic",
        feature_names=["vibration_ppv_mms", "energy_joules", "dominant_frequency_hz", "event_duration_s", "noise_ratio"],
    )

    # 2. Displacement
    disp_res = train_modality_pipeline(
        data_path="data/displacement_telemetry.csv",
        modality_name="displacement",
        feature_names=["velocity_mm_hr", "acceleration_mm_hr2", "inverse_velocity", "cumulative_displacement_mm", "creep_rate_ratio"],
    )

    # 3. Environmental
    env_res = train_modality_pipeline(
        data_path="data/environmental_telemetry.csv",
        modality_name="environmental",
        feature_names=["rainfall_3h_mm", "rainfall_24h_mm", "pore_pressure_kpa", "soil_moisture_pct", "temperature_celsius"],
    )

    all_results = [seismic_res, disp_res, env_res]

    # Save JSON report
    with open(os.path.join(REPORTS_DIR, "cross_validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save Markdown report
    md_content = "# Multi-Sensor Base Model Cross-Validation & Hyperparameter Report\n\n"
    md_content += "| Sensor Modality | Champion Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
    md_content += "|---|---|---|---|---|---|---|\n"

    for r in all_results:
        c = r["champion"]
        md_content += f"| **{r['modality'].capitalize()}** | {c['algorithm']} | {c['accuracy']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1_score']:.4f} | {c['roc_auc']:.4f} |\n"

    md_content += "\n## Confusion Matrices on Test Split (20% Hold-Out)\n\n"
    for r in all_results:
        c = r["champion"]
        cm = c["confusion_matrix"]
        md_content += f"### {r['modality'].capitalize()} Model\n"
        md_content += f"- True Negatives: `{cm[0][0]}` | False Positives: `{cm[0][1]}`\n"
        md_content += f"- False Negatives: `{cm[1][0]}` | True Positives: `{cm[1][1]}`\n\n"

    with open(os.path.join(REPORTS_DIR, "cross_validation_report.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[*] Full cross-validation report emitted to: {os.path.join(REPORTS_DIR, 'cross_validation_report.md')}")


if __name__ == "__main__":
    run_full_training_suite()
