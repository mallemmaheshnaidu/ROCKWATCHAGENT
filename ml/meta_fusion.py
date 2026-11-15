"""
meta_fusion.py - Option B: Priority-Weighted Hybrid Fusion Engine.

Implements Late Multimodal Sensor Fusion:
- Loads the champion gradient boosting models (Geotechnical Slope Stability/Radar, Weather Infiltration, Seismic)
  and the pit vision detector.
- Computes base probability vector [P_displacement, P_environmental, P_seismic, P_vision].
- Applies Priority-Weighted Hybrid Fusion:
  If any individual sensor screams danger (P_i >= 0.80), an automatic dominance override triggers,
  guaranteeing that extreme physical anomalies (e.g. radar velocity creep > 15 mm/hr, detachment bounding box)
  are never diluted or 'averaged out' by quiet sensors.
- Supports graceful sensor missingness via dynamic weight redistribution and uncertainty margin inflation.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import joblib
import numpy as np
from typing import Any, Dict, Optional
from ml.vision_detector import PitVisionDetector

MODELS_DIR = "ml/models"

from ml.risk_tiers import (  # noqa: E402  (lightweight, no heavy deps)
    apply_attenuation_floor,
    clamp_sensitivity,
    tier_from_probability,
)

_GEOTECH_FEATURES = [
    "unit_weight_kn_m3", "cohesion_kpa", "friction_angle_deg",
    "slope_angle_deg", "slope_height_m", "pore_water_ratio_ru", "velocity_mm_hr",
]
_ALLOWED_GEOTECH_FEATURES = set(_GEOTECH_FEATURES) | {"radar_velocity_mm_hr"}
_GEOTECH_DEFAULTS = {
    "unit_weight_kn_m3": 24.5, "cohesion_kpa": 180.0, "friction_angle_deg": 34.0,
    "slope_angle_deg": 52.0, "slope_height_m": 22.0, "pore_water_ratio_ru": 0.15,
}


def _number(value: Any, default: float) -> float:
    """Coerces an optional telemetry value, falling back to a documented default."""
    if value is None:
        return float(default)
    number = float(value)
    return number if np.isfinite(number) else float(default)


class MetaFusionEngine:
    """
    Late Multimodal Sensor Fusion Engine combining trained classical ML models
    and surveillance camera vision detectors across open-pit zones.
    """

    def __init__(self, models_dir: str = MODELS_DIR, dominance_threshold: float = 0.80):
        self.models_dir = models_dir
        self.dominance_threshold = dominance_threshold

        # Load trained independent champion models
        self.geotech_bundle = self._load_model("geotech_slope_model.joblib")
        self.weather_bundle = self._load_model("weather_model.joblib")
        self.seismic_bundle = self._load_model("seismic_model.joblib")

        # Fallback legacy models
        self.displacement_bundle = self._load_model("displacement_model.joblib")
        self.environmental_bundle = self._load_model("environmental_model.joblib")

        # Initialize vision detector
        self.vision_detector = PitVisionDetector()

        # Baseline ensemble weights
        self.base_weights = {
            "displacement": 0.40,
            "vision": 0.25,
            "environmental": 0.20,
            "seismic": 0.15,
        }

    def _load_model(self, filename: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"[!] Warning: Failed to load {filename}: {e}")
        return None

    def predict_displacement_probability(self, features_dict: Dict[str, Any]) -> Optional[float]:
        """
        Evaluates the Geotechnical Slope Stability & Radar Kinematics model.
        Returns None if radar displacement sensor is missing/offline.
        """
        v = features_dict.get("velocity_mm_hr")
        if v is None:
            v = features_dict.get("radar_velocity_mm_hr")
        if v is None:
            return None  # Sensor modality missing!

        v = float(v)
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"Invalid radar velocity for fusion: {v!r}")

        # Champion Geotechnical Slope model
        if self.geotech_bundle:
            model = self.geotech_bundle["model"]
            scaler = self.geotech_bundle.get("scaler")
            feature_names = list(self.geotech_bundle.get("features") or _GEOTECH_FEATURES)

            unknown = set(feature_names) - _ALLOWED_GEOTECH_FEATURES
            if unknown or not ({"velocity_mm_hr", "radar_velocity_mm_hr"} & set(feature_names)):
                raise ValueError(
                    f"Model feature contract mismatch: bundle={feature_names} "
                    f"allowed={sorted(_ALLOWED_GEOTECH_FEATURES)}"
                )

            row = {**_GEOTECH_DEFAULTS, "velocity_mm_hr": v, "radar_velocity_mm_hr": v}
            row.update({k: features_dict[k] for k in feature_names if k in features_dict})
            vector = np.array([[float(row[name]) for name in feature_names]])
            if scaler:
                vector = scaler.transform(vector)
            prob = float(model.predict_proba(vector)[0, 1])
            return prob

        # Fallback displacement model
        if self.displacement_bundle:
            import pandas as pd
            model = self.displacement_bundle["model"]
            f_names = self.displacement_bundle["features"]
            accel = _number(features_dict.get("acceleration_mm_hr2"), 0.0)
            inv_v = 1.0 / max(0.01, v)
            cumul = _number(features_dict.get("cumulative_displacement_mm"), v * 2.0)
            creep_ratio = _number(features_dict.get("creep_rate_ratio"), max(0.5, v / 1.5))
            vector = [v, accel, inv_v, cumul, creep_ratio]
            X = pd.DataFrame([vector], columns=f_names)
            return float(model.predict_proba(X)[0, 1])

        # No displacement model: report missing evidence, not a 5% risk score.
        return None

    def predict_environmental_probability(self, features_dict: Dict[str, Any]) -> Optional[float]:
        """
        Evaluates the Environmental Weather & Infiltration model.
        Returns None when the weather channel is missing or no model is loaded
        (missingness, not low risk).
        """
        if features_dict.get("rainfall_3h_mm") is None:
            return None
        if features_dict.get("pore_pressure_kpa") is None:
            return None

        rain_3h = _number(features_dict.get("rainfall_3h_mm"), 0.0)
        rain_24h = _number(features_dict.get("rainfall_24h_mm"), rain_3h * 2.5)
        pore = _number(features_dict.get("pore_pressure_kpa"), 35.0)
        moisture = _number(features_dict.get("soil_moisture_pct"), 25.0)
        temp = _number(features_dict.get("temperature_celsius"), 22.0)

        # Champion Weather Infiltration Model
        if self.weather_bundle:
            model = self.weather_bundle["model"]
            scaler = self.weather_bundle.get("scaler")

            p3 = _number(features_dict.get("precipitation_sum_last_3_days"), max(rain_3h, rain_24h * 1.5))
            p7 = _number(features_dict.get("precipitation_sum_last_7_days"), rain_24h * 3.0)
            p10 = _number(features_dict.get("precipitation_sum_last_10_days"), rain_24h * 4.0)
            t_max = _number(features_dict.get("temperature_2m_max_in_last_10_days"), temp + 5.0)
            t_min = _number(features_dict.get("temperature_2m_min_in_last_10_days"), max(0.0, temp - 8.0))
            sm = _number(features_dict.get("soil_moisture_0_to_7cm_mean_last_5_days"), moisture / 100.0)
            r_trend = _number(features_dict.get("rainfall_trend_last_5_days"), rain_3h * 0.4)
            d_gt20 = _number(features_dict.get("days_with_rain_gt_20mm_last_10_days"), 1.0 if rain_24h > 20.0 else 0.0)
            c_rain = _number(features_dict.get("consecutive_rainy_days"), 3.0 if rain_24h > 10.0 else 1.0)

            vector = np.array([[p3, p7, p10, t_max, t_min, sm, r_trend, d_gt20, c_rain]])
            if scaler:
                vector = scaler.transform(vector)
            return float(model.predict_proba(vector)[0, 1])

        # Fallback environmental model
        if self.environmental_bundle:
            import pandas as pd
            model = self.environmental_bundle["model"]
            f_names = self.environmental_bundle["features"]
            vector = [rain_3h, rain_24h, pore, moisture, temp]
            X = pd.DataFrame([vector], columns=f_names)
            return float(model.predict_proba(X)[0, 1])

        return None

    def predict_seismic_probability(self, features_dict: Dict[str, Any]) -> Optional[float]:
        """
        Evaluates the Seismic vibration and microseismic telemetry model.
        Returns None when the vibration channel is missing or no seismic model
        is loaded (missingness, not low risk).
        """
        if not self.seismic_bundle:
            return None
        if features_dict.get("vibration_ppv_mms") is None:
            return None
        import pandas as pd

        if "pipeline" in self.seismic_bundle:
            pipe = self.seismic_bundle["pipeline"]
            ppv = _number(features_dict.get("vibration_ppv_mms"), 1.0)
            energy_val = _number(features_dict.get("energy_joules"), max(0.0, ppv * 500.0))
            row = {
                "seismic": features_dict.get("seismic", "a"),
                "seismoacoustic": features_dict.get("seismoacoustic", "a"),
                "shift": features_dict.get("shift", "W"),
                "ghazard": features_dict.get("ghazard", "a"),
                "genergy": energy_val * 2.0,
                "gpuls": max(10.0, ppv * 25.0),
                "gdenergy": max(-50.0, min(100.0, ppv * 10.0 - 20.0)),
                "gdpuls": max(-50.0, min(100.0, ppv * 8.0 - 15.0)),
                "nbumps": 1.0 if ppv > 15.0 else 0.0,
                "nbumps2": 1.0 if ppv > 25.0 else 0.0,
                "nbumps3": 1.0 if ppv > 40.0 else 0.0,
                "nbumps4": 1.0 if ppv > 55.0 else 0.0,
                "nbumps5": 0.0,
                "nbumps6": 0.0,
                "nbumps7": 0.0,
                "nbumps89": 0.0,
                "energy": energy_val,
                "maxenergy": energy_val,
            }
            X = pd.DataFrame([row])
            return float(pipe.predict_proba(X)[0, 1])

        # Legacy bundle fallback
        model = self.seismic_bundle["model"]
        f_names = self.seismic_bundle["features"]
        vector = [
            _number(features_dict.get("vibration_ppv_mms"), 1.0),
            _number(features_dict.get("energy_joules"), 100.0),
            _number(features_dict.get("dominant_frequency_hz"), 35.0),
            _number(features_dict.get("event_duration_s"), 1.0),
            _number(features_dict.get("noise_ratio"), 0.1),
        ]
        X = pd.DataFrame([vector], columns=f_names)
        return float(model.predict_proba(X)[0, 1])

    def evaluate_zone_fusion(
        self,
        zone_id: str,
        telemetry: Dict[str, Any],
        camera_frame: Optional[np.ndarray] = None,
        sensitivity_multiplier: float = 1.0,
        sensor_uncertainty_penalty: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Executes Late Multimodal Fusion for a target zone:
        Evaluates base models, checks for dominant sensor spikes, and produces a calibrated hazard package.
        """
        # 1. Base Probabilities
        p_seismic = self.predict_seismic_probability(telemetry)
        p_disp = self.predict_displacement_probability(telemetry)
        p_env = self.predict_environmental_probability(telemetry)

        # Vision prediction
        if camera_frame is not None:
            vision_res = self.vision_detector.analyze_frame(zone_id, camera_frame)
            p_vision = vision_res["vision_risk_probability"]
            bounding_boxes = vision_res["bounding_boxes"]
            vision_status = vision_res["status"]
        else:
            p_vision = None          # offline camera is missing evidence, not 5% risk
            bounding_boxes = []
            vision_status = "CAMERA_STREAM_OFFLINE_OR_UNATTACHED"

        base_probabilities = {
            "seismic": round(p_seismic, 4) if p_seismic is not None else None,
            "displacement": round(p_disp, 4) if p_disp is not None else None,
            "environmental": round(p_env, 4) if p_env is not None else None,
            "vision": round(p_vision, 4) if p_vision is not None else None,
        }

        # 2. Dynamic Weight Redistribution for Missing Modalities
        active_weights = {}
        total_active_weight = 0.0
        for mod, weight in self.base_weights.items():
            if base_probabilities[mod] is not None:
                active_weights[mod] = weight
                total_active_weight += weight

        if total_active_weight > 0:
            normalized_weights = {k: v / total_active_weight for k, v in active_weights.items()}
            weighted_ensemble = sum(base_probabilities[k] * normalized_weights[k] for k in active_weights)
        else:
            weighted_ensemble = 0.05
            normalized_weights = {}

        # 3. Dominance Check (Option B Rule)
        # Any single modality >= dominance_threshold overrides the ensemble average!
        dominant_modality = None
        highest_p = 0.0
        for mod, p in base_probabilities.items():
            if p is not None and p > highest_p:
                highest_p = p
                dominant_modality = mod

        if highest_p >= self.dominance_threshold:
            final_prob = highest_p
            dominance_override = True
            fusion_rationale = (
                f"DOMINANCE OVERRIDE: {dominant_modality.upper()} sensor spiked to {highest_p:.1%}. "
                f"Elevated directly to prioritize critical anomaly without dilution."
            )
        else:
            final_prob = weighted_ensemble
            dominance_override = False
            fusion_rationale = "Weighted multimodal ensemble in normal consensus."

        # Modulate by approved sensitivity multiplier (if any desensitization proposal is active).
        # Attenuation is bounded and may never remove more than one severity tier.
        multiplier = clamp_sensitivity(sensitivity_multiplier)
        adjusted_prob = min(1.0, final_prob * multiplier)
        tier = apply_attenuation_floor(final_prob, adjusted_prob)

        # Uncertainty bounds (widened if sensors are degraded or missing)
        missing_count = sum(1 for p in base_probabilities.values() if p is None)
        uncertainty_margin = round(min(0.50, 0.08 + sensor_uncertainty_penalty + (0.05 * missing_count)), 3)
        lower_bound = round(max(0.0, adjusted_prob - uncertainty_margin), 3)
        upper_bound = round(min(1.0, adjusted_prob + uncertainty_margin), 3)

        return {
            "zone_id": zone_id,
            "raw_probability": round(float(final_prob), 4),
            "raw_risk_tier": tier_from_probability(final_prob),
            "sensitivity_applied": multiplier,
            "risk_probability": round(float(adjusted_prob), 4),
            "uncertainty_interval": [lower_bound, upper_bound],
            "uncertainty_margin": uncertainty_margin,
            "risk_tier": tier,
            "base_probabilities": base_probabilities,
            "model_status": {
                "displacement": "LOADED" if (self.geotech_bundle or self.displacement_bundle) else "UNAVAILABLE",
                "environmental": "LOADED" if (self.weather_bundle or self.environmental_bundle) else "UNAVAILABLE",
                "seismic": "LOADED" if self.seismic_bundle else "UNAVAILABLE",
                "vision": vision_status,
            },
            "dominant_sensor": dominant_modality if dominance_override else None,
            "dominance_override": dominance_override,
            "fusion_rationale": fusion_rationale,
            "vision_status": vision_status,
            "detected_bounding_boxes": bounding_boxes,
            "active_weights": {k: round(v, 2) for k, v in normalized_weights.items()},
        }
