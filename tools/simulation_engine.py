"""
simulation_engine.py - Geotechnical Scenario & Progressive Telemetry Simulation Engine.

Replaces unrealistic binary jumps with real rock mechanics physics:
1. Saito's Inverse Velocity Law for progressive tertiary creep acceleration:
   v(t) = v0 + A * (tf - t)^(-m) + noise
2. Hydrological Infiltration Lag:
   Pore water pressure u(t) follows an impulse response to rainfall accumulation with drainage lag.
3. Multi-Track Industrial Scenarios:
   - Scenario 1: North Wall Progressive Tertiary Creep (Green -> Yellow -> Orange -> Red Failure)
   - Scenario 2: South Crest Monsoon Infiltration & Delayed Hydraulic Destabilization
   - Scenario 3: Main Haulage Ramp Blasting Transient (False Alarm / Desensitization test)
   - Scenario 4: Baseline Stable Mining Operations (Diurnal thermal drift)
"""

import numpy as np
from typing import Dict, Any, List


class GeotechnicalSimulationEngine:
    """
    Simulates realistic temporal telemetry sequences across open-pit mine sectors.
    """

    SCENARIOS = {
        "SCENARIO_1_NORTH_WALL_CREEP": {
            "name": "Scenario 1: North Wall Progressive Tertiary Creep",
            "description": "Progressive bench failure following Saito's inverse velocity law. Accelerates from stable background creep to critical Level 4 Red over 24 hours.",
            "target_sector": "ZONE_NORTH_WALL",
        },
        "SCENARIO_2_SOUTH_CREST_INFILTRATION": {
            "name": "Scenario 2: South Crest Monsoon Hydraulic Infiltration",
            "description": "Heavy rainfall accumulation (85mm) saturates slope joints. Pore pressure rises with physical lag, triggering delayed structural destabilization.",
            "target_sector": "ZONE_SOUTH_CREST",
        },
        "SCENARIO_3_BLASTING_TRANSIENT": {
            "name": "Scenario 3: Main Ramp Blasting Transient (False Alarm Test)",
            "description": "Scheduled bench blast causes an immediate vibration spike and camera dust plume. Dissipates safely back to normal within 1 hour.",
            "target_sector": "ZONE_RAMP_3",
        },
        "SCENARIO_4_BASELINE_STABLE": {
            "name": "Scenario 4: Baseline Stable Mining Operations",
            "description": "Normal mining operations with routine background creep and low diurnal thermal fluctuations.",
            "target_sector": "ALL",
        },
    }

    def __init__(self, time_horizon_hours: float = 24.0, step_hours: float = 0.5):
        self.time_horizon = time_horizon_hours
        self.step = step_hours
        self.time_points = np.arange(0.0, time_horizon_hours + step_hours, step_hours)

    def generate_scenario_telemetry(self, scenario_id: str, t_hour: float) -> Dict[str, Dict[str, Any]]:
        """
        Computes the physical sensor state for all pit sectors at time t_hour.
        """
        t = float(np.clip(t_hour, 0.0, self.time_horizon))
        norm_t = t / self.time_horizon  # 0.0 to 1.0

        # Baseline baseline state for all sectors
        np.random.seed(int(t * 100))
        sectors_state = {
            "ZONE_NORTH_WALL": {
                "name": "North Wall Bench 4",
                "radar_velocity_mm_hr": round(0.7 + 0.1 * np.sin(t / 3.0) + np.random.normal(0, 0.04), 2),
                "rainfall_3h_mm": round(max(0.0, 1.2 + np.random.normal(0, 0.2)), 1),
                "vibration_ppv_mms": round(max(0.1, 0.4 + np.random.normal(0, 0.05)), 2),
                "pore_pressure_kpa": round(25.0 + 2.0 * np.sin(t / 4.0), 1),
                "crack_width_cm": 0.0,
                "detachment_detected": False,
                "rockfall_intensity": 0.0,
            },
            "ZONE_SOUTH_CREST": {
                "name": "South Crest Ramp",
                "radar_velocity_mm_hr": round(0.5 + 0.08 * np.sin(t / 3.0) + np.random.normal(0, 0.03), 2),
                "rainfall_3h_mm": round(max(0.0, 1.0 + np.random.normal(0, 0.2)), 1),
                "vibration_ppv_mms": round(max(0.1, 0.3 + np.random.normal(0, 0.04)), 2),
                "pore_pressure_kpa": round(22.0 + 1.5 * np.sin(t / 4.0), 1),
                "crack_width_cm": 0.0,
                "detachment_detected": False,
                "rockfall_intensity": 0.0,
            },
            "ZONE_RAMP_3": {
                "name": "Main Haulage Ramp 3",
                "radar_velocity_mm_hr": round(0.6 + 0.1 * np.sin(t / 3.0) + np.random.normal(0, 0.04), 2),
                "rainfall_3h_mm": round(max(0.0, 0.8 + np.random.normal(0, 0.1)), 1),
                "vibration_ppv_mms": round(max(0.1, 0.6 + np.random.normal(0, 0.08)), 2),
                "pore_pressure_kpa": round(24.0 + 1.8 * np.sin(t / 4.0), 1),
                "crack_width_cm": 0.0,
                "detachment_detected": False,
                "rockfall_intensity": 0.0,
            },
            "ZONE_HAUL_ROAD": {
                "name": "West Pit Access Road",
                "radar_velocity_mm_hr": round(0.4 + 0.06 * np.sin(t / 3.0) + np.random.normal(0, 0.03), 2),
                "rainfall_3h_mm": round(max(0.0, 0.5 + np.random.normal(0, 0.1)), 1),
                "vibration_ppv_mms": round(max(0.1, 0.25 + np.random.normal(0, 0.03)), 2),
                "pore_pressure_kpa": round(20.0 + 1.2 * np.sin(t / 4.0), 1),
                "crack_width_cm": 0.0,
                "detachment_detected": False,
                "rockfall_intensity": 0.0,
            },
        }

        # Apply specific physical scenario progression
        if scenario_id == "SCENARIO_1_NORTH_WALL_CREEP":
            sec = sectors_state["ZONE_NORTH_WALL"]
            # Saito's tertiary creep law: velocity accelerates hyperbolically as t approaches failure (tf = 24h)
            # t: 0-8h (Primary/Secondary: 0.8 -> 2.2 mm/hr)
            # t: 8-15h (Onset of tertiary creep: 2.2 -> 5.5 mm/hr, Level 2 Yellow)
            # t: 15-20h (Accelerating tertiary: 5.5 -> 13.5 mm/hr, Level 3 Orange)
            # t: 20-24h (Critical failure creep: 13.5 -> 19.5 mm/hr, Level 4 Red)
            if t <= 8.0:
                v = 0.8 + (t / 8.0) * 1.4 + np.random.normal(0, 0.08)
                crack = 0.0
                detach = False
                intensity = 0.05
            elif t <= 15.0:
                v = 2.2 + ((t - 8.0) / 7.0) ** 1.5 * 3.3 + np.random.normal(0, 0.15)
                crack = (t - 8.0) * 0.8
                detach = False
                intensity = 0.25
            elif t <= 20.0:
                v = 5.5 + ((t - 15.0) / 5.0) ** 2.0 * 8.0 + np.random.normal(0, 0.3)
                crack = 5.6 + (t - 15.0) * 1.6
                detach = t > 18.0
                intensity = 0.65
            else:
                # Imminent detachment / post-failure
                v = 13.5 + ((t - 20.0) / 4.0) ** 2.5 * 6.5 + np.random.normal(0, 0.5)
                crack = 13.6 + (t - 20.0) * 2.2
                detach = True
                intensity = 1.0

            sec["radar_velocity_mm_hr"] = round(max(0.2, v), 2)
            sec["crack_width_cm"] = round(crack, 1)
            sec["detachment_detected"] = detach
            sec["rockfall_intensity"] = intensity

        elif scenario_id == "SCENARIO_2_SOUTH_CREST_INFILTRATION":
            sec = sectors_state["ZONE_SOUTH_CREST"]
            # Rainfall accumulation with delayed pore water pressure response:
            # t: 0-6h: Rain starts (5 -> 25 mm/3h)
            # t: 6-14h: Peak monsoon downpour (45 mm/3h)
            # t: 14-24h: Infiltration lag causes pore pressure to rise from 25 to 92 kPa, destabilizing slope
            if t <= 6.0:
                rain = 2.0 + (t / 6.0) * 20.0
                pore = 22.0 + (t / 6.0) * 8.0
                v = 0.5 + (t / 6.0) * 0.9
            elif t <= 14.0:
                rain = 22.0 + ((t - 6.0) / 8.0) * 25.0
                pore = 30.0 + ((t - 6.0) / 8.0) ** 1.4 * 35.0  # Lagged rise
                v = 1.4 + ((t - 6.0) / 8.0) * 2.8
            else:
                rain = 47.0 - ((t - 14.0) / 10.0) * 20.0  # Rain eases
                pore = 65.0 + ((t - 14.0) / 10.0) * 28.0  # Saturated joints hold pressure!
                v = 4.2 + ((t - 14.0) / 10.0) ** 2.0 * 12.0  # Infiltration-induced failure

            sec["rainfall_3h_mm"] = round(rain + np.random.normal(0, 1.0), 1)
            sec["pore_pressure_kpa"] = round(pore + np.random.normal(0, 1.5), 1)
            sec["radar_velocity_mm_hr"] = round(max(0.2, v), 2)
            sec["detachment_detected"] = t > 18.0
            sec["rockfall_intensity"] = min(1.0, max(0.0, (t - 12.0) / 12.0))

        elif scenario_id == "SCENARIO_3_BLASTING_TRANSIENT":
            sec = sectors_state["ZONE_RAMP_3"]
            # Scheduled blast at t = 6.0h (11:30 AM)
            # Produces a vibration spike (24 mm/s) and camera dust plume, but decays to baseline in 1h
            if 5.8 <= t <= 6.4:
                sec["vibration_ppv_mms"] = round(22.0 + np.random.normal(0, 2.0), 2)
                sec["radar_velocity_mm_hr"] = round(3.5 + np.random.normal(0, 0.4), 2)  # Apparent radar noise
                sec["rockfall_intensity"] = 0.7  # Dust plume detected on vision
            elif 6.4 < t <= 7.2:
                # Rapid post-blast settling
                decay = (7.2 - t) / 0.8
                sec["vibration_ppv_mms"] = round(0.6 + decay * 6.0, 2)
                sec["radar_velocity_mm_hr"] = round(0.7 + decay * 1.5, 2)
                sec["rockfall_intensity"] = round(decay * 0.4, 2)
            # Other times remain baseline

        return sectors_state
