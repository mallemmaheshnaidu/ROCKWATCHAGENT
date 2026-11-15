"""
generate_training_data.py - Synthetic Grounded Mine Telemetry Dataset Generator.

Generates realistic historical shift records for 3 sensor modalities:
1. Seismic / Vibration telemetry (PPV, frequency, energy, duration)
2. Displacement / Radar telemetry (velocity, acceleration, inverse velocity, cumulative displacement)
3. Environmental / Weather telemetry (rainfall 3h/24h, pore pressure, moisture)

Incorporates realistic open-pit class imbalance (~10% true hazard precursors, 90% stable/normal).
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)


def generate_seismic_dataset(n_samples: int = 3000, out_path: str = "data/seismic_telemetry.csv") -> pd.DataFrame:
    """Generates vibration & microseismic sensor records."""
    # 90% normal operational shifts (blasting, truck traffic, ambient noise)
    # 10% seismic pre-rockfall clustering events
    is_hazard = np.random.binomial(1, 0.10, n_samples)

    # Normal shifts: low PPV, normal blast spikes, low cumulative energy
    ppv_normal = np.random.lognormal(mean=1.2, sigma=0.6, size=n_samples)
    energy_normal = np.random.exponential(scale=500.0, size=n_samples)
    freq_normal = np.random.normal(loc=35.0, scale=8.0, size=n_samples)
    duration_normal = np.random.uniform(0.5, 3.0, size=n_samples)

    # Precursor hazard shifts: high PPV, sustained energy surges, lower characteristic frequency
    ppv_hazard = np.random.lognormal(mean=3.2, sigma=0.5, size=n_samples)
    energy_hazard = np.random.exponential(scale=4500.0, size=n_samples) + 2000.0
    freq_hazard = np.random.normal(loc=18.0, scale=4.0, size=n_samples)
    duration_hazard = np.random.uniform(4.0, 15.0, size=n_samples)

    ppv = np.where(is_hazard == 1, ppv_hazard, ppv_normal)
    energy = np.where(is_hazard == 1, energy_hazard, energy_normal)
    freq = np.clip(np.where(is_hazard == 1, freq_hazard, freq_normal), 5.0, 80.0)
    duration = np.where(is_hazard == 1, duration_hazard, duration_normal)

    # Ambient noise ratio
    noise_ratio = np.random.uniform(0.05, 0.35, size=n_samples)

    df = pd.DataFrame({
        "vibration_ppv_mms": np.round(ppv, 2),
        "energy_joules": np.round(energy, 1),
        "dominant_frequency_hz": np.round(freq, 1),
        "event_duration_s": np.round(duration, 2),
        "noise_ratio": np.round(noise_ratio, 3),
        "hazard_label": is_hazard,
    })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[*] Seismic telemetry dataset written: {out_path} ({len(df)} samples, {sum(is_hazard)} hazards)")
    return df


def generate_displacement_dataset(n_samples: int = 3000, out_path: str = "data/displacement_telemetry.csv") -> pd.DataFrame:
    """Generates radar and crack extensometer displacement velocity records."""
    is_hazard = np.random.binomial(1, 0.12, n_samples)

    # Normal: sub-millimetre movement, near-zero acceleration
    v_normal = np.random.uniform(0.05, 1.2, size=n_samples)
    accel_normal = np.random.normal(0.0, 0.05, size=n_samples)
    cumul_normal = np.random.uniform(1.0, 20.0, size=n_samples)

    # Precursor hazard: accelerating creep, positive acceleration, large cumulative deformation
    v_hazard = np.random.uniform(3.5, 22.0, size=n_samples)
    accel_hazard = np.random.uniform(0.2, 4.0, size=n_samples)
    cumul_hazard = np.random.uniform(35.0, 180.0, size=n_samples)

    velocity = np.where(is_hazard == 1, v_hazard, v_normal)
    accel = np.where(is_hazard == 1, accel_hazard, accel_normal)
    cumul = np.where(is_hazard == 1, cumul_hazard, cumul_normal)

    # Fukuzono inverse velocity: 1 / velocity
    inv_v = 1.0 / np.clip(velocity, 0.01, 100.0)

    # Creep rate ratio: current velocity relative to 24h baseline
    creep_ratio = np.where(is_hazard == 1, np.random.uniform(2.5, 10.0, size=n_samples), np.random.uniform(0.8, 1.4, size=n_samples))

    df = pd.DataFrame({
        "velocity_mm_hr": np.round(velocity, 3),
        "acceleration_mm_hr2": np.round(accel, 4),
        "inverse_velocity": np.round(inv_v, 4),
        "cumulative_displacement_mm": np.round(cumul, 2),
        "creep_rate_ratio": np.round(creep_ratio, 2),
        "hazard_label": is_hazard,
    })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[*] Displacement telemetry dataset written: {out_path} ({len(df)} samples, {sum(is_hazard)} hazards)")
    return df


def generate_environmental_dataset(n_samples: int = 3000, out_path: str = "data/environmental_telemetry.csv") -> pd.DataFrame:
    """Generates rainfall, pore water pressure, and soil moisture records."""
    is_hazard = np.random.binomial(1, 0.10, n_samples)

    # Normal: low/moderate rain, stable pore pressure
    rain_3h_normal = np.random.exponential(scale=3.0, size=n_samples)
    rain_24h_normal = rain_3h_normal + np.random.exponential(scale=8.0, size=n_samples)
    pore_kpa_normal = np.random.normal(loc=40.0, scale=10.0, size=n_samples)
    moisture_normal = np.random.uniform(10.0, 35.0, size=n_samples)

    # Precursor hazard: heavy precipitation saturation, high pore pressure spike
    rain_3h_hazard = np.random.uniform(15.0, 55.0, size=n_samples)
    rain_24h_hazard = rain_3h_hazard + np.random.uniform(30.0, 110.0, size=n_samples)
    pore_kpa_hazard = np.random.normal(loc=115.0, scale=20.0, size=n_samples)
    moisture_hazard = np.random.uniform(45.0, 85.0, size=n_samples)

    rain_3h = np.where(is_hazard == 1, rain_3h_hazard, rain_3h_normal)
    rain_24h = np.where(is_hazard == 1, rain_24h_hazard, rain_24h_normal)
    pore = np.clip(np.where(is_hazard == 1, pore_kpa_hazard, pore_kpa_normal), 5.0, 200.0)
    moisture = np.clip(np.where(is_hazard == 1, moisture_hazard, moisture_normal), 5.0, 95.0)

    temp_c = np.random.normal(loc=22.0, scale=6.0, size=n_samples)

    df = pd.DataFrame({
        "rainfall_3h_mm": np.round(rain_3h, 2),
        "rainfall_24h_mm": np.round(rain_24h, 2),
        "pore_pressure_kpa": np.round(pore, 1),
        "soil_moisture_pct": np.round(moisture, 1),
        "temperature_celsius": np.round(temp_c, 1),
        "hazard_label": is_hazard,
    })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[*] Environmental telemetry dataset written: {out_path} ({len(df)} samples, {sum(is_hazard)} hazards)")
    return df


if __name__ == "__main__":
    generate_seismic_dataset()
    generate_displacement_dataset()
    generate_environmental_dataset()
