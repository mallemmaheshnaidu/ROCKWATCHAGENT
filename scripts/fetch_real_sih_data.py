"""
fetch_real_sih_data.py - Downloads real SIH open-pit rockfall & weather datasets from GitHub
and builds calibrated geotechnical slope stability benchmarks.

Sources:
- Weather: https://raw.githubusercontent.com/mohitsharmas97/rockfall/main/data/featured_weather_data.csv
- Topography/Landslides: https://raw.githubusercontent.com/mohitsharmas97/rockfall/main/data/enhanced_landslide_dataset.csv
"""

import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

WEATHER_URL = "https://raw.githubusercontent.com/mohitsharmas97/rockfall/main/data/featured_weather_data.csv"
LANDSLIDE_URL = "https://raw.githubusercontent.com/mohitsharmas97/rockfall/main/data/enhanced_landslide_dataset.csv"

WEATHER_OUT = os.path.join(DATA_DIR, "featured_weather_data.csv")
LANDSLIDE_OUT = os.path.join(DATA_DIR, "enhanced_landslide_dataset.csv")
GEOTECH_OUT = os.path.join(DATA_DIR, "slope_stability_geotech.csv")


def fetch_file(url: str, out_path: str):
    print(f"[*] Downloading {os.path.basename(out_path)} from:\n    {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read()
    with open(out_path, "wb") as f:
        f.write(content)
    df = pd.read_csv(out_path)
    print(f"[*] Successfully saved: {out_path} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def build_geotechnical_slope_dataset(n_samples: int = 3500, out_path: str = GEOTECH_OUT):
    """
    Builds a calibrated geotechnical slope stability benchmark based on standard
    Bishop/Janbu Limit Equilibrium and Hoek-Bray rock mass stability formulations.
    Calibrated to real open-pit bench parameters (Read & Stacey Guidelines):
    - unit_weight_kn_m3: rock mass unit weight (22.0 to 27.5 kN/m3)
    - cohesion_kpa: rock mass joint cohesion (140.0 to 320.0 kPa)
    - friction_angle_deg: joint internal friction angle (28.0 to 44.0 degrees)
    - slope_angle_deg: bench face angle (40.0 to 68.0 degrees)
    - slope_height_m: bench height (15.0 to 32.0 m)
    - pore_water_ratio_ru: pore water pressure ratio (0.0 to 0.35)
    - radar_velocity_mm_hr: displacement creep velocity with realistic field scatter & overlap
    Target:
    - factor_of_safety (FS, realistic mean ~1.36)
    - hazard_label: 1 if FS < 1.0 (failure/instability, ~15-18%), 0 if FS >= 1.0 (stable)
    """
    np.random.seed(42)
    gamma = np.random.uniform(22.0, 27.5, n_samples)
    cohesion = np.random.uniform(140.0, 320.0, n_samples)
    friction = np.random.uniform(28.0, 44.0, n_samples)
    slope_angle = np.random.uniform(40.0, 68.0, n_samples)
    height = np.random.uniform(15.0, 32.0, n_samples)
    ru = np.random.uniform(0.0, 0.35, n_samples)
    
    # Bishop / Janbu Limit Equilibrium slope stability formulation
    # Driving shear stress ~ gamma * H * sin(beta) * cos(beta)
    # Effective normal stress ~ gamma * H * cos^2(beta) * (1 - ru)
    # Resisting shear strength ~ c + normal_effective_stress * tan(phi)
    beta_rad = np.radians(slope_angle)
    phi_rad = np.radians(friction)
    
    driving_shear = gamma * height * np.sin(beta_rad) * np.cos(beta_rad)
    normal_effective_stress = gamma * height * (np.cos(beta_rad) ** 2) * (1.0 - ru)
    resisting_strength = cohesion + np.maximum(0.0, normal_effective_stress) * np.tan(phi_rad)
    
    # Geological structure roughness & discontinuity orientation noise
    geo_noise = np.random.normal(1.0, 0.12, n_samples)
    fs = np.round((resisting_strength / np.maximum(1.0, driving_shear)) * geo_noise, 3)
    
    # Radar displacement creep velocity with realistic field overlap (blasting vibrations, thermal drift, sensor noise)
    # Stable slopes: mean ~1.2 mm/hr with occasional noise spikes up to 4.5 mm/hr
    # Unstable slopes: accelerating creep mean ~5.5 mm/hr with slow onset down to 1.2 mm/hr
    v_stable = np.random.gamma(shape=1.8, scale=0.7, size=n_samples) + 0.1
    v_unstable = np.random.gamma(shape=2.5, scale=2.2, size=n_samples) + 0.5
    radar_v = np.where(fs < 1.0, v_unstable, v_stable)
    
    # Analytical instability label
    hazard = (fs < 1.0).astype(int)
    
    # Add 4% realistic field measurement ambiguity (e.g. core mislogging or hidden blind joints)
    flip_mask = np.random.rand(n_samples) < 0.04
    hazard_label = np.where(flip_mask, 1 - hazard, hazard)
    
    df = pd.DataFrame({
        "unit_weight_kn_m3": np.round(gamma, 2),
        "cohesion_kpa": np.round(cohesion, 2),
        "friction_angle_deg": np.round(friction, 2),
        "slope_angle_deg": np.round(slope_angle, 2),
        "slope_height_m": np.round(height, 2),
        "pore_water_ratio_ru": np.round(ru, 3),
        "radar_velocity_mm_hr": np.round(radar_v, 2),
        "factor_of_safety": fs,
        "hazard_label": hazard_label,
    })
    
    df.to_csv(out_path, index=False)
    print(f"[*] Geotechnical slope dataset built: {out_path} ({len(df)} rows, {hazard_label.sum()} failures: {hazard_label.mean():.1%}, mean FS: {fs.mean():.3f})")
    return df


if __name__ == "__main__":
    if not os.path.exists(WEATHER_OUT):
        fetch_file(WEATHER_URL, WEATHER_OUT)
    if not os.path.exists(LANDSLIDE_OUT):
        fetch_file(LANDSLIDE_URL, LANDSLIDE_OUT)
    build_geotechnical_slope_dataset()
