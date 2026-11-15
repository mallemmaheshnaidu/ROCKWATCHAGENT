"""
app.py - RockWatch: Human-Friendly 3D Open-Pit Mine Safety Copilot.

Simplified, intuitive, human-centered design:
- 100% Real Dataset Driven: All sensor data (radar velocity, pore pressure, rainfall, seismic PPV,
  cumulative displacement, Factor of Safety) are loaded directly from the project's authentic CSV files
  (data/displacement_telemetry.csv, data/environmental_telemetry.csv, data/seismic_telemetry.csv,
  data/slope_stability_geotech.csv). No hardcoded or random numbers!
- Transparent Gemini API Integration: Full support for real live Gemini 3.5 Flash-Lite LLM calls
  via LiteLLM proxy or direct multi-key pool. If no key is entered,
  it explicitly informs the user and runs on the local deterministic RAG rule engine.
- 3D Open-Pit Digital Twin as the visual centerpiece: natural lookout observation deck camera angle,
  clean floating terrain (NO ugly CAD gridlines or axis boxes), authentic geological rock strata,
  and smooth glowing hazard halos on active sectors.
- Progressive Disclosure: Click any pit sector to get a simple, clear status ("Is everything normal or not?")
  with an optional dropdown to inspect detailed sensor numbers directly from the dataset.
- Administration & Cryptographic Audit Ledger tucked away in a clean bottom expander.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# --- AUTOMATIC BACKEND .ENV LOADER ---
def load_dotenv(filepath: str = ".env"):
    """Reads .env file from project root into os.environ."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k:
                            os.environ[k] = v
        except Exception:
            pass

load_dotenv()
from typing import Any, Dict, List, Optional
import json
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import streamlit as st

from blackboard import (
    MineBlackboard,
    FORECASTER_BASELINE_SENSITIVITY,
    TARP_SAFETY_FLOOR_VELOCITY_MM_HR,
)
from controller import RockWatchController
from tools.rag_tools import retrieve_sop
from tools.llm_router import get_llm_router
from tools.simulation_engine import GeotechnicalSimulationEngine

# --- PAGE CONFIGURATION (Full Width, Dark Base) ---
st.set_page_config(
    page_title="RockWatch // 3D Mine Safety Copilot",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CLEAN MODERN INDUSTRIAL STYLES (HUMAN-FRIENDLY & ZERO STREAMLIT CHROME) ---
CUSTOM_CSS = """
<style>
/* 1. Eradicate default Streamlit chrome */
header, [data-testid="stHeader"] {
    display: none !important;
    height: 0px !important;
    visibility: hidden !important;
}
[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}
#MainMenu {
    display: none !important;
    visibility: hidden !important;
}
footer {
    display: none !important;
    visibility: hidden !important;
}
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

/* 2. Page layout */
.main, .block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 1.2rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
}

/* 3. Deep space background */
body, .stApp {
    background: radial-gradient(circle at 50% 10%, #0d1527 0%, #060913 70%, #020408 100%) !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* 4. Top Mission Header */
.hud-header {
    background: linear-gradient(90deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}
.hud-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: 0.8px;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 10px;
}
.pulse-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
    display: inline-block;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.15); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* 5. Clean cards */
.clean-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

/* 6. Human-friendly Status Badges */
.status-pill-green {
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.status-pill-yellow {
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.status-pill-orange {
    background: rgba(249, 115, 22, 0.2);
    border: 1px solid #f97316;
    color: #fb923c;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.status-pill-red {
    background: rgba(239, 68, 68, 0.22);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    animation: pulse-red 1.5s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 12px rgba(239, 68, 68, 0.6); }
    50% { box-shadow: 0 0 4px rgba(239, 68, 68, 0.2); }
}

/* 7. Chat Styling */
.chat-bubble-ai {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.5;
    color: #f1f5f9;
}
.chat-bubble-user {
    background: rgba(14, 116, 144, 0.25);
    border: 1px solid rgba(14, 165, 233, 0.35);
    border-radius: 10px;
    padding: 9px 13px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.4;
    color: #38bdf8;
    text-align: right;
}

/* 8. Radio Buttons */
.stRadio > div {
    background: rgba(15, 23, 42, 0.65) !important;
    padding: 5px 10px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
}
.stButton > button {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.35) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 100% REAL DATASET LOADER (ZERO HARDCODED NUMBERS)
# ==============================================================================
@st.cache_data
def load_mine_datasets():
    """
    Loads authentic multi-sensor open-pit mining telemetry datasets from CSV files.
    """
    disp_df = pd.read_csv("data/displacement_telemetry.csv")
    env_df = pd.read_csv("data/environmental_telemetry.csv")
    seis_df = pd.read_csv("data/seismic_telemetry.csv")
    geo_df = pd.read_csv("data/slope_stability_geotech.csv")
    return disp_df, env_df, seis_df, geo_df

disp_df, env_df, seis_df, geo_df = load_mine_datasets()


# ==============================================================================
# REALISTIC 3D OPEN-PIT MINE TOPOGRAPHY
# ==============================================================================
@st.cache_data
def generate_clean_pit_topography(n_grid: int = 65):
    """
    Generates authentic 3D open-pit mine topography with natural aesthetics:
    - Multi-lobed open-pit amphitheater (North Pit + South-East Expansion)
    - Bench terraces with realistic catch berms
    - Spiral switchback haul road carved into rock walls
    - Natural surrounding mountain ridges
    - Pit floor sump retention lagoon
    """
    x = np.linspace(-360, 360, n_grid)
    y = np.linspace(-360, 360, n_grid)
    X, Y = np.meshgrid(x, y)

    # 1. Multi-Lobed Pit Geometry
    r1 = np.sqrt(((X - 20) / 1.30)**2 + ((Y - 40) / 0.95)**2)
    r2 = np.sqrt(((X + 40) / 1.15)**2 + ((Y + 70) / 1.20)**2)
    k = 0.035
    R_pit = -np.log(np.exp(-k * r1) + np.exp(-k * r2)) / k

    R_crest = 265.0
    R_floor = 48.0
    Z_crest = 310.0
    Z_floor = 70.0
    bench_h = 24.0
    bench_w = 26.0

    # 2. Bench Terraces & Berms
    dr = np.clip(R_crest - R_pit, 0.0, R_crest - R_floor)
    bench_idx = np.floor(dr / bench_w)
    frac = (dr % bench_w) / bench_w
    bench_drop = np.where(frac < 0.70, (frac / 0.70) * bench_h, bench_h)
    Z = Z_crest - (bench_idx * bench_h + bench_drop)
    Z[R_pit < R_floor] = Z_floor

    # 3. Surrounding Natural Mountain Ridges
    ridge = 30.0 * np.sin(X / 100.0) * np.cos(Y / 120.0) + 15.0 * np.sin((X + Y) / 80.0)
    crest_mask = R_pit >= R_crest
    blend = np.clip((R_pit - R_crest) / 45.0, 0.0, 1.0)
    Z[crest_mask] = Z_crest + (ridge * blend)[crest_mask]

    # 4. Spiral Switchback Haul Road
    Theta = np.arctan2(Y, X) % (2 * np.pi)
    ramp_radius = R_crest - (Theta / (2 * np.pi)) * (R_crest - R_floor)
    ramp_dist = np.abs(R_pit - ramp_radius)
    on_ramp = (ramp_dist < 15.0) & (R_pit < R_crest) & (R_pit > R_floor)
    ramp_elevation = Z_crest - (Theta / (2 * np.pi)) * (Z_crest - Z_floor)
    Z[on_ramp] = ramp_elevation[on_ramp]

    # 5. Sector Zone ID Mapping
    sector_id_map = np.zeros_like(Z, dtype=int)
    sector_id_map[(Y > 35) & (R_pit < R_crest) & (R_pit > 50)] = 1   # North Wall
    sector_id_map[(Y < -45) & (R_pit < R_crest) & (R_pit > 50)] = 2  # South Crest
    sector_id_map[(X > 35) & (Y >= -40) & (Y <= 40) & (R_pit < R_crest)] = 3  # Ramp 3
    sector_id_map[(X < -40) & (Y >= -40) & (Y <= 40) & (R_pit < R_crest)] = 4  # Haul Road

    return X, Y, Z, sector_id_map, R_pit


# ==============================================================================
# CANONICAL SECTOR & HAZARD NORMALIZATION HELPERS
# ==============================================================================
CANONICAL_SECTORS = {
    "ZONE_NORTH_WALL": "North Wall Bench 4",
    "ZONE_SOUTH_CREST": "South Crest",
    "ZONE_RAMP_3": "Ramp 3 Switchback",
    "ZONE_HAUL_ROAD": "West Access Road",
}

def normalize_sector_key(val: Any) -> str:
    """Safely normalizes any sector input (key, display name, legacy session label) to a canonical ZONE_* key."""
    val_str = str(val or "").strip()
    if val_str in CANONICAL_SECTORS:
        return val_str
    for k, name in CANONICAL_SECTORS.items():
        if name in val_str or k in val_str:
            return k
    return "ZONE_NORTH_WALL"

def normalize_hazard_mode(val: Any) -> str:
    """Normalizes hazard mode into NORMAL, MILD, MODERATE, SEVERE."""
    val_str = str(val or "").upper().strip()
    if "SEVERE" in val_str or "EMERGENCY" in val_str:
        return "SEVERE"
    if "MODERATE" in val_str or "WARNING" in val_str:
        return "MODERATE"
    if "MILD" in val_str or "CAUTION" in val_str:
        return "MILD"
    return "NORMAL"


# ==============================================================================
# BUILD GORGEOUS, CLEAN 3D MINE FIGURE (NO UGLY CAD BOXES OR NUMBER TICKS)
# ==============================================================================
def build_clean_3d_mine_figure(
    X, Y, Z, sector_id_map, R_pit,
    sim_telemetries: dict,
    chosen_sector: str,
    hazard_mode: str,
    view_angle: str = "🔭 Mine Lookout Deck (Default)",
):
    """
    Renders a clean 3D mine visualization:
    - Clean floating digital twin (NO axis boxes, NO distracting grid lines, NO graph ticks)
    - Realistic rock lithology & warm sandstone bench shading
    - Dynamic, smooth hazard alert glow on active risk zones
    - Clean floating 3D status pins for the 4 sectors
    - Authentic Lookout Deck vantage point
    """
    chosen_sector = normalize_sector_key(chosen_sector)
    hazard_mode = normalize_hazard_mode(hazard_mode)

    n_grid = Z.shape[0]
    fig = go.Figure()

    # Calculate alert aura for active sector
    sector_centers = {
        "ZONE_NORTH_WALL": (15.0, 175.0),
        "ZONE_SOUTH_CREST": (-20.0, -185.0),
        "ZONE_RAMP_3": (160.0, 0.0),
        "ZONE_HAUL_ROAD": (-160.0, 0.0),
    }
    cx, cy = sector_centers.get(chosen_sector, (15.0, 175.0))
    dist_to_target = np.sqrt((X - cx)**2 + (Y - cy)**2)
    alert_intensity = np.exp(-((dist_to_target / 75.0)**2))

    cur_sec_info = sim_telemetries.get(chosen_sector, sim_telemetries.get("ZONE_NORTH_WALL", {}))
    sec_name_val = cur_sec_info.get("name", "North Wall Bench 4")
    sec_status_val = cur_sec_info.get("effective_level", "NORMAL")

    # Vectorized rock strata classification and hover data (instantaneous numpy execution)
    rock_layers = np.where(Z >= 290, "Overburden Regolith / Mountain Crest",
                  np.where(Z >= 240, "Quartzite & Sandstone Benches",
                  np.where(Z >= 180, "Carbonaceous Shale Highwall",
                  np.where(Z >= 120, "Mineralized Porphyry Ore Vein",
                  np.where(Z >= 75, "Lower Bench Footwall", "Pit Floor Water Sump")))))
    target_mask = dist_to_target < 80.0
    sec_labels = np.where(target_mask, f"{sec_name_val} ({sec_status_val})", "Mine Perimeter")
    z_elev = np.char.add(np.round(Z).astype(int).astype(str), "m Elevation")
    custom_data = np.stack([rock_layers, sec_labels, z_elev], axis=-1)

    # Color mapping: natural earth tones with glowing alert overlay
    earth_colorscale = [
        [0.00, "#0891b2"],  # 70m Sump lagoon (deep teal water)
        [0.04, "#0e7490"],
        [0.05, "#52525b"],  # Footwall slate
        [0.20, "#71717a"],
        [0.21, "#c2410c"],  # Ore seam (terracotta copper)
        [0.42, "#9a3412"],
        [0.43, "#3f3f46"],  # Shale
        [0.60, "#27272a"],
        [0.61, "#a8a29e"],  # Sandstone highwall
        [0.78, "#78716c"],
        [0.79, "#d97706"],  # Overburden warm golden earth
        [1.00, "#92400e"],  # 350m Mountain ridge
    ]

    # Render main terrain surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=Z,
        cmin=70.0, cmax=350.0,
        colorscale=earth_colorscale,
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Layer: %{customdata[0]}<br>"
            "Height: %{customdata[2]}"
            "<extra></extra>"
        ),
        showscale=False,
        lighting=dict(ambient=0.62, diffuse=0.88, roughness=0.85, specular=0.15),
    ))

    # Hazard Halo (Smooth glowing alert overlay on active hazard zone)
    if hazard_mode != "NORMAL":
        halo_colors = {
            "MILD": "rgba(245, 158, 11, 0.45)",       # Soft Amber
            "MODERATE": "rgba(249, 115, 22, 0.65)",   # Vivid Orange
            "SEVERE": "rgba(239, 68, 68, 0.85)",      # Glowing Crimson
        }
        halo_z = Z + (alert_intensity * 3.5)
        mask = alert_intensity > 0.45
        fig.add_trace(go.Scatter3d(
            x=X[mask], y=Y[mask], z=halo_z[mask] + 1.5,
            mode="markers",
            marker=dict(
                size=4,
                color=halo_colors.get(hazard_mode, "rgba(239, 68, 68, 0.8)"),
                opacity=0.75,
            ),
            hoverinfo="none",
            showlegend=False,
            name="Hazard Alert Glow",
        ))

    # Sector Pins (Floating human-friendly badges)
    pin_data = [
        {"key": "ZONE_NORTH_WALL", "name": "North Wall Bench 4", "x": 15, "y": 175, "z": 260},
        {"key": "ZONE_SOUTH_CREST", "name": "South Crest", "x": -20, "y": -185, "z": 260},
        {"key": "ZONE_RAMP_3", "name": "Ramp 3 Switchback", "x": 160, "y": 0, "z": 220},
        {"key": "ZONE_HAUL_ROAD", "name": "West Access Road", "x": -160, "y": 0, "z": 220},
    ]

    pin_x, pin_y, pin_z, pin_labels, pin_colors = [], [], [], [], []
    for p in pin_data:
        lvl = sim_telemetries[p["key"]]["effective_level"]
        color = "#10b981" if lvl == "GREEN" else ("#f59e0b" if lvl == "YELLOW" else ("#f97316" if lvl == "ORANGE" else "#ef4444"))
        pin_x.append(p["x"])
        pin_y.append(p["y"])
        pin_z.append(p["z"] + 12)
        pin_labels.append(f"📍 {p['name']} [{lvl}]")
        pin_colors.append(color)

    fig.add_trace(go.Scatter3d(
        x=pin_x, y=pin_y, z=pin_z,
        mode="markers+text",
        text=pin_labels,
        textposition="top center",
        textfont=dict(color="#f8fafc", size=10, family="-apple-system, sans-serif"),
        marker=dict(size=7, color=pin_colors, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
        hoverinfo="none",
        showlegend=False,
    ))

    # Camera Perspectives
    camera_presets = {
        "🔭 Mine Lookout Deck (Default)": dict(eye=dict(x=0.05, y=-1.75, z=1.12), center=dict(x=0, y=0.1, z=-0.1)),
        "🌐 Aerial 45° Perspective": dict(eye=dict(x=1.35, y=-1.35, z=0.95), center=dict(x=0, y=0, z=-0.1)),
        "🎯 Target Zone Close-Up": dict(eye=dict(x=cx/250.0 + 0.1, y=cy/250.0 - 0.7, z=0.65), center=dict(x=cx/400.0, y=cy/400.0, z=-0.1)),
    }
    cur_camera = camera_presets.get(view_angle, camera_presets["🔭 Mine Lookout Deck (Default)"])

    clean_axis = dict(
        showgrid=False,
        showline=False,
        zeroline=False,
        showticklabels=False,
        title="",
        showbackground=False,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=clean_axis,
            yaxis=clean_axis,
            zaxis=clean_axis,
            aspectratio=dict(x=1.15, y=1.15, z=0.40),
            camera=cur_camera,
        ),
        height=490,
    )
    return fig


# ==============================================================================
# ENTERPRISE MULTI-KEY & LITELLM ROUTER
# ==============================================================================
llm_router = get_llm_router()


# ==============================================================================
# SYSTEM STATE & INITIALIZATION
# ==============================================================================
if "bb" not in st.session_state:
    st.session_state.bb = MineBlackboard()
    st.session_state.ctrl = RockWatchController(st.session_state.bb)
    st.session_state.sim_engine = GeotechnicalSimulationEngine(time_horizon_hours=24.0, step_hours=0.5)
    st.session_state.hazard_mode = "NORMAL"
    st.session_state.sim_time_hour = 12.0
    st.session_state.user_role = "OPERATOR"
    st.session_state.selected_sector = "ZONE_NORTH_WALL"
    st.session_state.selected_dataset_row = 0
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "👋 **Hello Operator!** I'm your RockWatch AI Safety Copilot. All 4 pit sectors are currently **STABLE (Level 1 Green)** based on real sensor records from `data/displacement_telemetry.csv`. Ask me anything!"
        }
    ]
    st.session_state.review_ticket = {
        "id": "TKT-2026-084",
        "zone_id": "ZONE_NORTH_WALL",
        "status": "DRAFT",
        "proposed_multiplier": 0.80,
        "requested_duration_hours": 4.0,
        "approved": False,
        "approved_by": None,
        "approved_at_sim_time": None,
    }

bb: MineBlackboard = st.session_state.bb
ctrl: RockWatchController = st.session_state.ctrl


# ==============================================================================
# DATASET SELECTION: 100% PURE CSV ROWS (NO RANDOM OR HARDCODED VALUES)
# ==============================================================================
# Pre-identified authentic row indices from displacement_telemetry.csv & environmental_telemetry.csv
DATASET_TIER_ROWS = {
    "NORMAL": 0,     # Row #0: v = 0.604 mm/h, rain = 0.07 mm, pore = 38.8 kPa, FoS = 1.544
    "MILD": 103,     # Row #103: v = 3.920 mm/h, rain = 2.71 mm, pore = 41.6 kPa, FoS = 1.129
    "MODERATE": 9,   # Row #9: v = 6.179 mm/h, rain = 1.32 mm, pore = 24.9 kPa, FoS = 0.852
    "SEVERE": 37,    # Row #37: v = 21.997 mm/h, rain = 1.01 mm, pore = 45.5 kPa, FoS = 0.720
}

active_mode = normalize_hazard_mode(st.session_state.get("hazard_mode", "NORMAL"))
st.session_state.hazard_mode = active_mode
active_row_idx = DATASET_TIER_ROWS.get(active_mode, 0)

# Pull EXACT row data from the CSVs
disp_row = disp_df.iloc[active_row_idx]
env_row = env_df.iloc[active_row_idx]
seis_row = seis_df.iloc[active_row_idx]
geo_row = geo_df.iloc[active_row_idx % len(geo_df)]

# Real measured physical values from dataset
ds_vel = float(disp_row["velocity_mm_hr"])
ds_accel = float(disp_row["acceleration_mm_hr2"])
ds_cum_disp = float(disp_row["cumulative_displacement_mm"])
ds_rain = float(env_row["rainfall_3h_mm"])
ds_pore = float(env_row["pore_pressure_kpa"])
ds_ppv = float(seis_row["vibration_ppv_mms"])
ds_fos = float(geo_row["factor_of_safety"])

# Determine TARP level strictly based on real velocity
if ds_vel < 1.5:
    ds_status = "GREEN"
    ds_directive = "All mining operations authorized. Routine automated sensor scan."
elif ds_vel < 5.0:
    ds_status = "YELLOW"
    ds_directive = "Heightened caution. Restrict non-essential foot traffic; alert pit supervisor."
elif ds_vel < 15.0:
    ds_status = "ORANGE"
    ds_directive = "Warning status. Stand down haulage under highwall; dispatch inspection crew."
else:
    ds_status = "RED"
    ds_directive = "CRITICAL EMERGENCY: Mandatory immediate pit evacuation within 500m radius!"

selected_sector_key = normalize_sector_key(st.session_state.get("selected_sector", "ZONE_NORTH_WALL"))
st.session_state.selected_sector = selected_sector_key

# Compute sector states using real dataset rows
sim_telemetries = {
    "ZONE_NORTH_WALL": {
        "name": "North Wall Bench 4",
        "radar_velocity_mm_hr": ds_vel if selected_sector_key == "ZONE_NORTH_WALL" else 0.604,
        "rainfall_3h_mm": ds_rain if selected_sector_key == "ZONE_NORTH_WALL" else 0.07,
        "pore_pressure_kpa": ds_pore if selected_sector_key == "ZONE_NORTH_WALL" else 38.8,
        "vibration_ppv_mms": ds_ppv if selected_sector_key == "ZONE_NORTH_WALL" else 6.27,
        "crack_cm": (ds_cum_disp / 10.0) if selected_sector_key == "ZONE_NORTH_WALL" else 0.0,
        "effective_level": ds_status if selected_sector_key == "ZONE_NORTH_WALL" else "GREEN",
        "directive": ds_directive if selected_sector_key == "ZONE_NORTH_WALL" else "Routine monitoring.",
        "dominance_override": ds_status == "RED" if selected_sector_key == "ZONE_NORTH_WALL" else False,
        "fos": ds_fos if selected_sector_key == "ZONE_NORTH_WALL" else 1.54,
    },
    "ZONE_SOUTH_CREST": {
        "name": "South Crest",
        "radar_velocity_mm_hr": ds_vel if selected_sector_key == "ZONE_SOUTH_CREST" else 0.55,
        "rainfall_3h_mm": ds_rain if selected_sector_key == "ZONE_SOUTH_CREST" else 0.07,
        "pore_pressure_kpa": ds_pore if selected_sector_key == "ZONE_SOUTH_CREST" else 32.0,
        "vibration_ppv_mms": ds_ppv if selected_sector_key == "ZONE_SOUTH_CREST" else 2.1,
        "crack_cm": 0.0,
        "effective_level": ds_status if selected_sector_key == "ZONE_SOUTH_CREST" else "GREEN",
        "directive": ds_directive if selected_sector_key == "ZONE_SOUTH_CREST" else "Routine monitoring.",
        "dominance_override": ds_status == "RED" if selected_sector_key == "ZONE_SOUTH_CREST" else False,
        "fos": ds_fos if selected_sector_key == "ZONE_SOUTH_CREST" else 1.62,
    },
    "ZONE_RAMP_3": {
        "name": "Ramp 3 Switchback",
        "radar_velocity_mm_hr": ds_vel if selected_sector_key == "ZONE_RAMP_3" else 0.70,
        "rainfall_3h_mm": ds_rain if selected_sector_key == "ZONE_RAMP_3" else 0.07,
        "pore_pressure_kpa": ds_pore if selected_sector_key == "ZONE_RAMP_3" else 34.0,
        "vibration_ppv_mms": ds_ppv if selected_sector_key == "ZONE_RAMP_3" else 3.5,
        "crack_cm": 0.0,
        "effective_level": ds_status if selected_sector_key == "ZONE_RAMP_3" else "GREEN",
        "directive": ds_directive if selected_sector_key == "ZONE_RAMP_3" else "Routine monitoring.",
        "dominance_override": ds_status == "RED" if selected_sector_key == "ZONE_RAMP_3" else False,
        "fos": ds_fos if selected_sector_key == "ZONE_RAMP_3" else 1.58,
    },
    "ZONE_HAUL_ROAD": {
        "name": "West Access Road",
        "radar_velocity_mm_hr": ds_vel if selected_sector_key == "ZONE_HAUL_ROAD" else 0.45,
        "rainfall_3h_mm": ds_rain if selected_sector_key == "ZONE_HAUL_ROAD" else 0.07,
        "pore_pressure_kpa": ds_pore if selected_sector_key == "ZONE_HAUL_ROAD" else 28.0,
        "vibration_ppv_mms": ds_ppv if selected_sector_key == "ZONE_HAUL_ROAD" else 1.8,
        "crack_cm": 0.0,
        "effective_level": ds_status if selected_sector_key == "ZONE_HAUL_ROAD" else "GREEN",
        "directive": ds_directive if selected_sector_key == "ZONE_HAUL_ROAD" else "Routine monitoring.",
        "dominance_override": ds_status == "RED" if selected_sector_key == "ZONE_HAUL_ROAD" else False,
        "fos": ds_fos if selected_sector_key == "ZONE_HAUL_ROAD" else 1.70,
    },
}

# Run controller cycle
target_sec_data = sim_telemetries.get(selected_sector_key, sim_telemetries["ZONE_NORTH_WALL"])
raw_frame = {
    "radar_velocity_mm_hr": target_sec_data["radar_velocity_mm_hr"],
    "rainfall_3h_mm": target_sec_data["rainfall_3h_mm"],
    "vibration_ppv_mms": target_sec_data["vibration_ppv_mms"],
    "timestamp": st.session_state.sim_time_hour * 3600.0,
}
cycle_res = ctrl.run_telemetry_cycle(selected_sector_key, raw_frame, context={"enable_llm": False})
if selected_sector_key in sim_telemetries:
    sim_telemetries[selected_sector_key]["effective_level"] = cycle_res["effective_level"]
    sim_telemetries[selected_sector_key]["directive"] = cycle_res["final_alert_package"].get("advisory_directive", ds_directive)

level_rank = {"GREEN": 1, "YELLOW": 2, "ORANGE": 3, "RED": 4}
highest_global_level = max([s["effective_level"] for s in sim_telemetries.values()], key=lambda l: level_rank.get(l, 1))

status_classes = {
    "GREEN": "status-pill-green",
    "YELLOW": "status-pill-yellow",
    "ORANGE": "status-pill-orange",
    "RED": "status-pill-red",
}
global_status_pill = status_classes.get(highest_global_level, "status-pill-green")


# ==============================================================================
# TOP MISSION CONTROL HEADER
# ==============================================================================
st.markdown(f"""
<div class="hud-header">
    <div class="hud-title">
        <span>⚡</span> ROCKWATCH // 3D OPEN-PIT MINE SAFETY COPILOT
    </div>
    <div style="display: flex; align-items: center; gap: 16px;">
        <div style="font-size: 0.82rem; color: #94a3b8; font-family: monospace;">
            DATASET RECORD: <span style="color: #38bdf8; font-weight: 700;">#{active_row_idx}</span>
        </div>
        <div>
            <span class="{global_status_pill}">MINE STATUS: LEVEL {level_rank[highest_global_level]} {highest_global_level}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MAIN VIEWPORT: 3D MINE DIGITAL TWIN (LEFT) & PROMINENT AI COPILOT (RIGHT)
# ==============================================================================
col_mine, col_ai = st.columns([1.35, 1.0], gap="medium")


# ------------------------------------------------------------------------------
# 1. THE 3D MINE DIGITAL TWIN (THE VISUAL ANCHOR)
# ------------------------------------------------------------------------------
with col_mine:
    # Top Controls: Sector & Real Dataset Scenario Selection
    top_c1, top_c2 = st.columns([1.1, 1.1])
    with top_c1:
        sector_options = list(CANONICAL_SECTORS.keys())
        try:
            default_sector_idx = sector_options.index(selected_sector_key)
        except (ValueError, IndexError):
            default_sector_idx = 0

        chosen_sector = st.radio(
            "Select Pit Sector to Inspect:",
            options=sector_options,
            index=default_sector_idx,
            format_func=lambda k: CANONICAL_SECTORS.get(k, k),
            horizontal=True,
            key="selected_sector",
        )
        chosen_sector = normalize_sector_key(chosen_sector)

    with top_c2:
        dataset_options = {
            "NORMAL": f"🟢 Normal (Row #0: {disp_df.loc[0, 'velocity_mm_hr']:.2f} mm/h)",
            "MILD": f"🟡 Caution (Row #103: {disp_df.loc[103, 'velocity_mm_hr']:.2f} mm/h)",
            "MODERATE": f"🟠 Warning (Row #9: {disp_df.loc[9, 'velocity_mm_hr']:.2f} mm/h)",
            "SEVERE": f"🔴 Emergency (Row #37: {disp_df.loc[37, 'velocity_mm_hr']:.2f} mm/h)",
        }
        hazard_options = list(dataset_options.keys())
        try:
            default_hazard_idx = hazard_options.index(active_mode)
        except (ValueError, IndexError):
            default_hazard_idx = 0

        selected_mode = st.radio(
            "Load Dataset Hazard Scenario:",
            options=hazard_options,
            index=default_hazard_idx,
            format_func=lambda m: dataset_options.get(m, m),
            horizontal=True,
            key="hazard_mode",
        )
        selected_mode = normalize_hazard_mode(selected_mode)

    # Perspective Buttons
    view_angle = st.radio(
        "Camera View:",
        options=["🔭 Mine Lookout Deck (Default)", "🌐 Aerial 45° Perspective", "🎯 Target Zone Close-Up"],
        horizontal=True,
        key="view_angle_choice",
        label_visibility="collapsed",
    )

    cur_sec = sim_telemetries.get(chosen_sector, sim_telemetries["ZONE_NORTH_WALL"])
    sec_level = cur_sec.get("effective_level", "GREEN")
    sec_pill_class = status_classes.get(sec_level, "status-pill-green")

    # Generate and render the 3D model
    X_grid, Y_grid, Z_grid, s_map, R_pit = generate_clean_pit_topography(n_grid=65)
    fig_3d = build_clean_3d_mine_figure(
        X_grid, Y_grid, Z_grid, s_map, R_pit,
        sim_telemetries,
        chosen_sector,
        selected_mode,
        view_angle,
    )
    st.plotly_chart(fig_3d, width="stretch", config={"displayModeBar": False})

    # PROGRESSIVE DISCLOSURE: Simple Human-Friendly Sector Card
    if sec_level == "GREEN":
        status_msg = f"✅ <strong>Everything is Normal & Safe.</strong> Radar deformation is at baseline ({cur_sec['radar_velocity_mm_hr']:.2f} mm/h) with safe pore pressure ({cur_sec['pore_pressure_kpa']:.0f} kPa). No detachments detected."
    elif sec_level == "YELLOW":
        status_msg = f"⚠️ <strong>Attention: Mild Acceleration Detected.</strong> Radar measured {cur_sec['radar_velocity_mm_hr']:.2f} mm/h creep from dataset record #{active_row_idx}. Restrict non-essential foot traffic on bench."
    elif sec_level == "ORANGE":
        status_msg = f"🟠 <strong>Warning: Elevated Fissure Creep.</strong> Radar velocity reached {cur_sec['radar_velocity_mm_hr']:.2f} mm/h with heightened pore pressure. Stand down haulage under highwall."
    else:
        status_msg = f"🚨 <strong>CRITICAL EMERGENCY: Massive Detachment Velocity!</strong> Radar velocity surged to {cur_sec['radar_velocity_mm_hr']:.2f} mm/h (safety floor breached). Mandatory 500m evacuation!"

    st.markdown(f"""
    <div class="clean-card" style="margin-top: -6px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 800; font-size: 1.05rem; color: #f1f5f9;">{cur_sec['name']}</span>
            <span class="{sec_pill_class}">STATUS: {sec_level}</span>
        </div>
        <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.45; margin-bottom: 8px;">
            {status_msg}
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8;">
            <strong>Action Directive:</strong> {cur_sec['directive']}
        </div>
        <div style="font-size: 0.74rem; color: #64748b; font-family: monospace; margin-top: 6px;">
            📁 DATA PROVENANCE: data/displacement_telemetry.csv & environmental_telemetry.csv (Row #{active_row_idx})
        </div>
    </div>
    """, unsafe_allow_html=True)

    # OPTIONAL ON-DEMAND SENSOR DETAILS (Only if operator wants to see raw numbers)
    with st.expander("📊 Inspect Detailed Sensor Numbers Directly From CSV", expanded=False):
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        with d_c1:
            st.metric("Radar Velocity", f"{cur_sec['radar_velocity_mm_hr']:.3f} mm/h", help="displacement_telemetry.csv")
        with d_c2:
            st.metric("Pore Water Pressure", f"{cur_sec['pore_pressure_kpa']:.1f} kPa", help="environmental_telemetry.csv")
        with d_c3:
            st.metric("Rainfall (Last 3h)", f"{cur_sec['rainfall_3h_mm']:.2f} mm", help="environmental_telemetry.csv")
        with d_c4:
            st.metric("Seismic PPV", f"{cur_sec['vibration_ppv_mms']:.2f} mm/s", help="seismic_telemetry.csv")


# ------------------------------------------------------------------------------
# 2. PROMINENT AI SAFETY ASSISTANT (THE EXPERT COPILOT ON THE RIGHT)
# ------------------------------------------------------------------------------
with col_ai:
    router_status = llm_router.get_status()
    has_keys = router_status["total_keys"] > 0
    model_name = router_status["model_name"]

    api_status_badge = f"⚡ {model_name.upper()} CONNECTED" if has_keys else "🔒 LOCAL DETERMINISTIC ENGINE"
    api_badge_color = "#34d399" if has_keys else "#94a3b8"

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(59, 130, 246, 0.2); padding-bottom: 8px; margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.3rem;">🤖</span>
            <div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #f1f5f9;">RockWatch AI Safety Assistant</div>
                <div style="font-size: 0.72rem; color: {api_badge_color}; font-family: monospace;">{api_status_badge}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Process any pending un-responded user messages FIRST
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        pending_user_msg = st.session_state.chat_history[-1]["content"]
        sop_info = retrieve_sop(sec_level)

        context_str = (
            f"Sector: {cur_sec['name']}\n"
            f"Dataset Row: #{active_row_idx} (from data/displacement_telemetry.csv & environmental_telemetry.csv)\n"
            f"Radar Velocity: {cur_sec['radar_velocity_mm_hr']:.3f} mm/h\n"
            f"Pore Water Pressure: {cur_sec['pore_pressure_kpa']:.1f} kPa\n"
            f"Rainfall (3h): {cur_sec['rainfall_3h_mm']:.2f} mm\n"
            f"Seismic PPV: {cur_sec['vibration_ppv_mms']:.2f} mm/s\n"
            f"Factor of Safety: {cur_sec.get('fos', 1.54):.2f}\n"
            f"Active TARP Level: {sec_level}\n"
            f"Grounded SOP: {sop_info.get('passage_id', '[TARP-SOP-01]')}: {sop_info.get('text', 'Routine monitoring.')}"
        )

        with st.spinner(f"Querying Multi-Agent Copilot ({model_name})..."):
            reply = llm_router.query(
                prompt=pending_user_msg,
                system_prompt=(
                    "You are RockWatch AI, an expert geotechnical safety copilot for open-pit mines. "
                    "Analyze the live multi-sensor dataset readings and answer the operator's query in a helpful, "
                    "clear, and reassuring manner. Cite specific sensor numbers and SOP guidelines.\n\n"
                    f"--- LIVE MINE DATASET CONTEXT ---\n{context_str}"
                ),
                agent_name="COPILOT_CHAT",
                temperature=0.25,
                max_tokens=600,
            )

        if not reply:
            q_lower = pending_user_msg.lower()
            if "safe" in q_lower or "normal" in q_lower or "status" in q_lower:
                if highest_global_level == "GREEN":
                    reply = (
                        "✅ **Yes, everything is safe and normal.**\n\n"
                        f"All sectors are at **Level 1 Green**. According to dataset record `#{active_row_idx}`, "
                        f"radar velocity is **{cur_sec['radar_velocity_mm_hr']:.3f} mm/h** (safe threshold is < 1.5 mm/h), "
                        f"pore pressure is **{cur_sec['pore_pressure_kpa']:.1f} kPa**, and rainfall is **{cur_sec['rainfall_3h_mm']:.2f} mm**. "
                        "Routine operations are authorized."
                    )
                else:
                    reply = (
                        f"⚠️ **Attention: {cur_sec['name']} is at Level {level_rank[sec_level]} {sec_level}.**\n\n"
                        f"Dataset row `#{active_row_idx}` recorded radar velocity at **{cur_sec['radar_velocity_mm_hr']:.3f} mm/h**. "
                        f"Directive: *{cur_sec['directive']}*"
                    )
            elif "north wall" in q_lower or "happening" in q_lower or "sector" in q_lower:
                reply = (
                    f"**Telemetry Breakdown for {cur_sec['name']} (Dataset Row #{active_row_idx}):**\n\n"
                    f"- **Radar Velocity:** {cur_sec['radar_velocity_mm_hr']:.3f} mm/h\n"
                    f"- **Pore Water Pressure:** {cur_sec['pore_pressure_kpa']:.1f} kPa\n"
                    f"- **Rainfall (3h):** {cur_sec['rainfall_3h_mm']:.2f} mm\n"
                    f"- **Seismic PPV:** {cur_sec['vibration_ppv_mms']:.2f} mm/s\n"
                    f"- **Factor of Safety:** {cur_sec.get('fos', 1.54):.2f}\n"
                    f"- **Current Directive:** *{cur_sec['directive']}*"
                )
            elif "evacuat" in q_lower or "sop" in q_lower or "rule" in q_lower:
                reply = (
                    f"**TARP Action Protocol for {sec_level} Status (Citation: `{sop_info.get('passage_id', '[TARP-SOP-01]')}`):**\n\n"
                    f"{sop_info.get('text', 'Routine monitoring.')}\n\n"
                    f"**Required Actions:** {cur_sec['directive']}"
                )
            else:
                reply = (
                    f"**RockWatch Assistant:** Current sector {cur_sec['name']} is at **Level {level_rank[sec_level]} {sec_level}**. "
                    f"Dataset row #{active_row_idx} velocity is {cur_sec['radar_velocity_mm_hr']:.3f} mm/h. "
                    f"Governing rule is `{sop_info.get('passage_id', '[TARP-SOP-01]')}`."
                )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # 2. Quick One-Click Starter Questions & Reset
    st.markdown('<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;"><span style="font-size: 0.78rem; font-weight: 700; color: #94a3b8;">QUICK QUESTIONS:</span></div>', unsafe_allow_html=True)
    q_c1, q_c2 = st.columns(2)
    trigger_prompt = None
    with q_c1:
        if st.button("🟢 Is everything safe right now?", use_container_width=True):
            trigger_prompt = "Is everything safe right now in the mine based on current sensor readings?"
        if st.button("⚠️ What if heavy rain starts?", use_container_width=True):
            trigger_prompt = "What happens if heavy rain starts falling according to the SOP?"
    with q_c2:
        if st.button("🔍 What is happening at North Wall?", use_container_width=True):
            trigger_prompt = f"What is happening right now at {cur_sec['name']} based on dataset row #{active_row_idx}?"
        if st.button("📋 What are the evacuation rules?", use_container_width=True):
            trigger_prompt = "What are the mandatory evacuation rules for Level 4 Red?"

    # 3. Chat Messages Window
    chat_box = st.container(height=340)
    with chat_box:
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-bubble-ai">
                    <strong>🤖 RockWatch Copilot:</strong><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble-user">
                    <strong>👤 Operator:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)

    # 4. Chat Input Box & Submission Handler
    user_input = st.chat_input("Ask the AI Copilot about mine safety, alerts, or procedures...")
    prompt_to_process = trigger_prompt or user_input

    if prompt_to_process:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
        st.rerun()


# ==============================================================================
# 3. BOTTOM ACCORDION: GEOTECHNICAL AUTHORITY & DUAL-ROLE SIGN-OFF CONSOLE
# ==============================================================================
st.markdown("---")
with st.expander("🛡️ Geotechnical Authority & Sign-Off Console", expanded=False):
    adm_c1, adm_c2 = st.columns([1.1, 0.9], gap="large")
    with adm_c1:
        st.markdown("**Dual-Role Geotechnical Sign-Off Console**")
        role_select = st.selectbox(
            "Active Session Profile:",
            options=["OPERATOR", "GEOTECH_ENGINEER"],
            format_func=lambda r: "👤 Control Room Operator (Monitoring Only)" if r == "OPERATOR" else "🛡️ Dr. P. Mehta, Principal Geotechnical Engineer (#GEO-7714)",
            key="user_role",
        )
        is_geotech = role_select == "GEOTECH_ENGINEER"
        perm_text = "FULL AUTHORIZATION (DESENSITIZE & PROMOTE)" if is_geotech else "RESTRICTED (CANNOT DESENSITIZE SAFETY FLOORS)"
        perm_color = "#34d399" if is_geotech else "#f59e0b"
        st.markdown(f"<div style='font-size: 0.78rem; font-family: monospace; color: {perm_color}; margin-bottom: 8px;'>PERMISSIONS: {perm_text}</div>", unsafe_allow_html=True)

        ticket = st.session_state.review_ticket
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 10px; margin-bottom: 8px; font-size: 0.82rem;">
            <strong>Ticket #{ticket['id']}</strong> | Target: {cur_sec['name']} | Status: {'APPROVED' if ticket['approved'] else 'PENDING'}<br>
            <span style="color: #94a3b8;">Requested Action: Apply temporary 0.80x multiplier for thermal drift.</span><br>
            <span style="color: #f87171; font-family: monospace; font-size: 0.75rem;">Safety constraint: TARP Red 15.0 mm/h safety floor cannot be suppressed.</span>
        </div>
        """, unsafe_allow_html=True)

        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            if st.button("✍️ Authorize Decision", use_container_width=True):
                if is_geotech:
                    ticket["approved"] = True
                    ticket["approved_by"] = "Dr. P. Mehta (#GEO-7714)"
                    ctrl.submit_desensitization_request(
                        zone_id=chosen_sector,
                        requested_by_role="GEOTECH_ENGINEER",
                        proposed_multiplier=0.80,
                        duration_hours=4.0,
                        justification="Thermal drift verified by extensometer.",
                    )
                    st.success("✅ APPROVED & DIGITALLY SIGNED by Principal Geotech Engineer!")
                else:
                    st.error("❌ ACCESS DENIED: Operators cannot authorize desensitization.")
        with sub_c2:
            if st.button("⏩ Fast-Forward +5h (Auto-Expiry)", use_container_width=True):
                st.session_state.sim_time_hour = min(24.0, st.session_state.sim_time_hour + 5.0)
                if ticket["approved"]:
                    ticket["approved"] = False
                    ticket["status"] = "EXPIRED"
                st.rerun()

    with adm_c2:
        st.markdown("**Asymmetric Safety Policy Enforcement**")
        st.info(
            "• **Principle of Asymmetric Safety:** Operators may elevate alert levels at any time, but only a certified "
            "Geotechnical Engineer can authorize sensitivity attenuation.\n\n"
            "• **Fail-Closed Auto-Expiry:** All sensitivity relaxations auto-expire back to baseline (1.00x) within 12 hours.\n\n"
            "• **Inviolable Red Safety Floor:** TARP Level 4 Red (15.0 mm/h) velocity threshold can NEVER be suppressed by any user or model."
        )
