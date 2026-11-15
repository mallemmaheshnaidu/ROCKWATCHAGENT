# Project Implementation Checklist (`todo.md`)

## Phase 1: Foundational Documentation & Knowledge Corpus
- [x] Create `agent.md` with the 4 development principles (Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution).
- [x] Create `requirements.md` capturing all mandatory rubric criteria.
- [x] Create `architecture.md` specifying the 8-agent blackboard-controller blueprint.
- [x] Create `research.md` preserving the RockWatch SIH25071 geotechnical baseline.
- [x] Create `requirements.txt` and `.gitignore`.
- [x] Create `docs/` knowledge base with explicit passage IDs:
  - [x] `docs/tarp_sop_v1.md`
  - [x] `docs/geotech_guidelines.md`
  - [x] `docs/historical_incidents.md`
- [x] Verify Phase 1 completeness.

## Phase 2: Core State & Plain Python Tools
- [x] Implement `blackboard.py`:
  - [x] Partitioned sections (`raw_telemetry`, `sensor_health`, `clean_telemetry`, `surface_changes`, `tarp_status`, `forecast`, `advisor_notes`, `arbitration`, `proposals`, `golden_pool`).
  - [x] Read/write versioning.
  - [x] Fail-closed sensitivity lookup with injectable clock and expiry logging.
  - [x] Canonical hash-chained append-only logging to `logs/run_log.jsonl`.
- [x] Implement `tools/` (plain Python functions, >= 2 per agent):
  - [x] `sensor_tools.py` (`ingest_sensor_frame`, `assess_sensor_health`)
  - [x] `perception_tools.py` (`detect_surface_change`, `find_unalarmed_changes`)
  - [x] `geotech_tools.py` (`evaluate_tarp`, `compute_inverse_velocity`)
  - [x] `forecast_tools.py` (`compute_zone_risk`, `create_provenance_record`)
  - [x] `rag_tools.py` (`retrieve_sop`, `find_similar_incidents` with passage ID citations)
  - [x] `arbitration_tools.py` (`arbitrate_decision`, `build_alert_package`)
  - [x] `governance_tools.py` (`process_desensitization_proposal`, `propose_root_cause`)
  - [x] `evaluation_tools.py` (`evaluate_model_gate`, `sample_quiet_periods`)
- [x] Verify Phase 2 with unit assertions.

## Phase 3: Semantic Kernel Agent Modules & Controller
- [x] Implement 8 agent modules under `agents/` wrapping tools as SK plugins:
  - [x] `agents/health_agent.py`
  - [x] `agents/perception_agent.py`
  - [x] `agents/tarp_agent.py`
  - [x] `agents/forecaster_agent.py`
  - [x] `agents/advisor_agent.py` (SK chat completion with offline template fallback)
  - [x] `agents/arbiter_agent.py` (SK chat completion with offline template fallback)
  - [x] `agents/governance_agent.py`
  - [x] `agents/evaluator_agent.py`
- [x] Implement `controller.py`:
  - [x] Deterministic event routing.
  - [x] Injectable clock control.
  - [x] Safe error recovery.
- [x] Verify Phase 3 end-to-end execution.

## Phase 4: Test Suite & Verification
- [x] Implement `tests/test_scenarios.py`:
  - [x] Scenario 1: Primary radar offline -> health flags degradation -> caution elevated -> forecaster widens uncertainty.
  - [x] Scenario 2: Un-alarmed surface detachment discovered -> instant caution without approval -> event added to pending golden pool -> diagnostic run.
  - [x] Scenario 3: False alarm reported -> proposal drafted -> requires geotech auth -> auto-expires to baseline -> candidate model failing recall rejected by gate.
  - [x] Scenario 4: Middle tampering of `run_log.jsonl` detected by hash validator.
  - [x] Scenario 5: Out-of-corpus RAG query returns "not in documentation" with no hallucination.
  - [x] Scenario 6: Free LLM mock outage/rate limit falls back gracefully to templates.
- [x] Run pytest and verify all tests pass.

## Phase 5: Demo Notebook, Diagram & Documentation
- [x] Build `demo.ipynb` executing all scenarios and printing multi-agent traces and head hash anchor.
- [x] Generate clean ASCII / Mermaid architecture diagram.
- [x] Write `README.md` (≤ 2 pages, conforming to rubric structure).

## Phase 6: Interactive Streamlit Dashboard (Built Last)
- [x] Implement `app.py`:
  - Visual control room layout with zone risk status.
  - Sensor health cards and offline toggles.
  - Operator chat interface.
  - Geotech role login/toggle for proposal approvals.
- [x] Verify Streamlit runs cleanly without duplicate business logic.

## Phase 7: Multi-Sensor Independent Model Training & Cross-Validation Pipeline
- [x] Fetch real-world open-pit datasets (`featured_weather_data.csv`, `enhanced_landslide_dataset.csv`, `slope_stability_geotech.csv`).
- [x] Implement `ml/train_independent_models.py`:
  - 5-Fold Stratified Cross-Validation on Environmental Infiltration & Geotechnical Slope Stability.
  - Compare CatBoost, LightGBM, and XGBoost with hyperparameter sweeps.
  - Threshold / cutoff tuning for high safety recall.
  - Save champion models: `weather_model.joblib` and `geotech_slope_model.joblib`.
- [x] Generate comprehensive benchmark report: `ml/reports/multi_sensor_benchmark_report.md`.
- [x] Integrate champion models into `ml/meta_fusion.py` with Option B Priority-Weighted Late Fusion and Single-Sensor Dominance Override.
- [x] Verify 6/6 pytest scenarios pass, `demo.ipynb` runs cleanly, and `app.py` operates without error.
