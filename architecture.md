# System Architecture: RockWatch Multi-Agent Early Warning Platform

## 1. Architectural Overview

RockWatch uses a **Blackboard with Deterministic Controller** pattern built on **Microsoft Semantic Kernel**.

```
  Sensors (Simulated / Real) ──┐
  Operator / Geotech Chat ────┼──▶  CONTROLLER (Routes events, injects clock, handles fallback)
  Scan Differencing ──────────┘               │
                                              ▼
   ┌───────────────────────────── SHARED BLACKBOARD ─────────────────────────────┐
   │ raw_telemetry    │ sensor_health │ surface_changes │ tarp_status            │
   │ forecast         │ advisor_notes │ arbitration     │ proposals_and_auth     │
   │ golden_pool      │ gate_results  │ audit_log_ref   │ (all writes versioned) │
   └─────────────────────────────────────────────────────────────────────────────┘
      ▲        ▲         ▲        ▲        ▲         ▲        ▲         ▲
      │        │         │        │        │         │        │         │
   1 Health  2 Percep. 3 TARP   4 Fore-  5 Advisor 6 Arbiter 7 Gover-  8 Model
   Ingest    -tion     Engine   caster   (RAG+LLM) Responder  nance     Evaluator
```

### Core Axiom: Asymmetric Safety Authority
* **Increasing Caution (Misses, spikes, sensor failures):** Executed instantly with zero approvals needed.
* **Decreasing Caution (False alarms, desensitization):** Proposal only; requires authenticated Geotechnical Engineer sign-off; time-limited with automatic code-level expiry; cannot breach the hard baseline TARP floor.
* **Execution Boundary:** Code computes and decides safety paths; LLM explains; humans authorize risk reductions.

---

## 2. Agent Roster & Blackboard Contracts

Every agent is a distinct class/module under `agents/` registered as a Semantic Kernel Plugin. Each agent reads from and writes to the shared `MineBlackboard`.

| # | Agent Module | Primary Role | Reads from Blackboard | Writes to Blackboard | Tools (Plain Python Functions) |
|---|---|---|---|---|---|
| **1** | `health_agent.py` | Telemetry ingestion, data cleaning, sensor health auditing | `raw_telemetry` | `sensor_health`, `clean_telemetry` | `ingest_sensor_frame`, `assess_sensor_health` |
| **2** | `perception_agent.py` | Surface change detection & discovery of un-alarmed detachments | `clean_telemetry`, `alert_history` | `surface_changes`, `unalarmed_events` | `detect_surface_change`, `find_unalarmed_changes` |
| **3** | `tarp_agent.py` | Deterministic Trigger Action Response Plan execution | `clean_telemetry`, `sensor_health` | `tarp_status` | `evaluate_tarp`, `compute_inverse_velocity` |
| **4** | `forecaster_agent.py` | Calibrated horizon risk estimation with explicit uncertainty | `clean_telemetry`, `sensor_health`, `active_proposals` | `forecast` | `compute_zone_risk`, `create_provenance_record` |
| **5** | `advisor_agent.py` | Geotechnical SOP explanations & precedent lookup via RAG | `tarp_status`, `forecast`, `unalarmed_events` | `advisor_notes` | `retrieve_sop`, `find_similar_incidents` |
| **6** | `arbiter_agent.py` | Resolves model-TARP discrepancies, enforces caution rule, builds alert packages | `tarp_status`, `forecast`, `sensor_health`, `advisor_notes` | `arbitration_decision`, `alert_package` | `arbitrate_decision`, `build_alert_package` |
| **7** | `governance_agent.py` | Desensitization proposals, geotech authentication, auto-expiry, root-cause proposals | `submitted_proposals`, `unalarmed_events` | `active_proposals`, `root_cause_proposals` | `process_desensitization_proposal`, `propose_root_cause` |
| **8** | `evaluator_agent.py` | Golden test set management, non-inferiority model gating, quiet period auditing | `pending_events`, `alert_history` | `golden_set_vX`, `model_gate_results`, `audited_quiet_periods` | `evaluate_model_gate`, `sample_quiet_periods` |

---

## 3. Detailed Component Mechanisms

### A. Fail-Closed Desensitization & Expiry
1. Proposals only affect the **forecaster's alert sensitivity**, never the TARP floor.
2. In `blackboard.py`, reading sensitivity checks `current_time >= proposal.expires_at`.
3. If expired or upon any exception, it reverts immediately to `FORECASTER_BASELINE_SENSITIVITY`.
4. Injectable clock parameter allows deterministic time-travel in tests.
5. An audit log entry is emitted upon reversion.

### B. Grounded RAG with Passage Citations
* Local BM25/TF-IDF keyword retriever operating over `docs/*.md`.
* Every returned passage contains a unique ID (e.g., `[TARP-SOP-04]`).
* The Advisor cites passage IDs in answers.
* Queries ungrounded in the corpus return `"Information not found in geotechnical documentation."` (fail-closed).

### C. Append-Only Hash-Chained Run Log
* File: `logs/run_log.jsonl`.
* Stored using canonical JSON (`sort_keys=True`).
* Entry format: `{"entry_id": int, "prev_hash": str, "timestamp": str, "agent": str, "payload": dict, "hash": str}`.
* Detects middle edits, deletions, or insertions.
* Explicitly acknowledged limitation: does not detect complete-tail truncation without anchoring to an external head hash. Head hash is printed at test/demo completion.

### D. Honest Failure Mode
* **Trigger:** LLM API rate limits, network outage, or malformed generation.
* **Resilience:** Agents 5 and 6 fall back to deterministic string templates. The safety path (TARP evaluation, risk arbitration, alert package generation) executes with zero interruption because core decisions are made in deterministic Python code.
