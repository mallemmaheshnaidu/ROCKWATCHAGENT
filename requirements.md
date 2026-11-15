# Project Requirements & Submission Checklist

## 1. Core Mandatory Deliverables

* **Multi-Agent Architecture:**
  * Must be a multi-agent system of **at least four agents** (our system implements **8 distinct agents**).
  * Each agent must be a **separate module** with **one clear role**.
  * A single agent with more tools does not qualify.
  * Coordinated by the architecture named in the project (**Blackboard Pattern with Deterministic Controller**).

* **Tools Specification:**
  * Each working agent calls **at least two real tools written as plain Python functions**.
  * Tools must perform real calculations, evaluations, or lookups (no hardcoded stubs or trivial return strings).

* **Grounding and State:**
  * **RAG (Retrieval-Augmented Generation):** Grounding facts must come from a provided document set (`docs/`). Citations must reference specific passage IDs. Must fail closed ("not in the documents") when information is absent.
  * **Shared State / Blackboard:** Every working agent must explicitly **read and write the same shared object** (`blackboard.py`). No pure sources or pure sinks.

* **Framework:**
  * Built on **Microsoft Semantic Kernel (Python)**.
  * Agents registered as Semantic Kernel plugins with `@kernel_function` bindings.
  * Deterministic controller coordinates workflow and supports offline fallback if external LLM services are unavailable.

## 2. Submission Deliverables Checklist

- [ ] **Source Code:** One module per agent under `agents/`, tools under `tools/`, shared state in `blackboard.py`, orchestrator in `controller.py`.
- [ ] **README.md (≤ 2 Pages):**
  - Agent roster & responsibilities.
  - Architecture description and why it fits the domain.
  - Plain Python tools per agent.
  - Memory and retrieval design (Blackboard partitions, RAG corpus).
  - One honest failure handled (LLM outage/rate limit fallback to deterministic templated explanation with zero impact on safety paths).
- [ ] **Runnable Demo Notebook (`demo.ipynb`):**
  - Executes end-to-end multi-agent workflow.
  - Prints clean multi-agent trace.
  - Displays the canonical log head hash anchor.
- [ ] **Architecture Diagram:** Visual flowchart illustrating data flows between sensors, controller, blackboard, agents, and logs.
- [ ] **Append-Only Run Log (`logs/run_log.jsonl`):**
  - Canonical JSON (`sort_keys=True`) hash-chained entries (`prev_hash` validation).
  - Explicitly states audit guarantees and limitations.
- [ ] **Comprehensive Test Suite (`tests/test_scenarios.py`):**
  - Must cover at least three mandatory operational scenarios plus edge cases:
    1. **Scenario 1 (Degraded Sensors):** Primary radar offline; system continues on secondary sensors with elevated caution and widened uncertainty.
    2. **Scenario 2 (Missed Event Handling):** Post-hoc scan differencing discovers un-alarmed detachment; triggers instant caution without approval; logs to pending golden pool.
    3. **Scenario 3 (False Alarm Proposal & Model Gate Rejection):** Operator reports false alarm; drafts time-limited desensitization proposal requiring geotech role; model gate rejects a candidate model that drops recall on the golden set.
    4. **Scenario 4 (Log Tamper Detection):** Modifying a middle log entry is detected by hash validator.
    5. **Scenario 5 (RAG Grounding & Missing Citation):** Out-of-corpus query returns "not in documentation" rather than hallucinating.
    6. **Scenario 6 (LLM Outage Resilience):** Safety path executes deterministically when LLM fails.
