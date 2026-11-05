# ⚡ Engines Health Monitor — Industrial Predictive Maintenance System

**Neural Storm 2026 Hackathon Submission**  
**Track 01:** Future Tech (Intelligent Systems & Emerging Technology)  
**Problem Statement:** **FT-03: Predictive Maintenance for Industry (IoT time-series, Anomaly Detection)**  
**Team Name:** void main (`aniketmehetre`)  
**Hackathon Build Window:** November 2, 2026, 8:00 PM IST – November 4, 2026, 8:00 PM IST  

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Vite + Three.js](https://img.shields.io/badge/Vite-Three.js_3D-646CFF.svg)](https://threejs.org/)
[![Anime.js v4](https://img.shields.io/badge/Anime.js-v4.5-FF4B4B.svg)](https://animejs.com/)
[![Track: Future Tech](https://img.shields.io/badge/Neural_Storm_2026-Track_01:_FT--03-orange.svg)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-RUL_Regression-337AB7.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_+_SSE-009688.svg)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-FastMCP_Server-purple.svg)](https://modelcontextprotocol.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_v2-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/pytest-48_passed-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

### 🎥 Live 3D Digital Twin & Mission Control Experience
![Interactive Satellite Digital Twin Animated Walkthrough](docs/img/frontend_demo.gif)

*Figure: Real-time 3D Satellite Digital Twin ("Orbital Insertion & Exploded Assembly") — scrolling flies through orbital trajectories, separates spacecraft subsystems along flight axes, streams live IoT telemetry via FastAPI SSE, activates calibrated split-conformal bounds, and achieves \$29.54M verified fleet savings.*

| Satellite Bus & Telemetry HUD | Exploded Subsystem Assembly | Cost-Optimal Policy & Backtest |
| :---: | :---: | :---: |
| ![Satellite Twin](docs/img/satellite_space_01.png) | ![Exploded View](docs/img/satellite_space_02_exploded.png) | ![Decide with Dollars](docs/img/satellite_space_03_decide.png) |

> 💡 **Judge Demonstration Tool (`demo.py`):** Run `python demo.py` to trigger an automated walkthrough with live speaking scripts for video recording, or `python demo.py --verify` to test all live FastAPI backend endpoints!

---

## 📖 Table of Contents

1. [The 3-Minute Judge Pitch](#1-the-3-minute-judge-pitch)
2. [Problem Statement & Approach](#2-problem-statement--approach)
3. [3D Digital Twin Frontend Experience ("The Descent")](#3-3d-digital-twin-frontend-experience-the-descent)
4. [Target System Architecture](#4-target-system-architecture)
5. [Empirical Results & Verified Metrics](#5-empirical-results--verified-metrics)
   - [5.1 Multi-Metric Benchmark (FD001 Official Test Set)](#51-multi-metric-benchmark-fd001-official-test-set)
   - [5.2 Calibrated Split-Conformal Uncertainty](#52-calibrated-split-conformal-uncertainty)
   - [5.3 Anomaly & Degradation Onset Detection](#53-anomaly--degradation-onset-detection)
   - [5.4 3-Policy Fleet Money Backtest](#54-3-policy-fleet-money-backtest)
6. [Actionable Business Logic & Hangar Optimization](#6-actionable-business-logic--hangar-optimization)
7. [Agentic AI Layer v2 (LangGraph with Critic & Human Approval)](#7-agentic-ai-layer-v2-langgraph-with-critic--human-approval)
8. [Model Context Protocol (MCP) Server](#8-model-context-protocol-mcp-server)
9. [Cryptographic Audit Ledger](#9-cryptographic-audit-ledger)
10. [Tech Stack](#10-tech-stack)
11. [Setup & Quick Start (Zero API Key Offline Execution)](#11-setup--quick-start-zero-api-key-offline-execution)
12. [API & Streaming Reference](#12-api--streaming-reference)
13. [Built During the Hackathon Window](#13-built-during-the-hackathon-window)
14. [Honest Limitations](#14-honest-limitations)
15. [Hackathon Compliance & AI Tools Disclosure](#15-hackathon-compliance--ai-tools-disclosure)
16. [Credits & Acknowledgements](#16-credits--acknowledgements)

---

## 1. The 3-Minute Judge Pitch

1. **The Problem:** Reactive maintenance risks \$350k catastrophic in-service failures and AOG (Aircraft On Ground) groundings. Calendar-based fixed replacement wastes 32%+ of residual engine life.
2. **Detect Early:** Our unsupervised health-index detector (PCA + IsolationForest with CUSUM change-point tracking) flags degradation onset with a **median lead time of 164.5 cycles (~65 operating days) to failure**, alerting operators before RUL crosses 80 cycles in **98.9%** of cases.
3. **Calibrated Forecasts:** Rather than fragile point predictions, our system delivers finite-sample **80% split-conformal intervals** ($p_{10}$ to $p_{90}$, empirically validated at **84.0% coverage** on held-out engines) and derives mission-horizon failure probabilities $P(\text{failure within 30 cycles})$.
4. **Decide with Dollars:** A cost-optimal replacement policy and an **OR-Tools CP-SAT depot scheduler** optimize hangar slot allocations under capacity constraints. In our 100-engine fleet backtest, the predictive policy achieved **zero in-service failures and saved \$29,542,700 (84.4%)** over reactive maintenance.
5. **Act Safely:** A 7-node LangGraph agent with a **deterministic numeric critic** checks all claims against state, requires **human approval** before releasing simulated purchase orders, and records every action in a **SHA-256 hash-chained tamper-evident audit ledger**.
6. **Next-Gen 3D Digital Twin:** Built with **Three.js, Anime.js v4, and Lenis**, a cardless 3D digital twin renders procedural cutaway turbofan assemblies with real-time screen-projected callouts, procedural turbofan sound synthesis, and an atmospheric sky shader that visually reflects engine health.
7. **Plug in Anywhere:** The entire suite operates locally offline with zero API keys and exposes tools over both **FastAPI (REST + SSE)** and the **Model Context Protocol (MCP)**.

---

## 2. Problem Statement & Approach

### Problem Statement: FT-03 Predictive Maintenance for Industry
Turbofan engines operate under extreme thermal and mechanical stresses. Industrial operators need solutions that can process high-frequency multi-sensor IoT telemetry, isolate anomalous degradation trajectories early, provide mathematically trustworthy time-to-failure forecasts, and automatically schedule depot overhauls before in-service failure.

### Our Approach
Built on the **NASA CMAPSS FD001 dataset** (100 training engines, 100 test engines), our system bridges statistical time-series forecasting directly to enterprise maintenance logistics:

```mermaid
flowchart TD
    A["NASA CMAPSS IoT Telemetry (15 Sensors)"] --> B["Data Quality Guard (Trust Score)"]
    B --> C["Causal Feature Engine (90 Rolling Stats, Window=30)"]
    C --> D1["Anomaly Engine (PCA + IsolationForest)"]
    C --> D2["RUL Engine (Optuna-Tuned XGBoost)"]
    D1 --> E1["Health Index & CUSUM Onset Detector"]
    D2 --> E2["Split-Conformal Calibration (p10, p50, p90, P(fail<=30))"]
    E1 & E2 --> F["Cost-Optimal Policy & OR-Tools Hangar Scheduler"]
    F --> G["LangGraph Multi-Agent v2 (Triage -> Diagnostic -> Planner -> Critic -> Approval -> Dispatch)"]
    G --> H1["3D Digital Twin Frontend (Three.js + Anime.js v4)"]
    G --> H2["FastAPI REST & SSE Replay Streamer"]
    G --> H3["FastMCP Server (fleet://summary)"]
    G --> H4["Cryptographic Audit Ledger (SHA-256 Chain)"]
```

---

## 3. 3D Digital Twin Frontend Experience ("The Descent")

The frontend eliminates generic boxed card layouts in favor of an **Atmos-style continuous 3D coordinate space** and an **Anime.js v4 geometric Figure Layer**:

| Chapter | Journey Milestone | Mechanical 3D State | Atmospheric Sky Shader | Visual Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **01. Boarding** (`p=0.00-0.10`) | Stratospheric cruise | Whole turbofan on wing; nominal idle rotation | Clear day cyan ($uHealth=1.0$) | ![Boarding](docs/img/frontend_01_boarding.png) |
| **02. Anatomy** (`p=0.25-0.50`) | Mechanical deconstruction | Engine explodes along shaft into 6 cutaways (FAN, LPC, HPC, COMB, HPT, LPT) | Day to early dusk | ![Anatomy](docs/img/frontend_02_anatomy.png) |
| **03. Detect Early** (`p=0.50-0.65`) | Degradation onset | HPC thermal tinting; CUSUM onset marker at 164.5 cycles lead time | Amber twilight ($uHealth=0.6$) | ![Detect Early](docs/img/frontend_03_detect_early.png) |
| **04. Decide with Dollars** (`p=0.78-0.90`) | Financial optimization | High thermal stress; \$29.54M savings & \$35M vs \$5.5M comparative bars | Emergency red night ($uHealth=0.2$) | ![Decide](docs/img/frontend_04_decide_dollars.png) |
| **05. Dawn** (`p=0.90-1.00`) | Operational restoration | Replaced engine fitted; 7-node multi-agent pipeline & SHA-256 ledger verified | Cerulean dawn ($uHealth=1.0$) | ![Dawn](docs/img/frontend_05_dawn.png) |

### Key Frontend Capabilities
1. **Cardless 3D-Anchored Callouts ([`src/ui/callouts.js`](src/ui/callouts.js)):** Pinned to assembly 3D world anchors with screen-space camera matrix projection (`translate3d`), rendering live mini-telemetry waveforms without DOM jitter.
2. **Station Scrubber ([`src/ui/scrubber.js`](src/ui/scrubber.js)):** Slim interactive HUD along the bottom with 1-click seeking to any turbofan station (`FAN`, `LPC`, `HPC`, `COMB`, `HPT`, `LPT`).
3. **Kinetic Text Splitting ([`src/ui/headline.js`](src/ui/headline.js)):** Typography split into individual characters and staggered with anime.js v4 easing on chapter transitions.
4. **Web Audio API Engine Hum Synthesizer ([`src/main.js`](src/main.js)):** Procedural audio synthesis generating low-frequency spool hum and harmonic bypass whine dynamically modulated by scroll velocity (zero external audio files).
5. **Boarding Pass Loader:** Aviation-style procedural loading overlay with hardware capability detection and reduced-motion support.

---

## 4. Target System Architecture
4. [Empirical Results & Verified Metrics](#4-empirical-results--verified-metrics)
   - [4.1 Multi-Metric Benchmark (FD001 Official Test Set)](#41-multi-metric-benchmark-fd001-official-test-set)
   - [4.2 Calibrated Split-Conformal Uncertainty](#42-calibrated-split-conformal-uncertainty)
   - [4.3 Anomaly & Degradation Onset Detection](#43-anomaly--degradation-onset-detection)
   - [4.4 3-Policy Fleet Money Backtest](#44-3-policy-fleet-money-backtest)
5. [Actionable Business Logic & Hangar Optimization](#5-actionable-business-logic--hangar-optimization)
6. [Agentic AI Layer v2 (LangGraph with Critic & Human Approval)](#6-agentic-ai-layer-v2-langgraph-with-critic--human-approval)
7. [Model Context Protocol (MCP) Server](#7-model-context-protocol-mcp-server)
8. [Cryptographic Audit Ledger](#8-cryptographic-audit-ledger)
9. [Tech Stack](#9-tech-stack)
10. [Setup & Quick Start (Zero API Key Offline Execution)](#10-setup--quick-start-zero-api-key-offline-execution)
11. [API & Streaming Reference](#11-api--streaming-reference)
12. [Built During the Hackathon Window](#12-built-during-the-hackathon-window)
13. [Honest Limitations](#13-honest-limitations)
14. [Hackathon Compliance & AI Tools Disclosure](#14-hackathon-compliance--ai-tools-disclosure)
15. [Credits & Acknowledgements](#15-credits--acknowledgements)

---

## 1. The 3-Minute Judge Pitch

1. **The Problem:** Reactive maintenance risks \$350k catastrophic in-service failures and AOG (Aircraft On Ground) groundings. Calendar-based fixed replacement wastes 32%+ of residual engine life.
2. **Detect Early:** Our unsupervised health-index detector (PCA + IsolationForest with CUSUM change-point tracking) flags degradation onset with a **median lead time of 164.5 cycles (~65 operating days) to failure**, alerting operators before RUL crosses 80 cycles in **98.9%** of cases.
3. **Calibrated Forecasts:** Rather than fragile point predictions, our system delivers finite-sample **80% split-conformal intervals** ($p_{10}$ to $p_{90}$, empirically validated at **84.0% coverage** on held-out engines) and derives mission-horizon failure probabilities $P(\text{failure within 30 cycles})$.
4. **Decide with Dollars:** A cost-optimal replacement policy and an **OR-Tools CP-SAT depot scheduler** optimize hangar slot allocations under capacity constraints. In our 100-engine fleet backtest, the predictive policy achieved **zero in-service failures and saved \$29,542,700 (84.4%)** over reactive maintenance.
5. **Act Safely:** A 7-node LangGraph agent with a **deterministic numeric critic** checks all claims against state, requires **human approval** before releasing simulated purchase orders, and records every action in a **SHA-256 hash-chained tamper-evident audit ledger**.
6. **Plug in Anywhere:** The entire suite operates locally offline with zero API keys and exposes tools over both **FastAPI (REST + SSE)** and the **Model Context Protocol (MCP)**.

---

## 2. Problem Statement & Approach

### Problem Statement: FT-03 Predictive Maintenance for Industry
Turbofan engines operate under extreme thermal and mechanical stresses. Industrial operators need solutions that can process high-frequency multi-sensor IoT telemetry, isolate anomalous degradation trajectories early, provide mathematically trustworthy time-to-failure forecasts, and automatically schedule depot overhauls before in-service failure.

### Our Approach
Built on the **NASA CMAPSS FD001 dataset** (100 training engines, 100 test engines), our system bridges statistical time-series forecasting directly to enterprise maintenance logistics:

```mermaid
flowchart TD
    A["NASA CMAPSS IoT Telemetry (15 Sensors)"] --> B["Data Quality Guard (Trust Score)"]
    B --> C["Causal Feature Engine (90 Rolling Stats, Window=30)"]
    C --> D1["Anomaly Engine (PCA + IsolationForest)"]
    C --> D2["RUL Engine (Optuna-Tuned XGBoost)"]
    D1 --> E1["Health Index & CUSUM Onset Detector"]
    D2 --> E2["Split-Conformal Calibration (p10, p50, p90, P(fail<=30))"]
    E1 & E2 --> F["Cost-Optimal Policy & OR-Tools Hangar Scheduler"]
    F --> G["LangGraph Multi-Agent v2 (Triage -> Diagnostic -> Planner -> Critic -> Approval -> Dispatch)"]
    G --> H1["FastAPI REST & SSE Replay Streamer"]
    G --> H2["FastMCP Server (fleet://summary)"]
    G --> H3["Cryptographic Audit Ledger (SHA-256 Chain)"]
```

---

## 3. Target System Architecture

```text
CMAPSS FD001  +  Async Replay Streamer (Fault Injection: dropout / stuck / spike)
        │
   Ingest + Data-Quality Guard (trust_score)
        │
   src/features.py (Causal, Zero Train/Serve Skew)
        │
   ┌───────────────────────┬─────────────────────────────┐
   │ Anomaly Engine        │ RUL Engine                  │
   │ PCA + IsolationForest │ XGBoost Point Regressor     │
   │ EWMA + CUSUM Onset    │ + Split-Conformal Intervals │
   └───────────┬───────────┴──────────────┬──────────────┘
               │                          │
           Decision Engine: Cost-Optimal Policy, Replace-by Deadline,
           OR-Tools CP-SAT Hangar Bay Scheduler, 3-Policy Money Backtest
               │
   Agent Layer v2 (LangGraph): Triage ➔ Diagnostics ➔ Planner(Tools) ➔
   Critic(Numeric Check) ➔ Human Approval ➔ Simulated Dispatch ➔ Report
   LLM Backends: Offline (Default, Zero Keys) / Gemini / OpenAI
               │
   ┌────────────────────────┬────────────────────────────┬────────────────────────┐
   │ FastAPI (REST + SSE)   │ FastMCP Server             │ SHA-256 Audit Ledger   │
   │ /predict /anomaly /mro │ tools & fleet://summary    │ verify_chain() CLI     │
   └────────────────────────┴────────────────────────────┴────────────────────────┘
```

---

## 4. Empirical Results & Verified Metrics

Every metric below is generated directly from code and reproducible via `reports/*.json`:

### 4.1 Multi-Metric Benchmark (FD001 Official Test Set)
Evaluated on the official 100 test engines at their final observed cycles against ground-truth remaining life (`reports/benchmark.json`):

| Model | Official Test Set | RMSE (cycles) | MAE (cycles) | $R^2$ Score | NASA Asymmetric PHM Score |
|---|---|---|---|---|---|
| Baseline Linear Regression | 100 Engines (Last Cycle) | 23.15 | 18.40 | 0.6800 | 1,420.50 |
| **Our Optuna-Tuned XGBoost** | **100 Engines (Last Cycle)** | **16.74** | **12.33** | **0.8255** | **655.02** |

*NASA Asymmetric Scoring heavily penalizes dangerous late maintenance ($\hat{y} > y$) by a factor of $\exp(d/10)$ vs early maintenance ($\exp(-d/13)$).*

### 4.2 Calibrated Split-Conformal Uncertainty
Evaluated on held-out engines (`reports/calibration.json`):
- **Target Nominal Coverage:** 80.0% ($p_{10}$ to $p_{90}$ prediction band)
- **Measured Empirical Coverage on Held-Out Engines:** **84.0%** (Well within the 8-point tolerance gate)
- **Conformal Margin ($q_{80}$):** 22.48 cycles
- **Mean Interval Width:** 38.6 cycles
- **Dynamic Trust Widening:** When data-quality guard trust score drops below 0.80 (e.g. sensor dropout or stuck signal), conformal intervals automatically widen to prevent overconfident decisions.

### 4.3 Anomaly & Degradation Onset Detection
Evaluated across all test trajectories (`reports/anomaly.json`):
- **Onset Detection Rate:** **92.0%** of engines successfully flagged
- **Median Lead Time to Component Failure:** **164.5 cycles** (~65 days at 2.5 flights/day)
- **Median Lead Time to Test Cutoff:** **99.5 cycles**
- **Fraction Flagged Before RUL $\le$ 80 Cycles:** **98.9%**
- **False Alarm Rate (Healthy Phase, cycles $\le$ 25):** 2.0%

### 4.4 3-Policy Fleet Money Backtest
Simulated across all 100 engines using economic parameters (`config/costs.yaml`):
- Unplanned Failure Cost: \$350,000 | Planned Depot Overhaul: \$45,000 | Wasted Cycle Cost: \$150 | Utilization: 2.5 cycles/day (Assumed)

| Policy | Failures | Wasted Life % | Total Fleet Cost ($) | Net Savings vs Reactive |
|---|---|---|---|---|
| **1. Reactive (Run-to-Failure)** | 100 | 0.0% | \$35,000,000.00 | Baseline |
| **2. Fixed-Interval (140 Cycles)** | 0 | 32.2% | \$5,497,200.00 | \$29,502,800.00 (84.3%) |
| **3. Predictive AI (Our Policy)** | **0** | **30.9%** | **\$5,457,300.00** | **\$29,542,700.00 (84.4%)** |

*Our policy achieves zero catastrophic in-service failures while preserving 4,648 more operating cycles than fixed-interval maintenance.*

---

## 5. Actionable Business Logic & Hangar Optimization

### Automated Maintenance Strategy & Procurement
Classical predictive maintenance produces raw numbers; our system bridges prediction to real-world operations via `POST /maintenance-strategy` and `src/decision/policy.py`:
- **Subsystem Isolation:** Maps sensor degradation to physical modules (High-Pressure Compressor, Low-Pressure Turbine, Fan Assembly, Fuel Metering Unit, Bleed Valves, etc.).
- **Simulated Purchase Orders:** Emits simulated aerospace procurement orders with part descriptions, unit costs, lead times, and logistics hub distribution (e.g. ORD Central AeroDepot).
- **Work Order Specifications:** Lists mandatory technician licensing (FAA A&P Powerplant, EASA Part-66 B1, NDT Level II) and estimated hangar downtime.

### OR-Tools CP-SAT Depot Hangar Scheduling (`src/decision/scheduler.py`)
- Formulates a constraint satisfaction model with `AddCumulative` capacity constraints.
- Assigns at-risk aircraft engines onto limited hangar bays (default: 3 concurrent bays).
- Minimizes weighted tardiness against conservative $p_{10}$ RUL deadlines.

---

## 6. Agentic AI Layer v2 (LangGraph with Critic & Human Approval)

The upgraded multi-agent pipeline (`src/agents/fleet_agent.py`) replaces static prompt chaining with a robust state machine:

1. **Triage Node:** Ingests fleet summary, computes health distribution, filters at-risk engines.
2. **Parallel Diagnostic Node:** Computes SHAP-weighted subsystem sensitivity rankings across critical units.
3. **Planner Node:** Calls the Decision Policy and OR-Tools Hangar Scheduler as tools.
4. **Critic Node:** Validates that every count, RMSE, and dollar figure cited in drafts strictly matches state. Automatically triggers retries if discrepancy is detected.
5. **Approval Node (Human-in-the-Loop):** Halts procurement order dispatch until approved.
6. **Dispatch Node:** Issues simulated purchase orders (`SIMULATED-PO-2026-XXXX`).
7. **Report Writer Node:** Compiles the executive briefing with sign-off metadata.

*Supported LLM Providers (`src/llm/provider.py`):*
- **OfflineProvider (Default):** Runs 100% offline with zero API keys using domain heuristics.
- **GeminiProvider (`gemini-1.5-flash`):** Activated when `GEMINI_API_KEY` is present.
- **OpenAIProvider (`gpt-4o-mini`):** Activated when `OPENAI_API_KEY` is present.

---

## 7. Model Context Protocol (MCP) Server

The system implements the official **Model Context Protocol (MCP)** using `FastMCP` (`src/mcp_server.py`), enabling AI agents like Antigravity, Claude Desktop, or Cursor to query fleet status and orchestrate maintenance.

### Exposed Tools
- `list_fleet(status_filter)`
- `get_engine_status(engine_id)`
- `explain_engine(engine_id, top_n)`
- `recommend_maintenance(engine_id)`
- `schedule_hangar(engine_ids)`
- `run_backtest_summary()`
- Resource: `fleet://summary`

See [`docs/MCP.md`](docs/MCP.md) for Claude Desktop and Antigravity stdio configuration snippets.

---

## 8. Cryptographic Audit Ledger

To satisfy aerospace auditability requirements, `src/audit.py` maintains an append-only JSONL decision ledger (`reports/decision_ledger.jsonl`) with **SHA-256 hash chaining**:
- Every recommendation, approval decision, and simulated purchase order is cryptographically linked to the preceding entry's hash.
- Run verification CLI:
  ```bash
  python -m src.audit --verify
  ```
- Any unauthorized manual tampering with a logged cost or decision invalidates the hash chain and triggers an alert.

---

## 9. Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Frontend 3D** | `Three.js`, `anime.js` (v4), `Lenis`, `Vite` | Persistent 3D digital twin, Atmos scroll, kinetic Figure layer |
| **Language** | Python 3.11 / 3.12, Node.js 20+ | Backend ML & modern frontend build |
| **ML & Uncertainty** | `XGBoost`, `scikit-learn`, `scipy` | RUL regression, split-conformal calibration |
| **Anomaly Detection** | `PCA`, `IsolationForest`, CUSUM | Healthy baseline modeling and onset tracking |
| **Optimization** | `ortools` (CP-SAT) | Constrained hangar bay scheduling |
| **Agentic AI** | `LangGraph`, `pydantic` v2 | Stateful multi-agent graph with critic |
| **Protocol & API** | `FastAPI`, `sse-starlette`, `mcp` (FastMCP) | REST endpoints, SSE telemetry streaming, MCP server |
| **Testing & CI** | `pytest`, `httpx`, GitHub Actions | 48 automated tests, regression smoke tests |
| **Containerization** | `Docker`, `docker-compose` | Reproducible deployment images |

---

## 10. Setup & Quick Start (Zero API Key Offline Execution)

### Prerequisites
- Python 3.11 or 3.12
- Node.js 20+ (for 3D digital twin frontend)
- Git

### 1. Launch the 3D Digital Twin Frontend ("The Descent")
```bash
# Install frontend dependencies
npm install

# Start local dev server
npm run dev
```
Open **`http://localhost:5173`** in your browser to experience the 3D turbofan digital twin, atmospheric health transitions, and interactive audio synthesizer.

### 2. Python Environment & Dependencies Installation
```bash
git clone https://github.com/aniketmehetre/Neural-Storm-2026-FT03-Predictive-Maintenance.git
cd Neural-Storm-2026-FT03-Predictive-Maintenance

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # Linux / macOS

# Install package and dependencies
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### 3. Interactive Judge Demo & Walkthrough (`demo.py`)
```bash
# Launch interactive guided walkthrough with live narrator scripts
python demo.py

# Verify all live FastAPI backend endpoints (:8000) and model metrics
python demo.py --verify

# Generate / update animated walkthrough GIF (docs/img/frontend_demo.gif)
python demo.py --gif
```

### 3. Run the Full Test Suite (48 Tests)
```bash
pytest -v
```

### 4. Run Benchmark and Backtest
```bash
python -m src.benchmark --quick
python -m src.decision.backtest
python -m src.audit --verify
```

### 5. Start the FastAPI Service (REST + SSE)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API docs: `http://localhost:8000/docs`.

### 6. Start the Mission Control Dashboard (Streamlit)
```bash
streamlit run src/dashboard/app.py
```

### 7. Start the FastMCP Server
```bash
python -m src.mcp_server
```

---

## 11. API & Streaming Reference

### Endpoints
- `GET /health`: Liveness and readiness probe.
- `POST /predict`: Predicts RUL with conformal bounds (`rul_p10`, `rul_p50`, `rul_p90`) and failure probabilities (`prob_fail_10`, `prob_fail_30`).
- `POST /anomaly`: Evaluates sensor readings, returns `health_index`, `anomaly_score`, and `onset_detected`.
- `POST /ingest`: IoT gateway endpoint for single-cycle streaming ingestion.
- `GET /fleet`: Returns latest state of all active engines.
- `GET /stream/{unit_id}`: **Server-Sent Events (SSE)** endpoint streaming live cycle replay with optional fault injection (`?delay=0.1&fault=stuck&fault_cycle=15`).
- `POST /maintenance-strategy`: Generates plain-English strategy, action window, and simulated parts procurement order.

---

## 12. Built During the Hackathon Window

The following systems and modules were built and validated during the **Neural Storm 2026 48-hour build window**:
1. **Unsupervised Anomaly & Onset Engine ([`src/anomaly.py`](src/anomaly.py)):** PCA + IsolationForest fusion into a smoothed Health Index with CUSUM change-point tracking (92% onset detection rate).
2. **Split-Conformal Uncertainty Calibration ([`src/uncertainty.py`](src/uncertainty.py)):** Engine-level calibration yielding verified 84.0% empirical coverage and continuous $P(\text{failure} \le 30)$ failure probability.
3. **Financial Decision Policy & Backtest ([`src/decision/policy.py`](src/decision/policy.py), [`src/decision/backtest.py`](src/decision/backtest.py)):** Risk-threshold policy and 3-policy backtest proving \$29.5M savings over reactive maintenance.
4. **OR-Tools CP-SAT Hangar Bay Scheduler ([`src/decision/scheduler.py`](src/decision/scheduler.py)):** Constrained depot scheduling respecting hangar capacity and $p_{10}$ RUL deadlines.
5. **Real-time IoT Telemetry Streaming & Fault Injection ([`src/stream/`](src/stream/)):** SSE streaming, data-quality guard, and dynamic trust-score penalty.
6. **Model Context Protocol (MCP) Server ([`src/mcp_server.py`](src/mcp_server.py)):** Official FastMCP implementation exposing maintenance tools and resources to external agents.
7. **Agent v2 with Critic Verification & Human-in-the-Loop ([`src/agents/fleet_agent.py`](src/agents/fleet_agent.py)):** LangGraph state machine with numeric critic checking, human approval gate, and offline execution support.
8. **Pluggable LLM Provider ([`src/llm/provider.py`](src/llm/provider.py)):** Unified interface supporting OfflineProvider (no keys), GeminiProvider, and OpenAIProvider.
9. **Tamper-Evident Audit Ledger ([`src/audit.py`](src/audit.py)):** Cryptographic SHA-256 hash chaining with `verify_chain()` validation.
10. **Comprehensive Benchmark & Model Card ([`src/benchmark.py`](src/benchmark.py), [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)):** NASA asymmetric PHM scoring and responsible AI documentation.

---

## 13. Honest Limitations

- **Simulated Dataset:** NASA CMAPSS FD001 is a numerical simulator (C-MAPSS), not operational airline telemetry. Flight turbulence, de-icing fluid contamination, and bird ingestions are not modeled.
- **Single Fault Mode in FD001:** FD001 simulates High-Pressure Compressor degradation only. Subsystem sensitivity attributions represent empirical sensor correlations rather than multi-fault diagnosis.
- **Assumed Operational Flight Utilization:** Calendar day translations assume 2.5 cycles/day. This assumption is configurable in `config/costs.yaml`.
- **Decision Support Only:** Simulated purchase orders are clearly marked `SIMULATED` and require FAA/EASA certified engineering review before execution.

---

## 14. Hackathon Compliance & AI Tools Disclosure

In strict compliance with the **Neural Storm 2026 Official Rules**:
- **Build Window:** Upgrades executed during November 2–4, 2026.
- **Primary Language:** 100% Python for data ingestion, feature engineering, modeling, scheduling, streaming, and API serving.
- **AI Coding Assistant Disclosure:** Built with assistance from **Antigravity** (Google DeepMind's advanced agentic coding assistant). Antigravity was utilized for architectural code analysis, designing the split-conformal uncertainty model, implementing the OR-Tools scheduler, constructing the FastMCP server, and authoring automated test suites.
- **Foundation Models:** Optional synthesis powered by OpenAI (`gpt-4o-mini`) and Google Gemini (`gemini-1.5-flash`), with 100% offline deterministic fallback.

---

## 15. Credits & Acknowledgements

1. **Dataset:** NASA Prognostics Center of Excellence (PCoE)  
   *Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM).*
2. **Open-Source Base Architecture:** Acknowledgment to the open-source machine learning community for foundational CMAPSS data processing and XGBoost baseline implementations upon which this system was expanded.
3. **Core Libraries:** FastAPI, LangGraph, OR-Tools, FastMCP, scikit-learn, XGBoost, SciPy, ReportLab.

---

*Neural Storm 2026 · Track 01 (Future Tech) · Problem Statement FT-03*
