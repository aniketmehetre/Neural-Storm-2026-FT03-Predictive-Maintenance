# ⚡ Engines Health Monitor — Aerospace Predictive Maintenance System & 3D Digital Twin

**Neural Storm 2026 Hackathon Submission**  
**Track 01:** Future Tech (Intelligent Systems & Emerging Technology)  
**Problem Statement:** **FT-03: Predictive Maintenance for Industry (IoT time-series, Anomaly Detection)**  
**Team Name:** void main (`aniketmehetre`)  
**Repository:** [https://github.com/aniketmehetre/Neural-Storm-2026-FT03-Predictive-Maintenance](https://github.com/aniketmehetre/Neural-Storm-2026-FT03-Predictive-Maintenance)  
**Hackathon Build Window:** November 2, 2026 – November 5, 2026  

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

*Figure 1: Real-time 3D Aerospace Satellite Digital Twin — scrolling drives continuous orbital progression, explodes mechanical subsystems along flight axes, streams live IoT sensor cycles via FastAPI SSE (`:8000`), activates calibrated split-conformal uncertainty fan bands, and simulates autonomous AI parts procurement saving \$29.54M.*

| Satellite Bus & Telemetry HUD | Exploded Subsystem Assembly | Cost-Optimal Policy & Backtest |
| :---: | :---: | :---: |
| ![Satellite Twin](docs/img/satellite_space_01.png) | ![Exploded View](docs/img/satellite_space_02_exploded.png) | ![Decide with Dollars](docs/img/satellite_space_03_decide.png) |

> 💡 **Automated Judge Demonstration (`demo.py`):** Run `python demo.py` in your terminal for an interactive guided walkthrough with timed spoken scripts for recording, or `python demo.py --verify` to test all live FastAPI backend endpoints!

---

## 📖 Table of Contents

1. [Executive Summary & 3-Minute Judge Pitch](#1-executive-summary--3-minute-judge-pitch)
2. [Problem Statement & Industrial Realities](#2-problem-statement--industrial-realities)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Machine Learning & Predictive Modeling (RUL Engine)](#4-machine-learning--predictive-modeling-rul-engine)
5. [Finite-Sample Calibrated Conformal Uncertainty](#5-finite-sample-calibrated-conformal-uncertainty)
6. [Unsupervised Anomaly Detection & Degradation Onset](#6-unsupervised-anomaly-detection--degradation-onset)
7. [Decision Engine & Operations Research (OR-Tools)](#7-decision-engine--operations-research-or-tools)
8. [Agentic AI Layer v2 (LangGraph with Critic & Human Approval)](#8-agentic-ai-layer-v2-langgraph-with-critic--human-approval)
9. [Interactive 3D Digital Twin & Mission Control Deck](#9-interactive-3d-digital-twin--mission-control-deck)
10. [Model Context Protocol (MCP) Server](#10-model-context-protocol-mcp-server)
11. [Cryptographic Tamper-Evident Audit Ledger](#11-cryptographic-tamper-evident-audit-ledger)
12. [Empirical Results & Benchmark Verification](#12-empirical-results--benchmark-verification)
13. [Setup & Quick Start Guide (Zero API Key Offline Execution)](#13-setup--quick-start-guide-zero-api-key-offline-execution)
14. [Complete API & Streaming Reference](#14-complete-api--streaming-reference)
15. [Limitations, Compliance & AI Tools Disclosure](#15-limitations-compliance--ai-tools-disclosure)

---

## 1. Executive Summary & 3-Minute Judge Pitch

| Evaluation Dimension | Standard Hackathon Projects | **Our Solution (Team void main)** |
| :--- | :--- | :--- |
| **Prediction Reliability** | Single fragile point prediction (e.g. "RUL = 42") | **Finite-sample calibrated split-conformal intervals $[p_{10}, p_{90}]$** (verified 84.0% empirical coverage on held-out units) |
| **Early Warning** | Alerts only when failure is imminent (RUL < 20) | **Unsupervised PCA + IsolationForest with CUSUM change-point tracking** flagging degradation onset **164.5 cycles in advance** |
| **Business Impact** | Theoretical accuracy metrics (RMSE / $R^2$) | **OR-Tools CP-SAT depot hangar scheduler** achieving **\$29.54M (84.4%) fleet savings** and **zero in-service failures** |
| **AI Safety & Trust** | Unconstrained LLM output prone to hallucinations | **LangGraph state machine with deterministic Numeric Critic** and **Human-in-the-Loop approval gate** |
| **Auditability** | Ephemeral console prints or flat text logs | **Cryptographic SHA-256 hash-chained tamper-evident decision ledger** |
| **User Experience** | Static 2D dashboard tables | **Atmos-style 3D Satellite Digital Twin (Three.js + Anime.js v4 + Lenis)** streaming live IoT telemetry via FastAPI SSE |
| **Agentic Standards** | Isolated proprietary scripts | **Official Model Context Protocol (MCP)** server exposing fleet tools to external agents (Antigravity, Claude, Cursor) |

### The 3-Minute Judge Walkthrough
1. **The Industrial Pain:** Catastrophic in-service failures cost over \$350,000 per incident and risk catastrophic vehicle loss. Conversely, legacy fixed-interval scheduling retires components with 32%+ remaining useful life still left on the wing.
2. **Detect Early (Unsupervised):** Before supervised models can predict time-to-failure, physical systems depart from healthy baseline behavior. Our PCA + Isolation Forest engine fuses multi-sensor residuals into a normalized Health Index, detecting degradation onset with a **median lead time of 164.5 cycles (~65 operating days)** to failure.
3. **Calibrated Forecasts (Statistical Rigor):** Rather than gambling on fragile point predictions, our system delivers finite-sample **80% split-conformal prediction intervals**, validated at **84.0% coverage** on held-out test units, with mission-horizon failure probabilities $P(\text{failure within 30 cycles})$.
4. **Decide with Dollars (Operations Research):** A cost-optimal replacement policy paired with Google's **OR-Tools CP-SAT scheduler** optimizes multi-bay hangar slot allocations under capacity constraints. In our 100-engine fleet backtest, our policy achieved **zero in-service failures and \$29,542,700 in net savings**.
5. **Act Safely (Agentic AI):** A 7-node LangGraph agent with a **deterministic numeric critic** checks all claims against real state, requires **human approval** before releasing simulated purchase orders, and records every action in a **SHA-256 hash-chained tamper-evident audit ledger**.
6. **Next-Gen 3D Mission Control:** A continuous 3D coordinate space built with **Three.js, Anime.js v4, and Lenis** renders procedural spacecraft subsystems with screen-projected mini-waveforms, interactive raycasting, and real-time Server-Sent Events (SSE) telemetry streaming from FastAPI.

---

## 2. Problem Statement & Industrial Realities

### Problem Statement: FT-03 Predictive Maintenance for Industry
Turbofan engines and aerospace spacecraft operate under severe thermodynamic, aerodynamic, and structural stresses. Modern aerospace operators ingest high-frequency IoT multi-sensor telemetry across flight cycles. Industrial operators require an autonomous, reliable system that can:
1. Ingest real-time streaming IoT telemetry without lookahead bias.
2. Isolate subtle anomalous sensor degradation trajectories hundreds of hours prior to critical failure.
3. Deliver mathematically trustworthy, conformal time-to-failure forecasts with quantified uncertainty.
4. Translate statistical metrics into actionable business decisions, depot maintenance schedules, and parts procurement orders.
5. Operate transparently with zero API key dependencies in air-gapped mission environments.

### The NASA CMAPSS FD001 Benchmark
Our system is benchmarked on the gold-standard **NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS) FD001 dataset**:
- **Fleet Scale:** 100 training engines (run to failure) and 100 test engines (cut off at random points prior to failure).
- **Sensor Suite:** 14 high-variance operating sensors (LPC outlet temperature $T_{24}$, HPC outlet temperature $T_{30}$, LPT outlet temperature $T_{50}$, Bypass duct pressure $P_{15}$, HPC outlet pressure $P_{30}$, Physical fan speed $N_f$, Physical core speed $N_c$, Static pressure $P_{s30}$, Fuel flow ratio $\phi$, Corrected fan speed $N_{Rf}$, Corrected core speed $N_{Rc}$, Bypass ratio BPR, Bleed enthalpy $h_{tBleed}$, Coolant bleeds $W_{31}$, $W_{32}$).
- **Evaluation Criterion:** Tested against the official ground-truth Remaining Useful Life (RUL) of the 100 test engines at their final observed operating cycle.

---

## 3. End-to-End System Architecture

The following diagram illustrates the complete, reactive dataflow from raw multi-sensor IoT telemetry to decision logistics and 3D digital twin visualization:

```mermaid
flowchart TD
    subgraph SENSORS["1. IoT Telemetry Layer"]
        A["NASA CMAPSS FD001 Multi-Sensor Telemetry"] --> B["Data Quality Guard (Signal Quality & Trust Score)"]
        B --> C["Causal Feature Engine (90 Rolling Stats, Window=30, No Lookahead)"]
    end

    subgraph MODELS["2. AI & Statistical Engines"]
        C --> D1["Unsupervised Anomaly Engine (PCA + IsolationForest)"]
        C --> D2["Supervised RUL Engine (Optuna-Tuned XGBoost Regressor)"]
        D1 --> E1["Health Index & CUSUM Change-Point Detector"]
        D2 --> E2["Split-Conformal Calibration Engine ([p10, p90] Bands & P(fail<=30))"]
    end

    subgraph DECISION["3. Operations Research & Multi-Agent Layer"]
        E1 & E2 --> F["Cost-Optimal Replacement Policy (Replace-By Deadlines)"]
        F --> G["OR-Tools CP-SAT Depot Hangar Scheduler (Multi-Bay Optimization)"]
        G --> H["LangGraph Multi-Agent v2 (Triage ➔ Diagnostic ➔ Planner ➔ Numeric Critic ➔ Human Gate ➔ Dispatch)"]
    end

    subgraph OUTPUTS["4. Deployment & Visualization Deck"]
        H --> I1["FastAPI Backend Service (REST + Live SSE Replay :8000)"]
        H --> I2["3D Aerospace Satellite Digital Twin (Three.js + Anime.js v4 :5173)"]
        H --> I3["Model Context Protocol (FastMCP Server fleet://summary)"]
        H --> I4["Cryptographic Audit Ledger (SHA-256 Tamper-Evident Chain)"]
    end
```

---

## 4. Machine Learning & Predictive Modeling (RUL Engine)

### 4.1 Causal Feature Engineering (Zero Train/Serve Skew)
Machine learning models often fail in production due to subtle lookahead leakage during feature normalization. Our feature engine ([`src/features.py`](src/features.py)) enforces strict temporal causality:
- **Rolling Window Size ($W = 30$ cycles):** For every sensor $s_i$, we compute rolling summary statistics:
  $$\text{Mean: } \mu_t = \frac{1}{W} \sum_{k=0}^{W-1} x_{t-k}, \quad \text{Std: } \sigma_t = \sqrt{\frac{1}{W}\sum_{k=0}^{W-1} (x_{t-k} - \mu_t)^2}$$
  $$\text{Min/Max Range: } \Delta_t = \max_{k}(x_{t-k}) - \min_{k}(x_{t-k}), \quad \text{EWMA: } \tilde{x}_t = \alpha x_t + (1-\alpha) \tilde{x}_{t-1}$$
- **Sensor Rate of Change:** First-order backwards difference: $\Delta x_t = x_t - x_{t-1}$.
- **Zero Lookahead Guarantee:** Every feature vector at operating cycle $t$ depends strictly on cycles $\{t-W+1, \dots, t\}$. Expanding windows are used for cycles $t < 30$ with zero zero-padding artifacts. Total feature dimensionality: **90 causal features**.

### 4.2 Piecewise Linear RUL Target Formulation
Turbofan degradation exhibits an initial healthy plateau where wear is undetectable, followed by exponential degradation. In accordance with PHM literature, we clip RUL to a maximum plateau $R_{\text{clip}} = 125$ cycles:
$$y_t = \min(R_{\text{clip}}, \, \text{Total Cycles} - t)$$

### 4.3 Optuna Hyperparameter Optimization
Our XGBoost regressor was tuned using Bayesian optimization (Optuna, 50 trials, 5-fold grouped cross-validation grouped by engine `unit_id` to prevent cross-engine data leakage):
- `n_estimators`: 280
- `max_depth`: 4 (avoids leaf memorization)
- `learning_rate`: 0.038
- `subsample`: 0.82
- `colsample_bytree`: 0.74
- `reg_alpha`: 0.45, `reg_lambda`: 1.25

### 4.4 Official NASA Asymmetric PHM Scoring Function
In aerospace engineering, **late predictions are far more dangerous than early predictions**:
- **Early prediction ($\hat{y} < y$):** Engine serviced slightly early (minor economic waste).
- **Late prediction ($\hat{y} > y$):** Engine fails in flight (catastrophic disaster).

The official NASA PHM scoring function penalizes late predictions exponentially:
$$d = \hat{y} - y, \qquad S = \sum_{i=1}^{N} s_i \quad \text{where } s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \text{ (Early)} \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \text{ (Late)} \end{cases}$$

Our tuned XGBoost model achieved a NASA PHM score of **$S = 655.02$**, outperforming baseline linear regression ($S = 1420.50$) by over **53.8%**.

---

## 5. Finite-Sample Calibrated Conformal Uncertainty

### 5.1 Why Point Predictions Fail in Industry
Traditional ML outputs a single value $\hat{y} = 35.0$. If the engine fails at cycle 22, the aircraft suffers an in-flight shutdown. Operators need statistically guaranteed prediction intervals $[p_{10}, p_{90}]$ such that:
$$P\left(y \in [p_{10}, p_{90}]\right) \ge 1 - \alpha \quad (80\% \text{ target coverage})$$

### 5.2 Split-Conformal Calibration Algorithm ([`src/uncertainty.py`](src/uncertainty.py))
Using inductive split-conformal inference on a held-out calibration set of engines:
1. Compute absolute nonconformity residuals on the calibration set:
   $$R_i = |y_i - \hat{y}_i|$$
2. Compute the empirical $(1-\alpha)\left(1 + \frac{1}{n}\right)$-th quantile of residuals:
   $$\hat{q}_{0.80} = \text{Quantile}\left(\{R_1, \dots, R_n\}, \, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right) = 22.48 \text{ cycles}$$
3. For any new test engine telemetry reading, the calibrated prediction band is:
   $$[p_{10}, p_{90}] = \left[\max(0, \hat{y} - \hat{q}_{0.80}), \quad \hat{y} + \hat{q}_{0.80}\right]$$
4. **Failure Probability $P(\text{failure} \le 30 \text{ cycles})$:** Using kernel density estimation fitted on calibrated conformal residuals, we evaluate:
   $$P(y \le 30 \mid \hat{y}) = \int_{-\infty}^{30} \mathcal{N}\left(u; \, \hat{y}, \, \hat{\sigma}_{\text{res}}^2\right) du$$

### 5.3 Empirical Coverage Verification
On the official held-out test engines (`reports/calibration.json`), our empirical coverage was **84.0%** (target 80%), satisfying the rigorous 8-point aerospace safety margin.

---

## 6. Unsupervised Anomaly Detection & Degradation Onset

### 6.1 Multi-Model Residual Fusion ([`src/anomaly.py`](src/anomaly.py))
Before supervised RUL can be reliably predicted, the spacecraft must detect when healthy baseline operation has ended. We fuse two complementary unsupervised models:
1. **Principal Component Analysis (PCA):** Trained exclusively on healthy initial cycles ($t \le 25$). Measures reconstruction error across the orthogonal complement:
   $$\text{SPE}_t = \|x_t - W W^T x_t\|_2^2$$
2. **Isolation Forest:** Measures path length anomalies in feature space to capture non-linear sensor drift:
   $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
3. **Normalized Health Index ($HI_t \in [0, 1]$):** Fused and scaled such that $HI_t = 1.0$ represents pristine condition and $HI_t = 0.0$ represents severe anomaly.

### 6.2 CUSUM Change-Point Tracking
To prevent transient spikes from triggering false alarms, we apply Cumulative Sum (CUSUM) change-point tracking to the standardized Health Index residuals:
$$S_0 = 0, \qquad S_t = \max\left(0, \, S_{t-1} + (z_t - k)\right)$$
When $S_t > h$ (decision threshold $h = 5.0$), **degradation onset is formally declared**.
- **Onset Detection Rate:** **92.0%** of test engines successfully flagged.
- **Median Advance Warning:** **164.5 cycles** prior to component failure.
- **Advance Alerting:** **98.9%** flagged before RUL crossed 80 cycles.
- **False Alarm Rate:** Only **2.0%** during baseline operation.

---

## 7. Decision Engine & Operations Research (OR-Tools)

### 7.1 Cost-Optimal Replacement Policy
We model maintenance economics using operational airline cost parameters ([`config/costs.yaml`](config/costs.yaml)):
- **Catastrophic In-Service Failure Cost ($C_f$):** \$350,000 (AOG downtime, emergency recovery, unscheduled hangar displacement).
- **Scheduled Depot Overhaul Cost ($C_s$):** \$45,000 (Standard scheduled maintenance).
- **Wasted Life Cost ($C_w$):** \$150 / cycle discarded prematurely.

The cost-optimal trigger replaces an engine when the expected cost of running one additional cycle exceeds the expected cost of immediate replacement:
$$P(\text{failure in next cycle}) \times C_f \ge C_s + \text{RUL} \times C_w$$

### 7.2 OR-Tools CP-SAT Hangar Scheduler ([`src/decision/scheduler.py`](src/decision/scheduler.py))
Depot maintenance hangars have strictly constrained bay capacities. Our constraint satisfaction model:
- **Interval Variables:** Each maintenance job $j$ has duration $d_j \in [12, 48]$ hours.
- **Capacity Constraint:** A cumulative constraint ensures concurrent hangar bay usage never exceeds maximum capacity $B_{\max} = 3$:
  $$\text{AddCumulative}\left([I_1, \dots, I_m], \, [1, \dots, 1], \, B_{\max}\right)$$
- **Objective Function:** Minimize total weighted tardiness against conservative conformal deadlines $p_{10}$:
  $$\min \sum_{j} w_j \times \max(0, \, \text{End}_j - \text{Deadline}_j)$$

### 7.3 100-Engine Fleet Money Backtest Results

| Maintenance Strategy | In-Service Failures | Discarded Life % | Total Fleet Expenditure | Net Money Saved vs Reactive |
| :--- | :---: | :---: | :---: | :---: |
| **1. Reactive (Run-to-Failure)** | 100 | 0.0% | \$35,000,000.00 | Baseline (\$0.00) |
| **2. Fixed-Interval (140 Cycles)** | 0 | 32.2% | \$5,497,200.00 | \$29,502,800.00 (84.3%) |
| **3. Our Predictive AI Policy** | **0** | **30.9%** | **\$5,457,300.00** | **\$29,542,700.00 (84.4%)** |

*Key Takeaway: Our policy achieved **zero in-service failures** while capturing **\$29.54M in savings** and preserving **4,648 more usable flight cycles** than fixed-interval maintenance.*

---

## 8. Agentic AI Layer v2 (LangGraph with Critic & Human Approval)

The multi-agent orchestration layer ([`src/agents/fleet_agent.py`](src/agents/fleet_agent.py)) implements an enterprise LangGraph state machine:

```text
    ┌──────────────┐
    │ Triage Node  │ Ingests fleet status & isolates at-risk engines (RUL < 80)
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Diagnostic  │ Computes SHAP sensor attributions & identifies degrading subsystem
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │ Planner Node │ Executes OR-Tools Scheduler & Decision Policy as callable tools
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │ Numeric      │ Deterministically checks every count, RMSE, and dollar figure
    │ Critic Node  │ against state. Automatically retries if discrepancies exist.
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │ Human Gate   │ Pauses execution until licensed engineering approval is received
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │ Dispatch &   │ Issues simulated purchase orders (SIMULATED-PO-2026-XXXX)
    │ Report Node  │ and appends entry to SHA-256 cryptographic audit ledger
    └──────────────┘
```

- **Pluggable LLM Backends ([`src/llm/provider.py`](src/llm/provider.py)):**
  - **`OfflineProvider` (Default):** Operates 100% locally offline with zero API keys using domain heuristics.
  - **`GeminiProvider` (`gemini-1.5-flash`):** Activated automatically if `GEMINI_API_KEY` is set.
  - **`OpenAIProvider` (`gpt-4o-mini`):** Activated automatically if `OPENAI_API_KEY` is set.

---

## 9. Interactive 3D Digital Twin & Mission Control Deck

The frontend provides an Atmos-style continuous 3D digital twin built with **Three.js, Anime.js v4, and Lenis**:

### Key Features
1. **Procedural Aerospace PBR Materials:**
   - **Main Spacecraft Bus:** Quilted gold multilayer insulation (MLI) crinkled foil texture with titanium corner extrusion posts.
   - **Deployable Solar Array Wings:** Triple-junction photovoltaic cells with technical greyish slate-titanium finish, individual cell matrices, fine silver collection lines, and titanium deployment hinges.
   - **High-Gain Communication Dish:** Parabolic reflector dish modeled via mathematical lathe geometry ($z = a \cdot r^2$) with central feed horn and directional alignment.
   - **Earth Observation Sensor Payload:** High-aperture optical telescope with anti-reflective glass lenses.
   - **ADCS Subsystem:** Dual star-tracker camera housings and reaction wheel enclosures.
2. **Kinetic Typography & Clean Layout:**
   - Word-boundary tokenization (`.title-word` container) prevents mid-word breaks while preserving Anime.js v4 character reveal stagger animations.
   - Borderless, floating design sits seamlessly on the canvas with deep drop shadows.
3. **Live FastAPI Backend Integration (`:8000`):**
   - **Live Health Status Pill:** Displays `🟢 FASTAPI: ONLINE (:8000)`. Clicking it displays model validation metrics.
   - **Live IoT SSE Stream Button (`▶ LIVE IOT STREAM`):** Connects to `GET /stream/1`, replaying IoT cycles in real time and dynamically updating mini-waveforms and conformal RUL numbers.
   - **Automated AI MRO Button (`🤖 MRO AI STRATEGY`):** Triggers `POST /maintenance-strategy`, rendering real purchase order SKUs, downtime estimates, and maintenance narratives directly in the 3D inspection panel.

---

## 10. Model Context Protocol (MCP) Server

Our system implements the official **Model Context Protocol (MCP)** via `FastMCP` ([`src/mcp_server.py`](src/mcp_server.py)), allowing external AI agents (such as Antigravity, Claude Desktop, or Cursor) to interact with the maintenance suite:

### Exposed MCP Tools & Resources
- **Tool `list_fleet(status_filter)`:** Filter fleet units by status (`HEALTHY`, `WARNING`, `CRITICAL`).
- **Tool `get_engine_status(engine_id)`:** Get real-time RUL, conformal interval, and anomaly metrics.
- **Tool `explain_engine(engine_id, top_n)`:** Returns SHAP feature attributions explaining degradation.
- **Tool `recommend_maintenance(engine_id)`:** Generates actionable maintenance actions and action windows.
- **Tool `schedule_hangar(engine_ids)`:** Solves OR-Tools CP-SAT hangar slot allocations.
- **Resource `fleet://summary`:** Dynamic Markdown dashboard of fleet health distribution.

*Configuration for Claude Desktop / Antigravity (`claude_desktop_config.json`):*
```json
{
  "mcpServers": {
    "engine-monitor": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "C:/HACKATHONS/my-hackathon-project"
    }
  }
}
```

---

## 11. Cryptographic Tamper-Evident Audit Ledger

Aerospace regulatory compliance requires unalterable decision history. Our system implements an append-only JSONL decision ledger ([`src/audit.py`](src/audit.py)) with **SHA-256 hash chaining**:
- Every diagnostic record $E_i$ embeds the cryptographic hash of the preceding record:
  $$\text{Hash}_i = \text{SHA256}\left(\text{Hash}_{i-1} \,\|\, \text{Timestamp}_i \,\|\, \text{UnitID}_i \,\|\, \text{Decision}_i \,\|\, \text{Cost}_i\right)$$
- Genesis Block: `0000000000000000000000000000000000000000000000000000000000000000`.
- **Verification CLI:**
  ```bash
  python -m src.audit --verify
  ```
  *Output:* `[AUDIT LEDGER] Verified 14 entries. Cryptographic SHA-256 chain intact. Integrity: VALID.`

---

## 12. Empirical Results & Benchmark Verification

Every metric below is generated directly from source code and verified via `reports/*.json`:

| Metric Category | Target / Baseline | **Our Verified Result** | Verification Source |
| :--- | :--- | :--- | :--- |
| **Model Test RMSE** | 23.15 cycles (Linear Regression) | **16.74 cycles (Optuna XGBoost)** | `reports/benchmark.json` |
| **Test MAE** | 18.40 cycles | **12.33 cycles** | `reports/benchmark.json` |
| **$R^2$ Variance Explained** | 0.6800 | **0.8255** | `reports/benchmark.json` |
| **NASA Asymmetric PHM Score** | 1,420.50 (Penalty baseline) | **655.02 (-53.8% penalty)** | `reports/benchmark.json` |
| **Conformal Coverage (80% Band)** | 80.0% nominal target | **84.0% empirical coverage** | `reports/calibration.json` |
| **Anomaly Lead Time to Failure** | ~20 cycles (Thresholding) | **164.5 cycles median lead time** | `reports/anomaly.json` |
| **Early Warning Rate (RUL $\le$ 80)** | > 80.0% | **98.9% flagged in advance** | `reports/anomaly.json` |
| **Fleet In-Service Failures** | 100 / 100 (Reactive baseline) | **0 / 100 (Zero failures)** | `reports/backtest.json` |
| **Total Fleet Savings** | Baseline ($35.0M cost) | **$29,542,700 saved (-84.4%)** | `reports/backtest.json` |
| **Automated Test Suite** | 100% pass | **48 / 48 tests green** | `pytest -v` |

---

## 13. Setup & Quick Start Guide (Zero API Key Offline Execution)

### Prerequisites
- Python 3.11 or 3.12
- Node.js 20+ (for 3D digital twin frontend)
- Git

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/aniketmehetre/Neural-Storm-2026-FT03-Predictive-Maintenance.git
cd Neural-Storm-2026-FT03-Predictive-Maintenance

# Setup Python environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # Linux / macOS

pip install --upgrade pip
pip install -e .
pip install -r requirements.txt

# Setup Node.js frontend
npm install
```

### 2. Run the Full Test Suite (48 Tests)
```bash
pytest -v
```
*Expected: `48 passed in ~8.2s`*

### 3. Launch the Complete System in 2 Commands
**Terminal 1 — FastAPI Backend Server (`:8000`):**
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Access interactive Swagger UI:* [http://localhost:8000/docs](http://localhost:8000/docs)

**Terminal 2 — 3D Satellite Digital Twin (`:5173`):**
```bash
npm run dev
```
*Access 3D Mission Control:* [http://localhost:5173](http://localhost:5173)

### 4. Interactive Judge Demonstration Tool (`demo.py`)
```bash
# Launch interactive guided walkthrough with live narrator pitch scripts
python demo.py

# Verify live FastAPI backend endpoints and model metrics
python demo.py --verify

# Generate / refresh animated walkthrough GIF
python demo.py --gif
```

---

## 14. Complete API & Streaming Reference

### Endpoints
- `GET /health` — Liveness & readiness probe (returns `{"status": "ok", "model_loaded": true}`).
- `GET /` — API metadata, model validation RMSE, feature list, and active endpoints.
- `POST /predict` — Accepts recent sensor readings, outputs predicted RUL, conformal interval `[rul_p10, rul_p90]`, and failure probabilities `prob_fail_10`, `prob_fail_30`.
- `POST /anomaly` — Returns PCA + Isolation Forest anomaly metrics, `health_index`, and `onset_detected`.
- `POST /ingest` — IoT gateway single-cycle ingestion endpoint (<50ms processing).
- `GET /fleet` — Returns the real-time operational status of all ingested engines.
- `GET /stream/{unit_id}` — **Server-Sent Events (SSE)** endpoint streaming live engine telemetry cycles. Supports fault injection:
  ```bash
  curl -N "http://localhost:8000/stream/1?delay=0.1&fault=stuck&fault_cycle=15"
  ```
- `POST /maintenance-strategy` — AI agent generates structured MRO strategy, purchase orders, technician licensing requirements, and downtime estimates.

---

## 15. Limitations, Compliance & AI Tools Disclosure

### Honest Limitations
1. **Simulation Benchmark:** NASA CMAPSS is a thermodynamic simulation dataset. Real-world airline telemetry contains operational complexities like bird ingestion and de-icing fluid interference not present in the simulator.
2. **Single Primary Failure Mode in FD001:** FD001 simulates High-Pressure Compressor (HPC) degradation. Subsystem attributions represent empirical sensor correlations rather than multi-failure isolation.
3. **Decisions Are Advisory:** All emitted purchase orders are clearly marked `SIMULATED` and require sign-off by FAA A&P / EASA Part-66 certified powerplant engineers.

### Hackathon Compliance & AI Tools Disclosure
In strict adherence to the **Neural Storm 2026 Hackathon Official Rules**:
- **Build Window:** Built and upgraded during the hackathon window (November 2–5, 2026).
- **Primary Language:** 100% Python for data processing, feature engineering, modeling, scheduling, streaming, and API serving. Modern WebGL / Three.js for 3D visualization.
- **AI Coding Assistant Disclosure:** Built with assistance from **Antigravity** (Google DeepMind's advanced agentic coding assistant). Antigravity was utilized for architectural code analysis, designing the split-conformal uncertainty model, implementing the OR-Tools scheduler, constructing the FastMCP server, authoring automated test suites, and creating the 3D digital twin.
- **Foundation Models:** Optional synthesis powered by OpenAI (`gpt-4o-mini`) and Google Gemini (`gemini-1.5-flash`), with 100% offline deterministic fallback.

---

## 16. Team & Submission Credits

- **Team Name:** `void main`
- **Lead Developer:** Aniket Mehetre ([`@aniketmehetre`](https://github.com/aniketmehetre))
- **Track:** Track 01 — Future Tech (Intelligent Systems & Emerging Technology)
- **Problem Statement:** FT-03: Predictive Maintenance for Industry
- **License:** [MIT License](LICENSE)

*Neural Storm 2026 · Future Tech · Predictive Maintenance for Industry (FT-03)*
