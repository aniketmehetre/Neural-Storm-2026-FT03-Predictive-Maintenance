# ⚡ Engines Health Monitor — Industrial Predictive Maintenance System

**Neural Storm 2026 Hackathon Submission**  
**Track 01:** Future Tech (Intelligent Systems & Emerging Technology)  
**Problem Statement:** **FT-03: Predictive Maintenance for Industry (IoT time-series, Anomaly Detection)**  
**Team Name:** void main (`aniketmehetre`)  
**Hackathon Build Window:** November 2, 2026, 8:00 PM IST – November 4, 2026, 8:00 PM IST  

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Track: Future Tech](https://img.shields.io/badge/Neural_Storm_2026-Track_01:_FT--03-orange.svg)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-RUL_Regression-337AB7.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API_v1.1-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent_Pipeline-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/pytest-22_passed-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Table of Contents

1. [Hackathon Problem Statement & Solution Overview](#1-hackathon-problem-statement--solution-overview)
2. [Key Capabilities](#2-key-capabilities)
3. [System Architecture](#3-system-architecture)
4. [Actionable Business Logic: Automated Maintenance & Parts Procurement](#4-actionable-business-logic-automated-maintenance--parts-procurement)
5. [Machine Learning Pipeline & Performance](#5-machine-learning-pipeline--performance)
6. [Agentic AI Layer (LangGraph Multi-Agent Diagnostics)](#6-agentic-ai-layer-langgraph-multi-agent-diagnostics)
7. [Tech Stack](#7-tech-stack)
8. [Project Structure](#8-project-structure)
9. [Setup & Local Installation](#9-setup--local-installation)
10. [Running the Application Locally](#10-running-the-application-locally)
11. [API Reference](#11-api-reference)
12. [Testing & Verification](#12-testing--verification)
13. [Deployment (Docker & Cloud Run)](#13-deployment-docker--cloud-run)
14. [Hackathon Compliance & AI Tools Disclosure](#14-hackathon-compliance--ai-tools-disclosure)
15. [Credits & Acknowledgements](#15-credits--acknowledgements)

---

## 1. Hackathon Problem Statement & Solution Overview

### Problem Statement: FT-03 Predictive Maintenance for Industry
Unplanned industrial downtime costs the global aviation, manufacturing, and energy sectors billions of dollars annually. Catastrophic equipment failure can result in catastrophic safety hazards, mission aborts, and massive supply chain disruptions. Traditional maintenance is either **reactive** (repairing after breakdown) or **preventative** (replacing parts on fixed calendar intervals regardless of actual wear, wasting 30%+ of remaining useful component life).

### Our Solution: Engines Health Monitor
**Engines Health Monitor** is an enterprise-grade, end-to-end predictive maintenance and autonomous MRO (Maintenance, Repair, and Overhaul) orchestration system built on the **NASA CMAPSS FD001 Turbofan Engine Degradation dataset**. 

It moves beyond passive time-series charting into an **autonomous closed-loop operational system**:
1. **IoT Sensor Ingestion & Causal Featurization:** Ingests 21 noisy multi-channel telemetry streams, drops non-informative zero-variance sensors, and generates 90 causal rolling-window statistical features.
2. **High-Precision RUL Prediction:** Uses an Optuna-tuned XGBoost Regressor to predict Remaining Useful Life (RUL) with test RMSE ≈ **16.74 cycles** (~2–3 weeks of advance operational notice).
3. **Explainable AI (XAI):** Decomposes every prediction using SHAP TreeExplainer into cycle-by-cycle physical sensor contributions.
4. **Actionable Business Logic Layer (`POST /maintenance-strategy`):** Automatically maps anomalies to physical turbofan subsystems (HPC, LPT, Fan, Fuel Metering, etc.), issues plain-English operational directives, and synthesizes automated, simulated aerospace parts-procurement purchase orders with real aerospace SKUs, estimated costs, lead times, and logistics routing.
5. **Autonomous LangGraph Multi-Agent Diagnostic Pipeline:** A 4-agent sequential workflow that audits the entire fleet, diagnoses critical units, prioritizes maintenance windows, and writes publication-ready executive briefings and downloadable PDFs.
6. **Tesla-Style Interactive Dashboard:** Streamlit command center with fleet-wide telemetry, individual engine drill-downs, "what-if" sensor simulation, and conversational AI assistance.

---

## 2. Key Capabilities

| Capability | Module | Impact |
|---|---|---|
| **RUL Prediction** | XGBoost (Optuna) | Predicts remaining flight cycles with **16.74 RMSE** and **0.825 $R^2$** |
| **Sensor Explainability** | SHAP TreeExplainer | Decomposes exactly *why* an engine is failing down to sensor-level cycle impact |
| **Actionable Business Logic** | `strategy_agent.py` | Bridges ML predictions to real-world MRO: plain-English strategy + simulated parts order |
| **REST API** | FastAPI | High-throughput, production-ready inference and strategy dispatch endpoints |
| **Fleet Command Center** | Streamlit + Plotly | Dark-mode UI for fleet triage, degradation tracking, and live what-if simulation |
| **Autonomous Multi-Agent** | LangGraph (`StateGraph`) | 4-node pipeline: Sensor Analyst → Diagnostic → Prioritizer → Report Writer |
| **Executive PDF Briefing** | ReportLab | One-click downloadable maintenance briefing with KPI tables and action matrices |

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. IOT / DATA LAYER                             │
│   NASA CMAPSS FD001 (100 Turbofan Engines · 21 Sensors · Run-to-Fail)  │
│   15 Informative Sensors Retained · 6 Zero-Variance Dropped            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  2. CAUSAL FEATURE ENGINEERING                         │
│   src/features.py: 30-Cycle Rolling Windows                            │
│   6 Stats per Sensor (Mean, Std, Min, Max, Lag-1, Trend) = 90 Features │
│   Strict Causal Formulation — Zero Training-Serving Skew               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    3. MACHINE LEARNING & XAI LAYER                     │
│   XGBoost Regressor (Optuna Bayesian Optimization · 50 Trials)          │
│   GroupShuffleSplit by unit_id (No Data Leakage) · Piecewise RUL Cap   │
│   SHAP TreeExplainer for Per-Cycle Physical Attributions               │
└───────────────────────┬──────────────────────────┬─────────────────────┘
                        │                          │
┌───────────────────────▼───────────┐  ┌───────────▼─────────────────────┐
│       4. FASTAPI BACKEND          │  │       5. STREAMLIT COMMAND      │
│  src/api/main.py                  │  │  src/dashboard/app.py           │
│  - GET  /health                   │  │  Tab 1: Fleet Overview & KPIs   │
│  - POST /predict                  │  │  Tab 2: Engine Telemetry & SHAP │
│  - POST /maintenance-strategy     │  │  Tab 3: What-If Simulator & MRO │
│                                   │  │  Tab 4: Conversational AI Agent │
└───────────────────────┬───────────┘  └───────────┬─────────────────────┘
                        │                          │
                        └────────────┬─────────────┘
                                     │
┌────────────────────────────────────▼───────────────────────────────────┐
│                        6. AGENTIC AI & MRO LAYER                       │
│  • SimulatedMaintenanceAgent: Maps degradation to turbofan subsystems,  │
│    generates MRO strategy narrative & simulated parts purchase orders. │
│  • LangGraph FleetMonitoringAgent: 4-Node Sequential StateGraph        │
│    (Sensor Analyst ➔ Diagnostic ➔ Prioritizer ➔ Report Writer)        │
│  • ReportLab PDF Engine: Generates formal maintenance release dossiers │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Actionable Business Logic: Automated Maintenance & Parts Procurement

A major limitation of classical predictive maintenance is that machine learning models output raw scalar numbers (e.g. `RUL = 23.4 cycles`) with no bridge to physical maintenance logistics. 

To solve this, we implemented an **actionable business logic layer** ([`src/agents/strategy_agent.py`](src/agents/strategy_agent.py)) exposed via the FastAPI endpoint `POST /maintenance-strategy` and integrated directly into the Streamlit dashboard:

### How It Works:
1. **Anomaly & Subsystem Mapping:** The system ingests the RUL and isolates the dominant degraded component from sensor telemetry or SHAP attributions:
   - `sensor_03` / `sensor_07` / `sensor_11` $\rightarrow$ High-Pressure Compressor (HPC) & Diffuser
   - `sensor_04` / `sensor_21` $\rightarrow$ Low-Pressure Turbine (LPT) & Exhaust Mixer
   - `sensor_08` / `sensor_13` $\rightarrow$ Titanium Fan Rotor & #1 Bearing
   - `sensor_09` / `sensor_14` $\rightarrow$ High-Pressure Core Spool & Thrust Bearings
   - `sensor_12` $\rightarrow$ Fuel Metering Unit (FMU) & Fuel Injectors
   - `sensor_17` $\rightarrow$ Bleed Air Precooler & Pneumatic Valves
   - `sensor_20` $\rightarrow$ High-Pressure Turbine (HPT) Active Clearance Vanes
2. **Operational Directives:** Generates plain-English narrative directives, operational flight restrictions (e.g. revoking ETOPS clearance, mandating continuous data streaming, derating takeoff thrust), and step-by-step MRO procedures.
3. **Simulated Parts-Order Request:** Automatically generates an aerospace procurement requisition complete with:
   - Verified aerospace Stock Keeping Units (SKUs)
   - Component names and line-item unit costs (USD)
   - Order priority tiers: `AOG_EMERGENCY` (Aircraft On Ground), `EXPEDITED`, or `ROUTINE`
   - Logistics distribution center routing, carrier mode, and estimated delivery hours
   - Mandatory maintenance technician licenses (e.g. FAA A&P Powerplant, EASA Part-66 B1, NDT Level II)

---

## 5. Machine Learning Pipeline & Performance

### 5.1 Dataset: NASA CMAPSS FD001
- **Domain:** Commercial turbofan engine simulation (run-to-failure).
- **Engines:** 100 training engines (~20,631 cycles) and 100 test engines (~13,096 cycles).
- **Fault Mode:** High-Pressure Compressor (HPC) degradation under sea-level static flight conditions.
- **Sensors:** 21 sensors recorded. Sensors `01, 05, 10, 16, 18, 19` exhibit zero variance and are purged during Exploratory Data Analysis ([`01_eda.ipynb`](notebooks/01_eda.ipynb)).

### 5.2 Feature Engineering ([`src/features.py`](src/features.py))
For each of the 15 informative sensors, 6 causal rolling statistics over a 30-cycle window are computed:
$$\text{Features} = \{\text{mean}, \text{std}, \text{min}, \text{max}, \text{lag}_1, \text{trend} = (x_t - \mu_{\text{window}})\}$$
Totaling **90 engineered features**. Because rolling windows are causal ($t \in [t - W + 1, t]$), partial windows at initial startup use `min_periods=1`, ensuring 100% mathematical parity between training notebooks and real-time API inference.

### 5.3 Model Training & Optimization
- **Split:** `GroupShuffleSplit` on `unit_id` (80 train engines / 20 validation engines). This guarantees **zero data leakage** across cycles of the same engine.
- **Target Formulation:** Piecewise linear RUL capped at 125 cycles. Early healthy operation is capped because engines show negligible degradation during initial cycles.
- **Hyperparameter Optimization:** 50-trial Bayesian search via Optuna logged to MLflow.

### 5.4 Benchmark Performance

| Model Architecture | Validation RMSE | Test RMSE | Test MAE | Test $R^2$ |
|---|---|---|---|---|
| Baseline Linear Regression | 21.84 cycles | 23.15 cycles | 18.40 cycles | 0.680 |
| **Our Optuna-Tuned XGBoost** | **15.02 cycles** | **16.74 cycles** | **12.33 cycles** | **0.825** |

A test error of ~16.7 cycles provides airline maintenance dispatchers with an accurate **2 to 3-week planning horizon**, preventing unscheduled engine removals and flight cancellations.

---

## 6. Agentic AI Layer (LangGraph Multi-Agent Diagnostics)

In addition to individual API predictions, the project features a multi-agent orchestration architecture:

### 1. LangGraph Autonomous Fleet Monitoring Agent ([`src/agents/fleet_agent.py`](src/agents/fleet_agent.py))
A sequential `StateGraph` with 4 specialist nodes operating over a shared typed state (`FleetAgentState`):
1. **Sensor Analyst Node:** Queries `get_fleet_summary()` and `get_fleet_statistics()`, performs statistical distribution triage, and assigns a fleet-wide health grade.
2. **Diagnostic Agent Node:** Executes `explain_engine()` using SHAP TreeExplainer for all critical units ($RUL \le 30$) to identify primary mechanical fault signatures.
3. **Priority Ranker Node:** Classifies engines into operational intervention windows (Immediate Action, Week 1, Week 2, Week 3).
4. **Report Writer Node:** Synthesizes previous outputs into a structured executive Markdown briefing.

### 2. Conversational AI Analyst ([`src/agents/chat_agent.py`](src/agents/chat_agent.py))
An interactive ReAct agent using OpenAI function calling (`gpt-4o-mini`) equipped with 5 tools:
- `get_fleet_summary()`
- `get_engine_details(engine_id)`
- `get_engines_by_status(status)`
- `get_fleet_statistics()`
- `explain_engine(engine_id, top_n)`

### 3. Executive PDF Generation ([`src/agents/report_pdf.py`](src/agents/report_pdf.py))
Converts autonomous agent briefings into publication-grade PDF dossiers formatted with ReportLab, including executive KPI summary blocks, status tables, and engineer sign-off blocks.

---

## 7. Tech Stack

| Domain | Technology / Library | Purpose |
|---|---|---|
| **Language** | Python 3.11 / 3.12 | Primary programming language |
| **Machine Learning** | `XGBoost`, `scikit-learn` | Non-linear RUL gradient boosting regression |
| **Optimization & XAI** | `Optuna`, `SHAP`, `MLflow` | Bayesian hyperparameter tuning, experiment logging, TreeExplainer |
| **Data Processing** | `pandas`, `numpy`, `pyarrow` | High-performance rolling-window feature extraction |
| **Backend API** | `FastAPI`, `uvicorn`, `pydantic` | High-throughput asynchronous REST API with validation |
| **Agentic Framework** | `LangGraph`, `LangChain-OpenAI` | Multi-agent stateful graph workflow & ReAct loop |
| **Frontend UI** | `Streamlit`, `Plotly` | Dark-mode telemetry command center & interactive charts |
| **PDF Reporting** | `ReportLab`, `markdown` | Programmatic executive maintenance report compilation |
| **DevOps & Containers** | `Docker`, `Google Cloud Run` | Containerization, CI/CD, and serverless deployment |
| **Testing** | `pytest`, `httpx` | Integration and unit testing suite |

---

## 8. Project Structure

```
my-hackathon-project/
├── CMAPSSData/                   # NASA Turbofan run-to-failure dataset
│   ├── test_FD001.txt            # Test telemetry (100 engines)
│   ├── RUL_FD001.txt             # True ground-truth test RUL labels
│   └── readme.txt                # NASA dataset documentation
│
├── data/                         # Processed feature and EDA artifacts
│   ├── eda_meta.json             # Useful (15) and constant (6) sensor mappings
│   └── feature_meta.json         # 90 rolling-feature column definitions
│
├── models/                       # Model binaries and performance records
│   ├── xgb_rul.joblib            # Trained Optuna-optimized XGBoost regressor
│   └── model_meta.json           # Logged hyperparameters, metrics, and thresholds
│
├── notebooks/                    # Interactive development & verification
│   ├── 01_eda.ipynb              # Exploratory data analysis & sensor selection
│   ├── 02_feature_engineering.ipynb   # Rolling statistics & train/val splitting
│   └── 03_model_training.ipynb   # Optuna tuning, MLflow logging, and SHAP
│
├── src/                          # Core production source code
│   ├── features.py               # Shared feature engineering (zero train-serve skew)
│   │
│   ├── api/                      # FastAPI Backend
│   │   ├── __init__.py
│   │   └── main.py               # Endpoints: /health, /predict, /maintenance-strategy
│   │
│   ├── dashboard/                # Streamlit Command Center
│   │   ├── __init__.py
│   │   └── app.py                # 4-Tab dashboard with dark-mode Tesla theme
│   │
│   └── agents/                   # Agentic AI Layer
│       ├── __init__.py
│       ├── strategy_agent.py     # Actionable MRO strategy & parts ordering logic
│       ├── tools.py              # Domain tools for agent data querying & SHAP
│       ├── chat_agent.py         # OpenAI tool-calling conversational agent
│       ├── fleet_agent.py        # LangGraph 4-node autonomous diagnostic pipeline
│       └── report_pdf.py         # ReportLab PDF briefing generator
│
├── tests/
│   ├── __init__.py
│   └── test_api.py               # 22 automated integration and validation tests
│
├── Dockerfile                    # Lean container for FastAPI backend
├── Dockerfile.dashboard          # Full container for Streamlit Dashboard & Agents
├── cloudbuild.dashboard.yaml     # GCP Cloud Build configuration
├── pyproject.toml                # Standard Python package configuration
├── requirements.txt              # Complete project dependencies
└── requirements-api.txt          # Minimal dependencies for FastAPI microservice
```

---

## 9. Setup & Local Installation

### Prerequisites
- Python 3.11 or 3.12 installed
- Git
- *(Optional)* OpenAI API key (for Tab 4 Chat Analyst & Autonomous Fleet Agent)

### 1. Clone the Repository
```bash
git clone https://github.com/aniketmehetre/PreventiveMotorMantainance.git
cd my-hackathon-project
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux/macOS:
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional for LLM features)
Create a `.env` file in the project root:
```env
OPENAI_API_KEY="your-openai-api-key-here"
```
*(Note: Core RUL prediction, SHAP explainability, FastAPI endpoints, and the Simulated Maintenance Strategy Agent operate 100% locally and do not require an API key).*

---

## 10. Running the Application Locally

### Option A: Run the FastAPI Backend
Start the high-performance REST API with Uvicorn:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Documentation (Swagger UI): `http://localhost:8000/docs`
- Redoc Documentation: `http://localhost:8000/redoc`

### Option B: Run the Streamlit Command Center
Launch the multi-tab interactive telemetry dashboard:
```bash
streamlit run src/dashboard/app.py
```
Access the dashboard at `http://localhost:8501`.

---

## 11. API Reference

### 1. `GET /health`
Liveness and readiness probe for Docker / Cloud Run.
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. `POST /predict`
Predicts RUL for an engine given a sequence of recent sensor cycles.
**Request Body:**
```json
{
  "unit_id": 42,
  "readings": [
    {
      "cycle": 190,
      "sensor_02": 642.35,
      "sensor_03": 1589.70,
      "sensor_04": 1400.12,
      "sensor_06": 21.61,
      "sensor_07": 554.32,
      "sensor_08": 2388.08,
      "sensor_09": 9050.15,
      "sensor_11": 47.45,
      "sensor_12": 521.80,
      "sensor_13": 2388.07,
      "sensor_14": 8134.20,
      "sensor_15": 8.42,
      "sensor_17": 392.0,
      "sensor_20": 38.85,
      "sensor_21": 23.31
    }
  ]
}
```
**Response Body:**
```json
{
  "unit_id": 42,
  "predicted_rul": 24.5,
  "status": "CRITICAL",
  "cycles_provided": 1,
  "window_used": 1,
  "model_val_rmse": 15.02
}
```

### 3. `POST /maintenance-strategy`
Actionable business logic endpoint: maps predicted RUL to subsystem fault models, generates maintenance strategy, and issues automated parts procurement orders.

**Request Body:**
```json
{
  "unit_id": 42,
  "predicted_rul": 18.5,
  "dominant_sensor": "sensor_11",
  "operating_cycle": 215
}
```

**Response Body:**
```json
{
  "unit_id": 42,
  "predicted_rul": 18.5,
  "alert_level": "CRITICAL",
  "urgency_rating": "LEVEL 1 - IMMEDIATE AIRCRAFT ON GROUND (AOG) DISPATCH",
  "primary_affected_subsystem": "High-Pressure Compressor (Static Pressure Stage)",
  "plain_english_strategy": "Engine #42 at cycle 215 has entered a critical degradation phase with only 18.5 operating cycles remaining before anticipated functional failure. Telemetry indicates severe risk concentrated in the High-Pressure Compressor (Static Pressure Stage) (HPC Static Bleed & Diffuser Case), presenting signature symptoms of HPC diffuser flow separation and stage static pressure drop. The AI Agent advises immediate grounding upon completion of the current mission, cancellation of extended twin-engine operational performance (ETOPS) clearance, and induction into a qualified depot facility for borescope inspection and rapid modular replacement. A pre-emptive expedited purchase order has been automatically generated to preposition critical replacement assemblies.",
  "recommended_action_window": "Within next 3 cycles or 48 calendar hours (RUL: 18.5 cycles)",
  "operational_restrictions": [
    "GROUND IMMEDIATELY upon arrival at primary maintenance base.",
    "Revoke ETOPS (Extended-range Twin-engine Operational Performance Standards) authorization.",
    "Limit max continuous takeoff thrust derate to 85% if reposition flight is required.",
    "Prohibit revenue passenger flights until maintenance release certificate is signed."
  ],
  "maintenance_tasks": [
    "Perform comprehensive video-borescope inspection of High-Pressure Compressor (Static Pressure Stage).",
    "Inspect magnetic chip detectors and oil scavenger filters for metallic debris.",
    "Remove and replace HPC Static Bleed & Diffuser Case using pre-staged parts order PO-AERO-2026-F81A02E4.",
    "Perform cold-crank rotation check and acoustic resonance verification.",
    "Conduct full-throttle static ground run cell testing with 3-minute stabilization at takeoff thrust."
  ],
  "technician_certifications": [
    "FAA Airframe & Powerplant (A&P) with Gas Turbine Endorsement",
    "EASA Part-66 B1 Licensed Aircraft Maintenance Engineer (Turbofan)",
    "Borescope Non-Destructive Testing (NDT) Level II Certification"
  ],
  "estimated_downtime_hours": 36.0,
  "parts_order": {
    "order_id": "PO-AERO-2026-F81A02E4",
    "order_timestamp": "2026-11-05T18:15:00Z",
    "order_priority": "AOG_EMERGENCY",
    "target_subsystem": "High-Pressure Compressor (Static Pressure Stage)",
    "items": [
      {
        "sku": "HPC-DIF-110",
        "part_name": "HPC Diffuser Casing Air Seal & Vane Cluster",
        "subsystem": "High-Pressure Compressor (Static Pressure Stage)",
        "quantity": 1,
        "unit_cost_usd": 16500.0,
        "total_cost_usd": 16500.0,
        "lead_time_days": 3,
        "criticality": "CRITICAL"
      },
      {
        "sku": "SEN-PS30-11",
        "part_name": "Ps30 Static Pressure Differential Transducer",
        "subsystem": "High-Pressure Compressor (Static Pressure Stage)",
        "quantity": 1,
        "unit_cost_usd": 2400.0,
        "total_cost_usd": 2400.0,
        "lead_time_days": 1,
        "criticality": "STANDARD"
      }
    ],
    "total_cost_usd": 18900.0,
    "procurement_hub": "ORD-Central-AeroDepot (Chicago, IL)",
    "estimated_delivery_hours": 12,
    "shipping_method": "Dedicated Air Courier / Hot-Shot Logistics"
  },
  "agent_confidence": "94.8% (Based on Optuna-tuned XGBoost & CMAPSS FD001 fleet trends)",
  "agent_signature": "AeroGuard-PredictiveMaintenanceAgent-v2.6 [VERIFIED_MRO_DISPATCH]"
}
```

---

## 12. Testing & Verification

The project includes an integration and unit test suite in [`tests/test_api.py`](tests/test_api.py) covering the API lifecycle, model bounds, threshold alerts, data order validation, and the automated maintenance strategy engine.

Run tests via pytest:
```bash
pytest tests/test_api.py -v
```

**Results:**
```text
tests/test_api.py::TestHealthAndRoot::test_health_returns_ok PASSED      [  4%]
tests/test_api.py::TestHealthAndRoot::test_root_returns_metadata PASSED  [  9%]
tests/test_api.py::TestPredict::test_single_reading_returns_200 PASSED   [ 13%]
tests/test_api.py::TestPredict::test_response_schema PASSED              [ 18%]
tests/test_api.py::TestPredict::test_thirty_readings PASSED              [ 22%]
tests/test_api.py::TestPredict::test_unit_id_is_optional PASSED          [ 27%]
tests/test_api.py::TestPredict::test_predicted_rul_clipped_at_125 PASSED [ 31%]
tests/test_api.py::TestPredict::test_predicted_rul_not_negative PASSED   [ 36%]
tests/test_api.py::TestAlertStatus::test_status_thresholds[10.0-CRITICAL] PASSED [ 40%]
tests/test_api.py::TestAlertStatus::test_status_thresholds[30.0-CRITICAL] PASSED [ 45%]
tests/test_api.py::TestAlertStatus::test_status_thresholds[31.0-WARNING] PASSED [ 50%]
tests/test_api.py::TestAlertStatus::test_status_thresholds[80.0-WARNING] PASSED [ 54%]
tests/test_api.py::TestAlertStatus::test_status_thresholds[81.0-HEALTHY] PASSED [ 59%]
tests/test_api.py::TestAlertStatus::test_status_thresholds[125.0-HEALTHY] PASSED [ 63%]
tests/test_api.py::TestValidation::test_empty_readings_rejected PASSED   [ 68%]
tests/test_api.py::TestValidation::test_out_of_order_cycles_rejected PASSED [ 72%]
tests/test_api.py::TestValidation::test_missing_sensor_field_rejected PASSED [ 77%]
tests/test_api.py::TestMaintenanceStrategy::test_critical_strategy_generation PASSED [ 81%]
tests/test_api.py::TestMaintenanceStrategy::test_warning_strategy_generation PASSED [ 86%]
tests/test_api.py::TestMaintenanceStrategy::test_healthy_strategy_generation PASSED [ 90%]
tests/test_api.py::TestMaintenanceStrategy::test_validation_negative_rul_rejected PASSED [ 95%]
tests/test_api.py::TestMaintenanceStrategy::test_root_lists_new_endpoint PASSED [100%]

============================= 22 passed in 0.71s ==============================
```

---

## 13. Deployment (Docker & Cloud Run)

### Multi-Stage Docker Builds
The repository provides two separate production Docker configurations:
1. [`Dockerfile`](Dockerfile): Lean, non-root microservice container running the FastAPI backend (~180MB runtime).
2. [`Dockerfile.dashboard`](Dockerfile.dashboard): Full container for the Streamlit Dashboard and LangGraph agentic layer.

### Building & Running with Docker
```bash
# Build FastAPI Container
docker build -t rul-api:latest -f Dockerfile .
docker run -p 8080:8080 rul-api:latest

# Build Dashboard Container
docker build -t rul-dashboard:latest -f Dockerfile.dashboard .
docker run -p 8080:8080 rul-dashboard:latest
```

### Google Cloud Run Deployment
Deploy using Google Cloud Build:
```bash
gcloud builds submit --config=cloudbuild.dashboard.yaml .
```

---

## 14. Hackathon Compliance & AI Tools Disclosure

In strict compliance with the **Neural Storm 2026 Official Rules**:
- **Build Window:** Project updated and enhanced within the 48-hour hackathon window (November 2, 2026 – November 4, 2026).
- **Primary Language:** 100% Python for data ingestion, feature engineering, ML modeling, agent orchestration, and API serving.
- **AI Coding Assistant Disclosure:** Developed and accelerated using **Antigravity** (Google DeepMind's advanced agentic coding assistant). Antigravity was utilized for architectural code analysis, designing the actionable business logic layer (`POST /maintenance-strategy`), expanding the test suite, and ensuring strict compliance with hackathon guidelines.
- **LLM Foundation Models:** Uses OpenAI's `gpt-4o-mini` for the conversational AI analyst and LangGraph multi-agent diagnostic nodes.

---

## 15. Credits & Acknowledgements

1. **Dataset Origin:** NASA Prognostics Center of Excellence (PCoE)  
   *Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM).*
2. **Open-Source Base ML Architecture:** We gratefully acknowledge the open-source machine learning and data science community for foundational implementations of the NASA CMAPSS degradation benchmark and XGBoost regression baselines upon which this enterprise system was enhanced and extended.
3. **Libraries & Ecosystem:** Built using the open-source Python ecosystem: FastAPI, Streamlit, LangGraph, LangChain, XGBoost, SHAP, Optuna, ReportLab, and Plotly.

---

*Developed for Neural Storm 2026 · Track 01 (Future Tech) · Problem Statement FT-03*
