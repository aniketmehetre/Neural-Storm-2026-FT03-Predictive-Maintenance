# Project Codebase Audit

**Date:** 2026-11-05  
**Branch:** `winning-upgrade`  
**Purpose:** Code-level ground-truth verification of the current implementation before architectural upgrades.

---

## 1. Evaluation Protocol (Test RMSE, MAE, R²)

### Which Rows & Files
- Telemetry file: `CMAPSSData/test_FD001.txt` (100 test engines, 13,096 cycles total).
- Official target file: `CMAPSSData/RUL_FD001.txt` (100 rows, one scalar ground-truth RUL value per engine at the end of its observed test sequence).
- Row selection (`src/agents/tools.py` lines 119–125):
  ```python
  last = (
      result.sort_values(["unit_id", "cycle"])
      .groupby("unit_id")
      .tail(1)
      .sort_values("unit_id")
      .reset_index(drop=True)
  )
  ```
  Only the **single final recorded cycle** of each of the 100 test engines is evaluated (100 rows total).

### Target Transformation & Capping
- Capping rule (`src/agents/tools.py` lines 128–134):
  ```python
  preds = np.clip(model.predict(X_last), 0, RUL_CLIP)  # RUL_CLIP = 125
  rul_true = (
      pd.read_csv(DATA_RAW / "RUL_FD001.txt", header=None, names=["t"])
      ["t"]
      .clip(upper=RUL_CLIP)
      .values
  )
  ```
- Both predicted and true RUL values are clamped to $[0, 125]$.

### Metric Calculations
- RMSE (`src/agents/tools.py` line 154):
  $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{100} (\hat{y}_i - y_i)^2} \approx 16.74\text{ cycles}$$
- MAE (`src/agents/tools.py` line 155):
  $$\text{MAE} = \frac{1}{N}\sum_{i=1}^{100} |\hat{y}_i - y_i| \approx 12.33\text{ cycles}$$
- $R^2$:
  $$R^2 = 1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (y_i - \bar{y})^2} \approx 0.825$$
- **Note on NASA Score:** The official asymmetric NASA PHM score ($S = \sum (\exp(-d/13)-1)$ for $d < 0$ and $\sum (\exp(d/10)-1)$ for $d \ge 0$) was not implemented in the baseline code and will be added in P1.

---

## 2. Hardcoded Strings & Synthetic Confidence

- `src/agents/strategy_agent.py`:
  - **Line 251:** `confidence_score = 94.8 if alert_level == "CRITICAL" else (91.2 if alert_level == "WARNING" else 98.5)` (arbitrary hardcoded confidence percentage).
  - **Line 266:** `agent_confidence=f"{confidence_score:.1f}% (Based on Optuna-tuned XGBoost & CMAPSS FD001 fleet trends)"`.
  - **Line 267:** `agent_signature=f"{self.agent_name} [VERIFIED_MRO_DISPATCH]"` with `self.agent_name = "AeroGuard-PredictiveMaintenanceAgent-v2.6"`.
  - **Lines 28–205:** Aerospace component catalog and SKUs (`HPC-BLD-309X`, `SEN-TP-024`, etc.) are a static simulation catalog rather than real airline procurement integrations.
- `src/api/main.py`:
  - **Line 105 & 214:** Static API version `"1.0.0"` and `"1.1.0"`.

---

## 3. Path Audit (Absolute vs Relative)

- `README.md`:
  - Contained multiple `file:///c:/HACKATHONS/my-hackathon-project/...` Windows-specific absolute hyperlinks. These are broken when rendered on GitHub or non-Windows machines.
- `src/` codebase:
  - Clean: All files correctly use `Path(__file__).resolve().parents[...]` to anchor paths to the repository root.
- Docker & CloudBuild:
  - `Dockerfile` & `Dockerfile.dashboard` use relative paths `/app`.
  - `cloudbuild.dashboard.yaml` contains GCP project ID `project-a480ceee-48d1-45a9-aca`.

---

## 4. Agent Key Requirements & Failure Behavior

1. **`src/agents/chat_agent.py`:**
   - Requires `OPENAI_API_KEY`.
   - Behavior without key: In CLI / standalone usage, `OpenAI()` raises `openai.OpenAIError`. In dashboard, checks `os.getenv("OPENAI_API_KEY")` and renders a warning banner.
2. **`src/agents/fleet_agent.py`:**
   - Requires `OPENAI_API_KEY` (lines 75–78):
     ```python
     if not api_key:
         raise ValueError("OPENAI_API_KEY environment variable is not set.")
     ```
   - Behavior without key: Pipeline fails immediately with an unhandled `ValueError`.
3. **`src/agents/strategy_agent.py`:**
   - 100% offline, deterministic simulated maintenance logic. Requires no external keys.
