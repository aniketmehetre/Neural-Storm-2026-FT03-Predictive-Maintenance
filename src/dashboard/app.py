"""Streamlit Engines Health Monitor — Tesla-style dark theme.

Tabs
----
1. Fleet Overview   — KPIs, fleet RUL bar chart, status breakdown, attention table
2. Engine Analysis  — per-engine degradation curve + SHAP feature explanation
3. Live Prediction  — select any test engine, tweak sensors, get instant RUL
4. AI Analyst       — conversational tool-calling agent (requires OPENAI_API_KEY)

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env.config (or .env) from project root before anything else reads env vars
_ROOT_ENV = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT_ENV / ".env.config", override=False)
load_dotenv(_ROOT_ENV / ".env",        override=False)

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src.features import build_rolling_features, readings_to_feature_row
from src.agents.strategy_agent import SimulatedMaintenanceAgent

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
DATA_RAW   = ROOT / "CMAPSSData"
DATA_PROC  = ROOT / "data"
MODELS_DIR = ROOT / "models"

RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_set_{i}" for i in range(1, 4)]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)

SENSOR_LABELS: dict[str, str] = {
    # Source: Saxena et al. (2008) — CMAPSS variable definitions
    "sensor_02": "LPC outlet temperature (T24, °R)",
    "sensor_03": "HPC outlet temperature (T30, °R)",
    "sensor_04": "LPT outlet temperature (T50, °R)",
    "sensor_06": "Bypass-duct total pressure (P15, psia)",
    "sensor_07": "HPC outlet total pressure (P30, psia)",
    "sensor_08": "Physical fan speed (Nf, rpm)",
    "sensor_09": "Physical core speed (Nc, rpm)",
    "sensor_11": "HPC outlet static pressure (Ps30, psia)",
    "sensor_12": "Fuel flow / Ps30 ratio (phi)",
    "sensor_13": "Corrected fan speed (NRf, rpm)",
    "sensor_14": "Corrected core speed (NRc, rpm)",
    "sensor_15": "Bypass ratio (BPR)",
    "sensor_17": "Bleed enthalpy (htBleed)",
    "sensor_20": "HPT coolant bleed (W31)",
    "sensor_21": "LPT coolant bleed (W32)",
}

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_PRIMARY   = "#0a0e1a"
BG_CARD      = "#111827"
BG_CARD2     = "#1a2235"
BLUE_LIGHT   = "#3b82f6"
BLUE_GLOW    = "#60a5fa"
RED_LIGHT    = "#ef4444"
YELLOW_LIGHT = "#f59e0b"
GREEN_LIGHT  = "#10b981"
TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED   = "#94a3b8"
BORDER       = "#1e293b"

RUL_CLIP       = 125
ALERT_CRITICAL = 30
ALERT_WARNING  = 80
WINDOW         = 30

STATUS_COLOR = {"CRITICAL": RED_LIGHT, "WARNING": YELLOW_LIGHT, "HEALTHY": GREEN_LIGHT}
STATUS_EMOJI = {"CRITICAL": "🔴", "WARNING": "🟡", "HEALTHY": "🟢"}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    margin=dict(t=30, b=30, l=10, r=10),
)
_LEG = dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(color=TEXT_PRIMARY))


# ── CSS ───────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: {BG_PRIMARY}; color: {TEXT_PRIMARY}; }}
    section[data-testid="stSidebar"] {{ background:{BG_CARD}; border-right:1px solid {BORDER}; }}
    [data-testid="stMetric"] {{
        background:{BG_CARD}; border:1px solid {BORDER};
        border-radius:12px; padding:20px 24px;
    }}
    [data-testid="stMetricLabel"] {{
        color:{TEXT_MUTED} !important; font-size:0.75rem !important;
        font-weight:500 !important; letter-spacing:0.08em !important; text-transform:uppercase;
    }}
    [data-testid="stMetricValue"] {{
        color:{TEXT_PRIMARY} !important; font-size:2rem !important; font-weight:700 !important;
    }}
    h1,h2,h3 {{ color:{TEXT_PRIMARY} !important; }}
    h3 {{ font-size:0.95rem !important; text-transform:uppercase;
           letter-spacing:0.06em; color:{TEXT_MUTED} !important; }}
    hr {{ border-color:{BORDER} !important; margin:1.5rem 0 !important; }}
    [data-testid="stSelectbox"] > div > div {{
        background:{BG_CARD} !important; border:1px solid {BORDER} !important;
        color:{TEXT_PRIMARY} !important; border-radius:8px !important;
    }}
    [data-testid="stNumberInput"] input {{
        background:{BG_CARD2} !important; color:{TEXT_PRIMARY} !important;
        border:1px solid {BORDER} !important; border-radius:6px !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background:{BG_CARD}; border-radius:10px; padding:4px; gap:4px;
        border:1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background:transparent; color:{TEXT_MUTED}; border-radius:7px;
        padding:8px 20px; font-weight:500; font-size:0.88rem;
    }}
    .stTabs [aria-selected="true"] {{
        background:{BLUE_LIGHT} !important; color:white !important;
    }}
    .stButton > button {{
        background:{BLUE_LIGHT}; color:white; border:none;
        border-radius:8px; font-weight:600; padding:10px 24px;
        font-size:0.9rem; width:100%;
    }}
    .stButton > button:hover {{ background:{BLUE_GLOW}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color:{TEXT_MUTED} !important; font-size:0.78rem !important;
    }}
    [data-testid="stChatMessage"] {{
        background:{BG_CARD2} !important; border:1px solid {BORDER} !important;
        border-radius:10px !important; margin-bottom:8px !important;
    }}
    [data-testid="stChatInputContainer"] {{
        background:{BG_CARD} !important; border-top:1px solid {BORDER} !important;
    }}
    </style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _status(rul: float) -> str:
    if rul <= ALERT_CRITICAL: return "CRITICAL"
    if rul <= ALERT_WARNING:  return "WARNING"
    return "HEALTHY"


_STAT_SUFFIX: dict[str, str] = {
    "_mean":  "recent average",
    "_std":   "how much it fluctuates",
    "_min":   "lowest recent reading",
    "_max":   "peak recent reading",
    "_lag1":  "value last cycle",
    "_trend": "rate of change",
}


def _feature_label(feat: str) -> str:
    """Convert 'sensor_11_max' → 'HPC outlet static pressure (peak recent reading)'."""
    for key, name in SENSOR_LABELS.items():
        if feat.startswith(key):
            suffix_raw      = feat[len(key):]
            suffix_readable = _STAT_SUFFIX.get(suffix_raw, suffix_raw)
            short           = name.split("(")[0].strip()
            return f"{short}  ({suffix_readable})"
    if feat.startswith("op_set_"):
        return f"Operating condition {feat[-1]}"
    return feat


def _shap_glossary() -> None:
    """Render an expandable plain-English glossary below a SHAP chart."""
    with st.expander("📖  What do these labels mean?"):
        st.markdown(
            f'<div style="font-size:0.82rem;color:{TEXT_MUTED};line-height:1.8;">'
            f'<b style="color:{TEXT_PRIMARY};">Sensor names</b> come from the NASA CMAPSS '
            f'turbofan engine dataset (Saxena et al., 2008). The model analyses each '
            f"sensor's behaviour over the <b>last 30 operating cycles</b> and extracts "
            f"six statistical snapshots per sensor:"
            f"</div>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        measures = [
            ("Recent average", "Typical sensor level over the last 30 cycles — "
             "smooths out noise to show the overall trend"),
            ("Peak recent reading", "Highest value recorded in the last 30 cycles — "
             "catches dangerous spikes"),
            ("Lowest recent reading", "Minimum value in the window — "
             "useful for detecting pressure drops or speed loss"),
            ("How much it fluctuates", "Standard deviation — a high value means the "
             "sensor is unstable, which often signals wear"),
            ("Rate of change", "Is the sensor getting worse or better? "
             "A negative trend in pressure or speed is a degradation warning"),
            ("Value last cycle", "The reading from the immediately previous cycle — "
             "captures sudden short-term shifts"),
        ]
        for i, (term, desc) in enumerate(measures):
            col = col_a if i % 2 == 0 else col_b
            col.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER};'
                f'border-radius:8px;padding:10px 12px;margin-bottom:8px;">'
                f'<div style="color:{BLUE_GLOW};font-weight:600;font-size:0.8rem;">'
                f'{term}</div>'
                f'<div style="color:{TEXT_MUTED};font-size:0.76rem;margin-top:2px;">'
                f'{desc}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.8rem;color:{TEXT_MUTED};line-height:1.7;">'
            f'<b style="color:{TEXT_PRIMARY};">Key sensors you will often see:</b><br>'
            f'• <b style="color:{TEXT_PRIMARY};">HPC outlet static pressure</b> — pressure '
            f'exiting the High-Pressure Compressor. Drops as the compressor fouls or blades wear.<br>'
            f'• <b style="color:{TEXT_PRIMARY};">Physical / Corrected core speed</b> — how fast '
            f'the engine core rotates. Slowing down = power loss.<br>'
            f'• <b style="color:{TEXT_PRIMARY};">LPT / HPC outlet temperature</b> — exhaust heat. '
            f'Rising temperature = the engine is working harder to produce the same thrust.<br>'
            f'• <b style="color:{TEXT_PRIMARY};">Fuel flow / Ps30 ratio</b> — fuel efficiency. '
            f'A rising ratio means the engine burns more fuel for the same pressure output.<br>'
            f'• <b style="color:{TEXT_PRIMARY};">Bypass ratio</b> — share of air bypassing the '
            f'combustion core. Shifts with fan blade health.'
            f'</div>',
            unsafe_allow_html=True,
        )


def _shap_legend_html(base: float, pred: float, color: str) -> str:
    """Return a compact HTML explainer for SHAP bar charts."""
    return (
        f'<div style="background:#0f1929;border:1px solid #1e293b;border-radius:8px;'
        f'padding:14px 18px;margin-top:4px;font-size:0.78rem;line-height:1.7;">'
        f'<div style="display:flex;gap:28px;flex-wrap:wrap;margin-bottom:8px;">'
        f'<span>📍 <b style="color:#94a3b8;">Base value</b> '
        f'<b style="color:#60a5fa;">{base:.1f} cycles</b> — '
        f'average prediction across all training engines</span>'
        f'<span>🎯 <b style="color:#94a3b8;">Final prediction</b> '
        f'<b style="color:{color};">{pred:.1f} cycles</b></span>'
        f'</div>'
        f'<div style="display:flex;gap:20px;">'
        f'<span>🔵 <b style="color:#3b82f6;">Blue bar</b> — '
        f'this sensor reading is <b>healthier than usual</b>, adding cycles to the estimate</span>'
        f'<span>🔴 <b style="color:#ef4444;">Red bar</b> — '
        f'this sensor reading is <b>worse than usual</b>, reducing the estimate</span>'
        f'</div>'
        f'<div style="margin-top:6px;color:#64748b;font-size:0.72rem;">'
        f'Bar length = how many cycles each sensor contributes or removes from the prediction. '
        f'Features sorted by absolute impact (most influential at top).'
        f'</div>'
        f'</div>'
    )


def _card(html: str) -> None:
    st.markdown(
        f'<div style="background:{BG_CARD2};border:1px solid {BORDER};'
        f'border-radius:12px;padding:20px;">{html}</div>',
        unsafe_allow_html=True,
    )


def _section(title: str) -> None:
    st.markdown(f"### {title}")


# ── Cached data loaders ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _load_artifacts():
    model     = joblib.load(MODELS_DIR / "xgb_rul.joblib")
    meta      = json.loads((MODELS_DIR / "model_meta.json").read_text())
    eda_meta  = json.loads((DATA_PROC  / "eda_meta.json").read_text())
    feat_meta = json.loads((DATA_PROC  / "feature_meta.json").read_text())
    explainer = shap.TreeExplainer(model)
    return model, meta, eda_meta, feat_meta, explainer


@st.cache_data(show_spinner="Computing fleet predictions…")
def _fleet_predictions() -> pd.DataFrame:
    model, _, eda_meta, feat_meta, _ = _load_artifacts()
    sensors  = eda_meta["useful_sensors"]
    drop     = eda_meta["constant_sensors"]
    feat_cols = feat_meta["feature_cols"]

    df = (pd.read_csv(DATA_RAW / "test_FD001.txt",
                      sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
          .drop(columns=drop)
          .sort_values(["unit_id", "cycle"]).reset_index(drop=True))

    feat_df = pd.concat([
        df.groupby("unit_id", group_keys=False)
          .apply(lambda g, s=s: build_rolling_features(g.sort_values("cycle"), [s], WINDOW))
        for s in sensors
    ], axis=1)

    result = pd.concat(
        [df[["unit_id", "cycle"]].reset_index(drop=True),
         feat_df.reset_index(drop=True)], axis=1)

    last = (result.sort_values(["unit_id", "cycle"])
                  .groupby("unit_id").tail(1)
                  .sort_values("unit_id").reset_index(drop=True))

    preds    = np.clip(model.predict(last[feat_cols].values), 0, RUL_CLIP)
    rul_true = (pd.read_csv(DATA_RAW / "RUL_FD001.txt", header=None, names=["t"])
                ["t"].clip(upper=RUL_CLIP).values)

    last = last[["unit_id", "cycle"]].copy()
    last["predicted_rul"] = preds.round(1)
    last["true_rul"]      = rul_true
    last["error"]         = (last["predicted_rul"] - last["true_rul"]).round(1)
    last["status"]        = last["predicted_rul"].apply(_status)
    return last


@st.cache_data(show_spinner="Loading degradation curves…")
def _full_test() -> pd.DataFrame:
    model, _, eda_meta, feat_meta, _ = _load_artifacts()
    sensors  = eda_meta["useful_sensors"]
    drop     = eda_meta["constant_sensors"]
    feat_cols = feat_meta["feature_cols"]

    df = (pd.read_csv(DATA_RAW / "test_FD001.txt",
                      sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
          .drop(columns=drop)
          .sort_values(["unit_id", "cycle"]).reset_index(drop=True))

    feat_df = pd.concat([
        df.groupby("unit_id", group_keys=False)
          .apply(lambda g, s=s: build_rolling_features(g.sort_values("cycle"), [s], WINDOW))
        for s in sensors
    ], axis=1)

    result = pd.concat(
        [df[["unit_id", "cycle"]].reset_index(drop=True),
         feat_df.reset_index(drop=True)], axis=1)
    result["predicted_rul"] = np.clip(
        model.predict(result[feat_cols].values), 0, RUL_CLIP).round(1)
    result["status"] = result["predicted_rul"].apply(_status)
    return result


@st.cache_data(show_spinner="Loading raw test readings…")
def _raw_test() -> pd.DataFrame:
    _, _, eda_meta, _, _ = _load_artifacts()
    drop = eda_meta["constant_sensors"]
    return (pd.read_csv(DATA_RAW / "test_FD001.txt",
                        sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
            .drop(columns=drop)
            .sort_values(["unit_id", "cycle"]).reset_index(drop=True))


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engines Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_css()

model, model_meta, eda_meta, feat_meta, explainer = _load_artifacts()
fleet_df  = _fleet_predictions()
full_df   = _full_test()
raw_test  = _raw_test()

n_crit = int((fleet_df["status"] == "CRITICAL").sum())
n_warn = int((fleet_df["status"] == "WARNING").sum())
n_ok   = int((fleet_df["status"] == "HEALTHY").sum())
rmse   = float(np.sqrt(((fleet_df["predicted_rul"] - fleet_df["true_rul"]) ** 2).mean()))

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
  <div>
    <span style="font-size:1.6rem;font-weight:700;color:{TEXT_PRIMARY};letter-spacing:-0.02em;">
      ⚙️ Engines Health Monitor
    </span><br>
    <span style="font-size:0.82rem;color:{TEXT_MUTED};">
      Predictive Maintenance · NASA CMAPSS FD001 · XGBoost (Optuna tuned)
    </span>
  </div>
  <div style="text-align:right;">
    <span style="font-size:0.78rem;color:{TEXT_MUTED};">MODEL PERFORMANCE</span><br>
    <span style="font-size:0.9rem;color:{BLUE_GLOW};font-weight:600;">
      Val RMSE {model_meta['val_metrics']['rmse']:.2f} &nbsp;·&nbsp;
      Test RMSE {model_meta['test_metrics']['rmse']:.2f} cycles
    </span>
  </div>
</div><hr>""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("TOTAL ENGINES", len(fleet_df))
k2.metric("🔴 CRITICAL",   n_crit)
k3.metric("🟡 WARNING",    n_warn)
k4.metric("🟢 HEALTHY",    n_ok)
k5.metric("FLEET RMSE",    f"{rmse:.1f} cyc")

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar — Fleet Diagnostic Agent ──────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="color:{BLUE_GLOW};font-size:1rem;font-weight:700;'
        f'margin-bottom:0.5rem;">🚁 Autonomous Fleet Diagnostic</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{TEXT_MUTED};font-size:0.78rem;margin-bottom:1rem;">'
        "Runs a 4-stage LangGraph agent pipeline:<br>"
        "Sensor Analyst → Diagnostics → Prioritizer → Report Writer"
        "</div>",
        unsafe_allow_html=True,
    )
    run_agent_btn = st.button("▶  Run Fleet Diagnostic", key="run_agent")

    if run_agent_btn:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Set OPENAI_API_KEY environment variable first.")
        else:
            with st.spinner("Running 4-agent LangGraph pipeline…"):
                try:
                    from src.agents.fleet_agent import FleetMonitoringAgent
                    result = FleetMonitoringAgent().run()
                    st.session_state["fleet_report"] = result["final_report"]
                    st.session_state["fleet_agent_log"] = [
                        m.content for m in result["messages"]
                    ]
                    st.session_state.pop("fleet_pdf_bytes", None)  # force PDF regeneration
                    st.success("Diagnostic complete!")
                except Exception as exc:
                    st.error(f"Agent error: {exc}")

    if st.session_state.get("fleet_report"):
        with st.expander("📋 Latest Fleet Report", expanded=True):
            st.markdown(st.session_state["fleet_report"])

        # ── PDF download ───────────────────────────────────────────────────────
        try:
            import importlib
            import src.agents.report_pdf as _pdf_mod
            importlib.reload(_pdf_mod)          # always use latest code on disk
            from src.agents.tools import get_fleet_summary

            _report_hash = hash(st.session_state["fleet_report"])
            if st.session_state.get("_pdf_report_hash") != _report_hash:
                st.session_state.pop("fleet_pdf_bytes", None)

            if "fleet_pdf_bytes" not in st.session_state:
                with st.spinner("Generating PDF…"):
                    st.session_state["fleet_pdf_bytes"] = _pdf_mod.build_pdf(
                        markdown_report=st.session_state["fleet_report"],
                        fleet_stats=get_fleet_summary(),
                    )
                    st.session_state["_pdf_report_hash"] = _report_hash

            ts_label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
            st.download_button(
                label="⬇  Download PDF Report",
                data=st.session_state["fleet_pdf_bytes"],
                file_name=f"fleet_maintenance_briefing_{ts_label}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _pdf_exc:
            st.warning(f"PDF generation failed: {_pdf_exc}")

        if st.session_state.get("fleet_agent_log"):
            with st.expander("🔍 Agent Trace"):
                for msg in st.session_state["fleet_agent_log"]:
                    st.markdown(
                        f'<div style="color:{TEXT_MUTED};font-size:0.75rem;'
                        f'margin-bottom:4px;">{msg}</div>',
                        unsafe_allow_html=True,
                    )
        if st.button("🗑  Clear Report", key="clear_report"):
            for _k in ("fleet_report", "fleet_agent_log", "fleet_pdf_bytes"):
                st.session_state.pop(_k, None)
            st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭  Fleet Overview",
    "🔍  Engine Analysis",
    "⚡  Live Prediction",
    "🤖  AI Analyst",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FLEET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Platform guide ─────────────────────────────────────────────────────────
    capabilities = [
        ("🏭", "Fleet Overview",
         "Real-time RUL predictions for all 100 engines. Bar chart, status donut, "
         "attention table, and predicted-vs-true accuracy scatter."),
        ("🔍", "Engine Analysis",
         "Select any engine to inspect its full degradation curve and a SHAP "
         "waterfall explaining <i>why</i> the model gave that prediction."),
        ("⚡", "Live Prediction",
         "Auto-fill sensor values from any test engine, tweak individual readings, "
         "and get an instant RUL estimate with its own SHAP breakdown."),
        ("🤖", "AI Analyst",
         "Chat with a GPT-4o-mini agent that calls live tools: fleet summary, "
         "per-engine details, SHAP explanations, and fleet statistics. "
         "Requires <code>OPENAI_API_KEY</code>."),
        ("🚁", "Fleet Diagnostic (sidebar)",
         "One-click LangGraph pipeline: Sensor Analyst → Diagnostics → "
         "Prioritizer → Report Writer. Produces a full Markdown maintenance "
         "briefing in ~30 s."),
    ]

    cap_html = "".join(f"""
      <div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:14px;">
        <div style="font-size:1.4rem;flex-shrink:0;margin-top:2px;">{icon}</div>
        <div>
          <div style="color:{TEXT_PRIMARY};font-weight:600;font-size:0.88rem;
                      margin-bottom:2px;">{title}</div>
          <div style="color:{TEXT_MUTED};font-size:0.80rem;line-height:1.5;">
            {desc}</div>
        </div>
      </div>""" for icon, title, desc in capabilities)

    quick_steps = [
        ("1", "Start here", "Scan the KPI cards and the bar chart above. "
         "Red bars = engines that need immediate action."),
        ("2", "Drill into an engine", "Switch to <b>Engine Analysis</b>, pick any "
         "engine from the drop-down, and read the SHAP chart to understand the root cause."),
        ("3", "Simulate a scenario", "Go to <b>Live Prediction</b>, choose an engine, "
         "adjust any sensor, and hit <b>Predict RUL</b> to see the impact instantly."),
        ("4", "Ask the AI", "Open <b>AI Analyst</b> and type a plain-English question "
         "such as <i>\"Which engines fail within 20 cycles?\"</i> or "
         "<i>\"Why is engine 42 critical?\"</i>"),
        ("5", "Full diagnostic", "Click <b>▶ Run Fleet Diagnostic</b> in the sidebar "
         "for an autonomous 4-agent report with a prioritised maintenance schedule."),
    ]

    steps_html = "".join(f"""
      <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:12px;">
        <div style="background:{BLUE_LIGHT};color:white;border-radius:50%;
                    width:22px;height:22px;display:flex;align-items:center;
                    justify-content:center;font-size:0.72rem;font-weight:700;
                    flex-shrink:0;margin-top:1px;">{n}</div>
        <div>
          <span style="color:{TEXT_PRIMARY};font-weight:600;font-size:0.82rem;">
            {step} — </span>
          <span style="color:{TEXT_MUTED};font-size:0.82rem;line-height:1.5;">
            {desc}</span>
        </div>
      </div>""" for n, step, desc in quick_steps)

    guide_col, steps_col = st.columns(2, gap="large")

    with guide_col:
        st.markdown(
            f'<div style="background:{BG_CARD2};border:1px solid {BORDER};'
            f'border-radius:12px;padding:20px 22px;margin-bottom:20px;">'
            f'<div style="color:{BLUE_GLOW};font-size:0.72rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px;">'
            f'Platform Capabilities</div>'
            f'{cap_html}</div>',
            unsafe_allow_html=True,
        )

    with steps_col:
        st.markdown(
            f'<div style="background:{BG_CARD2};border:1px solid {BORDER};'
            f'border-radius:12px;padding:20px 22px;margin-bottom:20px;">'
            f'<div style="color:{BLUE_GLOW};font-size:0.72rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px;">'
            f'Quick Start Guide</div>'
            f'{steps_html}'
            f'<div style="margin-top:16px;padding-top:14px;border-top:1px solid {BORDER};'
            f'color:{TEXT_MUTED};font-size:0.75rem;">'
            f'🔴 CRITICAL ≤ 30 cycles &nbsp;·&nbsp; '
            f'🟡 WARNING ≤ 80 cycles &nbsp;·&nbsp; '
            f'🟢 HEALTHY &gt; 80 cycles'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    col_bar, col_pie = st.columns([3, 1], gap="large")

    with col_bar:
        _section("Fleet RUL — All 100 Engines")
        fig = go.Figure()
        for status, color in STATUS_COLOR.items():
            sub = fleet_df[fleet_df["status"] == status].sort_values("unit_id")
            fig.add_trace(go.Bar(
                x=sub["unit_id"], y=sub["predicted_rul"],
                name=status, marker_color=color,
                hovertemplate="<b>Engine %{x}</b><br>Pred RUL: %{y:.0f} cycles<extra></extra>",
            ))
        fig.add_hline(y=ALERT_CRITICAL, line_dash="dot", line_color=RED_LIGHT,
                      line_width=1.5, annotation_text="CRITICAL",
                      annotation_font_color=RED_LIGHT, annotation_font_size=10)
        fig.add_hline(y=ALERT_WARNING,  line_dash="dot", line_color=YELLOW_LIGHT,
                      line_width=1.5, annotation_text="WARNING",
                      annotation_font_color=YELLOW_LIGHT, annotation_font_size=10)
        fig.update_layout(**PLOTLY_BASE, barmode="overlay", height=340,
                          xaxis_title="Engine ID", yaxis_title="Predicted RUL (cycles)",
                          legend={**_LEG, "orientation": "h",
                                  "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1})
        st.plotly_chart(fig, use_container_width=True)

    with col_pie:
        _section("Status Breakdown")
        fig2 = go.Figure(go.Pie(
            labels=["CRITICAL", "WARNING", "HEALTHY"],
            values=[n_crit, n_warn, n_ok],
            marker_colors=[RED_LIGHT, YELLOW_LIGHT, GREEN_LIGHT],
            hole=0.55, textinfo="percent",
            textfont=dict(size=12, color=TEXT_PRIMARY),
        ))
        fig2.add_annotation(text=f"<b>{len(fleet_df)}</b>", x=0.5, y=0.5,
                            showarrow=False, font=dict(size=18, color=TEXT_PRIMARY))
        fig2.update_layout(**PLOTLY_BASE, height=340, showlegend=True,
                           legend={**_LEG, "orientation": "h",
                                   "yanchor": "top", "y": -0.05})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Attention table
    attention = fleet_df[fleet_df["status"] != "HEALTHY"].sort_values("predicted_rul")
    col_t, col_i = st.columns([2, 1], gap="large")

    with col_t:
        _section(f"Engines Requiring Attention ({len(attention)})")
        if attention.empty:
            st.success("All engines are operating within healthy parameters.")
        else:
            disp = attention[["unit_id","predicted_rul","true_rul","error","status"]].copy()
            disp.columns = ["Engine","Pred RUL","True RUL","Error (cyc)","Status"]
            disp["Status"] = disp["Status"].map(lambda s: f"{STATUS_EMOJI[s]} {s}")
            st.dataframe(disp, use_container_width=True, hide_index=True,
                         column_config={
                             "Pred RUL":    st.column_config.NumberColumn(format="%.0f"),
                             "True RUL":    st.column_config.NumberColumn(format="%.0f"),
                             "Error (cyc)": st.column_config.NumberColumn(format="%.1f"),
                         })

    with col_i:
        _section("Alert Thresholds")
        _card(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
          <div style="width:10px;height:10px;border-radius:50%;
                      background:{RED_LIGHT};flex-shrink:0;"></div>
          <div>
            <div style="color:{TEXT_PRIMARY};font-weight:600;">CRITICAL</div>
            <div style="color:{TEXT_MUTED};font-size:0.78rem;">
              RUL ≤ {ALERT_CRITICAL} cycles — schedule maintenance now</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
          <div style="width:10px;height:10px;border-radius:50%;
                      background:{YELLOW_LIGHT};flex-shrink:0;"></div>
          <div>
            <div style="color:{TEXT_PRIMARY};font-weight:600;">WARNING</div>
            <div style="color:{TEXT_MUTED};font-size:0.78rem;">
              RUL {ALERT_CRITICAL}–{ALERT_WARNING} cycles — monitor closely</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
          <div style="width:10px;height:10px;border-radius:50%;
                      background:{GREEN_LIGHT};flex-shrink:0;"></div>
          <div>
            <div style="color:{TEXT_PRIMARY};font-weight:600;">HEALTHY</div>
            <div style="color:{TEXT_MUTED};font-size:0.78rem;">
              RUL > {ALERT_WARNING} cycles — operating normally</div>
          </div>
        </div>""")

    # Accuracy scatter
    st.markdown("<hr>", unsafe_allow_html=True)
    _section("Model Accuracy — Predicted vs True RUL")
    fig3 = go.Figure()
    for status, color in STATUS_COLOR.items():
        sub = fleet_df[fleet_df["status"] == status]
        fig3.add_trace(go.Scatter(
            x=sub["true_rul"], y=sub["predicted_rul"], mode="markers",
            name=status, marker=dict(color=color, size=9, opacity=0.85,
                                     line=dict(color=BG_CARD, width=1)),
            hovertemplate="<b>Engine %{customdata}</b><br>"
                          "True: %{x:.0f} · Pred: %{y:.0f}<extra></extra>",
            customdata=sub["unit_id"],
        ))
    lim = int(fleet_df[["true_rul","predicted_rul"]].max().max()) + 8
    fig3.add_trace(go.Scatter(x=[0,lim], y=[0,lim], mode="lines",
                              line=dict(color=TEXT_MUTED, dash="dash", width=1),
                              name="Perfect prediction", hoverinfo="skip"))
    fig3.update_layout(**PLOTLY_BASE, height=380, xaxis_title="True RUL (cycles)",
                       yaxis_title="Predicted RUL (cycles)",
                       legend={**_LEG, "orientation": "h",
                               "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1})
    st.plotly_chart(fig3, use_container_width=True)
    mae = float((fleet_df["predicted_rul"] - fleet_df["true_rul"]).abs().mean())
    st.markdown(
        f'<div style="color:{TEXT_MUTED};font-size:0.78rem;text-align:center;margin-top:-12px;">'
        f'Test RMSE = <b style="color:{BLUE_GLOW}">{rmse:.2f} cycles</b> &nbsp;·&nbsp;'
        f'MAE = <b style="color:{BLUE_GLOW}">{mae:.2f} cycles</b> &nbsp;·&nbsp;'
        f'Above diagonal = overestimated (safe) · Below = underestimated (risky)'
        f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ENGINE ANALYSIS + SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_sel, _ = st.columns([2, 3])
    with col_sel:
        selected = st.selectbox(
            "Select engine",
            options=sorted(fleet_df["unit_id"].tolist()),
            format_func=lambda uid: (
                f"Engine {uid:>3}  ·  "
                f"{STATUS_EMOJI[fleet_df.loc[fleet_df['unit_id']==uid,'status'].values[0]]}  "
                f"{fleet_df.loc[fleet_df['unit_id']==uid,'status'].values[0]}  "
                f"— Pred RUL: "
                f"{fleet_df.loc[fleet_df['unit_id']==uid,'predicted_rul'].values[0]:.0f} cyc"
            ),
            label_visibility="collapsed",
            key="tab2_engine",
        )

    eng_row   = fleet_df[fleet_df["unit_id"] == selected].iloc[0]
    eng_full  = full_df[full_df["unit_id"] == selected].sort_values("cycle")
    s_color   = STATUS_COLOR[eng_row["status"]]

    # Engine summary bar
    st.markdown(f"""
    <div style="background:{BG_CARD2};border:1px solid {s_color};border-radius:10px;
                padding:14px 20px;margin-bottom:16px;display:flex;gap:40px;align-items:center;">
      <div><div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;">
           Engine</div>
           <div style="color:{TEXT_PRIMARY};font-size:1.4rem;font-weight:700;">#{selected}</div></div>
      <div><div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;">
           Predicted RUL</div>
           <div style="color:{s_color};font-size:1.4rem;font-weight:700;">
           {eng_row['predicted_rul']:.0f} cycles</div></div>
      <div><div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;">
           True RUL</div>
           <div style="color:{TEXT_PRIMARY};font-size:1.4rem;font-weight:700;">
           {eng_row['true_rul']:.0f} cycles</div></div>
      <div><div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;">
           Status</div>
           <div style="color:{s_color};font-size:1.2rem;font-weight:700;">
           {STATUS_EMOJI[eng_row['status']]} {eng_row['status']}</div></div>
    </div>""", unsafe_allow_html=True)

    # Degradation curve
    col_deg, col_shap = st.columns([1, 1], gap="large")

    with col_deg:
        _section("Degradation Curve")
        fig_d = go.Figure()
        fig_d.add_hrect(y0=0, y1=ALERT_CRITICAL,
                        fillcolor=RED_LIGHT, opacity=0.07, line_width=0)
        fig_d.add_hrect(y0=ALERT_CRITICAL, y1=ALERT_WARNING,
                        fillcolor=YELLOW_LIGHT, opacity=0.07, line_width=0)
        fig_d.add_trace(go.Scatter(
            x=eng_full["cycle"], y=eng_full["predicted_rul"], mode="lines",
            line=dict(color=BLUE_LIGHT, width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            hovertemplate="Cycle %{x}<br>RUL: %{y:.0f}<extra></extra>",
        ))
        fig_d.add_hline(y=ALERT_CRITICAL, line_dash="dot", line_color=RED_LIGHT,
                        line_width=1.2, annotation_text="CRITICAL",
                        annotation_font_color=RED_LIGHT, annotation_font_size=10)
        fig_d.add_hline(y=ALERT_WARNING, line_dash="dot", line_color=YELLOW_LIGHT,
                        line_width=1.2, annotation_text="WARNING",
                        annotation_font_color=YELLOW_LIGHT, annotation_font_size=10)
        fig_d.update_layout(**PLOTLY_BASE, height=360,
                            xaxis_title="Operating Cycle",
                            yaxis_title="Predicted RUL (cycles)", showlegend=False)
        st.plotly_chart(fig_d, use_container_width=True)

    with col_shap:
        _section("SHAP — Why this prediction?")

        feat_cols   = feat_meta["feature_cols"]
        last_row_df = eng_full.iloc[[-1]][feat_cols]
        shap_vals   = explainer(last_row_df)
        sv          = shap_vals.values[0]
        base_val    = float(shap_vals.base_values[0])
        pred_val    = float(eng_row["predicted_rul"])

        top_n   = 12
        indices = np.argsort(np.abs(sv))[::-1][:top_n]
        raw_names     = [feat_cols[i] for i in indices]
        readable_names = [_feature_label(n) for n in raw_names]
        values        = [sv[i] for i in indices]
        colors        = [BLUE_LIGHT if v > 0 else RED_LIGHT for v in values]

        # Hover shows both the readable label and the raw feature code
        hover_texts = [
            f"<b>{rl}</b><br>"
            f"<span style='color:#94a3b8;font-size:11px;'>{rn}</span><br>"
            f"SHAP: {v:+.2f} cycles"
            for rl, rn, v in zip(readable_names, raw_names, values)
        ]

        fig_sh = go.Figure(go.Bar(
            x=values[::-1],
            y=readable_names[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:+.1f}" for v in values[::-1]],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_MUTED),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts[::-1],
        ))
        fig_sh.add_vline(x=0, line_color=TEXT_MUTED, line_width=1)
        fig_sh.update_layout(
            **PLOTLY_BASE, height=390,
            xaxis_title="Cycles added (+) or removed (−) from the prediction",
            yaxis_title=None, showlegend=False,
        )
        fig_sh.update_yaxes(tickfont=dict(size=11))
        st.plotly_chart(fig_sh, use_container_width=True)
        st.markdown(_shap_legend_html(base_val, pred_val, s_color), unsafe_allow_html=True)
        _shap_glossary()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f'<p style="color:{TEXT_MUTED};font-size:0.88rem;margin-bottom:1.2rem;">'
        "Select an engine to auto-fill its latest sensor readings, "
        "then adjust any value and click Predict to get an instant RUL estimate."
        "</p>", unsafe_allow_html=True)

    sensors = eda_meta["useful_sensors"]

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        _section("Sensor Inputs")

        sel_engine = st.selectbox(
            "Auto-fill from test engine",
            options=sorted(raw_test["unit_id"].unique().tolist()),
            format_func=lambda uid: f"Engine {uid}",
            key="live_engine",
        )

        # Get last 30 cycles of selected engine for auto-fill
        eng_raw = (raw_test[raw_test["unit_id"] == sel_engine]
                   .sort_values("cycle").tail(WINDOW))
        last_values = eng_raw.iloc[-1]

        st.markdown(
            f'<div style="color:{TEXT_MUTED};font-size:0.78rem;margin-bottom:12px;">'
            f"Showing last observed cycle: {int(last_values['cycle'])}"
            f"</div>", unsafe_allow_html=True)

        # Two-column grid for sensor inputs
        input_vals: dict[str, float] = {}
        sensor_items = [(s, SENSOR_LABELS.get(s, s)) for s in sensors]
        cols_a, cols_b = st.columns(2)
        for i, (sensor, label) in enumerate(sensor_items):
            default = float(last_values[sensor])
            col = cols_a if i % 2 == 0 else cols_b
            input_vals[sensor] = col.number_input(
                label=label,
                value=default,
                format="%.4f",
                key=f"live_{sensor}",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡  Predict RUL", key="predict_btn")

    with col_result:
        _section("Prediction Result")

        if predict_clicked:
            # Build the 30-cycle history: use last 29 cycles from raw data
            # + the user-edited final cycle
            history = eng_raw.head(WINDOW - 1)
            history_dicts = [
                {"cycle": int(row["cycle"]),
                 **{s: float(row[s]) for s in sensors}}
                for _, row in history.iterrows()
            ]
            last_cycle = int(last_values["cycle"])
            edited_reading = {"cycle": last_cycle, **input_vals}
            history_dicts.append(edited_reading)

            feat_vec = readings_to_feature_row(
                cycle_readings=history_dicts,
                sensors=sensors,
                feature_cols=feat_meta["feature_cols"],
                window=WINDOW,
            )
            raw_pred = float(model.predict(feat_vec.reshape(1, -1))[0])
            pred_rul = round(float(np.clip(raw_pred, 0, RUL_CLIP)), 1)
            status   = _status(pred_rul)
            s_color  = STATUS_COLOR[status]

            # Big result card
            pct = pred_rul / RUL_CLIP
            st.markdown(f"""
            <div style="background:{BG_CARD2};border:2px solid {s_color};
                        border-radius:16px;padding:32px;text-align:center;">
              <div style="color:{TEXT_MUTED};font-size:0.78rem;
                          text-transform:uppercase;letter-spacing:0.1em;
                          margin-bottom:8px;">Remaining Useful Life</div>
              <div style="color:{s_color};font-size:4rem;font-weight:800;
                          line-height:1;margin-bottom:8px;">{pred_rul:.0f}</div>
              <div style="color:{TEXT_MUTED};font-size:1rem;margin-bottom:20px;">
                operating cycles</div>
              <div style="background:{BORDER};border-radius:8px;
                          height:8px;width:100%;margin-bottom:16px;">
                <div style="background:{s_color};border-radius:8px;
                            height:8px;width:{pct*100:.0f}%;
                            transition:width 0.5s ease;"></div>
              </div>
              <div style="display:inline-block;background:{s_color}22;
                          border:1px solid {s_color};border-radius:20px;
                          padding:6px 20px;color:{s_color};font-weight:700;
                          font-size:1rem;letter-spacing:0.05em;">
                {STATUS_EMOJI[status]} {status}
              </div>
            </div>""", unsafe_allow_html=True)

            # SHAP explanation for this prediction
            st.markdown("<br>", unsafe_allow_html=True)
            _section("Why this prediction?")
            live_feat_cols = feat_meta["feature_cols"]
            shap_vals = explainer(
                pd.DataFrame([feat_vec], columns=live_feat_cols)
            )
            sv      = shap_vals.values[0]
            base_v  = float(shap_vals.base_values[0])
            top_idx = np.argsort(np.abs(sv))[::-1][:10]

            live_raw_names      = [live_feat_cols[i] for i in top_idx]
            live_readable_names = [_feature_label(n) for n in live_raw_names]
            live_values         = [sv[i] for i in top_idx]
            live_colors         = [BLUE_LIGHT if v > 0 else RED_LIGHT for v in live_values]
            live_hover = [
                f"<b>{rl}</b><br>"
                f"<span style='color:#94a3b8;font-size:11px;'>{rn}</span><br>"
                f"SHAP: {v:+.2f} cycles"
                for rl, rn, v in zip(live_readable_names, live_raw_names, live_values)
            ]

            fig_live = go.Figure(go.Bar(
                x=live_values[::-1],
                y=live_readable_names[::-1],
                orientation="h",
                marker_color=live_colors[::-1],
                text=[f"{v:+.1f}" for v in live_values[::-1]],
                textposition="outside",
                textfont=dict(size=10, color=TEXT_MUTED),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=live_hover[::-1],
            ))
            fig_live.add_vline(x=0, line_color=TEXT_MUTED, line_width=1)
            fig_live.update_layout(
                **PLOTLY_BASE, height=360,
                xaxis_title="Cycles added (+) or removed (−) from the prediction",
                yaxis_title=None, showlegend=False,
            )
            fig_live.update_yaxes(tickfont=dict(size=11))
            st.plotly_chart(fig_live, use_container_width=True)
            st.markdown(_shap_legend_html(base_v, pred_rul, s_color), unsafe_allow_html=True)
            _shap_glossary()

            # ── Actionable Business Logic: Automated Maintenance Strategy & Procurement ─
            st.markdown("<br>", unsafe_allow_html=True)
            _section("🤖 Automated Maintenance Strategy & Procurement")

            # Determine dominant degrading sensor from negative SHAP contributions
            neg_features = [live_raw_names[i] for i in range(len(live_raw_names)) if live_values[i] < 0]
            dom_sensor = None
            if neg_features:
                for s in sensors:
                    if neg_features[0].startswith(s):
                        dom_sensor = s
                        break

            strat_agent = SimulatedMaintenanceAgent()
            strat_res = strat_agent.generate_strategy(
                predicted_rul=pred_rul,
                unit_id=sel_engine,
                dominant_sensor=dom_sensor,
                operating_cycle=last_cycle,
            )

            # Strategy Card
            st.markdown(f"""
            <div style="background:{BG_CARD2};border:1px solid {BORDER};border-left:4px solid {s_color};
                        border-radius:10px;padding:20px;margin-bottom:16px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="color:{s_color};font-weight:700;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.05em;">
                  {strat_res.urgency_rating}
                </span>
                <span style="color:{TEXT_MUTED};font-size:0.75rem;">
                  Subsystem: <b style="color:{TEXT_PRIMARY};">{strat_res.primary_affected_subsystem}</b>
                </span>
              </div>
              <div style="color:{TEXT_PRIMARY};font-size:0.88rem;line-height:1.6;margin-bottom:12px;">
                {strat_res.plain_english_strategy}
              </div>
              <div style="color:{TEXT_MUTED};font-size:0.8rem;margin-bottom:8px;">
                ⏱ <b>Action Window:</b> <span style="color:{TEXT_PRIMARY};">{strat_res.recommended_action_window}</span>
                &nbsp;|&nbsp; 🛠 <b>Estimated Downtime:</b> <span style="color:{TEXT_PRIMARY};">{strat_res.estimated_downtime_hours:.1f} hrs</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"📦 Simulated Parts Order: {strat_res.parts_order.order_id} ({strat_res.parts_order.order_priority})", expanded=(status == "CRITICAL")):
                st.markdown(f"""
                <div style="font-size:0.8rem;color:{TEXT_MUTED};margin-bottom:10px;">
                  <b>Procurement Hub:</b> {strat_res.parts_order.procurement_hub} &nbsp;|&nbsp;
                  <b>Logistics Modality:</b> {strat_res.parts_order.shipping_method} &nbsp;|&nbsp;
                  <b>Estimated Delivery:</b> {strat_res.parts_order.estimated_delivery_hours} hours
                </div>
                """, unsafe_allow_html=True)

                parts_data = [
                    {
                        "SKU": p.sku,
                        "Part Name": p.part_name,
                        "Subsystem": p.subsystem,
                        "Qty": p.quantity,
                        "Unit Cost ($)": f"${p.unit_cost_usd:,.2f}",
                        "Total ($)": f"${p.total_cost_usd:,.2f}",
                        "Lead Time": f"{p.lead_time_days} days",
                        "Criticality": p.criticality,
                    }
                    for p in strat_res.parts_order.items
                ]
                st.dataframe(pd.DataFrame(parts_data), use_container_width=True, hide_index=True)

                st.markdown(f"""
                <div style="text-align:right;font-size:0.9rem;font-weight:700;color:{BLUE_GLOW};margin-top:6px;">
                  Total Procurement Valuation: ${strat_res.parts_order.total_cost_usd:,.2f}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<b>Mandatory Technician Licenses:</b>", unsafe_allow_html=True)
                for cert in strat_res.technician_certifications:
                    st.markdown(f"- `{cert}`")

                st.markdown("<b>MRO Procedure Action Checklist:</b>", unsafe_allow_html=True)
                for task in strat_res.maintenance_tasks:
                    st.markdown(f"- {task}")

        else:
            _card(f"""
            <div style="text-align:center;padding:40px 0;">
              <div style="font-size:3rem;margin-bottom:16px;">⚡</div>
              <div style="color:{TEXT_PRIMARY};font-weight:600;font-size:1rem;
                          margin-bottom:8px;">Ready to predict</div>
              <div style="color:{TEXT_MUTED};font-size:0.85rem;">
                Select an engine, adjust sensor values if needed,<br>
                then click <b>Predict RUL</b>
              </div>
            </div>""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI ANALYST (tool-calling chat agent)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    # ── API key check ──────────────────────────────────────────────────────────
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))

    if not api_key_present:
        st.markdown(f"""
        <div style="background:{BG_CARD2};border:1px solid {YELLOW_LIGHT};
                    border-radius:12px;padding:32px;text-align:center;margin-top:20px;">
          <div style="font-size:2.5rem;margin-bottom:16px;">🔑</div>
          <div style="color:{TEXT_PRIMARY};font-weight:700;font-size:1.1rem;
                      margin-bottom:12px;">OpenAI API Key Required</div>
          <div style="color:{TEXT_MUTED};font-size:0.88rem;line-height:1.6;">
            The AI Analyst uses GPT-4o-mini with tool calling to answer questions<br>
            about your engine fleet using live model predictions and SHAP explanations.<br><br>
            <code style="background:{BG_CARD};padding:4px 10px;border-radius:4px;
                         color:{BLUE_GLOW};font-family:monospace;">
              export OPENAI_API_KEY=sk-...
            </code><br><br>
            Then restart the dashboard.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Session state init ─────────────────────────────────────────────────
        # chat_display:    clean list of {role, content} shown to the user
        # chat_api_history: full OpenAI message list (incl. tool-call objects)
        #                   — sent on every turn so the model has full context
        if "chat_display"      not in st.session_state:
            st.session_state.chat_display = []
        if "chat_api_history"  not in st.session_state:
            st.session_state.chat_api_history = []
        if "chat_agent"        not in st.session_state:
            try:
                from src.agents.chat_agent import ChatAgent
                st.session_state.chat_agent = ChatAgent(model="gpt-4o-mini")
                st.session_state.chat_agent_error = None
            except Exception as exc:
                st.session_state.chat_agent = None
                st.session_state.chat_agent_error = str(exc)

        if st.session_state.chat_agent_error:
            st.error(f"Could not initialise agent: {st.session_state.chat_agent_error}")
        else:
            # ── Header ─────────────────────────────────────────────────────────
            col_hdr, col_clear = st.columns([5, 1])
            with col_hdr:
                st.markdown(
                    f'<div style="color:{TEXT_MUTED};font-size:0.85rem;">'
                    "Ask anything about the fleet — I have access to live RUL predictions, "
                    "SHAP explanations, and fleet statistics. Powered by GPT-4o-mini + "
                    "function calling."
                    "</div>",
                    unsafe_allow_html=True,
                )
            with col_clear:
                if st.button("🗑 Clear", key="clear_chat"):
                    st.session_state.chat_display = []
                    st.session_state.chat_api_history = []
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Helper: run one agent turn and persist results ─────────────────
            def _run_agent_turn(user_text: str) -> tuple[str, list[str]]:
                """Append user message, call agent, update both histories."""
                user_msg = {"role": "user", "content": user_text}
                # API history already contains prior turns; append new user msg
                api_hist = st.session_state.chat_api_history + [user_msg]
                reply, updated_api_hist, tools_used = (
                    st.session_state.chat_agent.run_turn(api_hist)
                )
                # Persist full API history (includes tool call objects)
                st.session_state.chat_api_history = updated_api_hist
                # Persist display-only messages (plain text, no tool objects)
                st.session_state.chat_display.append(user_msg)
                st.session_state.chat_display.append(
                    {"role": "assistant", "content": reply, "tools": tools_used}
                )
                return reply, tools_used

            # ── Starter prompts (shown only on empty chat) ─────────────────────
            if not st.session_state.chat_display:
                st.markdown(
                    f'<div style="color:{TEXT_MUTED};font-size:0.82rem;'
                    f'margin-bottom:8px;">Try asking:</div>',
                    unsafe_allow_html=True,
                )
                starter_cols = st.columns(3)
                starters = [
                    ("🔴 Which engines need immediate maintenance?",
                     "Which engines need immediate maintenance?"),
                    ("🔍 Explain why engine 55 is at risk",
                     "Explain why engine 55 is at risk"),
                    ("📊 Give me a full fleet health summary",
                     "Give me a full fleet health summary with all key statistics"),
                ]
                for col, (label, prompt) in zip(starter_cols, starters):
                    if col.button(label, key=f"starter_{label[:10]}"):
                        with st.spinner("Analysing…"):
                            _run_agent_turn(prompt)
                        st.rerun()

            # ── Chat history (display only) ────────────────────────────────────
            for msg in st.session_state.chat_display:
                role = msg.get("role")
                content = msg.get("content", "")
                if role not in ("user", "assistant") or not content:
                    continue
                with st.chat_message(role):
                    st.markdown(content)
                    tools_used = msg.get("tools", [])
                    if role == "assistant" and tools_used:
                        tools_str = ", ".join(
                            f"`{t.replace('_', ' ')}`" for t in tools_used
                        )
                        st.caption(f"Tools called: {tools_str}")

            # ── Input ──────────────────────────────────────────────────────────
            user_input = st.chat_input("Ask about your engine fleet…")
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        reply, tools_used = _run_agent_turn(user_input)
                    st.markdown(reply)
                    if tools_used:
                        tools_str = ", ".join(
                            f"`{t.replace('_', ' ')}`" for t in tools_used
                        )
                        st.caption(f"Tools called: {tools_str}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    f'<div style="display:flex;justify-content:space-between;'
    f'color:{TEXT_MUTED};font-size:0.75rem;">'
    f"<span>⚙️ Engines Health Monitor · NASA CMAPSS FD001</span>"
    f"<span>XGBoost (Optuna) · 90 features · window=30 · FastAPI + Streamlit · "
    f"GPT-4o-mini + LangGraph</span>"
    f"</div>", unsafe_allow_html=True)
