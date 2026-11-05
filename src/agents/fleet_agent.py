"""Autonomous Fleet Monitoring Agent — LangGraph multi-agent pipeline.

Architecture: sequential pipeline with 4 specialist nodes.

  START → sensor_analyst → diagnostic → prioritizer → report_writer → END

Each node calls domain tools directly (Python function calls), then uses an LLM
to synthesize an analysis from the raw data. The LangGraph state carries all
intermediate results forward, giving each node full context from previous stages.

Usage:
    from src.agents.fleet_agent import FleetMonitoringAgent

    agent  = FleetMonitoringAgent()
    report = agent.run()
    print(report["final_report"])
"""

from __future__ import annotations

import json
import operator
import os
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agents.tools import (
    explain_engine,
    get_engine_details,
    get_engines_by_status,
    get_fleet_statistics,
    get_fleet_summary,
)


# ── Agent state ────────────────────────────────────────────────────────────────
class FleetAgentState(TypedDict):
    """State passed between all nodes in the fleet monitoring pipeline."""

    # Accumulated message log for observability
    messages: Annotated[list[BaseMessage], operator.add]
    # Structured outputs from each specialist node
    fleet_overview:   str   # JSON — sensor analyst output
    diagnostics:      str   # JSON — per-engine SHAP diagnostics
    priority_ranking: str   # JSON — urgency-ranked engine list
    final_report:     str   # Markdown — maintenance briefing


# ── Structured output schema ───────────────────────────────────────────────────
class EngineAlert(BaseModel):
    engine_id:    int
    predicted_rul: float
    status:       str
    top_risk_factor: str
    recommended_action: str


class FleetReport(BaseModel):
    timestamp:      str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fleet_health:   str
    critical_alerts: list[EngineAlert] = []
    warning_alerts:  list[EngineAlert] = []
    summary:         str
    recommended_actions: list[str] = []
    model_confidence: str


# ── LLM factory ───────────────────────────────────────────────────────────────
def _llm(temperature: float = 0) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=api_key)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Sensor Analyst
# Collects fleet-wide data and identifies at-risk engines
# ══════════════════════════════════════════════════════════════════════════════
def sensor_analyst_node(state: FleetAgentState) -> dict:
    """Analyze all fleet sensors, classify engine health, identify anomalies."""
    summary  = get_fleet_summary()
    stats    = get_fleet_statistics()
    critical = get_engines_by_status("CRITICAL")
    warning  = get_engines_by_status("WARNING")

    prompt = f"""You are a Sensor Analysis Agent for an industrial turbofan engine fleet.
You have access to real-time RUL predictions from an XGBoost model (NASA CMAPSS FD001).

FLEET SUMMARY:
{json.dumps(summary, indent=2)}

RUL DISTRIBUTION STATISTICS:
{json.dumps(stats, indent=2)}

CRITICAL ENGINES (RUL ≤ 30 cycles):
{json.dumps(critical, indent=2)}

WARNING ENGINES (RUL ≤ 80 cycles):
{json.dumps(warning, indent=2)}

Perform a thorough sensor analysis:
1. Overall fleet health assessment (use precise percentages and counts)
2. Key patterns in the RUL distribution
3. Which engines pose the highest immediate risk and why
4. Concerning trends across the fleet

Be data-driven and specific. Output valid JSON with keys:
  "health_grade", "key_findings" (list), "immediate_risks" (list of engine_ids),
  "fleet_summary_text" (2-3 sentences)"""

    response = _llm().invoke([HumanMessage(content=prompt)])
    overview = response.content

    return {
        "fleet_overview": json.dumps(
            {
                "raw": {
                    "summary":  summary,
                    "stats":    stats,
                    "critical": critical,
                    "warning":  warning,
                },
                "analysis": overview,
            }
        ),
        "messages": [
            AIMessage(content=f"[Sensor Analyst] Fleet analyzed. "
                              f"{summary['critical_count']} critical, "
                              f"{summary['warning_count']} warning engines identified.")
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Diagnostic Agent
# Deep-dives into each critical engine using SHAP explanations
# ══════════════════════════════════════════════════════════════════════════════
def diagnostic_node(state: FleetAgentState) -> dict:
    """Explain WHY each critical engine is at risk using SHAP feature importance."""
    overview_data = json.loads(state["fleet_overview"])
    raw_critical  = overview_data["raw"]["critical"]["engines"]

    engine_diagnoses = []
    for eng in raw_critical[:10]:  # cap at 10 to limit API calls
        eid        = int(eng["unit_id"])
        details    = get_engine_details(eid)
        shap_data  = explain_engine(eid, top_n=5)
        engine_diagnoses.append(
            {
                "engine_id": eid,
                "details":   details,
                "shap":      shap_data,
            }
        )

    if not engine_diagnoses:
        return {
            "diagnostics": json.dumps({"message": "No critical engines found.", "diagnoses": []}),
            "messages":    [AIMessage(content="[Diagnostic Agent] No critical engines to diagnose.")],
        }

    prompt = f"""You are a Diagnostic Agent specializing in turbofan engine failure analysis.
For each critical engine below, synthesize the SHAP explanation into a plain-English diagnosis.

CRITICAL ENGINE DIAGNOSTICS:
{json.dumps(engine_diagnoses, indent=2)}

For each engine produce:
- A one-sentence diagnosis (what is degrading and how)
- The primary fault indicator (top SHAP feature)
- Recommended maintenance action (specific and actionable)

Output valid JSON: a list of objects with keys:
  "engine_id", "diagnosis", "primary_fault_indicator", "recommended_action", "urgency_score" (1-10)"""

    response = _llm().invoke([HumanMessage(content=prompt)])

    return {
        "diagnostics": json.dumps(
            {"raw_diagnoses": engine_diagnoses, "llm_analysis": response.content}
        ),
        "messages": [
            AIMessage(
                content=f"[Diagnostic Agent] Completed SHAP diagnosis for "
                        f"{len(engine_diagnoses)} critical engines."
            )
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Priority Ranker
# Ranks all at-risk engines by maintenance urgency
# ══════════════════════════════════════════════════════════════════════════════
def prioritizer_node(state: FleetAgentState) -> dict:
    """Rank all at-risk engines by urgency and create a maintenance schedule."""
    overview_data = json.loads(state["fleet_overview"])
    diagnostics   = json.loads(state["diagnostics"])

    prompt = f"""You are a Maintenance Prioritization Agent for an industrial fleet.
Using the fleet analysis and SHAP diagnostics below, create a prioritized maintenance schedule.

FLEET OVERVIEW:
{overview_data.get("analysis", "")}

CRITICAL ENGINE DIAGNOSTICS:
{diagnostics.get("llm_analysis", "No critical engines.")}

WARNING ENGINES:
{json.dumps(overview_data["raw"]["warning"]["engines"][:20], indent=2)}

Create a prioritized maintenance plan:
1. Rank all critical engines by urgency (lowest RUL = highest priority)
2. Group warning engines into two bands: 30-50 cycles and 51-80 cycles
3. Assign maintenance windows (Week 1, Week 2, Week 3)

Output valid JSON with keys:
  "immediate_action" (list of engine_ids, RUL < 20),
  "week_1" (list of engine_ids),
  "week_2" (list of engine_ids),
  "week_3" (list of engine_ids),
  "total_engines_requiring_action" (int),
  "priority_rationale" (string)"""

    response = _llm().invoke([HumanMessage(content=prompt)])

    return {
        "priority_ranking": json.dumps(
            {
                "llm_ranking": response.content,
                "fleet_stats": overview_data["raw"]["stats"],
            }
        ),
        "messages": [
            AIMessage(content="[Priority Ranker] Maintenance schedule prioritized.")
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Report Writer
# Synthesizes all previous outputs into a final maintenance briefing
# ══════════════════════════════════════════════════════════════════════════════
def report_writer_node(state: FleetAgentState) -> dict:
    """Write the final structured maintenance briefing in Markdown."""
    overview_data = json.loads(state["fleet_overview"])
    summary       = overview_data["raw"]["summary"]
    ts            = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    prompt = f"""You are a Chief Maintenance Engineer writing an executive fleet health report.
Synthesize all analysis below into a professional, actionable maintenance briefing.

FLEET OVERVIEW ANALYSIS:
{overview_data.get("analysis", "")}

DIAGNOSTIC FINDINGS:
{json.loads(state["diagnostics"]).get("llm_analysis", "No critical findings.")}

PRIORITY RANKING & SCHEDULE:
{json.loads(state["priority_ranking"]).get("llm_ranking", "")}

RAW FLEET METRICS:
- Total engines: {summary["total_engines"]}
- Critical: {summary["critical_count"]} | Warning: {summary["warning_count"]} | Healthy: {summary["healthy_count"]}
- Fleet prediction RMSE: {summary["fleet_rmse_cycles"]} cycles

Write a complete Markdown maintenance briefing with these sections:
1. **Executive Summary** (3-4 sentences, key numbers)
2. **Fleet Health Status** (status table or bullet points)
3. **Critical Engine Alerts** (engine IDs, RUL, primary fault, action required)
4. **Maintenance Schedule** (immediate / Week 1 / Week 2 / Week 3)
5. **Model Confidence Note** (brief note on RMSE and what it means for reliability)
6. **Recommended Next Steps** (numbered action list)

Tone: professional, precise, direct. Use exact numbers. Timestamp: {ts}"""

    response = _llm(temperature=0.2).invoke([HumanMessage(content=prompt)])

    return {
        "final_report": response.content,
        "messages": [
            AIMessage(content="[Report Writer] Maintenance briefing complete.")
        ],
    }


# ── Graph compilation ──────────────────────────────────────────────────────────
def _build_graph() -> StateGraph:
    g = StateGraph(FleetAgentState)

    g.add_node("sensor_analyst", sensor_analyst_node)
    g.add_node("diagnostic",     diagnostic_node)
    g.add_node("prioritizer",    prioritizer_node)
    g.add_node("report_writer",  report_writer_node)

    g.add_edge(START,             "sensor_analyst")
    g.add_edge("sensor_analyst",  "diagnostic")
    g.add_edge("diagnostic",      "prioritizer")
    g.add_edge("prioritizer",     "report_writer")
    g.add_edge("report_writer",   END)

    return g.compile()


# ── Public API ─────────────────────────────────────────────────────────────────
class FleetMonitoringAgent:
    """Autonomous fleet monitoring agent.

    Runs a 4-stage LangGraph pipeline that analyzes the full engine fleet,
    diagnoses critical engines with SHAP, prioritizes maintenance, and
    produces a structured Markdown briefing.

    Usage:
        agent  = FleetMonitoringAgent()
        result = agent.run()
        print(result["final_report"])
    """

    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self) -> dict:
        """Execute the full fleet monitoring pipeline.

        Returns:
            The final LangGraph state dict, including ``"final_report"``
            (Markdown), ``"fleet_overview"``, ``"diagnostics"``,
            ``"priority_ranking"``, and ``"messages"``.
        """
        initial_state: FleetAgentState = {
            "messages":       [],
            "fleet_overview": "",
            "diagnostics":    "",
            "priority_ranking": "",
            "final_report":   "",
        }
        return self._graph.invoke(initial_state)
