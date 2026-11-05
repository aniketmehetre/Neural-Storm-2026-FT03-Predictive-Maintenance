"""Tool-calling chat agent for the Engines Health Monitor dashboard.

Uses the OpenAI chat completion API with function calling.
Each ``run_turn`` call is stateless — the caller (Streamlit) owns the history.
"""

from __future__ import annotations

import json
import os
from typing import Generator

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from src.agents.tools import (
    explain_engine,
    get_engine_details,
    get_engines_by_status,
    get_fleet_statistics,
    get_fleet_summary,
)

# ── Tool schemas for OpenAI function calling ───────────────────────────────────
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_fleet_summary",
            "description": (
                "Get a high-level summary of the entire fleet health: "
                "how many engines are CRITICAL / WARNING / HEALTHY, "
                "the top at-risk engines, and fleet-level prediction accuracy (RMSE, MAE)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_engine_details",
            "description": (
                "Get detailed information for a specific engine: predicted RUL, "
                "true RUL, prediction error, current health status, and urgency note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "engine_id": {
                        "type": "integer",
                        "description": "Engine identifier, integer from 1 to 100.",
                    }
                },
                "required": ["engine_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_engines_by_status",
            "description": (
                "Return all engines with a given health status. "
                "CRITICAL = RUL ≤ 30 cycles (imminent failure), "
                "WARNING = RUL ≤ 80 cycles (degradation detected), "
                "HEALTHY = RUL > 80 cycles (normal operation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["CRITICAL", "WARNING", "HEALTHY"],
                        "description": "Health status to filter engines by.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_statistics",
            "description": (
                "Get statistical summary of the fleet's RUL distribution: "
                "mean, median, min, max, standard deviation, percentiles, "
                "and count of engines below critical thresholds."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_engine",
            "description": (
                "Get a SHAP-based explanation for why the model predicted a specific RUL "
                "for an engine. Returns the top features driving the prediction up or down "
                "with their SHAP contribution values in cycles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "engine_id": {
                        "type": "integer",
                        "description": "Engine identifier, integer from 1 to 100.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top features to return (default 8).",
                        "default": 8,
                    },
                },
                "required": ["engine_id"],
            },
        },
    },
]

_TOOL_DISPATCH = {
    "get_fleet_summary":     lambda args: get_fleet_summary(),
    "get_engine_details":    lambda args: get_engine_details(**args),
    "get_engines_by_status": lambda args: get_engines_by_status(**args),
    "get_fleet_statistics":  lambda args: get_fleet_statistics(),
    "explain_engine":        lambda args: explain_engine(**args),
}

SYSTEM_PROMPT = """\
You are an expert AI maintenance analyst for an industrial turbofan engine fleet.
You have access to real-time Remaining Useful Life (RUL) predictions from an \
XGBoost model trained on the NASA CMAPSS FD001 dataset.

Your responsibilities:
- Answer questions about fleet health, individual engine status, and failure risk
- Explain model predictions using SHAP feature importance
- Provide actionable, prioritized maintenance recommendations
- Cite exact numbers from tool results — never estimate or guess
- Be concise but precise; engineers rely on your accuracy

Alert thresholds:
  🔴 CRITICAL  — RUL ≤ 30 cycles  → immediate maintenance required
  🟡 WARNING   — RUL ≤ 80 cycles  → schedule inspection
  🟢 HEALTHY   — RUL > 80 cycles  → operating normally

Dataset: 100 test turbofan engines · XGBoost (Optuna tuned) · 90 rolling-window features · window=30 cycles
"""


def _execute_tool(tool_call: ChatCompletionMessageToolCall) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments or "{}")
    try:
        result = _TOOL_DISPATCH[name](args)
    except Exception as exc:
        result = {"error": str(exc)}
    return json.dumps(result, default=str)


class ChatAgent:
    """Stateless tool-calling agent.

    The caller (Streamlit session) owns the conversation history and passes
    it in full on every turn.

    Args:
        model: OpenAI model name. ``"gpt-4o-mini"`` is fast and cheap;
               use ``"gpt-4o"`` for more nuanced analysis.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Run: export OPENAI_API_KEY=sk-..."
            )
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def run_turn(
        self,
        messages: list[dict],
        max_tool_rounds: int = 6,
    ) -> tuple[str, list[dict], list[str]]:
        """Execute one conversational turn with an automatic tool-calling loop.

        Args:
            messages:        Full conversation history (without system prompt).
            max_tool_rounds: Maximum tool-call → result cycles per turn.

        Returns:
            ``(reply, updated_messages, tool_names_called)``
            where ``updated_messages`` is the history extended with this turn.
        """
        full = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        tools_called: list[str] = []

        for _ in range(max_tool_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            full.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                reply = msg.content or ""
                # Return history without the system prompt
                return reply, full[1:], tools_called

            for tc in msg.tool_calls:
                tools_called.append(tc.function.name)
                result = _execute_tool(tc)
                full.append(
                    {
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    }
                )

        return (
            "I reached the maximum number of tool calls for this query. "
            "Please try a more focused question.",
            full[1:],
            tools_called,
        )
