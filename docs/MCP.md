# Model Context Protocol (MCP) Server Integration Guide

The **Engines Health Monitor** exposes full fleet diagnostic, prognostic, and scheduling capabilities over the official **Model Context Protocol (MCP)** using `FastMCP`.

---

## 1. Available Tools & Resources

### Resources
- `fleet://summary`: Read-only snapshot of fleet health counts, RUL distributions, and risk categories.

### Tools
- `list_fleet(status_filter)`: Returns fleet engine inventory, filterable by `CRITICAL`, `WARNING`, or `HEALTHY`.
- `get_engine_status(engine_id)`: Fetches real-time RUL, conformal uncertainty intervals ($p_{10}, p_{50}, p_{90}$), and failure probabilities.
- `explain_engine(engine_id, top_n)`: Returns cycle-by-cycle SHAP feature contributions for root-cause diagnosis.
- `recommend_maintenance(engine_id)`: Evaluates cost-optimal replacement timing and safe replace-by deadlines.
- `schedule_hangar(engine_ids)`: Runs CP-SAT optimization to schedule engines onto limited hangar bays without capacity violations.
- `run_backtest_summary()`: Runs the 3-policy empirical backtest showing dollar savings over reactive/fixed strategies.

---

## 2. Configuration for Antigravity

Add the following block to your Antigravity project configuration:

```json
{
  "mcpServers": {
    "engines-health-monitor": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

---

## 3. Configuration for Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "engines-health-monitor": {
      "command": "python",
      "args": [
        "-m", "src.mcp_server"
      ],
      "cwd": "C:/HACKATHONS/my-hackathon-project"
    }
  }
}
```

---

## 4. Testing the Server Locally

Run the MCP server in standard I/O mode:
```bash
python -m src.mcp_server
```
