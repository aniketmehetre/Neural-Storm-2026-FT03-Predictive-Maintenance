"""Global configuration and path management for Engines Health Monitor.

Ensures strict path relativity to repo root without machine-specific or absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

# ── Base Directory Anchors ───────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "CMAPSSData"
DATA_PROC_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"
CONFIG_DIR = ROOT_DIR / "config"

# Ensure output directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """Runtime environment configuration."""

    # Operational thresholds (cycles)
    alert_critical: int = Field(default=30, description="Critical RUL threshold")
    alert_warning: int = Field(default=80, description="Warning RUL threshold")
    rul_clip: int = Field(default=125, description="Piecewise linear upper cap on RUL")
    window_size: int = Field(default=30, description="Rolling window size in cycles")

    # LLM provider settings (supports offline, gemini, openai)
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "offline").lower()
    )
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    gemini_api_key: str | None = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    openai_model: str = Field(default="gpt-4o-mini")
    gemini_model: str = Field(default="gemini-1.5-flash")

    # API and server settings
    api_port: int = Field(default=8000)
    api_host: str = Field(default="0.0.0.0")

    model_config = {"arbitrary_types_allowed": True}


settings = Settings()
