"""
server/app.py — FastAPI application entry point for CyberSec-SOC-OpenEnv.

Passes a factory function (not an instance) to create_fastapi_app, as
required by openenv.core.env_server.http_server.HTTPEnvServer.

Environment variables:
    TASK_LEVEL            (str)  Difficulty: easy|medium|hard. Default: medium.
    SEED                  (int)  Random seed. Default: 42.
    ENABLE_WEB_INTERFACE  (1|0)  Mount Gradio web UI at /web. Default: 0.
"""

import os

from openenv.core.env_server import create_fastapi_app

from ..models import SOCAction, SOCObservation
from .soc_environment import SOCEnvironment

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

task_level: str = os.environ.get("TASK_LEVEL", "medium")
seed: int = int(os.environ.get("SEED", "42"))


# ── FACTORY FUNCTION ──────────────────────────────────────────────────────────
# create_fastapi_app requires a *callable* that produces a fresh Environment
# instance per session — it must NOT receive an instance directly.

def make_env() -> SOCEnvironment:
    """Factory that creates a fresh SOCEnvironment for each WebSocket session."""
    return SOCEnvironment(task_level=task_level, seed=seed)


# ── APP ───────────────────────────────────────────────────────────────────────

app = create_fastapi_app(make_env, SOCAction, SOCObservation)

# Web UI is enabled when ENABLE_WEB_INTERFACE=1 (Gradio mounted at /web).
