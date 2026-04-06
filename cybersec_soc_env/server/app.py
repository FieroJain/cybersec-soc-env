"""
server/app.py - FastAPI application entry point for CyberSec-SOC-OpenEnv.
"""

import os
import logging as _logging
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from openenv.core.env_server import create_fastapi_app

from ..models import SOCAction, SOCObservation
from .soc_environment import SOCEnvironment, TASK_CONFIG

_log = _logging.getLogger(__name__)

# CONFIG
task_level: str = os.environ.get("TASK_LEVEL", "medium")
seed: int = int(os.environ.get("SEED", "42"))


def make_env() -> SOCEnvironment:
    """Factory that creates a fresh SOCEnvironment for each WebSocket session."""
    return SOCEnvironment(task_level=task_level, seed=seed)


app = create_fastapi_app(make_env, SOCAction, SOCObservation)


@app.get("/", response_class=HTMLResponse)
def root():
    """Embed Gradio dashboard directly at root via iframe."""
    return """<!DOCTYPE html>
<html style="height:100%;margin:0;padding:0;">
<head>
    <title>CyberSec-SOC-OpenEnv</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { height: 100%; width: 100%; overflow: hidden; }
        .topbar {
            background: #0d1a2d;
            border-bottom: 2px solid #00ff88;
            padding: 6px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            height: 40px;
        }
        .topbar h1 {
            color: #00ff88;
            font-family: monospace;
            font-size: 0.95rem;
            font-weight: bold;
        }
        .topbar .links {
            margin-left: auto;
            display: flex;
            gap: 10px;
        }
        .topbar a {
            color: #00aaff;
            font-family: monospace;
            font-size: 0.78rem;
            padding: 3px 10px;
            border: 1px solid #00aaff;
            border-radius: 4px;
            text-decoration: none;
        }
        .topbar a:hover { background: #00aaff22; }
        iframe {
            width: 100%;
            height: calc(100vh - 40px);
            border: none;
            display: block;
        }
    </style>
</head>
<body>
    <div class="topbar">
        <h1>🛡️ CyberSec-SOC-OpenEnv &nbsp;|&nbsp; AI Threat Defense Simulator</h1>
        <div class="links">
            <a href="/docs" target="_blank">API Docs</a>
            <a href="/web" target="_blank">Full Screen</a>
            <a href="https://github.com/FieroJain/cybersec-soc-env" target="_blank">GitHub</a>
        </div>
    </div>
    <iframe
        src="/web"
        title="CyberSec SOC Command Center"
        allow="same-origin"
    ></iframe>
</body>
</html>"""


def _run_grader_episode(env: SOCEnvironment) -> Dict[str, Any]:
    """Run a single grader episode with deterministic rule-based agent."""
    obs = env.reset()
    steps: int = 0
    max_steps: int = TASK_CONFIG[env.task_level]["max_steps"]

    while not obs.done and steps < max_steps:
        candidates = sorted(
            [n for n in obs.node_statuses if not n["is_isolated"]],
            key=lambda n: n["alert_score"],
            reverse=True,
        )

        action = SOCAction(action_type="nothing", target_node_id=-1)
        for node in candidates:
            if node["visible_compromise"]:
                action = SOCAction(
                    action_type="isolate",
                    target_node_id=int(node["id"]),
                )
                break
            if node["alert_score"] > 0.4:
                action = SOCAction(
                    action_type="scan",
                    target_node_id=int(node["id"]),
                )
                break

        obs = env.step(action)
        steps += 1

    return {
        "defender_wins": bool(obs.defender_wins),
        "attack_stage": int(obs.attack_stage),
        "business_impact": float(obs.business_impact_score),
        "steps": int(steps),
        "false_isolations": int(env.state.false_isolations),
    }


def _compute_episode_score(result: Dict[str, Any], level: str) -> float:
    """Map one episode result to a score in [0.0, 1.0]."""
    max_steps = TASK_CONFIG[level]["max_steps"]
    score = 0.0
    if result["defender_wins"]:
        score += 0.5
    if result["attack_stage"] <= 2:
        score += 0.2
    elif result["attack_stage"] == 3:
        score += 0.1
    if result["business_impact"] < 1.0:
        score += 0.2
    elif result["business_impact"] < 2.0:
        score += 0.1
    efficiency = 1.0 - (result["steps"] / max_steps)
    score += 0.1 * max(0.0, efficiency)
    return round(min(1.0, max(0.0, score)), 3)


@app.get("/tasks", response_class=JSONResponse)
def list_tasks() -> Dict[str, Any]:
    """Return the catalogue of available task levels."""
    tasks = [
        {
            "id": "easy",
            "description": "5 nodes, 1 infected, max 20 steps",
            "max_steps": 20,
            "nodes": 5,
            "start_compromised": 1,
        },
        {
            "id": "medium",
            "description": "10 nodes, 2 infected, max 35 steps",
            "max_steps": 35,
            "nodes": 10,
            "start_compromised": 2,
        },
        {
            "id": "hard",
            "description": "20 nodes, 3 infected, max 50 steps",
            "max_steps": 50,
            "nodes": 20,
            "start_compromised": 3,
        },
    ]
    return {"tasks": tasks}


@app.post("/tasks/{task_id}/grade", response_class=JSONResponse)
def grade_task(task_id: str, n_episodes: int = 3) -> Dict[str, Any]:
    """Run grader episodes and return score 0.0-1.0."""
    if task_id not in TASK_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Must be one of {list(TASK_CONFIG)}.",
        )

    env = SOCEnvironment(task_level=task_id, seed=seed)
    episode_scores = []
    wins = 0
    total_steps = 0
    total_false_isolations = 0

    for _ in range(n_episodes):
        result = _run_grader_episode(env)
        ep_score = _compute_episode_score(result, task_id)
        episode_scores.append(ep_score)
        if result["defender_wins"]:
            wins += 1
        total_steps += result["steps"]
        total_false_isolations += result["false_isolations"]

    avg_score = round(sum(episode_scores) / len(episode_scores), 3)
    containment_rate = round(wins / n_episodes, 3)
    avg_steps = round(total_steps / n_episodes, 1)

    return {
        "task_id": task_id,
        "score": avg_score,
        "episodes_run": n_episodes,
        "details": {
            "containment_rate": containment_rate,
            "avg_steps": avg_steps,
            "false_isolations": total_false_isolations,
        },
    }


@app.post("/reset/{task_id}", response_class=JSONResponse)
def reset_at_task(task_id: str) -> Dict[str, Any]:
    """Reset environment at specific task level and return initial observation."""
    if task_id not in TASK_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Must be one of {list(TASK_CONFIG)}.",
        )

    env = SOCEnvironment(task_level=task_id, seed=seed)
    obs = env.reset()

    return {
        "task_id": task_id,
        "topology": str(obs.topology_type),
        "nodes": int(len(obs.node_statuses)),
        "attack_stage": int(obs.attack_stage),
        "observation": obs.model_dump(),
    }


# GRADIO DASHBOARD mounted at /web
try:
    import gradio as _gr
    from .gradio_dashboard import demo as _gradio_demo
    _gr.mount_gradio_app(app, _gradio_demo, path="/web")
    _log.info("Gradio SOC dashboard mounted at /web")
except Exception as _e:
    _log.warning("Gradio dashboard not mounted: %s", _e)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()