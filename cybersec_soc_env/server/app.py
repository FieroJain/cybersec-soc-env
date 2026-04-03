"""
server/app.py — FastAPI application entry point for CyberSec-SOC-OpenEnv.

Passes a factory function (not an instance) to create_fastapi_app, as
required by openenv.core.env_server.http_server.HTTPEnvServer.

Additional endpoints (added below the base app):
    GET  /tasks                  — list all available task levels
    POST /tasks/{task_id}/grade  — run 3-episode rule-based grader, score 0.0-1.0
    POST /reset/{task_id}        — reset environment at a specific task level

Environment variables:
    TASK_LEVEL            (str)  Difficulty: easy|medium|hard. Default: medium.
    SEED                  (int)  Random seed. Default: 42.
    ENABLE_WEB_INTERFACE  (1|0)  Mount Gradio web UI at /web. Default: 0.
"""

import os
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from openenv.core.env_server import create_fastapi_app

from ..models import SOCAction, SOCObservation
from .soc_environment import SOCEnvironment, TASK_CONFIG

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

task_level: str = os.environ.get("TASK_LEVEL", "medium")
seed: int = int(os.environ.get("SEED", "42"))


# ── FACTORY FUNCTION ──────────────────────────────────────────────────────────
# create_fastapi_app requires a *callable* that produces a fresh Environment
# instance per session — it must NOT receive an instance directly.

def make_env() -> SOCEnvironment:
    """Factory that creates a fresh SOCEnvironment for each WebSocket session."""
    return SOCEnvironment(task_level=task_level, seed=seed)


# ── BASE APP ──────────────────────────────────────────────────────────────────

app = create_fastapi_app(make_env, SOCAction, SOCObservation)

# Web UI is enabled when ENABLE_WEB_INTERFACE=1 (Gradio mounted at /web).


# ── HELPER: rule-based grader agent ──────────────────────────────────────────

def _run_grader_episode(env: SOCEnvironment) -> Dict[str, Any]:
    """
    Run a single grader episode with a deterministic rule-based agent.

    Strategy: scan the highest-alert non-isolated node each step;
    isolate immediately when a compromise is confirmed via scan.

    Returns dict with defender_wins, attack_stage, business_impact,
    steps, and false_isolations.
    """
    obs = env.reset()
    steps: int = 0
    max_steps: int = TASK_CONFIG[env.task_level]["max_steps"]

    while not obs.done and steps < max_steps:
        # Sort non-isolated nodes by alert_score descending.
        candidates = sorted(
            [n for n in obs.node_statuses if not n["is_isolated"]],
            key=lambda n: n["alert_score"],
            reverse=True,
        )

        action = SOCAction(action_type="nothing", target_node_id=-1)
        for node in candidates:
            if node["visible_compromise"]:
                # Confirmed compromise → isolate immediately.
                action = SOCAction(
                    action_type="isolate",
                    target_node_id=int(node["id"]),
                )
                break
            if node["alert_score"] > 0.4:
                # High alert but unscanned → scan to confirm.
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
    """
    Map one episode result to a score in [0.0, 1.0].

    Components:
    - 0.5  defender wins
    - 0.2  attack stage reached ≤ 2 (early kill-chain stop)
    - 0.1  attack stage reached == 3 (lateral movement stopped)
    - 0.2  business impact < 1.0
    - 0.1  business impact < 2.0 (partial)
    - 0.1  efficiency (unused step budget)
    """
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


# ── ENDPOINT 1: GET /tasks ────────────────────────────────────────────────────

@app.get("/tasks", response_class=JSONResponse)
def list_tasks() -> Dict[str, Any]:
    """
    Return the catalogue of available task levels.

    Response schema::

        {
          "tasks": [
            {"id": "easy",   "description": "...", "max_steps": 20},
            {"id": "medium", "description": "...", "max_steps": 35},
            {"id": "hard",   "description": "...", "max_steps": 50}
          ]
        }
    """
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


# ── ENDPOINT 2: POST /tasks/{task_id}/grade ───────────────────────────────────

@app.post("/tasks/{task_id}/grade", response_class=JSONResponse)
def grade_task(task_id: str, n_episodes: int = 3) -> Dict[str, Any]:
    """
    Run *n_episodes* (default 3) of the environment at *task_id* using
    a deterministic rule-based agent and return an aggregated score.

    Path parameter:
        task_id: one of easy | medium | hard

    Query parameter:
        n_episodes: number of episodes to average (default 3)

    Response schema::

        {
          "task_id": "easy",
          "score": 0.417,
          "episodes_run": 3,
          "details": {
            "containment_rate": 0.33,
            "avg_steps": 18.0,
            "false_isolations": 1
          }
        }

    Score is in [0.0, 1.0]. Raises 400 for unknown task_id.
    """
    if task_id not in TASK_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Must be one of {list(TASK_CONFIG)}.",
        )

    env = SOCEnvironment(task_level=task_id, seed=seed)
    episode_scores: list[float] = []
    wins: int = 0
    total_steps: int = 0
    total_false_isolations: int = 0

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


# ── ENDPOINT 3: POST /reset/{task_id} ────────────────────────────────────────

@app.post("/reset/{task_id}", response_class=JSONResponse)
def reset_at_task(task_id: str) -> Dict[str, Any]:
    """
    Reset a fresh environment instance at the specified task level and
    return the initial observation.

    Path parameter:
        task_id: one of easy | medium | hard

    Response schema::

        {
          "task_id": "medium",
          "topology": "mesh",
          "nodes": 10,
          "attack_stage": 1,
          "observation": { ... SOCObservation fields ... }
        }

    Raises 400 for unknown task_id. This endpoint creates an isolated
    environment instance and does not affect live WebSocket sessions.
    """
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

   def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()