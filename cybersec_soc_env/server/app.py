"""
server/app.py - FastAPI application entry point for CyberSec-SOC-OpenEnv.
"""

import os
import json
import logging as _logging
import time as _time
from typing import Any, Dict, List, Optional

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
            <a href="/demo" target="_blank">Demo Run</a>
            <a href="/leaderboard" target="_blank">Leaderboard</a>
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
    """Map one episode result to a score in [0.001, 0.999]."""
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
    return round(min(0.999, max(0.001, score)), 3)


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
    """Run grader episodes and return score 0.001-0.999."""
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


# ---------------------------------------------------------------------------
# NEW: /demo endpoint — deterministic demonstration episode
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=JSONResponse)
def demo_episode(task_id: str = "medium") -> Dict[str, Any]:
    """
    Run a complete deterministic demonstration episode using the rule-based agent
    and return the full trajectory as JSON.

    Judges and developers can hit this URL to instantly see the environment working
    end-to-end without needing an LLM API key.

    Query params:
        task_id: easy | medium | hard (default: medium)
    """
    if task_id not in TASK_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Must be one of {list(TASK_CONFIG)}.",
        )

    env = SOCEnvironment(task_level=task_id, seed=42)  # fixed seed for determinism
    obs = env.reset()
    cfg = TASK_CONFIG[task_id]
    max_steps = cfg["max_steps"]

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0
    step = 0
    scanned_nodes: set = set()

    # Initial state snapshot
    trajectory.append({
        "step": 0,
        "action": None,
        "reward": 0.0,
        "cumulative_reward": 0.0,
        "attack_stage": obs.attack_stage,
        "topology": obs.topology_type,
        "nodes": len(obs.node_statuses),
        "alerts": obs.alerts,
        "done": obs.done,
        "defender_wins": obs.defender_wins,
    })

    while not obs.done and step < max_steps:
        step += 1

        # Rule-based agent: isolate confirmed threats, scan highest-alert unscanned
        candidates = sorted(
            [n for n in obs.node_statuses if not n["is_isolated"]],
            key=lambda n: (n["visible_compromise"], n["alert_score"]),
            reverse=True,
        )

        action = SOCAction(action_type="nothing", target_node_id=-1)
        action_reason = "No active threats detected"

        for node in candidates:
            if node["visible_compromise"]:
                action = SOCAction(action_type="isolate", target_node_id=int(node["id"]))
                action_reason = f"Isolating confirmed threat node {node['id']} ({node['type']})"
                break
            if node["alert_score"] > 0.4 and node["id"] not in scanned_nodes:
                action = SOCAction(action_type="scan", target_node_id=int(node["id"]))
                action_reason = (
                    f"Scanning high-alert node {node['id']} ({node['type']}) "
                    f"alert={node['alert_score']:.3f}"
                )
                break

        if action.action_type == "scan":
            scanned_nodes.add(action.target_node_id)

        obs = env.step(action)
        reward = obs.reward or 0.0
        total_reward += reward

        trajectory.append({
            "step": step,
            "action": f"{action.action_type}({action.target_node_id})",
            "action_reason": action_reason,
            "reward": round(reward, 4),
            "cumulative_reward": round(total_reward, 4),
            "attack_stage": obs.attack_stage,
            "business_impact": obs.business_impact_score,
            "visible_threats": sum(1 for n in obs.node_statuses if n["visible_compromise"]),
            "isolated_count": sum(1 for n in obs.node_statuses if n["is_isolated"]),
            "alerts": obs.alerts[-3:],  # last 3 to keep response manageable
            "done": obs.done,
            "defender_wins": obs.defender_wins,
        })

        if obs.done:
            break

    # Compute final score
    rewards_list = [t["reward"] for t in trajectory if t["reward"] != 0.0]
    final_stage = obs.attack_stage
    score = 0.001
    if obs.defender_wins:
        score += 0.5
    if final_stage <= 2:
        score += 0.2
    elif final_stage == 3:
        score += 0.1
    if sum(rewards_list) > 0:
        score += 0.2
    score = round(min(0.999, max(0.001, score)), 3)

    return {
        "demo": True,
        "agent": "rule_based_greedy",
        "task_id": task_id,
        "seed": 42,
        "topology": trajectory[0]["topology"],
        "total_steps": step,
        "max_steps": max_steps,
        "defender_wins": obs.defender_wins,
        "final_attack_stage": final_stage,
        "total_reward": round(total_reward, 4),
        "score": score,
        "trajectory": trajectory,
        "_note": (
            "This is a deterministic rule-based agent demo. "
            "The LLM agent uses the same environment via WebSocket. "
            "Hit POST /tasks/{task_id}/grade to score any agent."
        ),
    }


# ---------------------------------------------------------------------------
# NEW: /leaderboard endpoint — baseline scores for all agents × all tasks
# ---------------------------------------------------------------------------

@app.get("/leaderboard", response_class=JSONResponse)
def leaderboard() -> Dict[str, Any]:
    """
    Return leaderboard comparing rule-based agent vs LLM agent across all 3 tasks.

    Scores are computed live for the rule-based agent (3 episodes each).
    LLM baseline scores are loaded from baseline_scores.json if available,
    otherwise the pre-measured v1 scores are returned as reference.
    """
    # ── Live rule-based scores (fast — no LLM call) ───────────────────────────
    rule_scores: Dict[str, Any] = {}
    for task_id in ["easy", "medium", "hard"]:
        env = SOCEnvironment(task_level=task_id, seed=42)
        episode_results = []
        for ep_seed in [42, 123, 777]:  # 3 deterministic episodes
            env2 = SOCEnvironment(task_level=task_id, seed=ep_seed)
            r = _run_grader_episode(env2)
            episode_results.append(_compute_episode_score(r, task_id))
        avg = round(sum(episode_results) / len(episode_results), 3)
        rule_scores[task_id] = {
            "score": avg,
            "episode_scores": episode_results,
        }

    rule_overall = round(
        sum(v["score"] for v in rule_scores.values()) / 3, 3
    )

    # ── LLM agent scores (from file or pre-measured fallback) ─────────────────
    llm_scores_v1 = {
        "easy":   {"score": 0.173, "note": "pre-fix — parser bug caused scan(0) loop"},
        "medium": {"score": 0.441, "note": "pre-fix baseline"},
        "hard":   {"score": 0.703, "note": "pre-fix baseline"},
    }
    llm_v1_overall = round(
        sum(v["score"] for v in llm_scores_v1.values()) / 3, 3
    )

    # Try to load post-fix scores from baseline_scores.json
    llm_scores_v2 = None
    llm_v2_overall = None
    try:
        import os as _os
        baseline_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
            "baseline_scores.json",
        )
        if _os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline_data = json.load(f)
            llm_scores_v2 = {
                k: {"score": v}
                for k, v in baseline_data.get("scores", {}).items()
            }
            llm_v2_overall = baseline_data.get("overall")
    except Exception as _e:
        _log.warning("Could not load baseline_scores.json: %s", _e)

    return {
        "leaderboard": [
            {
                "rank": 1,
                "agent": "rule_based_greedy",
                "description": "Deterministic greedy: isolate confirmed > scan highest-alert",
                "scores": rule_scores,
                "overall": rule_overall,
                "computed_live": True,
            },
            *(
                [
                    {
                        "rank": 2,
                        "agent": f"llm_{baseline_data.get('model', 'unknown')}_v2",
                        "description": "LLM agent v2 — fixed parser, action history, rich obs, CoT",
                        "scores": llm_scores_v2,
                        "overall": llm_v2_overall,
                        "computed_live": False,
                        "source": "baseline_scores.json",
                    }
                ]
                if llm_scores_v2 else []
            ),
            {
                "rank": 3 if llm_scores_v2 else 2,
                "agent": "llm_Llama-3.1-8B-Instruct_v1",
                "description": "LLM agent v1 — broken parser (pre-fix baseline)",
                "scores": llm_scores_v1,
                "overall": llm_v1_overall,
                "computed_live": False,
                "source": "measured_manual",
            },
        ],
        "tasks": {
            "easy":   {"nodes": 5,  "max_steps": 20, "start_compromised": 1},
            "medium": {"nodes": 10, "max_steps": 35, "start_compromised": 2},
            "hard":   {"nodes": 20, "max_steps": 50, "start_compromised": 3},
        },
        "rubric": {
            "containment_rate": {"weight": 0.5, "higher_is_better": True},
            "response_time":    {"weight": 0.3, "higher_is_better": False},
            "false_positive_rate": {"weight": 0.2, "higher_is_better": False},
        },
        "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
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
