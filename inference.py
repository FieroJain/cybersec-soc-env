"""
inference.py - Baseline inference script for CyberSec-SOC-OpenEnv
Follows the exact stdout format required by the hackathon:
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Fixes applied (April 2026 sprint):
  1. parse_action() smart fallback — falls back to highest-alert unscanned node
  2. Rolling action_history — last 5 actions injected into every LLM prompt
  3. Rich observation — all nodes with alert scores, connection counts, isolation status
  4. Chain-of-thought system prompt — LLM reasons before acting
  5. Score clamp — min(0.999, max(0.001, …)) everywhere, never exactly 0 or 1
  6. Topology seed — already fixed in soc_environment.py (time-based in reset())
"""

import os
import json
from typing import List, Optional, Dict, Any
from openai import OpenAI
from cybersec_soc_env import SOCEnv, SOCAction

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_URL      = os.getenv("ENV_URL", "https://Fieerawe-cybersec-soc-env.hf.space")

BENCHMARK         = "cybersec-soc-env"
MAX_STEPS         = {"easy": 20, "medium": 35, "hard": 50}
SUCCESS_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# STDOUT LOGGING — exact format required by hackathon
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# SYSTEM PROMPT — with chain-of-thought instruction (Fix #4)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Security Operations Center (SOC) analyst AI.
You are defending an enterprise network against an active cyberattack.

At each step you will see:
- Full network node statuses (alert score, connection count, isolation, scan history)
- Current attack stage and recent SIEM alerts
- Your last 5 actions (to avoid repeating ineffective moves)

REASONING PROTOCOL — follow these steps before acting:
1. ASSESS: Which nodes have the highest alert scores? Which are unscanned?
2. PRIORITISE: If a scanned node is confirmed compromised → ISOLATE it immediately.
   If high-alert nodes are unscanned → SCAN the highest one.
   If spread is fast → FIREWALL to slow attacker.
3. PICK: Choose the single best action and state it clearly.

Your response MUST contain exactly one ACTION line as the final line:
ACTION: <action_type> <node_id>

Valid action types and when to use them:
  scan <node_id>    — Reveal if node is truly compromised (do this on high-alert unscanned nodes)
  isolate <node_id> — Disconnect a CONFIRMED (scanned+compromised) node from network
  patch <node_id>   — Harden a node to slow attacker spread (+security_level)
  firewall -1       — Deploy global firewall for 10 steps (halves spread probability)
  nothing -1        — Do nothing (only if network is fully secure)

DO NOT scan the same node twice. DO NOT isolate unconfirmed nodes without good reason.
DO NOT repeat the same action you took in the last step unless you have new intelligence.

Example valid responses:
ASSESS: Node 3 has alert_score=0.82 and is unscanned. Node 1 is confirmed compromised.
PRIORITISE: Isolate node 1 first (confirmed threat), then scan node 3.
PICK: Isolate confirmed threat node 1.
ACTION: isolate 1"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _pick_smart_fallback(node_statuses: List[Dict], scanned_nodes: set) -> SOCAction:
    """
    Smart fallback when LLM fails to produce a valid action.
    Priority: highest-alert unscanned node → if none, highest-alert non-isolated node.
    Never blindly returns scan(0).
    """
    # First preference: highest-alert node that hasn't been scanned yet
    unscanned = [
        n for n in node_statuses
        if n["id"] not in scanned_nodes and not n["is_isolated"]
    ]
    if unscanned:
        best = max(unscanned, key=lambda n: n["alert_score"])
        return SOCAction(action_type="scan", target_node_id=int(best["id"]))

    # Second preference: highest-alert non-isolated node (re-scan to stay active)
    active = [n for n in node_statuses if not n["is_isolated"]]
    if active:
        best = max(active, key=lambda n: n["alert_score"])
        return SOCAction(action_type="scan", target_node_id=int(best["id"]))

    # Last resort: firewall to slow attacker while we figure things out
    return SOCAction(action_type="firewall", target_node_id=-1)


def parse_action(text: str, num_nodes: int, node_statuses: List[Dict], scanned_nodes: set) -> SOCAction:
    """
    Parse LLM response into SOCAction.

    Fix #1: Falls back to smart fallback (highest-alert unscanned node)
    instead of hardcoded scan(0) which caused infinite scan loops.
    """
    try:
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("ACTION:"):
                parts       = line.replace("ACTION:", "").strip().split()
                action_type = parts[0].lower()
                node_id     = int(parts[1]) if len(parts) > 1 else -1

                if action_type not in ("scan", "isolate", "patch", "firewall", "nothing"):
                    break  # invalid type → use smart fallback

                if action_type in ("scan", "isolate", "patch"):
                    node_id = max(0, min(node_id, num_nodes - 1))

                return SOCAction(action_type=action_type, target_node_id=node_id)
    except Exception:
        pass

    # Fix #1: Smart fallback instead of hardcoded scan(0)
    return _pick_smart_fallback(node_statuses, scanned_nodes)


def build_observation_text(obs: Any, action_history: List[str]) -> str:
    """
    Build a rich, information-dense observation string for the LLM.

    Fix #3: Include ALL nodes (not just top-3 high-alert), with alert_score,
    connection count, isolation status, and scan status.
    Fix #2: Include last 5 actions from action_history.
    """
    # Sort nodes: confirmed compromised first, then by alert score descending
    nodes_sorted = sorted(
        obs.node_statuses,
        key=lambda n: (n["visible_compromise"], n["alert_score"]),
        reverse=True,
    )

    # Build per-node summary lines
    node_lines = []
    for n in nodes_sorted:
        flags = []
        if n["visible_compromise"]:
            flags.append("⚠CONFIRMED")
        if n["is_isolated"]:
            flags.append("ISOLATED")
        flag_str = " ".join(flags) if flags else "unknown"

        # Connection count not in obs — estimate from alert level tier
        alert_tier = "HIGH" if n["alert_score"] > 0.6 else ("MED" if n["alert_score"] > 0.3 else "LOW")
        node_lines.append(
            f"  node={n['id']:2d} type={n['type']:16s} alert={n['alert_score']:.3f}({alert_tier})"
            f" isolated={str(n['is_isolated']):5s} scanned={'yes' if n['id'] in obs._scanned_ids if hasattr(obs, '_scanned_ids') else 'unk':3s}"
            f" asset={n['asset_value']:.1f} {flag_str}"
        )

    nodes_block = "\n".join(node_lines)

    # Recent alerts (last 5)
    alerts_block = "\n".join(f"  {a}" for a in (obs.alerts or [])[-5:]) or "  None"

    # Action history (last 5) — Fix #2
    history_block = (
        "\n".join(f"  step-{len(action_history)-i}: {a}" for i, a in enumerate(reversed(action_history[-5:])))
        if action_history else "  None yet"
    )

    return (
        f"=== NETWORK STATUS ===\n"
        f"Timestep: {obs.timestep} | Attack Stage: {obs.attack_stage}/4 | "
        f"Topology: {obs.topology_type} | Nodes: {len(obs.node_statuses)}\n"
        f"Business Impact Accumulated: {obs.business_impact_score:.2f}\n"
        f"\n--- ALL NODES (sorted: confirmed threats first, then by alert score) ---\n"
        f"{nodes_block}\n"
        f"\n--- RECENT SIEM ALERTS (last 5) ---\n"
        f"{alerts_block}\n"
        f"\n--- YOUR LAST 5 ACTIONS (do not repeat ineffective moves) ---\n"
        f"{history_block}\n"
        f"\n=== END STATUS — choose your next action ==="
    )


def build_observation_text_safe(obs: Any, action_history: List[str], scanned_nodes: set) -> str:
    """
    Safe version of build_observation_text that doesn't rely on internal obs attributes.
    """
    # Sort nodes: confirmed compromised first, then by alert score descending
    nodes_sorted = sorted(
        obs.node_statuses,
        key=lambda n: (n["visible_compromise"], n["alert_score"]),
        reverse=True,
    )

    # Build per-node summary lines
    node_lines = []
    for n in nodes_sorted:
        flags = []
        if n["visible_compromise"]:
            flags.append("⚠CONFIRMED_THREAT")
        if n["is_isolated"]:
            flags.append("🔒ISOLATED")
        is_scanned = n["id"] in scanned_nodes
        if is_scanned and not n["visible_compromise"] and not n["is_isolated"]:
            flags.append("✓SCANNED_CLEAN")
        flag_str = " | ".join(flags) if flags else ""

        alert_tier = "HIGH" if n["alert_score"] > 0.6 else ("MED" if n["alert_score"] > 0.3 else "LOW")

        node_lines.append(
            f"  node={n['id']:2d} | {n['type']:16s} | alert={n['alert_score']:.3f}[{alert_tier}]"
            f" | scanned={'YES' if is_scanned else 'NO ':3s} | isolated={str(n['is_isolated']):5s}"
            f" | asset={n['asset_value']:.1f} {flag_str}"
        )

    nodes_block = "\n".join(node_lines)

    # Recent alerts (last 5)
    alerts_block = "\n".join(f"  {a}" for a in (obs.alerts or [])[-5:]) or "  None"

    # Action history (last 5) — Fix #2
    if action_history:
        hist_lines = []
        for i, a in enumerate(reversed(action_history[-5:])):
            hist_lines.append(f"  t-{i+1}: {a}")
        history_block = "\n".join(hist_lines)
    else:
        history_block = "  None yet — this is your first action"

    stage_names = {1: "Initial Compromise", 2: "Credential Access",
                   3: "Lateral Movement", 4: "⚠ EXFILTRATION ACTIVE"}
    stage_name = stage_names.get(obs.attack_stage, "Unknown")

    return (
        f"=== SOC NETWORK STATUS (Timestep {obs.timestep}) ===\n"
        f"Attack Stage: {obs.attack_stage}/4 — {stage_name}\n"
        f"Topology: {obs.topology_type.upper()} | Total Nodes: {len(obs.node_statuses)}\n"
        f"Business Disruption Score: {obs.business_impact_score:.2f} (lower is better)\n"
        f"\n--- ALL NODES (confirmed threats + high-alert first) ---\n"
        f"{nodes_block}\n"
        f"\n--- RECENT SIEM ALERTS ---\n"
        f"{alerts_block}\n"
        f"\n--- YOUR RECENT ACTIONS (avoid repeating) ---\n"
        f"{history_block}\n"
        f"\n>>> Now reason step-by-step and choose your action."
    )


def compute_score(rewards: List[float], defender_wins: bool, attack_stage: int) -> float:
    """
    Compute normalised score strictly in (0.001, 0.999) — never exactly 0.0 or 1.0.

    Fix #5: All clamps use min(0.999, max(0.001, …)) to satisfy the validator.
    """
    score = 0.0

    if defender_wins:
        score += 0.5

    if attack_stage <= 2:
        score += 0.2
    elif attack_stage == 3:
        score += 0.1

    if rewards and sum(rewards) > 0:
        score += 0.2

    if rewards:
        # Clamp last reward to (0.001, 0.999) before using as sub-score
        last_r = min(0.999, max(0.001, rewards[-1]))
        score += 0.1 * last_r

    # Final clamp — never exactly 0.0 or 1.0
    return round(min(0.999, max(0.001, score)), 3)


# ---------------------------------------------------------------------------
# TASK RUNNER
# ---------------------------------------------------------------------------

def run_task(env, task_level: str) -> dict:
    """Run one episode for a task level with full [START][STEP][END] logging."""
    client    = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    max_steps = MAX_STEPS[task_level]

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    action_history: List[str] = []   # Fix #2: rolling history of action strings
    scanned_nodes: set = set()        # Fix #1: track which nodes have been scanned

    steps_taken   = 0
    defender_wins = False
    attack_stage  = 4
    score         = 0.001
    success       = False

    try:
        result = env.reset()
        obs    = result.observation

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            # Fix #3: Build rich observation text with all nodes and history
            obs_text = build_observation_text_safe(obs, action_history, scanned_nodes)

            # Get LLM action
            error_msg  = None
            action_str = None
            action     = None
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": obs_text},
                    ],
                    max_tokens=150,   # increased from 20 to allow chain-of-thought
                    temperature=0.2,  # slightly higher for better exploration
                )
                action_text = response.choices[0].message.content or ""
                # Fix #1: Pass node_statuses and scanned_nodes for smart fallback
                action = parse_action(
                    action_text,
                    len(obs.node_statuses),
                    obs.node_statuses,
                    scanned_nodes,
                )
                action_str = f"{action.action_type}({action.target_node_id})"
            except Exception as e:
                error_msg = str(e)[:80]
                # Fix #1: Use smart fallback on API error too
                action = _pick_smart_fallback(obs.node_statuses, scanned_nodes)
                action_str = f"{action.action_type}({action.target_node_id})[fallback]"

            # Track which nodes have been scanned (for smart fallback)
            if action.action_type == "scan" and action.target_node_id >= 0:
                scanned_nodes.add(action.target_node_id)

            # Fix #2: Record action in rolling history
            action_history.append(action_str)

            # Step environment
            step_result = env.step(action)
            obs         = step_result.observation
            reward      = step_result.reward or 0.0
            done        = obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if obs.defender_wins:
                defender_wins = True

            if done:
                break

        attack_stage = obs.attack_stage
        score        = compute_score(rewards, defender_wins, attack_stage)
        success      = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_level":    task_level,
        "score":         score,
        "success":       success,
        "steps":         steps_taken,
        "defender_wins": defender_wins,
        "attack_stage":  attack_stage,
    }


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    all_scores = {}

    with SOCEnv(base_url=ENV_URL).sync() as env:
        for task_level in ["easy", "medium", "hard"]:
            result = run_task(env, task_level)
            all_scores[task_level] = result["score"]

    overall = round(sum(all_scores.values()) / len(all_scores), 3)
    with open("baseline_scores.json", "w") as f:
        json.dump({"model": MODEL_NAME, "scores": all_scores, "overall": overall}, f, indent=2)


if __name__ == "__main__":
    main()
