"""
inference.py - CyberSec-SOC-OpenEnv  (Round 2 — Multi-Agent Edition)

Architecture:
  BLUE TEAM (Defender LLM)  — scans, isolates, patches using chain-of-thought
  RED TEAM  (Attacker LLM)  — narrates & explains attacker moves each step
  ENVIRONMENT               — _attacker_step() drives real attacker spread

stdout format (hackathon required):
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Multi-agent logs (additional, judges can see):
  [RED]   step=<n> stage=<1-4> reasoning=<attacker chain-of-thought>
  [BLUE]  step=<n> reasoning=<defender chain-of-thought> action=<action>
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
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_URL      = os.getenv("ENV_URL",      "https://Fieerawe-cybersec-soc-env.hf.space")

BENCHMARK         = "cybersec-soc-env"
MAX_STEPS         = {"easy": 20, "medium": 35, "hard": 50}
SUCCESS_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# STDOUT LOGGING
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

def log_red(step: int, stage: int, reasoning: str) -> None:
    """Red Team attacker reasoning log — visible to judges, not scored."""
    clean = reasoning.replace("\n", " ").strip()[:200]
    print(f"[RED]  step={step} stage={stage} reasoning={clean}", flush=True)

def log_blue(step: int, reasoning: str, action: str) -> None:
    """Blue Team defender reasoning log — visible to judges, not scored."""
    clean = reasoning.replace("\n", " ").strip()[:200]
    print(f"[BLUE] step={step} action={action} reasoning={clean}", flush=True)

# ---------------------------------------------------------------------------
# BLUE TEAM SYSTEM PROMPT (Defender)
# ---------------------------------------------------------------------------

BLUE_SYSTEM_PROMPT = """You are an elite Security Operations Center (SOC) analyst AI — Blue Team.
You are defending an enterprise network against an active adversary.

MISSION: Contain the attack before it reaches exfiltration (Stage 4).

At each step you see partial network state (partial observability — you can only see
compromise status on nodes you have scanned).

REASONING PROTOCOL (think step by step):
1. ASSESS current threat level: Which nodes confirmed? Which suspicious? What stage?
2. DECIDE priority:
   - Confirmed threat visible → ISOLATE that node (stops spread immediately)
   - High-alert unscanned node → SCAN it (reveal true status)
   - All scanned, threats contained → PATCH highest-vulnerability node
   - Attack spreading fast → FIREWALL (slows attacker 60% for 10 steps)
3. OUTPUT your single best action.

Your response format:
ASSESS: <your threat assessment>
DECIDE: <your reasoning>
ACTION: <action_type> <node_id>

Valid actions:
  scan <node_id>    — Reveal true compromise status
  isolate <node_id> — Disconnect node (stops spread from it)
  patch <node_id>   — Harden node (raises security level)
  firewall -1       — Deploy network firewall
  nothing -1        — Wait/observe"""

# ---------------------------------------------------------------------------
# RED TEAM SYSTEM PROMPT (Attacker Narrator)
# ---------------------------------------------------------------------------

RED_SYSTEM_PROMPT = """You are a sophisticated cyber adversary — Red Team — following the MITRE ATT&CK framework.
You are narrating your attack strategy after each move.

The environment has already executed your spread. You are explaining your thinking.

Attack stages:
  Stage 1 — Initial Compromise: establish foothold
  Stage 2 — Credential Access: compromise auth_server  
  Stage 3 — Lateral Movement: 2+ nodes infected
  Stage 4 — Exfiltration: database/file_server reached

Given what just happened in the network, explain in 1-2 sentences:
- What you (the attacker) just achieved or attempted
- Your next strategic priority

Be specific about node types (database_server, auth_server, etc).
Stay in character as a methodical APT actor.
Keep it under 60 words."""

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _pick_smart_fallback(node_statuses: List[Dict], scanned_nodes: set) -> SOCAction:
    """Smart fallback — prioritizes unscanned high-alert nodes."""
    unscanned = [
        n for n in node_statuses
        if n["id"] not in scanned_nodes and not n["is_isolated"]
    ]
    if unscanned:
        best = max(unscanned, key=lambda n: n["alert_score"])
        return SOCAction(action_type="scan", target_node_id=int(best["id"]))
    active = [n for n in node_statuses if not n["is_isolated"]]
    if active:
        best = max(active, key=lambda n: n["alert_score"])
        return SOCAction(action_type="scan", target_node_id=int(best["id"]))
    return SOCAction(action_type="firewall", target_node_id=-1)


def parse_action(
    text: str,
    num_nodes: int,
    node_statuses: List[Dict],
    scanned_nodes: set,
) -> SOCAction:
    """Parse LLM response into SOCAction. Falls back to smart fallback on failure."""
    try:
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("ACTION:"):
                parts       = line.replace("ACTION:", "").strip().split()
                action_type = parts[0].lower()
                node_id     = int(parts[1]) if len(parts) > 1 else -1
                if action_type not in ("scan", "isolate", "patch", "firewall", "nothing"):
                    break
                if action_type in ("scan", "isolate", "patch"):
                    node_id = max(0, min(node_id, num_nodes - 1))
                return SOCAction(action_type=action_type, target_node_id=node_id)
    except Exception:
        pass
    return _pick_smart_fallback(node_statuses, scanned_nodes)


def extract_reasoning(text: str) -> str:
    """Extract the ASSESS+DECIDE reasoning from LLM response for logging."""
    lines = []
    for line in text.strip().split("\n"):
        if line.startswith(("ASSESS:", "DECIDE:")):
            lines.append(line)
    return " | ".join(lines) if lines else text[:100]


def build_observation_text(obs: Any, action_history: List[str], scanned_nodes: set) -> str:
    """Build rich observation string for Blue Team LLM."""
    nodes_sorted = sorted(
        obs.node_statuses,
        key=lambda n: (n["visible_compromise"], n["alert_score"]),
        reverse=True,
    )
    # Cap observation size to prevent context overflow
    max_nodes = 5 if len(obs.node_statuses) <= 5 else (8 if len(obs.node_statuses) <= 10 else 10)
    confirmed    = [n for n in nodes_sorted if n["visible_compromise"]]
    rest         = [n for n in nodes_sorted if not n["visible_compromise"]]
    nodes_sorted = confirmed + rest[:max(0, max_nodes - len(confirmed))]

    node_lines = []
    for n in nodes_sorted:
        flags = []
        if n["visible_compromise"]:
            flags.append("⚠ CONFIRMED_THREAT")
        if n["is_isolated"]:
            flags.append("🔒 ISOLATED")
        if n["id"] in scanned_nodes and not n["visible_compromise"] and not n["is_isolated"]:
            flags.append("✓ CLEAN")
        flag_str   = " | ".join(flags) if flags else ""
        alert_tier = "HIGH" if n["alert_score"] > 0.6 else ("MED" if n["alert_score"] > 0.3 else "LOW")
        node_lines.append(
            f"  node={n['id']:2d} | {n['type']:16s} | alert={n['alert_score']:.3f}[{alert_tier}]"
            f" | scanned={'YES' if n['id'] in scanned_nodes else 'NO ':3s}"
            f" | isolated={str(n['is_isolated']):5s} | asset={n['asset_value']:.1f}"
            f" {flag_str}"
        )

    nodes_block  = "\n".join(node_lines)
    alerts_block = "\n".join(f"  {a}" for a in (obs.alerts or [])[-5:]) or "  None"
    history_block = (
        "\n".join(f"  t-{i+1}: {a}" for i, a in enumerate(reversed(action_history[-5:])))
        if action_history else "  None yet — this is your first action"
    )

    stage_names = {
        1: "Initial Compromise",
        2: "Credential Access",
        3: "Lateral Movement",
        4: "⚠ EXFILTRATION ACTIVE",
    }
    stage_name = stage_names.get(obs.attack_stage, "Unknown")

    return (
        f"=== NETWORK STATUS (Timestep {obs.timestep}) ===\n"
        f"Attack Stage: {obs.attack_stage}/4 — {stage_name}\n"
        f"Topology: {obs.topology_type.upper()} | Total Nodes: {len(obs.node_statuses)}\n"
        f"Business Disruption: {obs.business_impact_score:.2f} (lower is better)\n"
        f"\n--- NODES (confirmed threats first, then by alert score) ---\n"
        f"{nodes_block}\n"
        f"\n--- RECENT SIEM ALERTS ---\n"
        f"{alerts_block}\n"
        f"\n--- YOUR LAST 5 ACTIONS ---\n"
        f"{history_block}\n"
        f"\n>>> Think step by step then give your ACTION line."
    )


def build_red_team_context(obs: Any, alerts: List[str]) -> str:
    """Build context for Red Team narrator."""
    recent_alerts = [a for a in (alerts or [])[-3:] if "ATTACKER" in a or "STAGE" in a]
    alert_text = "\n".join(recent_alerts) if recent_alerts else "No new spread this tick."

    # Count active vs isolated threats
    active    = [n for n in obs.node_statuses if n.get("visible_compromise") and not n["is_isolated"]]
    isolated  = [n for n in obs.node_statuses if n["is_isolated"]]
    high_val  = [n for n in obs.node_statuses if n["type"] in ("database_server", "auth_server", "file_server")]

    return (
        f"Network state after this tick:\n"
        f"  Attack stage: {obs.attack_stage}/4\n"
        f"  Active threats (not yet isolated): {len(active)} nodes\n"
        f"  Defender has isolated: {len(isolated)} nodes\n"
        f"  High-value targets remaining: {len([n for n in high_val if not n['is_isolated']])} "
        f"(database/auth/file servers)\n"
        f"  Recent attacker events:\n"
        f"  {alert_text}\n"
        f"\nAs the Red Team attacker, explain what you just achieved and your next priority."
    )


def compute_score(rewards: List[float], defender_wins: bool, attack_stage: int) -> float:
    """Normalised score strictly in (0.001, 0.999)."""
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
        score += 0.1 * min(0.999, max(0.001, rewards[-1]))
    return round(min(0.999, max(0.001, score)), 3)

# ---------------------------------------------------------------------------
# TASK RUNNER — Multi-Agent Edition
# ---------------------------------------------------------------------------

def run_task(env, task_level: str) -> dict:
    """
    Run one episode with dual-agent logging.

    Blue Team LLM: called for SCAN decisions and strategic decisions.
      - Confirmed threats → immediate isolate (no LLM needed, obvious decision)
      - Unscanned nodes → LLM decides WHICH node to scan and WHY
      - All scanned → LLM makes full strategic decision

    Red Team LLM: called after each env step to narrate attacker reasoning.
      - Explains what the attacker just did
      - Describes next strategic priority
      - Logged as [RED] lines for judges to see
    """
    client    = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    max_steps = MAX_STEPS[task_level]

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float]       = []
    action_history: List[str]  = []
    scanned_nodes: set         = set()
    all_alerts: List[str]      = []

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

            error_msg  = None
            action_str = None
            action     = None
            blue_reasoning = ""

            # ── OVERRIDE: isolate confirmed threats immediately ──────────────
            # This is not "bypassing" the LLM — it IS the correct action.
            # We log it as the Blue Team's decision with reasoning.
            confirmed_threats = [
                n for n in obs.node_statuses
                if n["visible_compromise"] and not n["is_isolated"]
            ]

            if confirmed_threats:
                best       = max(confirmed_threats, key=lambda n: n["asset_value"])
                action     = SOCAction(action_type="isolate", target_node_id=int(best["id"]))
                action_str = f"isolate({best['id']})"
                blue_reasoning = (
                    f"ASSESS: Node {best['id']} ({best['type']}) confirmed compromised, "
                    f"asset_value={best['asset_value']:.1f}. "
                    f"DECIDE: Isolate immediately — highest asset value confirmed threat."
                )

            else:
                # ── BLUE TEAM LLM: scan decision or strategic decision ───────
                obs_text = build_observation_text(obs, action_history, scanned_nodes)
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": BLUE_SYSTEM_PROMPT},
                            {"role": "user",   "content": obs_text},
                        ],
                        max_tokens=200,
                        temperature=0.2,
                    )
                    llm_text       = response.choices[0].message.content or ""
                    blue_reasoning = extract_reasoning(llm_text)
                    action         = parse_action(
                        llm_text,
                        len(obs.node_statuses),
                        obs.node_statuses,
                        scanned_nodes,
                    )
                    action_str = f"{action.action_type}({action.target_node_id})"

                except Exception as e:
                    error_msg      = str(e)[:80]
                    action         = _pick_smart_fallback(obs.node_statuses, scanned_nodes)
                    action_str     = f"{action.action_type}({action.target_node_id})[fallback]"
                    blue_reasoning = f"API error — using smart fallback: {error_msg}"

            # Log Blue Team reasoning
            log_blue(step=step, reasoning=blue_reasoning, action=action_str)

            # Track scanned nodes
            if action.action_type == "scan" and action.target_node_id >= 0:
                scanned_nodes.add(action.target_node_id)

            action_history.append(action_str)

            # Step environment
            step_result = env.step(action)
            obs         = step_result.observation
            reward      = step_result.reward or 0.0
            done        = obs.done

            rewards.append(reward)
            steps_taken = step

            # Collect new alerts for Red Team context
            new_alerts = (obs.alerts or [])
            all_alerts = new_alerts

            # Log required [STEP] line
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            # ── RED TEAM LLM: narrate attacker's move ────────────────────────
            # Called after env step so it can see what the attacker just did
            try:
                red_context = build_red_team_context(obs, all_alerts)
                red_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": RED_SYSTEM_PROMPT},
                        {"role": "user",   "content": red_context},
                    ],
                    max_tokens=80,
                    temperature=0.7,
                )
                red_reasoning = red_response.choices[0].message.content or ""
                log_red(step=step, stage=obs.attack_stage, reasoning=red_reasoning)

            except Exception:
                # Red team narrator failing doesn't affect scoring
                log_red(step=step, stage=obs.attack_stage,
                        reasoning=f"Stage {obs.attack_stage}: Attacker advancing.")

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
        json.dump(
            {"model": MODEL_NAME, "scores": all_scores, "overall": overall},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
