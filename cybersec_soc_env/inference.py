"""
inference.py - Baseline inference script for CyberSec-SOC-OpenEnv
Follows the exact stdout format required by the hackathon:
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
import time
from typing import List, Optional
from openai import OpenAI
from cybersec_soc_env import SOCEnv, SOCAction

# CONFIG
# Defaults set ONLY for API_BASE_URL and MODEL_NAME (NOT HF_TOKEN)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_URL      = os.getenv("ENV_URL", "https://Fieerawe-cybersec-soc-env.hf.space")

BENCHMARK    = "cybersec-soc-env"
MAX_STEPS    = {"easy": 20, "medium": 35, "hard": 50}
SUCCESS_THRESHOLD = 0.4


# STDOUT LOGGING - exact format required by hackathon
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
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


# SYSTEM PROMPT
SYSTEM_PROMPT = """You are a Security Operations Center (SOC) analyst.
You must defend a network against a cyberattack.
At each step you will see network node statuses and recent alerts.

Choose ONE action from:
- scan <node_id>   : reveal if a node is truly compromised
- isolate <node_id>: disconnect a node from the network
- patch <node_id>  : harden a node against attack
- firewall         : deploy firewall to slow attacker
- nothing          : do nothing

Reply with ONLY one line in this exact format:
ACTION: <action_type> <node_id>
Examples:
ACTION: scan 3
ACTION: isolate 7
ACTION: firewall -1
ACTION: nothing -1"""


def parse_action(text: str, num_nodes: int) -> SOCAction:
    """Parse LLM response into SOCAction. Falls back safely."""
    try:
        for line in text.strip().split("\n"):
            if line.startswith("ACTION:"):
                parts = line.replace("ACTION:", "").strip().split()
                action_type = parts[0].lower()
                node_id = int(parts[1]) if len(parts) > 1 else -1
                if action_type in ("scan", "isolate", "patch"):
                    node_id = max(0, min(node_id, num_nodes - 1))
                return SOCAction(action_type=action_type, target_node_id=node_id)
    except Exception:
        pass
    return SOCAction(action_type="scan", target_node_id=0)


def compute_score(rewards: List[float], defender_wins: bool, attack_stage: int) -> float:
    """Compute normalized score 0.0 to 1.0."""
    score = 0.0
    if defender_wins:
        score += 0.5
    if attack_stage <= 2:
        score += 0.2
    elif attack_stage == 3:
        score += 0.1
    if sum(rewards) > 0:
        score += 0.2
    if len(rewards) > 0:
        score += 0.1 * min(0.999, max(0.001, rewards[-1] if rewards else 0))
    return round(min(0.999, max(0.001, score)), 3)


def run_task(env, task_level: str) -> dict:
    """Run one episode for a task level with full [START][STEP][END] logging."""
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    max_steps = MAX_STEPS[task_level]

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    defender_wins = False
    attack_stage = 4
    score = 0.0
    success = False

    try:
        result = env.reset()
        obs = result.observation

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            # Build observation for LLM
            high_alert = [
                n for n in obs.node_statuses
                if n["alert_score"] > 0.5 and not n["is_isolated"]
            ]
            obs_text = (
                f"TIMESTEP: {obs.timestep} | "
                f"ATTACK STAGE: {obs.attack_stage}/4 | "
                f"TOPOLOGY: {obs.topology_type} | "
                f"HIGH ALERT NODES: {json.dumps(high_alert[:3])} | "
                f"ALERTS: {obs.alerts[-2:]}"
            )

            # Get LLM action
            error_msg = None
            action_str = "scan 0"
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs_text},
                    ],
                    max_tokens=20,
                    temperature=0.1,
                )
                action_text = response.choices[0].message.content or ""
                action = parse_action(action_text, len(obs.node_statuses))
                action_str = f"{action.action_type}({action.target_node_id})"
            except Exception as e:
                error_msg = str(e)[:50]
                action = SOCAction(action_type="scan", target_node_id=0)
                action_str = "scan(0)"

            # Step environment
            step_result = env.step(action)
            obs = step_result.observation
            reward = step_result.reward or 0.0
            done = obs.done

            rewards.append(reward)
            steps_taken = step

            # Log step in required format
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

            if obs.defender_wins:
                defender_wins = True

            if done:
                break

        attack_stage = obs.attack_stage
        score = compute_score(rewards, defender_wins, attack_stage)
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_level": task_level,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "defender_wins": defender_wins,
        "attack_stage": attack_stage,
    }


def main():
    all_scores = {}

    with SOCEnv(base_url=ENV_URL).sync() as env:
        for task_level in ["easy", "medium", "hard"]:
            result = run_task(env, task_level)
            all_scores[task_level] = result["score"]

    # Save scores
    overall = round(sum(all_scores.values()) / len(all_scores), 3)
    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "scores": all_scores,
            "overall": overall,
        }, f, indent=2)


if __name__ == "__main__":
    main()
