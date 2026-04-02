"""
inference.py — Baseline inference script for CyberSec-SOC-OpenEnv
Uses OpenAI client to run an LLM agent against the SOC environment.
Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
"""

import os
import json
import time
from openai import OpenAI
from cybersec_soc_env import SOCEnv, SOCAction

# ── CONFIG ────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "https://Fieerawe-cybersec-soc-env.hf.space")

SYSTEM_PROMPT = """You are a Security Operations Center (SOC) analyst.
You must defend a network against a cyberattack.
At each step you will see:
- node_statuses: list of nodes with alert_score (0=clean, 1=compromised)
- attack_stage: 1=Initial, 2=Credential, 3=Lateral, 4=Exfiltration
- alerts: recent security alerts
- business_impact_score: cost of your isolation actions

Choose ONE action from:
- scan <node_id>      : reveal if a node is truly compromised
- isolate <node_id>   : disconnect a node from the network
- patch <node_id>     : harden a node against attack
- firewall            : deploy firewall to slow attacker
- nothing             : do nothing (use rarely)

Reply with ONLY one line in this exact format:
ACTION: <action_type> <node_id>
Examples:
ACTION: scan 3
ACTION: isolate 7
ACTION: firewall -1
ACTION: nothing -1
"""

def parse_llm_action(text: str, num_nodes: int) -> SOCAction:
    """Parse LLM response into SOCAction. Falls back to scan node 0."""
    try:
        for line in text.strip().split("\n"):
            if line.startswith("ACTION:"):
                parts = line.replace("ACTION:", "").strip().split()
                action_type = parts[0].lower()
                node_id = int(parts[1]) if len(parts) > 1 else -1
                # Validate node_id range
                if action_type in ("scan", "isolate", "patch"):
                    node_id = max(0, min(node_id, num_nodes - 1))
                return SOCAction(action_type=action_type, target_node_id=node_id)
    except Exception:
        pass
    # Safe fallback
    return SOCAction(action_type="scan", target_node_id=0)

def run_episode(env, task_level: str, max_steps: int = 50) -> dict:
    """Run one episode and return results."""
    result = env.reset()
    obs = result.observation
    
    total_reward = 0.0
    steps = 0
    defender_wins = False
    
    while not obs.done and steps < max_steps:
        # Build observation text for LLM
        high_alert_nodes = [
            n for n in obs.node_statuses 
            if n["alert_score"] > 0.5 and not n["is_isolated"]
        ]
        obs_text = f"""
TIMESTEP: {obs.timestep}
ATTACK STAGE: {obs.attack_stage}/4
TOPOLOGY: {obs.topology_type}
BUSINESS IMPACT: {obs.business_impact_score}

HIGH ALERT NODES (alert_score > 0.5):
{json.dumps(high_alert_nodes[:5], indent=2)}

ALL NODES SUMMARY:
Total nodes: {len(obs.node_statuses)}
Isolated: {sum(1 for n in obs.node_statuses if n['is_isolated'])}
Visible compromises: {sum(1 for n in obs.node_statuses if n['visible_compromise'])}

RECENT ALERTS:
{chr(10).join(obs.alerts[-3:])}
"""
        # LLM decides action
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs_text}
                ],
                max_tokens=50,
                temperature=0.1,
            )
            action_text = response.choices[0].message.content
        except Exception as e:
            print(f"  LLM error: {e}, using fallback action")
            action_text = "ACTION: scan 0"
        
        action = parse_llm_action(action_text, len(obs.node_statuses))
        step_result = env.step(action)
        obs = step_result.observation
        total_reward += step_result.reward or 0.0
        steps += 1
        
        if obs.defender_wins:
            defender_wins = True
    
    return {
        "task_level": task_level,
        "steps_taken": steps,
        "total_reward": round(total_reward, 3),
        "defender_wins": defender_wins,
        "attack_stage_reached": obs.attack_stage,
        "business_impact": obs.business_impact_score,
    }

def compute_score(episode_result: dict, task_level: str) -> float:
    """
    Compute grader score 0.0 to 1.0.
    
    Score components:
    - 0.5 if defender wins
    - 0.2 for stopping at stage <= 2 (early containment)
    - 0.2 for low business impact (< 1.0)
    - 0.1 for efficiency (fewer steps = better)
    
    Max steps per task: easy=20, medium=35, hard=50
    """
    max_steps = {"easy": 20, "medium": 35, "hard": 50}[task_level]
    score = 0.0
    
    # Win condition
    if episode_result["defender_wins"]:
        score += 0.5
    
    # Early containment bonus
    if episode_result["attack_stage_reached"] <= 2:
        score += 0.2
    elif episode_result["attack_stage_reached"] == 3:
        score += 0.1
    
    # Business impact penalty avoidance
    if episode_result["business_impact"] < 1.0:
        score += 0.2
    elif episode_result["business_impact"] < 2.0:
        score += 0.1
    
    # Efficiency bonus
    efficiency = 1.0 - (episode_result["steps_taken"] / max_steps)
    score += 0.1 * max(0.0, efficiency)
    
    return round(min(1.0, max(0.0, score)), 3)

def main():
    print("=" * 60)
    print("CyberSec-SOC-OpenEnv — Baseline Inference Script")
    print("=" * 60)
    print(f"Model:   {MODEL_NAME}")
    print(f"API URL: {API_BASE_URL}")
    print(f"Env URL: {ENV_URL}")
    print("=" * 60)
    
    scores = {}
    
    with SOCEnv(base_url=ENV_URL).sync() as env:
        for task_level in ["easy", "medium", "hard"]:
            print(f"\n[TASK: {task_level.upper()}]")
            start = time.time()
            
            # Run 3 episodes per task, average the scores
            task_scores = []
            for episode in range(3):
                print(f"  Episode {episode + 1}/3 ...", end=" ", flush=True)
                result = run_episode(env, task_level)
                score = compute_score(result, task_level)
                task_scores.append(score)
                print(f"score={score:.3f} | win={result['defender_wins']} | "
                      f"stage={result['attack_stage_reached']}")
            
            avg_score = round(sum(task_scores) / len(task_scores), 3)
            scores[task_level] = avg_score
            elapsed = round(time.time() - start, 1)
            print(f"  → Average score: {avg_score:.3f} ({elapsed}s)")
    
    print("\n" + "=" * 60)
    print("BASELINE SCORES")
    print("=" * 60)
    for level, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {level:8s}: {score:.3f}  [{bar:<20}]")
    
    overall = round(sum(scores.values()) / len(scores), 3)
    print(f"\n  OVERALL : {overall:.3f}")
    print("=" * 60)
    
    # Save scores to file for judges
    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "scores": scores,
            "overall": overall,
        }, f, indent=2)
    print("Scores saved to baseline_scores.json")

if __name__ == "__main__":
    main()
