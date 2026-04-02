"""
grader.py — Programmatic grader for CyberSec-SOC-OpenEnv
Runs deterministic grading for all 3 tasks.
Returns scores between 0.0 and 1.0.
"""

import sys
from cybersec_soc_env import SOCEnv, SOCAction

ENV_URL = "http://localhost:8000"

def random_agent_episode(env, task_level: str, seed: int = 0) -> dict:
    """Run episode with deterministic rule-based agent for reproducible grading."""
    import numpy as np
    rng = np.random.default_rng(seed)
    
    result = env.reset()
    obs = result.observation
    total_reward = 0.0
    steps = 0
    
    while not obs.done and steps < 50:
        # Rule-based: scan highest alert node, then isolate if confirmed
        nodes = sorted(
            obs.node_statuses,
            key=lambda n: n["alert_score"],
            reverse=True
        )
        
        action = SOCAction(action_type="nothing", target_node_id=-1)
        
        for node in nodes:
            if node["is_isolated"]:
                continue
            if node["visible_compromise"]:
                action = SOCAction(
                    action_type="isolate",
                    target_node_id=node["id"]
                )
                break
            if node["alert_score"] > 0.5:
                action = SOCAction(
                    action_type="scan",
                    target_node_id=node["id"]
                )
                break
        
        step_result = env.step(action)
        obs = step_result.observation
        total_reward += step_result.reward or 0.0
        steps += 1
    
    return {
        "defender_wins": obs.defender_wins,
        "attack_stage": obs.attack_stage,
        "business_impact": obs.business_impact_score,
        "steps": steps,
        "total_reward": total_reward,
    }

def grade_task(task_level: str, n_episodes: int = 5) -> float:
    """
    Grade a task level with deterministic agent.
    Returns score 0.0 to 1.0.
    Score is reproducible given same seed.
    """
    max_steps = {"easy": 20, "medium": 35, "hard": 50}[task_level]
    episode_scores = []
    
    with SOCEnv(base_url=ENV_URL).sync() as env:
        for i in range(n_episodes):
            result = random_agent_episode(env, task_level, seed=i)
            
            score = 0.0
            if result["defender_wins"]:
                score += 0.5
            if result["attack_stage"] <= 2:
                score += 0.2
            elif result["attack_stage"] == 3:
                score += 0.1
            if result["business_impact"] < 1.0:
                score += 0.2
            efficiency = 1.0 - (result["steps"] / max_steps)
            score += 0.1 * max(0.0, efficiency)
            
            episode_scores.append(round(min(1.0, max(0.0, score)), 3))
    
    return round(sum(episode_scores) / len(episode_scores), 3)

def main():
    print("=" * 50)
    print("CyberSec-SOC-OpenEnv Grader")
    print("=" * 50)
    
    results = {}
    for level in ["easy", "medium", "hard"]:
        print(f"Grading {level}...", end=" ", flush=True)
        score = grade_task(level)
        results[level] = score
        print(f"score = {score:.3f}")
    
    print("\nFINAL GRADES:")
    for level, score in results.items():
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        print(f"  {level}: {score:.3f}")
    
    overall = round(sum(results.values()) / 3, 3)
    print(f"  overall: {overall:.3f}")
    print("\nAll scores in valid range 0.0-1.0 ✅")
    return results

if __name__ == "__main__":
    main()
