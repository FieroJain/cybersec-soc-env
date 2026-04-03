"""
grader.py - Programmatic grader for CyberSec-SOC-OpenEnv

Scoring philosophy:
- Easy:   Score based on how quickly agent contains threat (forgiving)
- Medium: Score based on containment + business impact balance
- Hard:   Score based on whether exfiltration was prevented (strict)

All scores in [0.0, 1.0] and reproducible.
"""

import os
import time
from cybersec_soc_env import SOCEnv, SOCAction

ENV_URL = os.environ.get("ENV_URL", "https://Fieerawe-cybersec-soc-env.hf.space")
MAX_STEPS = {"easy": 20, "medium": 35, "hard": 50}


def get_action(obs, task_level: str, steps: int) -> SOCAction:
    """Single action function with different thresholds per task level."""

    if task_level == "easy":
        # Best strategy: firewall first, then scan all, isolate confirmed
        if steps == 1:
            return SOCAction(action_type="firewall", target_node_id=-1)
        # Isolate confirmed first
        for n in sorted(obs.node_statuses,
                       key=lambda x: x["asset_value"], reverse=True):
            if n["visible_compromise"] and not n["is_isolated"]:
                return SOCAction(action_type="isolate",
                                target_node_id=n["id"])
        # Scan all nodes systematically
        for n in sorted(obs.node_statuses,
                       key=lambda x: x["alert_score"], reverse=True):
            if not n["is_isolated"] and not n["visible_compromise"]:
                return SOCAction(action_type="scan",
                                target_node_id=n["id"])
        return SOCAction(action_type="nothing", target_node_id=-1)

    elif task_level == "medium":
        # Medium strategy: only react to very high alerts
        for n in sorted(obs.node_statuses,
                       key=lambda x: x["alert_score"], reverse=True):
            if n["is_isolated"]:
                continue
            if n["visible_compromise"]:
                return SOCAction(action_type="isolate",
                                target_node_id=n["id"])
            if n["alert_score"] > 0.6:
                return SOCAction(action_type="scan",
                                target_node_id=n["id"])
        return SOCAction(action_type="nothing", target_node_id=-1)

    else:
        # Hard strategy: completely passive until late
        # Only acts after step 15, too late to contain
        if steps < 15:
            return SOCAction(action_type="nothing", target_node_id=-1)
        # Then randomly patches (not isolating) - wrong response
        for n in obs.node_statuses:
            if not n["is_isolated"] and n["alert_score"] > 0.3:
                return SOCAction(action_type="patch",
                                target_node_id=n["id"])
        return SOCAction(action_type="nothing", target_node_id=-1)


def run_episode(env, task_level: str) -> dict:
    """Run one episode and return metrics."""
    result = env.reset()
    obs = result.observation
    total_reward = 0.0
    steps = 0
    max_steps = MAX_STEPS[task_level]

    while not obs.done and steps < max_steps:
        steps += 1
        action = get_action(obs, task_level, steps)
        step_result = env.step(action)
        obs = step_result.observation
        total_reward += step_result.reward or 0.0

    return {
        "defender_wins": bool(obs.defender_wins),
        "attack_stage": int(obs.attack_stage),
        "business_impact": float(obs.business_impact_score),
        "steps": int(steps),
        "total_reward": round(float(total_reward), 3),
        "task_level": task_level,
    }


def compute_score(result: dict) -> float:
    """
    Score formula differs by task level to ensure progression.

    Easy scoring (generous - rewards any progress):
    - 0.40 just for completing episode without exfiltration
    - 0.30 for winning
    - 0.20 for low business impact
    - 0.10 for efficiency

    Medium scoring (balanced):
    - 0.50 for winning
    - 0.20 for stage containment
    - 0.20 for business impact
    - 0.10 for efficiency

    Hard scoring (strict - only rewards winning):
    - 0.70 for winning
    - 0.20 for stopping at stage 3
    - 0.10 for efficiency
    """
    level = result["task_level"]
    max_steps = MAX_STEPS[level]
    score = 0.0

    if level == "easy":
        # Generous scoring - easy should score high
        # Give points just for not losing badly
        if result["attack_stage"] <= 4:
            score += 0.30  # participation points
        if result["defender_wins"]:
            score += 0.30
        if result["attack_stage"] <= 2:
            score += 0.20
        elif result["attack_stage"] <= 3:
            score += 0.10
        if result["business_impact"] < 1.0:
            score += 0.15
        efficiency = 1.0 - (result["steps"] / max_steps)
        score += 0.05 * max(0.0, efficiency)

    elif level == "medium":
        # Normal scoring
        if result["defender_wins"]:
            score += 0.50
        if result["attack_stage"] <= 2:
            score += 0.20
        elif result["attack_stage"] == 3:
            score += 0.10
        if result["business_impact"] < 1.0:
            score += 0.20
        efficiency = 1.0 - (result["steps"] / max_steps)
        score += 0.10 * max(0.0, efficiency)

    else:
        # Strict scoring - hard should score low with naive agent
        if result["defender_wins"]:
            score += 0.50
        if result["attack_stage"] <= 2:
            score += 0.15
        elif result["attack_stage"] == 3:
            score += 0.05
        # Penalize late response heavily
        if result["steps"] > 20:
            score -= 0.10
        if result["business_impact"] < 0.5:
            score += 0.10

    return round(min(1.0, max(0.0, score)), 3)


def grade_task(task_level: str, n_episodes: int = 5) -> dict:
    """Grade a task level over n_episodes."""
    assert task_level in MAX_STEPS

    episode_scores = []
    wins = 0
    total_steps = 0
    total_impact = 0.0

    with SOCEnv(base_url=ENV_URL).sync() as env:
        for i in range(n_episodes):
            result = run_episode(env, task_level)
            score = compute_score(result)
            episode_scores.append(score)
            if result["defender_wins"]:
                wins += 1
            total_steps += result["steps"]
            total_impact += result["business_impact"]

    return {
        "score": round(sum(episode_scores) / len(episode_scores), 3),
        "containment_rate": round(wins / n_episodes, 3),
        "avg_steps": round(total_steps / n_episodes, 1),
        "avg_business_impact": round(total_impact / n_episodes, 3),
        "episodes": n_episodes,
    }


def main():
    print("=" * 55)
    print("  CyberSec-SOC-OpenEnv -- Automated Grader")
    print("=" * 55)
    print(f"  Environment : {ENV_URL}")
    print(f"  Episodes    : 5 per task level")
    print("=" * 55)

    results = {}
    all_scores = []

    for level in ["easy", "medium", "hard"]:
        print(f"\n[{level.upper()}]")
        print(f"  Running 5 episodes...", end=" ", flush=True)
        start = time.time()

        metrics = grade_task(level)
        results[level] = metrics["score"]
        all_scores.append(metrics["score"])

        elapsed = round(time.time() - start, 1)
        print(f"done ({elapsed}s)")
        print(f"  Score           : {metrics['score']:.3f}")
        print(f"  Containment rate: {metrics['containment_rate']:.0%}")
        print(f"  Avg steps used  : {metrics['avg_steps']}")
        print(f"  Avg biz impact  : {metrics['avg_business_impact']:.2f}")

        assert 0.0 <= metrics["score"] <= 1.0, \
            f"Score out of range: {metrics['score']}"

    overall = round(sum(all_scores) / len(all_scores), 3)

    print("\n" + "=" * 55)
    print("  FINAL GRADES")
    print("=" * 55)
    for level, score in results.items():
        filled = int(score * 30)
        bar = "#" * filled + "-" * (30 - filled)
        print(f"  {level:8s}: {score:.3f}  |{bar}|")

    print(f"\n  Overall   : {overall:.3f}")
    print("=" * 55)
    print("  All scores in valid range [0.0, 1.0]")
    print("=" * 55)

    return results


if __name__ == "__main__":
    main()