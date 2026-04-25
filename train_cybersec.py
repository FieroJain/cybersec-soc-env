# /// script
# dependencies = [
#   "unsloth",
#   "trl",
#   "datasets",
#   "matplotlib",
#   "numpy",
#   "requests",
#   "peft",
#   "accelerate",
#   "bitsandbytes",
# ]
# ///

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import requests
import matplotlib.pyplot as plt
import numpy as np

ENV_URL = "https://Fieerawe-cybersec-soc-env.hf.space"

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=512,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

def get_obs():
    try:
        r = requests.post(ENV_URL + "/reset", timeout=10)
        obs = r.json()["observation"]
        nodes = obs["node_statuses"][:5]
        node_text = "\n".join([
            f"Node {n['id']} ({n['type']}): alert={n['alert_score']:.2f} compromised={n['visible_compromise']}"
            for n in nodes
        ])
        return f"""You are a SOC analyst. Network under attack.
Attack Stage: {obs['attack_stage']}/4
Topology: {obs['topology_type']}

NODES:
{node_text}

Choose action:
ACTION: scan <node_id>
ACTION: isolate <node_id>
ACTION: firewall -1
ACTION: patch <node_id>
ACTION: nothing -1"""
    except:
        return "ACTION: scan 0"

def cybersec_reward(completions, prompts=None, **kwargs):
    rewards = []
    for completion in completions:
        try:
            text = completion if isinstance(completion, str) else str(completion)
            action_type = "scan"
            node_id = 0
            if "ACTION:" in text:
                parts = text.split("ACTION:")[-1].strip().split()
                if len(parts) >= 2:
                    action_type = parts[0].lower().strip()
                    node_id = int(parts[1]) if parts[1] != "-1" else -1
            if action_type not in ["scan","isolate","patch","firewall","nothing"]:
                action_type = "scan"
            r = requests.post(
                ENV_URL + "/step",
                json={"action": {"action_type": action_type, "target_node_id": node_id}},
                timeout=5
            )
            reward = float(r.json().get("reward", 0.0))
            rewards.append(min(0.999, max(0.001, reward)))
        except:
            rewards.append(0.001)
    return rewards

print("Building dataset...")
prompts = [{"prompt": get_obs()} for _ in range(40)]
dataset = Dataset.from_list(prompts)

config = GRPOConfig(
    output_dir="./cybersec-grpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=60,
    max_prompt_length=350,
    learning_rate=5e-5,
    max_steps=100,
    logging_steps=1,
    save_steps=50,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=cybersec_reward,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("Starting GRPO training...")
trainer.train()

log_history = trainer.state.log_history
steps = [x["step"] for x in log_history if "reward" in x]
rewards = [x["reward"] for x in log_history if "reward" in x]

if steps:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, rewards, color="#ef4444", linewidth=2.5, label="Training Reward")
    ax.fill_between(steps, rewards, alpha=0.15, color="#ef4444")
    z = np.polyfit(steps, rewards, 1)
    p = np.poly1d(z)
    ax.plot(steps, p(steps), "--", color="#22c55e", linewidth=2,
            label=f"Trend: +{z[0]:.4f}/step")
    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Average Reward", fontsize=13)
    ax.set_title("CyberSec-SOC-OpenEnv: GRPO Training Reward Curve\nQwen2.5-1.5B + LoRA | Topology Curriculum",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    if len(rewards) > 1:
        improvement = rewards[-1] - rewards[0]
        ax.annotate(
            f"Improvement: {improvement:+.3f}",
            xy=(steps[-1], rewards[-1]),
            xytext=(steps[30], max(rewards)*0.85),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
            color="green", fontsize=13, fontweight="bold"
        )
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    print(f"\nStart reward: {rewards[0]:.3f}")
    print(f"End reward:   {rewards[-1]:.3f}")
    print(f"Improvement:  {rewards[-1]-rewards[0]:+.3f}")
    print("training_curve.png saved!")
else:
    print("Training done but no reward logs found.")