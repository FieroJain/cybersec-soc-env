from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from huggingface_hub import HfApi
import requests
import os
import matplotlib.pyplot as plt
import numpy as np

ENV_URL = "https://Fieerawe-cybersec-soc-env.hf.space"
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = "Fieerawe"
MODEL_REPO = f"{HF_USERNAME}/cybersec-soc-defender"

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
            f"Node {n['id']} ({n['type']}): alert={n['alert_score']:.2f}"
            for n in nodes
        ])
        return f"""You are a SOC analyst defending a network.
Attack Stage: {obs['attack_stage']}/4
Topology: {obs['topology_type']}
NODES:
{node_text}
Choose: ACTION: scan/isolate/firewall/patch/nothing <node_id>"""
    except:
        return "ACTION: firewall -1"

def cybersec_reward(completions, prompts=None, **kwargs):
    rewards = []
    for completion in completions:
        try:
            text = str(completion)
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
                json={"action": {
                    "action_type": action_type,
                    "target_node_id": node_id
                }},
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
    output_dir="./cybersec-grpo-v2",
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

print("Training...")
trainer.train()

# Save reward curve
log_history = trainer.state.log_history
steps = [x["step"] for x in log_history if "reward" in x]
rewards_list = [x["reward"] for x in log_history if "reward" in x]

if steps:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, rewards_list, color="#ef4444", linewidth=2.5, label="Training Reward")
    ax.fill_between(steps, rewards_list, alpha=0.15, color="#ef4444")
    z = np.polyfit(steps, rewards_list, 1)
    p = np.poly1d(z)
    ax.plot(steps, p(steps), "--", color="#22c55e", linewidth=2,
            label=f"Trend: +{z[0]:.4f}/step")
    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Average Reward", fontsize=13)
    ax.set_title("CyberSec-SOC-OpenEnv: GRPO Training Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    if len(rewards_list) > 1:
        improvement = rewards_list[-1] - rewards_list[0]
        ax.annotate(
            f"Improvement: {improvement:+.3f}",
            xy=(steps[-1], rewards_list[-1]),
            xytext=(steps[len(steps)//3], max(rewards_list)*0.85),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
            color="green", fontsize=13, fontweight="bold"
        )
    plt.tight_layout()
    plt.savefig("training_curve_v2.png", dpi=150, bbox_inches="tight")
    print(f"Start: {rewards_list[0]:.3f} End: {rewards_list[-1]:.3f}")
    print(f"Improvement: {rewards_list[-1]-rewards_list[0]:+.3f}")

# Push model to HF Hub
print("Pushing trained model to HF Hub...")
model.push_to_hub(
    MODEL_REPO,
    token=HF_TOKEN,
    commit_message="CyberSec SOC GRPO trained defender"
)
tokenizer.push_to_hub(
    MODEL_REPO,
    token=HF_TOKEN,
    commit_message="tokenizer"
)
print(f"Model saved at: https://huggingface.co/{MODEL_REPO}")
print("DONE!")