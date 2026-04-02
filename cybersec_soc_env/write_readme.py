content = """---
title: CyberSec-SOC-OpenEnv
emoji: \U0001f6e1
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# CyberSec-SOC-OpenEnv

AI agent plays SOC analyst defending an enterprise network against a cyberattack.
Built for Meta x Scaler PyTorch OpenEnv Hackathon 2026.

## Why This Matters

Every large company employs SOC analysts to monitor networks and respond to
cyberattacks. This environment simulates exactly that task giving RL agents
a realistic environment to learn threat detection and containment strategy.

## Live API

- Swagger UI: https://Fieerawe-cybersec-soc-env.hf.space/docs
- Reset: POST https://Fieerawe-cybersec-soc-env.hf.space/reset
- Step: POST https://Fieerawe-cybersec-soc-env.hf.space/step
- State: GET https://Fieerawe-cybersec-soc-env.hf.space/state
- Tasks: GET https://Fieerawe-cybersec-soc-env.hf.space/tasks
- Grade: POST https://Fieerawe-cybersec-soc-env.hf.space/tasks/{task_id}/grade

## Features

| Feature | Description |
|---|---|
| Procedural topology | 4 network types random every episode |
| Partial observability | Compromise hidden until agent scans |
| Alert noise | 5% false alert rate like real SOC |
| Business impact | Isolating servers costs disruption points |
| MITRE ATT&CK | 4-stage attack progression |
| 3 difficulty levels | Easy / Medium / Hard |

## Action Space

| Action | Parameters | Description |
|---|---|---|
| scan | target_node_id | Reveal if node is compromised |
| isolate | target_node_id | Disconnect node from network |
| patch | target_node_id | Harden node against attack |
| firewall | -1 | Slow attacker spread for 10 steps |
| nothing | -1 | Take no action |

## Observation Space

| Field | Type | Description |
|---|---|---|
| node_statuses | List[dict] | Per-node alert score and status |
| attack_stage | int | Current attack stage 1-4 |
| timestep | int | Current step number |
| alerts | List[str] | Last 5 security alerts |
| topology_type | str | star/mesh/segmented/hierarchical |
| business_impact_score | float | Disruption cost from isolations |
| defender_wins | bool | True if all threats contained |

## Reward Function

| Event | Reward |
|---|---|
| Isolate confirmed threat | +1.0 |
| Scan reveals compromise | +0.5 |
| Patch a node | +0.3 |
| False positive isolation | -0.2 |
| Per timestep penalty | -0.05 |
| Attacker contained win | +5.0 |
| Exfiltration succeeds loss | -5.0 |
| Perfect containment bonus | +2.0 |

## Tasks

| Task | Nodes | Infected | Max Steps | Description |
|---|---|---|---|---|
| easy | 5 | 1 | 20 | Isolate single infected node |
| medium | 10 | 2 | 35 | Stop lateral movement |
| hard | 20 | 3 | 50 | Prevent data exfiltration |

## Baseline Scores

| Task | Rule-based Agent | LLM Agent best episode |
|---|---|---|
| easy | 0.417 | 0.275 |
| medium | 0.427 | 0.475 |
| hard | 0.431 | 0.892 |
| overall | 0.425 | 0.462 |

## Setup
```bash
pip install openenv-core networkx numpy fastapi uvicorn pydantic openai
pip install -e .
```

## Run Server Locally
```bash
uvicorn cybersec_soc_env.server.app:app --host 0.0.0.0 --port 8000
```

## Quick Test
```python
import requests
url = "https://Fieerawe-cybersec-soc-env.hf.space"
r = requests.post(url + "/reset")
print("Topology:", r.json()["observation"]["topology_type"])
action = {"action": {"action_type": "scan", "target_node_id": 0}}
r2 = requests.post(url + "/step", json=action)
print("Reward:", r2.json()["reward"])
```

## Run Grader
```bash
python grader.py
```

## Run Inference
```bash
set HF_TOKEN=your_token_here
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py
```

## Project Structure
```
cybersec_soc_env/
    __init__.py
    models.py
    client.py
    openenv.yaml
    pyproject.toml
    Dockerfile
    server/
        soc_environment.py
        app.py
        requirements.txt
inference.py
grader.py
.env.example
```

## Environment Variables

| Variable | Description |
|---|---|
| API_BASE_URL | LLM API endpoint |
| MODEL_NAME | Model identifier |
| HF_TOKEN | Hugging Face token |
| ENV_URL | Environment server URL |

## Author

Built for Meta x Scaler PyTorch OpenEnv Hackathon 2026.
GitHub: https://github.com/FieroJain/cybersec-soc-env
HF Space: https://huggingface.co/spaces/Fieerawe/cybersec-soc-env
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('README written!')