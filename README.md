# 🛡️ CyberSec-SOC-OpenEnv

> AI agent plays SOC analyst defending an enterprise network against a cyberattack.  
> Built for the **Meta × Scaler PyTorch OpenEnv Hackathon 2026**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Live-009688?style=flat-square&logo=fastapi&logoColor=white)](https://Fieerawe-cybersec-soc-env.hf.space/docs)
[![Hugging Face](https://img.shields.io/badge/HF%20Space-Live-FFD21F?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Fieerawe/cybersec-soc-env)
[![License](https://img.shields.io/badge/License-MIT-red?style=flat-square)](LICENSE)

---

## 📖 Table of Contents

- [Research Motivation](#-research-motivation)
- [Features](#-features)
- [Environment Design](#-environment-design)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Reward Function](#reward-function)
- [Tasks](#-tasks)
- [Baseline Performance](#-baseline-performance)
- [Live API](#-live-api)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Environment Variables](#️-environment-variables)

---

## 🔬 Research Motivation

Training AI agents for cybersecurity defense is one of the most valuable open problems in AI safety. Human SOC analysts are overwhelmed — the average enterprise receives **10,000+ security alerts per day**, with **45% being false positives**.

**CyberSec-SOC-OpenEnv** is the first environment in the OpenEnv ecosystem designed for this domain, enabling researchers to train and evaluate LLM agents on:

- Realistic threat detection
- Containment strategy
- Risk-vs-disruption tradeoffs

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌐 Procedural Topology | 4 network types randomly generated every episode |
| 👁️ Partial Observability | Compromise is hidden until the agent scans |
| 📡 Alert Noise | 5% false alert rate — mirrors real SOC conditions |
| 💼 Business Impact | Isolating servers costs disruption points |
| ⚔️ MITRE ATT&CK | Full 4-stage attack progression |
| 🎯 Difficulty Levels | Easy / Medium / Hard |

---

## 🎮 Environment Design

### Action Space

| Action | Parameters | Description |
|---|---|---|
| `scan` | `target_node_id` | Reveal if a node is compromised |
| `isolate` | `target_node_id` | Disconnect node from the network |
| `patch` | `target_node_id` | Harden a node against attack |
| `firewall` | `-1` | Slow attacker spread for 10 steps |
| `nothing` | `-1` | Take no action this step |

### Observation Space

| Field | Type | Description |
|---|---|---|
| `node_statuses` | `List[dict]` | Per-node alert score and status |
| `attack_stage` | `int` | Current attack stage (1–4) |
| `timestep` | `int` | Current step number |
| `alerts` | `List[str]` | Last 5 security alerts |
| `topology_type` | `str` | `star` / `mesh` / `segmented` / `hierarchical` |
| `business_impact_score` | `float` | Disruption cost from isolations |
| `defender_wins` | `bool` | `True` if all threats are contained |

### Reward Function

| Event | Reward |
|---|---|
| Isolate confirmed threat | `+1.0` |
| Scan reveals compromise | `+0.5` |
| Patch a node | `+0.3` |
| False positive isolation | `-0.2` |
| Per-timestep penalty | `-0.05` |
| Attacker fully contained | `+5.0` |
| Exfiltration succeeds | `-5.0` |
| Perfect containment bonus | `+2.0` |

---

## 📋 Tasks

| Task | Nodes | Infected | Max Steps | Objective |
|---|---|---|---|---|
| 🟢 `easy` | 5 | 1 | 20 | Isolate single infected node |
| 🟡 `medium` | 10 | 2 | 35 | Stop lateral movement |
| 🔴 `hard` | 20 | 3 | 50 | Prevent data exfiltration |

---

## 📊 Baseline Performance

| Task | Rule-Based Agent | LLM Agent |
|---|---|---|
| `easy` | 0.507 | 0.173 |
| `medium` | 0.387 | 0.485 |
| `hard` | 0.080 | 0.703 |
| **Overall** | **0.325** | **0.454** |

> 💡 LLM agents significantly outperform rule-based agents on complex tasks, demonstrating the value of reasoning under uncertainty.

---

## 🌐 Live API

**Base URL:** `https://Fieerawe-cybersec-soc-env.hf.space`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | [`/docs`](https://Fieerawe-cybersec-soc-env.hf.space/docs) | Interactive Swagger UI |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Get current environment state |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/tasks/{task_id}/grade` | Grade agent on a task |

---

## ⚡ Quick Start

### 1. Install

```bash
pip install openenv-core networkx numpy fastapi uvicorn pydantic openai
pip install -e .
```

### 2. Run Server Locally

```bash
uvicorn cybersec_soc_env.server.app:app --host 0.0.0.0 --port 8000
```

### 3. Test the API

```python
import requests

url = "https://Fieerawe-cybersec-soc-env.hf.space"

# Start a new episode
r = requests.post(url + "/reset")
print("Topology:", r.json()["observation"]["topology_type"])

# Take an action
action = {"action": {"action_type": "scan", "target_node_id": 0}}
r2 = requests.post(url + "/step", json=action)
print("Reward:", r2.json()["reward"])
```

### 4. Run the Grader

```bash
python grader.py
```

### 5. Run LLM Inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

python inference.py
```

---

## 🗂️ Project Structure

```
cybersec-soc-env/
│
├── cybersec_soc_env/
│   ├── __init__.py
│   ├── models.py               # Pydantic data models
│   ├── client.py               # API client wrapper
│   ├── openenv.yaml            # OpenEnv spec
│   ├── pyproject.toml
│   ├── Dockerfile
│   │
│   └── server/
│       ├── soc_environment.py  # Core simulation logic
│       ├── app.py              # FastAPI application
│       └── requirements.txt
│
├── inference.py                # LLM agent runner
├── grader.py                   # Evaluation script
└── .env.example                # Environment variable template
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face access token |
| `ENV_URL` | Environment server URL |

```env
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=your_huggingface_token
ENV_URL=https://Fieerawe-cybersec-soc-env.hf.space
```

---

## 👤 Author

Built for the **Meta × Scaler PyTorch OpenEnv Hackathon 2026**.

- GitHub: [FieroJain/cybersec-soc-env](https://github.com/FieroJain/cybersec-soc-env)
- HF Space: [Fieerawe/cybersec-soc-env](https://huggingface.co/spaces/Fieerawe/cybersec-soc-env)

---

*If this helped your research, consider leaving a ⭐ on [GitHub](https://github.com/FieroJain/cybersec-soc-env)!*
