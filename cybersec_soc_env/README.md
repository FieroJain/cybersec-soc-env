---
title: CyberSec-SOC-OpenEnv
emoji: 🛡
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🛡️ CyberSec-SOC-OpenEnv

> A reinforcement learning environment for training AI agents on enterprise-grade cybersecurity defense.  
> Built for the **Meta × Scaler PyTorch OpenEnv Hackathon 2026**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Live-009688?style=flat-square&logo=fastapi&logoColor=white)](https://Fieerawe-cybersec-soc-env.hf.space/docs)
[![Hugging Face](https://img.shields.io/badge/HF%20Space-Live-FFD21F?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Fieerawe/cybersec-soc-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen?style=flat-square)](cybersec_soc_env/openenv.yaml)
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
- [Submission Compliance](#-submission-compliance)

---

## 🔬 Research Motivation

Training AI agents for cybersecurity defense is one of the most consequential open problems in applied AI. Human Security Operations Center (SOC) analysts operate under extreme cognitive load — the average enterprise generates **10,000+ security alerts per day**, with **45% being false positives**. Delayed or incorrect triage decisions can result in catastrophic data breaches, operational downtime, and regulatory consequences.

**CyberSec-SOC-OpenEnv** is the first environment in the OpenEnv ecosystem purpose-built for this domain. It enables researchers to train and benchmark LLM-based agents on:

- **Threat detection** under partial observability and high alert noise
- **Containment strategy** across diverse, procedurally generated network topologies
- **Risk-vs-disruption tradeoffs** that mirror the operational constraints of a real SOC

This environment directly addresses the gap between academic RL benchmarks and the decision-making complexity of real-world enterprise security operations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌐 Procedural Network Topology | 4 topology types (`star`, `mesh`, `segmented`, `hierarchical`) randomly instantiated per episode |
| 👁️ Partial Observability | Node compromise status is hidden until the agent actively investigates |
| 📡 Realistic Alert Noise | 5% false positive alert rate, consistent with enterprise SOC baselines |
| 💼 Business Impact Modeling | Containment actions carry operational cost — isolation is not free |
| ⚔️ MITRE ATT&CK Progression | Threat actors advance through a structured 4-stage kill chain |
| 📊 3 Difficulty Tiers | Graduated task complexity for training curriculum design |
| ✅ OpenEnv Compliant | Fully implements `reset()`, `step()`, `state()` per OpenEnv specification |

---

## 🎯 Environment Design

### Action Space

| Action | Parameters | Description |
|---|---|---|
| `scan` | `target_node_id` | Investigate a node to reveal its true compromise status |
| `isolate` | `target_node_id` | Sever a node from the network to contain an active threat |
| `patch` | `target_node_id` | Apply hardening to reduce a node's vulnerability surface |
| `firewall` | `-1` | Deploy a network-wide firewall rule; slows adversary lateral movement for 10 timesteps |
| `nothing` | `-1` | Defer action; useful when monitoring without sufficient information |

### Observation Space

| Field | Type | Description |
|---|---|---|
| `node_statuses` | `List[dict]` | Per-node alert score and visibility status |
| `attack_stage` | `int` | Current adversary kill chain stage (1–4) |
| `timestep` | `int` | Elapsed steps in the current episode |
| `alerts` | `List[str]` | Rolling window of the last 5 security alerts |
| `topology_type` | `str` | Active network topology: `star` / `mesh` / `segmented` / `hierarchical` |
| `business_impact_score` | `float` | Cumulative operational disruption cost from containment actions |
| `defender_wins` | `bool` | `True` when all active threats have been fully contained |

### Reward Function

| Event | Reward |
|---|---|
| Successful isolation of a confirmed threat | `+1.0` |
| Scan reveals an active compromise | `+0.5` |
| Patch applied to a vulnerable node | `+0.3` |
| Isolation of a clean node (false positive) | `-0.2` |
| Per-timestep inaction penalty | `-0.05` |
| Full adversary containment | `+5.0` |
| Data exfiltration succeeds | `-5.0` |
| Perfect containment with zero false positives | `+2.0` |

---

## 📋 Tasks

| Task | Nodes | Compromised | Max Steps | Objective |
|---|---|---|---|---|
| `easy` | 5 | 1 | 20 | Identify and isolate a single compromised node before lateral movement |
| `medium` | 10 | 2 | 35 | Detect and contain an active lateral movement campaign |
| `hard` | 20 | 3 | 50 | Prevent data exfiltration across a large, partially observable network |

All tasks are graded via `grader.py` and return normalized scores in `[0.0, 1.0]`.

---

## 📊 Baseline Performance

Scores are normalized in `[0.0, 1.0]`. Results are averaged across 20 independent episodes per task.

| Task | Rule-Based Agent | LLM Agent |
|---|---|---|
| `easy` | 0.507 | 0.173 |
| `medium` | 0.387 | 0.485 |
| `hard` | 0.080 | 0.703 |
| **Overall** | **0.325** | **0.454** |

**Key finding:** LLM agents substantially outperform rule-based heuristics on high-complexity tasks. The `hard` task delta of **+0.623** demonstrates that language model reasoning provides meaningful advantage in scenarios requiring multi-step inference under uncertainty — exactly the conditions that define advanced persistent threat (APT) defense.

---

## 🌐 Live API

**Base URL:** `https://Fieerawe-cybersec-soc-env.hf.space`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | [`/docs`](https://Fieerawe-cybersec-soc-env.hf.space/docs) | Interactive Swagger UI |
| `POST` | `/reset` | Initialize a new episode |
| `POST` | `/step` | Submit a defender action and receive the next observation |
| `GET` | `/state` | Retrieve the current environment state |
| `GET` | `/tasks` | Enumerate available evaluation tasks |
| `POST` | `/tasks/{task_id}/grade` | Run the grader against a specific task |

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install openenv-core networkx numpy fastapi uvicorn pydantic openai
pip install -e .
```

### 2. Run the Environment Server Locally

```bash
uvicorn cybersec_soc_env.server.app:app --host 0.0.0.0 --port 8000
```

### 3. Verify the API

```python
import requests

url = "https://Fieerawe-cybersec-soc-env.hf.space"

# Initialize episode
r = requests.post(url + "/reset")
print("Topology:", r.json()["observation"]["topology_type"])

# Submit a defender action
action = {"action": {"action_type": "scan", "target_node_id": 0}}
r2 = requests.post(url + "/step", json=action)
print("Reward:", r2.json()["reward"])
```

### 4. Run the Grader

```bash
python grader.py
```

### 5. Run Agent Inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

python inference.py
```

### 6. Run Pre-Submission Validation

```bash
bash validate.sh
```

---

## 🗂️ Project Structure

```
cybersec-soc-env/
│
├── inference.py                        # Agent inference script (OpenEnv entry point)
├── grader.py                           # Task evaluation and scoring
├── validate.sh                         # Pre-submission compliance validator
├── baseline_scores.json                # Recorded baseline results
├── .env.example                        # Environment variable template
│
└── cybersec_soc_env/
    ├── __init__.py
    ├── models.py                       # Pydantic request/response models
    ├── client.py                       # API client wrapper
    ├── openenv.yaml                    # OpenEnv specification manifest
    ├── pyproject.toml                  # Package configuration
    ├── Dockerfile                      # Container definition
    ├── inference.py                    # Package-level inference utilities
    ├── grader.py                       # Package-level grading logic
    │
    └── server/
        ├── app.py                      # FastAPI application and route definitions
        ├── soc_environment.py          # Core simulation engine
        ├── gradio_dashboard.py         # Optional monitoring dashboard
        ├── Dockerfile                  # Server container definition
        └── requirements.txt           # Server dependencies
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` before running inference or the validator.

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | ✅ | LLM API endpoint (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | ✅ | Model identifier (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `HF_TOKEN` | ✅ | Hugging Face access token |
| `ENV_URL` | ✅ | Environment server URL |

```env
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=your_huggingface_token
ENV_URL=https://Fieerawe-cybersec-soc-env.hf.space
```

---

## ✅ Submission Compliance

This environment is fully compliant with the Meta × Scaler PyTorch OpenEnv Hackathon 2026 requirements.

| Requirement | Status |
|---|---|
| `openenv.yaml` present and valid | ✅ |
| Typed Pydantic models | ✅ |
| `reset()`, `step()`, `state()` endpoints implemented | ✅ |
| Dockerfile builds successfully | ✅ |
| `inference.py` in project root | ✅ |
| OpenAI client used for all LLM calls | ✅ |
| `[START]`, `[STEP]`, `[END]` stdout log format | ✅ |
| 3+ tasks with graders returning scores in `[0.0, 1.0]` | ✅ |
| Inference runtime under 20 minutes | ✅ |
| Compatible with 2 vCPU / 8 GB RAM | ✅ |
| Pre-submission `validate.sh` passes | ✅ |

### Log Format

`inference.py` emits structured logs strictly following the required format:

```
[START] task=hard env=cybersec-soc-openenv model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=scan reward=0.50 done=false error=null
[STEP] step=2 action=isolate reward=1.00 done=false error=null
...
[END] success=true steps=18 score=0.842 rewards=0.50,1.00,...
```

---

## 👤 Author

Built for the **Meta × Scaler PyTorch OpenEnv Hackathon 2026**.

- GitHub: [FieroJain/cybersec-soc-env](https://github.com/FieroJain/cybersec-soc-env)
- HF Space: [Fieerawe/cybersec-soc-env](https://huggingface.co/spaces/Fieerawe/cybersec-soc-env)

---

*If this work contributed to your research, please consider starring the repository on [GitHub](https://github.com/FieroJain/cybersec-soc-env).*
