---
title: CyberSec-SOC-OpenEnv
emoji: 🛡
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# CyberSec-SOC-OpenEnv

> AI agent plays SOC analyst defending a procedurally generated enterprise
> network against a multi-stage cyberattack.
> Built for the PyTorch OpenEnv Hackathon.

## Why This Environment Is Different

| Feature | CyberSec-SOC | Typical submission |
|---|---|---|
| Topology | Procedural (4 types, random each episode) | Fixed grid |
| Observability | Partial — scan to reveal | Full |
| Alert system | Noisy — 5% false alert rate | Clean |
| Business impact | Yes — isolation has cost | No |
| Attack model | MITRE ATT&CK 4-stage | Simple spread |

## Setup

```bash
pip install openenv-core
cd cybersec_soc_env
pip install -e .
```

## Run Server Locally

```bash
uvicorn cybersec_soc_env.server.app:app --host 0.0.0.0 --port 8000
```

## Use The Client

```python
# Sync (notebooks/testing)
from cybersec_soc_env import SOCEnv, SOCAction

with SOCEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset()
    print(f"Topology: {obs.topology_type}")
    print(f"Nodes: {len(obs.node_statuses)}")

    while not obs.done:
        result = env.step(SOCAction(action_type="scan", target_node_id=0))
        obs = result.observation
        print(f"Stage: {obs.attack_stage} | Reward: {result.reward}")
        print(f"Alerts: {obs.alerts[-1]}")
```

```python
# Async (TRL/GRPO training)
import asyncio
from cybersec_soc_env import SOCEnv, SOCAction

async def train():
    async with SOCEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        result = await env.step(
            SOCAction(action_type="isolate", target_node_id=2)
        )
        print(f"Business impact: {result.observation.business_impact_score}")

asyncio.run(train())
```

## Web UI (Built-in Gradio)

```bash
ENABLE_WEB_INTERFACE=1 uvicorn cybersec_soc_env.server.app:app \
  --host 0.0.0.0 --port 8000
# Open http://localhost:8000/web
```

## Deploy to Hugging Face

```bash
openenv push --repo-id YOUR_USERNAME/cybersec-soc-env
```

## Task Levels

| Level | Nodes | Start Infected | Max Steps |
|---|---|---|---|
| easy | 5 | 1 | 20 |
| medium | 10 | 2 | 35 |
| hard | 20 | 3 | 50 |

Override at runtime:
```bash
TASK_LEVEL=hard SEED=123 uvicorn cybersec_soc_env.server.app:app \
  --host 0.0.0.0 --port 8000
```

## Reward Table

| Event | Reward |
|---|---|
| Isolate confirmed threat (scan first) | +1.0 |
| Scan reveals hidden compromise | +0.5 |
| Patch a node | +0.3 |
| False positive isolation | -0.2 |
| Per timestep (urgency) | -0.05 |
| Do nothing while 2+ active threats | -0.1 |
| Attacker fully contained | **+5.0** |
| Exfiltration succeeds | **-5.0** |
| Perfect containment (0 false isolations) | **+2.0** |

## Environment Design Decisions

**Procedural topology**: Every episode the network is regenerated with a
randomly selected topology (star, mesh, segmented, hierarchical). This
forces agents to generalise rather than memorise.

**Partial observability**: Compromise status is hidden until a `scan` action
is taken on that node. Agents must reason about uncertainty just like real
SOC analysts.

**Alert noise**: 5 % of clean nodes generate false alerts each step, mixed
with Gaussian noise on true signals. Agents must distinguish signal from
noise — exactly like real SOC work.

**Business impact**: Isolating high-value nodes (database, auth server)
carries a disruption cost that is added to `business_impact_score`. Agents
must balance security vs. operational continuity.

**MITRE ATT&CK stages**:

| Stage | Trigger | Consequence |
|---|---|---|
| 1 – Initial Compromise | Episode start | Spread begins |
| 2 – Credential Access | auth_server compromised | Faster spread |
| 3 – Lateral Movement | 2+ nodes compromised | Wider attack surface |
| 4 – Exfiltration | db/file_server at stage 3 | 5-step countdown |

## Action Space

| Action | Target | Effect |
|---|---|---|
| `scan` | node id | Reveals true compromise status |
| `isolate` | node id | Disconnects node; stops spread through it |
| `patch` | node id | Raises security_level +0.2, reduces vulnerabilities -1 |
| `firewall` | -1 (global) | Halves spread probability for 10 steps |
| `nothing` | -1 (global) | No-op; penalty if >=2 active threats |

## File Structure

```
cybersec_soc_env/
??? __init__.py          # Public exports
??? models.py            # SOCAction, SOCObservation, SOCState
??? client.py            # SOCEnv WebSocket client
??? README.md
??? openenv.yaml         # Registry metadata
??? pyproject.toml       # Build configuration
??? server/
    ??? __init__.py
    ??? soc_environment.py  # Core RL environment
    ??? app.py              # FastAPI entry point
    ??? requirements.txt
    ??? Dockerfile
```
