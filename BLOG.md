\# CyberSec-SOC-OpenEnv: Training AI Agents to Defend Enterprise Networks



\*\*Meta x Scaler PyTorch OpenEnv Hackathon 2026 — Team Peak\*\*



\## The Problem



Enterprise Security Operations Centers face 10,000 alerts per day.

45% are false positives. One wrong isolation decision takes down a

database server and costs hundreds of thousands in downtime.



We built the AI training ground to solve this.



\## The Environment



Two LLM agents in real adversarial conflict:



\*\*Red Team\*\* — follows MITRE ATT\&CK kill chain:

\- Stage 1: Initial Compromise

\- Stage 2: Credential Access

\- Stage 3: Lateral Movement

\- Stage 4: Exfiltration



\*\*Blue Team\*\* — SOC analyst AI under partial observability:

\- Actions: scan, isolate, patch, firewall

\- Cannot see compromise status until node is scanned

\- Must reason under uncertainty with false positive noise



\## The Discovery



We ran 90 controlled experiments and found something nobody had measured.



| Topology | Win Rate | Avg Score |

|---|---|---|

| Mesh | 86% | 0.731 |

| Star | 73% | 0.614 |

| Hierarchical | 44% | 0.509 |

| Segmented | 0% | 0.219 |



\*\*3.33× performance gap. Same agent. Same task.\*\*



Network topology is an adversarial attack surface. This became

our training curriculum — mesh first, segmented last.



\## Training Results



Trained Llama-3.1-8B-Instruct with Unsloth + TRL.



Loss: 4.41 → 0.097 (97% reduction in 30 steps)



Before training: random actions, no reasoning

After training: "Immediately isolate Node 4 (database\_server) —

highest asset value confirmed threat. Prevent lateral movement."



\## What We Cover



\- Multi-Agent: Red Team + Blue Team + Coalition Formation

\- Long-Horizon: 50-step MITRE ATT\&CK episodes

\- World Modeling: 4 procedural topologies, partial observability

\- Self-Improving: Topology curriculum from empirical finding

\- Fleet AI bonus: Scalable oversight auditor

\- Patronus bonus: Schema drift across episodes



\## Links



\- Live environment: https://huggingface.co/spaces/Fieerawe/cybersec-soc-env

\- GitHub: https://github.com/FieroJain/cybersec-soc-env

\- Training notebook: YOUR\_COLAB\_LINK

\- Demo video: YOUR\_YOUTUBE\_LINK

