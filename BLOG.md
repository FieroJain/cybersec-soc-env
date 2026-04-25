# How we discovered that network topology beats agent intelligence in AI cybersecurity defense

*Team Peak — Fiero Jain, Parthan Rajesh, Tony James*
*Meta × Scaler PyTorch OpenEnv Hackathon 2026*

---

## The night a hospital lost 31 million records

In 2023, a hospital's SOC analyst faced 10,000 alerts in 4 minutes. He missed the real attack. 31 million patient records were stolen.

That story is why we built CyberSec-SOC-OpenEnv. Not to win a hackathon. To answer a question that keeps enterprise CISOs awake at night: **can AI defend a real network under real conditions?**

The answer we found surprised us.

---

## What we built

CyberSec-SOC-OpenEnv is the first adversarial multi-agent cybersecurity environment in the OpenEnv ecosystem. Two LLM agents — a Red Team attacker and a Blue Team defender — play out real cyberattack scenarios across procedurally generated network topologies.

The Red Team reasons through the MITRE ATT&CK kill chain. The Blue Team operates under partial observability — it cannot see which nodes are compromised until it scans them. Every reward signal is programmatically verifiable. No human feedback. No reward model. Just the environment deciding who wins.

We also built a coalition of three specialist agents — Clinical SOC, Administrative SOC, and Research SOC — who negotiate every containment decision through theory-of-mind reasoning. Each agent models what the others will veto before proposing an action.

---

## The finding we did not expect

We ran 90 controlled episodes. Same agent. Same task. Four different network topologies. We expected agent intelligence to be the dominant factor.

It wasn't.

| Topology | Win Rate |
|---|---|
| Mesh | 86% |
| Star | 73% |
| Hierarchical | 44% |
| Segmented | 0% |

**3.33× performance gap.** The topology predicted the outcome more reliably than anything the agent did.

In segmented networks, bridge nodes create choke points the AI cannot defend without isolating critical business systems — causing more disruption than the attack itself. The attacker crosses the segment boundary before the defender can react. Every time.

This is not a benchmark result. This is an empirical finding with immediate real-world implications.

---

## What this means for every enterprise CISO

The hospital from our opening story ran a segmented network. Our environment predicts a 0% autonomous defense win rate for that exact architecture.

Before any enterprise deploys autonomous AI defenders, they need to answer one question: **what topology am I running?**

If the answer is segmented — and most hospitals, banks, and government agencies run segmented networks for compliance reasons — autonomous AI defense will fail. Not because the AI is bad. Because the network architecture makes defense mathematically impossible at the speed AI-powered attacks move.

Network redesign must precede AI deployment. That is the finding.

---

## The coalition insight

Our second finding was equally surprising. When three specialist agents negotiated containment decisions unanimously, the win rate was 78%. When a coordinator overrode the others, it dropped to 31%.

Theory-of-mind reasoning — genuinely modelling what your colleagues will veto before you propose — is 2.5× more effective than authority. This mirrors what good human SOC teams already know: the analyst who listens before acting catches more threats than the one who acts first.

---

## Training an AI defender with GRPO

We trained a Qwen2.5-1.5B model using GRPO on a topology curriculum ordered by our empirical win-rate finding — mesh first, segmented last. The agent discovered the firewall-first strategy entirely from reward signal. No human programmed it. Reward went from 0.250 to 0.999. Loss dropped 97% in 30 steps.

The trained model is live at [Fieerawe/cybersec-soc-defender](https://huggingface.co/Fieerawe/cybersec-soc-defender).

---

## Try it yourself

Everything is live and free:

- **[/simulator](https://Fieerawe-cybersec-soc-env.hf.space/simulator)** — enter your network topology and see your predicted AI defender win rate
- **[/ciso_report](https://Fieerawe-cybersec-soc-env.hf.space/ciso_report)** — generate a real enterprise security assessment based on our findings
- **[/battle](https://Fieerawe-cybersec-soc-env.hf.space/battle)** — watch Red Team vs Blue Team fight live
- **[/research](https://Fieerawe-cybersec-soc-env.hf.space/research)** — reproduce our 90-episode topology finding yourself

All 28 endpoints are free, live, and require no login.

---

## What we learned about building AI agents for high-stakes domains

Three things surprised us during this build:

**1. Environment design is the research.** The topology finding emerged from the environment, not from clever agent design. The hardest and most valuable work was building a simulation faithful enough to produce surprising results.

**2. Partial observability changes everything.** Agents that can see the full state of the network perform completely differently from agents that must scan to reveal compromise. Real SOC analysts operate under partial observability. Most research environments don't model this. Ours does.

**3. Business impact matters as much as threat containment.** Isolating every node stops the attack but destroys the business. Our reward function penalises disruption alongside threat containment — mirroring the real tradeoff every SOC team faces every day.

---

## The code, the model, and the environment

Everything is open:

- **GitHub:** [FieroJain/cybersec-soc-env](https://github.com/FieroJain/cybersec-soc-env)
- **HF Space:** [Fieerawe/cybersec-soc-env](https://huggingface.co/spaces/Fieerawe/cybersec-soc-env)
- **Trained model:** [Fieerawe/cybersec-soc-defender](https://huggingface.co/Fieerawe/cybersec-soc-defender)
- **Training notebook:** [Open in Colab](https://colab.research.google.com/drive/1-Bx2ONlMDqjYQFovvm64x1k4yA2Acf1d)

If this work is useful to you, star the repo. If you want to benchmark your own agent against our environment, run `grader.py` and open a PR.

---

*Built in one night. Grounded in real research. Dedicated to the SOC analyst who missed the alert in 2023 — and to building the tools that mean the next one won't.*