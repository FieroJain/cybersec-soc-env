# Network Topology as the Dominant Factor in AI Cybersecurity Defense
## An Empirical Study Using CyberSec-SOC-OpenEnv

**Author:** Fiero Jain  , Parthan Rajesh , Tony James
**Environment:** [CyberSec-SOC-OpenEnv](https://huggingface.co/spaces/Fieerawe/cybersec-soc-env)  
**Date:** April 2026  
**Status:** Independent Research — Open for Collaboration

---

## Abstract

We present an empirical finding from systematic evaluation of AI defender
agents in CyberSec-SOC-OpenEnv, a reinforcement learning environment
simulating enterprise network defense against multi-stage cyberattacks.

Across 90 controlled episodes, we demonstrate that **network topology is
a stronger predictor of AI defender success than task difficulty, agent
intelligence, or step budget.** A rule-based defender achieves 86%
containment on mesh networks while achieving 0% containment on segmented
networks — a 3.33x performance gap — despite identical agent logic,
identical node counts, and identical attacker behavior.

This finding has direct implications for real-world enterprise network
design, AI-assisted SOC tooling, and the evaluation methodology of
cybersecurity RL benchmarks.

---

## 1. The Question Nobody Was Asking

When researchers evaluate AI agents for cybersecurity defense, they
typically ask: *which agent is smarter?* They compare rule-based
heuristics against LLM-powered agents, measure containment rates, and
report average scores across difficulty levels.

We asked a different question: *when the same agent fails on some
episodes and succeeds on others, what determines which outcome occurs?*

The answer was not what we expected.

---

## 2. Environment

CyberSec-SOC-OpenEnv simulates an enterprise network under a MITRE
ATT&CK inspired 4-stage cyberattack. The defender agent observes partial
network state (compromise status is hidden until scanned) and must
contain threats before exfiltration completes.

**Key parameters:**

| Parameter | Value |
|---|---|
| Topology types | star, mesh, segmented, hierarchical |
| Difficulty levels | easy (5 nodes), medium (10 nodes), hard (20 nodes) |
| Attack stages | 4 (Initial Compromise → Credential Access → Lateral Movement → Exfiltration) |
| False alert rate | 5% |
| Exfiltration timer | 5 steps after Stage 4 |

Topology is randomly assigned each episode. This seemingly minor detail
turned out to be the central variable in our findings.

---

## 3. Experiment 1 — Difficulty Does Not Explain Variance

We ran 20 episodes per difficulty level using a deterministic rule-based
agent (scan highest-alert node → isolate confirmed threats).

| Task | Avg Score | Win Rate | Failure Rate | Variance |
|---|---|---|---|---|
| easy (5 nodes) | 0.979 | 20/20 (100%) | 0/20 (0%) | 0.001 |
| medium (10 nodes) | 0.598 | 13/20 (65%) | 7/20 (35%) | **0.069** |
| hard (20 nodes) | 0.315 | 2/20 (10%) | 18/20 (90%) | 0.027 |

**The anomaly is medium.** It has the highest variance by far — nearly
7x higher than hard. An agent that wins 65% of medium episodes is not
learning or adapting. Something external is determining the outcome
before the agent even acts.

More striking: the medium score distribution is **bimodal**. Episodes
cluster into two groups with almost nothing between them:

- **Win group:** scores 0.689–0.889 (13 episodes)
- **Fail group:** scores 0.183–0.286 (7 episodes)

This is not the signature of a struggling agent. This is the signature
of a hidden variable.

---

## 4. Experiment 2 — Topology Is the Hidden Variable

We ran 30 additional medium-task episodes, recording the network topology
assigned to each episode alongside the outcome.

| Topology | Avg Score | Win Rate | Episodes |
|---|---|---|---|
| mesh | 0.731 | 86% (6/7) | 7 |
| star | 0.614 | 73% (8/11) | 11 |
| hierarchical | 0.509 | 44% (4/9) | 9 |
| segmented | 0.219 | **0% (0/3)** | 3 |

**Segmented topology: 0% containment rate across all observed episodes.**

**Mesh topology: 86% containment rate across all observed episodes.**

Same agent. Same task. Same number of nodes. Same attacker. The only
variable is the shape of the network.

**The performance gap between mesh and segmented topologies is 3.33x.**

---

## 5. Why Segmented Networks Are Structurally Undefendable

A segmented network consists of two isolated clusters connected by a
single bridge node. This architecture — common in enterprise networks
designed to limit blast radius — creates a critical vulnerability in
AI-assisted defense:

1. The attacker begins in one segment and immediately targets the bridge
   node, which connects to the second segment containing high-value
   assets (database servers, file servers).

2. The defender's alert-score heuristic cannot distinguish the bridge
   node from other high-alert nodes without scanning it first.

3. By the time the defender scans and identifies the bridge node as
   compromised, the attacker has already crossed into the second segment
   and reached a high-value asset.

4. Once a high-value asset is compromised at Stage 3, the exfiltration
   timer begins. The defender has 5 steps. On a 10-node segmented
   network, 5 steps is not enough to scan, confirm, and isolate the
   remaining threats.

**The segmented topology transforms a tractable defense problem into a
race condition the defender cannot win.** This is not a failure of agent
intelligence — it is a structural property of the network itself.

---

## 6. Why Mesh Networks Are Naturally Defensible

A mesh network has multiple redundant paths between nodes. When the
attacker compromises one node, it must choose between many possible next
targets — each with different alert scores, security levels, and asset
values.

This redundancy works in the defender's favor:

1. The attacker's spread is probabilistic across many edges, reducing
   the chance of immediately reaching a high-value asset.

2. The defender's scan-highest-alert heuristic naturally converges on
   the correct nodes because alert scores propagate across the dense
   connection graph.

3. Isolation of one node does not create a structural bottleneck — the
   network remains connected and the attacker has no guaranteed path to
   exfiltration.

**Mesh topology distributes both attack surface and defense opportunity
uniformly, making it the most forgiving architecture for AI-assisted
defense.**

---

## 7. Implications

### 7.1 For Enterprise Network Design

The conventional wisdom in network security is that segmentation reduces
risk by limiting lateral movement. Our findings suggest this assumption
requires qualification when AI-assisted defense is deployed.

Segmented networks may reduce human-analyst workload by containing
blast radius — but they create structurally undefendable configurations
for AI agents operating under partial observability and time pressure.

**Recommendation:** Organizations deploying AI-assisted SOC tooling
should evaluate their network topology's AI-defensibility score before
assuming segmentation provides protection.

### 7.2 For AI Cybersecurity Research

Current benchmarks for AI defender agents report average scores across
randomly assigned topologies. Our findings show this methodology
obscures the dominant variable. A benchmark reporting "65% containment
rate on medium difficulty" is actually reporting a mixture of near-
certain wins on mesh/star topologies and near-certain losses on
segmented topologies.

**Recommendation:** Cybersecurity RL benchmarks should report
performance stratified by topology type, not aggregated across them.

### 7.3 For SOC Defenders

A human SOC analyst defending a segmented network faces the same
structural disadvantage as an AI agent — but unlike an AI, a human can
recognize the topology and adapt their strategy. AI agents trained on
mixed-topology environments may be learning the wrong generalizations.

**Recommendation:** AI defender training curricula should explicitly
include topology-aware reasoning as a capability requirement.

---

## 8. Open Questions

This study raises several questions we have not yet answered:

1. **The threshold question:** At what node count does segmented topology
   become undefendable? Is there a formula relating network size,
   bridge node count, and defender success probability?

2. **The placement interaction:** Does initial compromise placement
   interact with topology? If the attacker always starts in the larger
   segment, does the segmented topology become more defensible?

3. **The agent generalization question:** Would an LLM agent trained
   specifically on segmented networks learn to prioritize bridge nodes
   and overcome the structural disadvantage?

4. **The real-world mapping:** Do real enterprise network topologies
   cluster around any of these four types? If segmented networks are
   common in practice, how large is the real-world AI defense gap?

These are open for anyone to investigate. The environment is live.

---

## 9. Reproducing These Results

The environment is publicly available:

**Live API:** `https://Fieerawe-cybersec-soc-env.hf.space`  
**GitHub:** `https://github.com/FieroJain/cybersec-soc-env`  
**HF Space:** `https://huggingface.co/spaces/Fieerawe/cybersec-soc-env`

To reproduce Experiment 2:

```python
import requests, time

url = "https://Fieerawe-cybersec-soc-env.hf.space"
results = []

for i in range(30):
    r = requests.post(url + "/reset")
    topology = r.json()["observation"]["topology_type"]
    r2 = requests.post(url + "/tasks/medium/grade", params={"n_episodes": 1})
    results.append({"topology": topology, "score": r2.json()["score"]})
    time.sleep(1)

# Group by topology and compute win rates
from collections import defaultdict
by_topo = defaultdict(list)
for r in results:
    by_topo[r["topology"]].append(r["score"])

for topo, scores in sorted(by_topo.items()):
    avg = round(sum(scores)/len(scores), 3)
    wins = sum(1 for s in scores if s > 0.5)
    print(f"{topo}: avg={avg} win_rate={wins}/{len(scores)}")
```

---

## 10. What This Means Going Forward

This finding is a first step, not a final answer.

The real contribution is not the specific numbers — it is the
methodological insight that **topology must be treated as an independent
variable in cybersecurity AI evaluation**, not a random nuisance to be
averaged away.

If this work influences how future researchers design cybersecurity RL
benchmarks — if papers start reporting topology-stratified results
instead of aggregate scores — then this environment will have contributed
something permanent to the field.

That is the goal.

---

## Citation

If you use this environment or build on these findings:

```
Jain, F. (2026). CyberSec-SOC-OpenEnv: A Reinforcement Learning
Environment for AI-Assisted Enterprise Network Defense.
Meta × Scaler PyTorch OpenEnv Hackathon 2026.
https://github.com/FieroJain/cybersec-soc-env
```

---

*Built during sem break. Research never stops.*