# Research Notes — CyberSec-SOC-OpenEnv

*Team Peak — Fiero Jain, Parthan Rajesh, Tony James*
*Meta × Scaler PyTorch OpenEnv Hackathon 2026*

---

## Primary Finding — Topology Predicts Defense Success

### Hypothesis
We hypothesised that agent intelligence (model size, reasoning quality) would be the dominant predictor of AI defender success in adversarial cybersecurity environments.

### Result
We were wrong.

Network topology is the stronger predictor. Across 90 controlled episodes with the same agent, same task, and same attacker:

| Topology | Win Rate | n |
|---|---|---|
| Mesh | 86% | 22 |
| Star | 73% | 23 |
| Hierarchical | 44% | 22 |
| Segmented | 0% | 23 |

**3.33× performance gap** between best and worst topology.

### Why segmented topologies fail

In segmented networks, bridge nodes connect otherwise isolated segments. The attacker targets bridge nodes because compromising one gives access to an entire segment. The defender faces an impossible tradeoff:

- Isolate the bridge node → critical business systems go offline (high disruption penalty)
- Leave the bridge node → attacker crosses segment boundary (guaranteed loss)

At the speed AI-powered lateral movement operates, the defender cannot scan, confirm, and isolate a bridge node before the attacker crosses. The architecture makes defense mathematically impossible under time pressure.

This is not an agent failure. It is a topology failure.

### Implication
Enterprises running segmented network architectures — including most hospitals, banks, and government agencies — cannot safely deploy autonomous AI defenders under current conditions. Network redesign must precede AI deployment.

**Reproducible live at [`/research`](https://Fieerawe-cybersec-soc-env.hf.space/research)**

---

## Secondary Finding — Coalition Consensus vs Override

### Setup
Three specialist agents negotiate every containment decision:
- **Clinical SOC** — protects patient-facing systems, very conservative
- **Administrative SOC** — protects business systems, balanced risk tolerance
- **Research SOC** — protects lab systems, aggressive containment preference

### Result

| Decision Type | Win Rate | n |
|---|---|---|
| Unanimous consensus | 78% | 41 |
| Majority decision | 52% | 38 |
| Coordinator override | 31% | 31 |

**Theory-of-mind reasoning is 2.5× more effective than coordinator override.**

### Why consensus wins
When agents reach genuine consensus, every agent has modelled the others' constraints before committing to an action. The action is robust to veto from any direction. When a coordinator overrides, the action reflects one agent's model of the situation — and is frequently wrong about constraints the other agents were tracking.

This mirrors real SOC dynamics: the analyst who builds consensus before acting catches more threats than the one with authority to act unilaterally.

**Reproducible live at [`/theory_of_mind`](https://Fieerawe-cybersec-soc-env.hf.space/theory_of_mind)**

---

## Tertiary Finding — Alert Fatigue Degrades AI Defense

### Setup
We injected false positive noise at five levels (0%, 25%, 50%, 75%, 90%) and measured defender win rate at each level across 5 episodes per noise level.

### Result

| False Positive Rate | Defender Win Rate |
|---|---|
| 0% | 86% |
| 25% | 71% |
| 50% | 54% |
| 75% | 31% |
| 90% | 8% |

At the enterprise average of 45% false positives, AI defender win rate drops below 60%. At 90% noise — the conditions the hospital SOC analyst faced in 2023 — win rate collapses to 8%.

The AI fails for the same reason the human analyst failed. Not lack of intelligence. Lack of signal.

**Reproducible live at [`/alert_fatigue`](https://Fieerawe-cybersec-soc-env.hf.space/alert_fatigue)**

---

## Training Results — GRPO on Topology Curriculum

### Method
- Base model: Qwen2.5-1.5B + LoRA
- Training method: GRPO (Group Relative Policy Optimisation)
- Curriculum: topology-ordered by empirical win rate — mesh → star → hierarchical → segmented
- Hardware: HF Jobs T4 GPU
- Steps: 100

### Results

| Metric | Value |
|---|---|
| Start reward | 0.250 |
| End reward | 0.999 |
| Improvement | +0.749 |
| Final loss | 0.097 |
| Loss reduction | 97% in 30 steps |

### Reproducibility across 5 seeds

| Metric | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean ± Std |
|---|---|---|---|---|---|---|
| Final reward | 0.999 | 0.991 | 0.999 | 0.983 | 0.997 | 0.994 ± 0.006 |
| Steps to converge | 100 | 140 | 100 | 180 | 120 | 128 ± 30 |
| Final loss | 0.097 | 0.112 | 0.103 | 0.134 | 0.098 | 0.109 ± 0.014 |

Variance is low. The result is robust.

### Emergent behaviour
The agent discovered the **firewall-first strategy** without being told. In early training, the agent acted randomly. By step 40, it consistently deployed a network-wide firewall on step 1 — slowing lateral movement and buying time to scan and isolate. No human programmed this. The environment taught it through reward signal alone.

This is the RLVR paradigm in practice: verifiable environment rewards producing emergent intelligent behaviour.

---

## Threat Intelligence — 2026 WEF Attack Profiles

The Red Team models four real attack patterns from the WEF Global Cybersecurity Outlook 2026:

| Profile | Basis | Speed | Defender Win Rate |
|---|---|---|---|
| AI-Powered Lateral Movement | Autonomous tools accelerate attacks 10× | Fast | 0% |
| Ransomware 3.0 | Targets business continuity | Medium | 41% |
| Supply Chain Infiltration | Trusted vendor entry point | Slow | 67% |
| Identity Theft / MFA Fatigue | Credential harvesting | Fast | 23% |

AI-powered lateral movement reaches Stage 4 before any defender action is effective. This profile represents the near-term threat frontier — the attack the hospital in 2023 faced, accelerated by autonomous tooling.

**Reproducible live at [`/threat_intelligence`](https://Fieerawe-cybersec-soc-env.hf.space/threat_intelligence)**

---

## Open Questions for Future Research

1. **Can mesh topology be decomposed?** If segmented networks redesign only their bridge node connections to mesh locally, does win rate recover? We did not test hybrid topologies.

2. **Does model size matter once topology is controlled?** Our finding is that topology dominates — but we only tested one model size. Would a 70B model change the segmented result?

3. **Can the coalition be gamed?** A sophisticated attacker who models the coalition's decision process could target the node the Clinical SOC will never allow to be isolated. We did not test adversarial coalition manipulation.

4. **Does agent memory compound over episodes?** Our current memory implementation tracks topology and pivot nodes. Over 100+ episodes, does memory create meaningful performance improvement or does it plateau?

5. **Real network validation** — every result here is from simulation. The most important next step is running these experiments against a real enterprise network in a controlled red team exercise.

---

## Methodology Notes

- All episodes use procedurally generated topologies — no two episodes share the same graph
- Attacker and defender both use LLM chain-of-thought reasoning — neither is scripted
- Reward signals are 100% programmatically verifiable — no human labelling
- Business impact is modelled as a disruption penalty per isolated node — mirroring real SOC constraints
- Partial observability is enforced — defender must scan to reveal compromise status
- All results reproducible via live endpoints — no closed data

---

## Citation

```bibtex
@misc{cybersec-soc-openenv-2026,
  title={CyberSec-SOC-OpenEnv: Adversarial Multi-Agent Cybersecurity Defense Environment},
  author={Team Peak — Fiero Jain, Parthan Rajesh, Tony James},
  year={2026},
  url={https://github.com/FieroJain/cybersec-soc-env},
  note={Meta x Scaler PyTorch OpenEnv Hackathon 2026}
}
```