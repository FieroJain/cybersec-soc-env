"""
server/soc_environment.py — Full SOC analyst RL environment implementation.

Implements the SOCEnvironment class, which drives a procedurally generated
enterprise network under a multi-stage cyberattack. The agent plays the role
of a Security Operations Center (SOC) analyst.

Features:
    - 4 procedural network topologies (star, mesh, segmented, hierarchical)
    - Partial observability (compromise visible only after scan)
    - Alert noise simulation (5 % false-alert rate)
    - Business impact scoring (isolation of high-value nodes has a cost)
    - MITRE ATT&CK inspired 4-stage attack model with exfiltration timer
    - Three difficulty levels: easy / medium / hard
"""

import uuid
from typing import Optional
import time

import networkx as nx
import numpy as np

from openenv.core.env_server.interfaces import Environment

from ..models import SOCAction, SOCObservation, SOCState


# ── CONSTANTS ────────────────────────────────────────────────────────────────

TASK_CONFIG: dict = {
    "easy":   {"nodes": 5,  "start_compromised": 1, "max_steps": 20},
    "medium": {"nodes": 10, "start_compromised": 2, "max_steps": 35},
    "hard":   {"nodes": 20, "start_compromised": 3, "max_steps": 50},
}

NODE_TYPES: list[str] = [
    "workstation",
    "database_server",
    "web_server",
    "auth_server",
    "file_server",
]

HIGH_VALUE_TYPES: set[str] = {"database_server", "file_server", "auth_server"}

TOPOLOGIES: list[str] = ["star", "mesh", "segmented", "hierarchical"]

# Business impact weights per node type (isolation disruption cost)
BUSINESS_WEIGHT: dict[str, float] = {
    "workstation":     0.1,
    "web_server":      0.3,
    "auth_server":     0.5,
    "database_server": 0.8,
    "file_server":     0.6,
}


# ── ENVIRONMENT ──────────────────────────────────────────────────────────────


class SOCEnvironment(Environment):
    """
    SOC analyst RL environment.

    The agent defends a procedurally generated enterprise network against
    an autonomous attacker that follows a MITRE ATT&CK inspired kill-chain:
        Stage 1 – Initial Compromise
        Stage 2 – Credential Access  (auth_server hit)
        Stage 3 – Lateral Movement   (2+ nodes compromised)
        Stage 4 – Exfiltration       (database/file_server hit at stage 3)

    The episode ends when:
        - All active threats are contained (defender wins → +5 reward)
        - Exfiltration timer reaches 5 steps (attacker wins → -5 reward)
        - The step limit is reached (defender loses on points)

    Observation is partial: the agent can only see whether a node is
    compromised after performing a scan action on that node.
    """

    def __init__(self, task_level: str = "medium", seed: int = 42) -> None:
        """
        Initialise the SOC environment.

        Args:
            task_level: Difficulty level — one of 'easy', 'medium', 'hard'.
            seed: Base random seed. Each episode randomises from this base.
        """
        super().__init__()
        assert task_level in TASK_CONFIG, (
            f"task_level must be one of {list(TASK_CONFIG)}, got '{task_level}'"
        )
        self.task_level: str = task_level
        self.base_seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._graph: nx.DiGraph = nx.DiGraph()
        self._topology_type: str = "mesh"
        self._step_count: int = 0
        self._alerts_log: list[str] = []
        self._state: SOCState = SOCState()
        self._business_impact: float = 0.0
        self._episode_seed: int = seed
        # Attacker adaptive state
        self._attacker_last_action_step: int = 0
        self._attacker_evasion_mode: bool = False
        self._attacker_quiet_steps: int = 0

    # ── NETWORK GENERATION ────────────────────────────────────────────────────

    def _build_network(self, n: int) -> None:
        """
        Procedurally generate a directed network graph with a random topology.

        Topology is drawn uniformly from: star, mesh, segmented, hierarchical.
        Each node is assigned a random type, vulnerability count, security
        level, asset value, and state flags (compromised, isolated, scanned).

        Args:
            n: Number of nodes (from TASK_CONFIG for the current level).
        """
        self._graph = nx.DiGraph()
        topology: str = str(self.rng.choice(TOPOLOGIES))
        self._topology_type = topology

        for i in range(n):
            node_type: str = NODE_TYPES[i % len(NODE_TYPES)]
            self._graph.add_node(i, **{
                "type": node_type,
                "vulnerabilities": int(self.rng.integers(0, 6)),
                "security_level": float(
                    np.clip(self.rng.normal(0.6, 0.2), 0.1, 1.0)
                ),
                "asset_value": BUSINESS_WEIGHT[node_type],
                "compromised": False,
                "isolated": False,
                "scanned": False,
                "_was_visible": False,
            })

        if topology == "star":
            # Hub-and-spoke: node 0 is the central hub.
            for i in range(1, n):
                self._graph.add_edge(0, i)
                self._graph.add_edge(i, 0)

        elif topology == "mesh":
            # Linear backbone with random extra edges for density.
            for i in range(n - 1):
                self._graph.add_edge(i, i + 1)
                self._graph.add_edge(i + 1, i)
            extra: int = max(2, n // 3)
            for _ in range(extra):
                a, b = self.rng.integers(0, n, size=2)
                if int(a) != int(b):
                    self._graph.add_edge(int(a), int(b))

        elif topology == "segmented":
            # Two isolated segments bridged by a single link.
            mid: int = n // 2
            for i in range(mid - 1):
                self._graph.add_edge(i, i + 1)
                self._graph.add_edge(i + 1, i)
            for i in range(mid, n - 1):
                self._graph.add_edge(i, i + 1)
                self._graph.add_edge(i + 1, i)
            self._graph.add_edge(mid - 1, mid)
            self._graph.add_edge(mid, mid - 1)

        elif topology == "hierarchical":
            # Binary tree: each node i has children at 2i+1 and 2i+2.
            for i in range(n):
                for child in [2 * i + 1, 2 * i + 2]:
                    if child < n:
                        self._graph.add_edge(i, child)
                        self._graph.add_edge(child, i)

    # ── RESET ─────────────────────────────────────────────────────────────────

    def reset(self) -> SOCObservation:
        """
        Start a new episode.

        Generates a fresh network with a new random topology, seeds the
        attacker on `start_compromised` random nodes, and returns the initial
        partial observation. The agent cannot see compromise status on any node
        until it performs a scan action.

        Returns:
            SOCObservation with reward=0.0 and done=False.
        """
        self._episode_seed = int(self.rng.integers(0, 2 ** 31))
        
        self.rng = np.random.default_rng(int(time.time() * 1000) % (2**31))

        cfg: dict = TASK_CONFIG[self.task_level]
        n: int = cfg["nodes"]

        self._build_network(n)
        self._step_count = 0
        self._alerts_log = []
        self._business_impact = 0.0

        # Seed the attacker on randomly chosen start nodes.
        start_nodes: list[int] = [
            int(x)
            for x in self.rng.choice(n, cfg["start_compromised"], replace=False)
        ]
        for node_id in start_nodes:
            self._graph.nodes[node_id]["compromised"] = True

        self._state = SOCState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            true_compromised=start_nodes,
            task_level=self.task_level,
            topology_type=self._topology_type,
        )
        # Reset attacker adaptive state each episode
        self._attacker_last_action_step = 0
        self._attacker_evasion_mode = False
        self._attacker_quiet_steps = 0

        self._alerts_log.append(
            f"[t=0] New episode. Topology={self._topology_type}. "
            f"Nodes={n}. Threats={len(start_nodes)}."
        )
        return self._make_observation(reward=0.0, done=False)

    # ── STEP ──────────────────────────────────────────────────────────────────

    def step(self, action: SOCAction) -> SOCObservation:
        """
        Apply a defender action, advance the attacker, and compute the reward.

        The method never raises an exception for any valid action_type value.
        Unknown action types are treated as 'nothing'. All values returned
        in the observation are JSON-serialisable Python primitives.

        Args:
            action: SOCAction chosen by the agent.

        Returns:
            SOCObservation with current reward and done flag embedded.
        """
        self._step_count += 1
        self._state.step_count = self._step_count
        reward: float = -0.05  # per-step urgency penalty

        nid: int = action.target_node_id
        node: Optional[dict] = self._graph.nodes.get(nid) if nid >= 0 else None

        # ── APPLY DEFENDER ACTION ─────────────────────────────────────────────

        if action.action_type == "scan" and node is not None:
            node["scanned"] = True
            if node["compromised"] and not node.get("_was_visible", False):
                reward += 0.5
                node["_was_visible"] = True
                self._alerts_log.append(
                    f"[t={self._step_count}] SCAN node {nid} "
                    f"({node['type']}): COMPROMISE CONFIRMED"
                )
            else:
                # 5 % false-alert rate even on clean nodes.
                if float(self.rng.random()) < 0.05:
                    self._alerts_log.append(
                        f"[t={self._step_count}] SCAN node {nid}: "
                        f"false alert triggered"
                    )
                else:
                    self._alerts_log.append(
                        f"[t={self._step_count}] SCAN node {nid}: clean"
                    )

        elif action.action_type == "isolate" and node is not None:
            if not node["isolated"]:
                node["isolated"] = True
                # Remove all edges to/from this node.
                edges_to_remove = (
                    list(self._graph.in_edges(nid))
                    + list(self._graph.out_edges(nid))
                )
                self._graph.remove_edges_from(edges_to_remove)

                biz_cost: float = float(node["asset_value"])
                self._business_impact += biz_cost
                self._state.business_impact_score = round(self._business_impact, 3)

                if node["compromised"] and node["scanned"]:
                    reward += 1.0  # true positive isolation
                    self._alerts_log.append(
                        f"[t={self._step_count}] ISOLATE node {nid} "
                        f"({node['type']}): CONTAINED. "
                        f"Biz cost: {biz_cost:.2f}"
                    )
                else:
                    reward -= 0.2  # false positive
                    reward -= biz_cost * 0.1  # extra penalty for high-value FP
                    self._state.false_isolations += 1
                    self._alerts_log.append(
                        f"[t={self._step_count}] ISOLATE node {nid} "
                        f"({node['type']}): FALSE POSITIVE. "
                        f"Biz cost: {biz_cost:.2f}"
                    )

        elif action.action_type == "patch" and node is not None:
            node["security_level"] = float(
                min(1.0, node["security_level"] + 0.2)
            )
            node["vulnerabilities"] = int(max(0, node["vulnerabilities"] - 1))
            reward += 0.3
            self._alerts_log.append(
                f"[t={self._step_count}] PATCH node {nid} "
                f"({node['type']}): hardened."
            )

        elif action.action_type == "firewall":
            self._state.firewall_active = True
            self._state.firewall_steps_remaining = 10
            self._alerts_log.append(
                f"[t={self._step_count}] FIREWALL deployed (10 steps)."
            )

        else:
            # 'nothing' action or unknown type: apply spreading penalty.
            active_threats: int = sum(
                1
                for nid_check in self._graph.nodes
                if self._graph.nodes[nid_check]["compromised"]
                and not self._graph.nodes[nid_check]["isolated"]
            )
            if active_threats >= 2:
                reward -= 0.1

        # ── ADVANCE ATTACKER ──────────────────────────────────────────────────
        self._attacker_step()

        # ── UPDATE FIREWALL TIMER ─────────────────────────────────────────────
        if self._state.firewall_active:
            self._state.firewall_steps_remaining -= 1
            if self._state.firewall_steps_remaining <= 0:
                self._state.firewall_active = False
                self._alerts_log.append(
                    f"[t={self._step_count}] FIREWALL expired."
                )

        # ── UPDATE TRUE STATE ─────────────────────────────────────────────────
        active: list[int] = [
            int(nd)
            for nd in self._graph.nodes
            if self._graph.nodes[nd]["compromised"]
            and not self._graph.nodes[nd]["isolated"]
        ]
        self._state.true_compromised = active

        cfg = TASK_CONFIG[self.task_level]

        # ── TERMINATION: attacker completes exfiltration ──────────────────────
        if self._state.exfiltration_timer >= 5:
            reward -= 5.0
            self._alerts_log.append(
                f"[t={self._step_count}] "
                f"*** ATTACKER WINS: exfiltration complete ***"
            )
            self._state.total_reward += reward
            return self._make_observation(
                reward=reward, done=True, defender_wins=False
            )

        # ── TERMINATION: all threats contained ───────────────────────────────
        if len(active) == 0:
            reward += 5.0
            if self._state.false_isolations == 0:
                reward += 2.0  # perfect containment bonus
            self._state.containment_success = True
            self._alerts_log.append(
                f"[t={self._step_count}] "
                f"*** DEFENDER WINS: all threats contained ***"
            )
            self._state.total_reward += reward
            return self._make_observation(
                reward=reward, done=True, defender_wins=True
            )

        # ── TERMINATION: step limit reached ──────────────────────────────────
        if self._step_count >= cfg["max_steps"]:
            self._state.total_reward += reward
            return self._make_observation(
                reward=reward, done=True, defender_wins=False
            )

        self._state.total_reward += reward
        return self._make_observation(reward=reward, done=False)

    # ── ATTACKER SIMULATION ───────────────────────────────────────────────────

    def _attacker_step(self) -> None:
        new_infections: list[int] = []
        for src in list(self._graph.nodes):
            src_data: dict = self._graph.nodes[src]
            if not src_data["compromised"] or src_data["isolated"]:
                continue
            for dst in self._graph.successors(src):
                dst_data: dict = self._graph.nodes[dst]
                if dst_data["compromised"] or dst_data["isolated"]:
                    continue
                p: float = (
                    0.15
                    * (1.0 - dst_data["security_level"])
                    * (1.0 + src_data["vulnerabilities"] * 0.1)
                )
                if self._state.firewall_active:
                    p *= 0.4
                p = float(np.clip(p, 0.0, 0.95))
                if float(self.rng.random()) < p:
                    new_infections.append(int(dst))

        for dst in new_infections:
            self._graph.nodes[dst]["compromised"] = True
            self._alerts_log.append(
                f"[t={self._step_count}] ATTACKER spreads to "
                f"node {dst} ({self._graph.nodes[dst]['type']})"
            )

        for nid in self._graph.nodes:
            nd: dict = self._graph.nodes[nid]
            if not nd["compromised"] and not nd["isolated"]:
                if float(self.rng.random()) < 0.05:
                    self._alerts_log.append(
                        f"[t={self._step_count}] ALERT node {nid}: "
                        f"suspicious traffic (false positive)"
                    )

        active: list[int] = [
            n for n in self._graph.nodes
            if self._graph.nodes[n]["compromised"]
            and not self._graph.nodes[n]["isolated"]
        ]
        types_hit: set[str] = {self._graph.nodes[n]["type"] for n in active}

        if "auth_server" in types_hit and self._state.attack_stage < 2:
            self._state.attack_stage = 2
            self._alerts_log.append(
                f"[t={self._step_count}] STAGE 2: Credential Access achieved"
            )
        if len(active) >= 2 and self._state.attack_stage < 3:
            self._state.attack_stage = 3
            self._alerts_log.append(
                f"[t={self._step_count}] STAGE 3: Lateral Movement active"
            )
        if (
            types_hit & {"database_server", "file_server"}
            and self._state.attack_stage >= 3
            and self._state.attack_stage < 4
        ):
            self._state.attack_stage = 4
            self._alerts_log.append(
                f"[t={self._step_count}] STAGE 4: Exfiltration STARTING"
            )
        if self._state.attack_stage == 4:
            self._state.exfiltration_timer += 1

    # ── OBSERVATION BUILDER ───────────────────────────────────────────────────

    def _make_observation(
        self,
        reward: float,
        done: bool,
        defender_wins: bool = False,
    ) -> SOCObservation:
        """
        Build a partial-observability observation for the agent.

        alert_score per node = true_signal (0.6 if compromised, else 0.0)
        + Gaussian noise (mean=0, σ=0.08), clipped to [0, 1].

        Compromise status (visible_compromise) is only True if BOTH:
            - The node is compromised.
            - The node has been scanned by the agent.

        All values are plain Python primitives (int, float, bool, str, list).

        Args:
            reward:        Step reward to embed in the observation.
            done:          Episode termination flag.
            defender_wins: True only when all threats are contained.

        Returns:
            SOCObservation suitable for JSON serialisation.
        """
        statuses: list[dict] = []
        for nid in self._graph.nodes:
            nd: dict = self._graph.nodes[nid]
            base_alert: float = 0.6 if nd["compromised"] else 0.0
            noise: float = float(
                np.clip(self.rng.normal(0, 0.08), -0.2, 0.2)
            )
            alert_score: float = float(np.clip(base_alert + noise, 0.0, 1.0))
            statuses.append({
                "id":                 int(nid),
                "type":               str(nd["type"]),
                "alert_score":        round(alert_score, 3),
                "is_isolated":        bool(nd["isolated"]),
                "visible_compromise": bool(nd["compromised"] and nd["scanned"]),
                "asset_value":        round(float(nd["asset_value"]), 3),
            })

        return SOCObservation(
            node_statuses=statuses,
            attack_stage=int(self._state.attack_stage),
            timestep=int(self._step_count),
            alerts=self._alerts_log[-5:],
            topology_type=str(self._topology_type),
            business_impact_score=round(float(self._business_impact), 3),
            done=bool(done),
            reward=round(float(reward), 4),
            defender_wins=bool(defender_wins),
        )

    # ── STATE PROPERTY ────────────────────────────────────────────────────────

    @property
    def state(self) -> SOCState:
        """
        Return the full ground-truth state for grading.

        This property is exposed via the /state endpoint and never
        sent to the agent during normal training.

        Returns:
            SOCState with all internal flags and counters.
        """
        return self._state


