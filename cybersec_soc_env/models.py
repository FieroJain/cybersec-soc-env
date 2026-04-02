"""
models.py — Pydantic data models for CyberSec-SOC-OpenEnv.

Defines the Action, Observation, and State classes using the official
OpenEnv specification base classes.
"""

from typing import List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


# ── ACTION ──────────────────────────────────────────────────────────────────


class SOCAction(Action):
    """
    Defender action in the SOC environment.

    The agent selects one of five action types and optionally specifies a
    target node. Global actions (firewall, nothing) use target_node_id=-1.

    Action types:
        scan      — Reveal true compromise status on one node.
        isolate   — Disconnect a node from the network.
        patch     — Harden a node (reduce vulnerabilities, raise security_level).
        firewall  — Deploy network-wide firewall for 10 steps (halves spread prob).
        nothing   — No-op action (incurs extra penalty if threats are active).
    """

    action_type: str = Field(
        default="nothing",
        description="scan|isolate|patch|firewall|nothing",
    )
    target_node_id: int = Field(
        default=-1,
        description="Target host ID. -1 for global actions.",
    )


# ── OBSERVATION ─────────────────────────────────────────────────────────────


class SOCObservation(Observation):
    """
    Partial-observability observation returned to the agent each step.

    Inherited fields from Observation:
        done:   bool             — Episode termination flag.
        reward: Optional[float]  — Step reward.

    Compromise status is **only** visible after a scan action has been
    taken on that node (partial observability). Alert scores include
    Gaussian noise and a 5 % base false-alert rate.
    """

    node_statuses: List[dict] = Field(
        default_factory=list,
        description=(
            "List of dicts per node: "
            "{id, type, alert_score, is_isolated, visible_compromise, asset_value}"
        ),
    )
    attack_stage: int = Field(
        default=1,
        description="1=Initial 2=Credential 3=Lateral 4=Exfiltration",
    )
    timestep: int = Field(default=0)
    alerts: List[str] = Field(
        default_factory=list,
        description="Last 5 alert strings (may include false positives)",
    )
    topology_type: str = Field(
        default="mesh",
        description="Network topology: star|mesh|segmented|hierarchical",
    )
    business_impact_score: float = Field(
        default=0.0,
        description="Cumulative business disruption from isolations",
    )
    defender_wins: bool = Field(default=False)


# ── STATE ────────────────────────────────────────────────────────────────────


class SOCState(State):
    """
    Full ground truth state used by the grader.

    Never sent directly to the agent — only accessible via the /state
    endpoint. Inherited fields from State:
        step_count: int   — Current step within the episode.
        episode_id: str   — UUID for this episode.
    """

    true_compromised: List[int] = Field(
        default_factory=list,
        description="Node IDs that are truly compromised and not isolated.",
    )
    attack_stage: int = Field(
        default=1,
        description="Current MITRE ATT&CK stage (1–4).",
    )
    exfiltration_timer: int = Field(
        default=0,
        description="Steps elapsed in stage 4 (exfiltration). Game over at 5.",
    )
    false_isolations: int = Field(
        default=0,
        description="Count of nodes isolated without confirmed compromise.",
    )
    task_level: str = Field(
        default="medium",
        description="Difficulty level: easy|medium|hard.",
    )
    topology_type: str = Field(
        default="mesh",
        description="Network topology chosen for this episode.",
    )
    firewall_active: bool = Field(
        default=False,
        description="Whether the global firewall is currently deployed.",
    )
    firewall_steps_remaining: int = Field(
        default=0,
        description="Steps remaining on the active firewall.",
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward accumulated so far in the episode.",
    )
    business_impact_score: float = Field(
        default=0.0,
        description="Cumulative business disruption from isolation actions.",
    )
    containment_success: bool = Field(
        default=False,
        description="True if all threats were contained before time limit.",
    )
