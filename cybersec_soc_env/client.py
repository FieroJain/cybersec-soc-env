"""
client.py — WebSocket client for CyberSec-SOC-OpenEnv.

Provides both async (TRL/GRPO training) and sync (notebooks/testing) access
to the SOC environment server via the OpenEnv EnvClient base class.

Import note: EnvClient lives in openenv.core (re-exported from openenv.core.env_client).
"""

from typing import Any, Dict

from openenv.core import EnvClient  # correct path for openenv 0.2.x
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SOCAction, SOCObservation, SOCState


class SOCEnv(EnvClient[SOCAction, SOCObservation, SOCState]):
    """
    WebSocket client for CyberSec-SOC-OpenEnv.

    Connects to a running ``cybersec_soc_env.server.app:app`` server and
    exposes the standard OpenEnv client interface with typed request and
    response models.

    Async usage (RL training with TRL / GRPO)::

        async with SOCEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            obs = result.observation
            print(f"Topology: {obs.topology_type}")
            result = await env.step(
                SOCAction(action_type="scan", target_node_id=0)
            )
            print(result.observation.alerts)
            print(result.observation.business_impact_score)

    Sync usage (notebooks / testing)::

        with SOCEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            obs = result.observation
            result = env.step(
                SOCAction(action_type="isolate", target_node_id=2)
            )
            print(result.observation.topology_type)
    """

    # ── REQUIRED ABSTRACT METHOD IMPLEMENTATIONS ──────────────────────────────

    def _step_payload(self, action: SOCAction) -> Dict[str, Any]:
        """
        Convert a SOCAction to the JSON payload expected by the env server.

        Args:
            action: SOCAction instance with action_type and target_node_id.

        Returns:
            Dictionary with serialised action fields.
        """
        return {
            "action_type": action.action_type,
            "target_node_id": action.target_node_id,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SOCObservation]:
        """
        Parse the server response dict into a typed StepResult[SOCObservation].

        The server encodes the full observation inside ``payload["observation"]``.
        All SOCObservation fields are already plain Python primitives, so they
        can be unpacked directly from the dict.

        Args:
            payload: Raw JSON response data from the server step/reset endpoint.

        Returns:
            StepResult containing a typed SOCObservation.
        """
        obs_data: Dict[str, Any] = payload.get("observation", {})
        observation = SOCObservation(
            node_statuses=obs_data.get("node_statuses", []),
            attack_stage=obs_data.get("attack_stage", 1),
            timestep=obs_data.get("timestep", 0),
            alerts=obs_data.get("alerts", []),
            topology_type=obs_data.get("topology_type", "mesh"),
            business_impact_score=obs_data.get("business_impact_score", 0.0),
            defender_wins=obs_data.get("defender_wins", False),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SOCState:
        """
        Parse the server state response into a typed SOCState.

        Args:
            payload: JSON response from the /state WebSocket message.

        Returns:
            SOCState with full ground-truth environment information.
        """
        return SOCState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            true_compromised=payload.get("true_compromised", []),
            attack_stage=payload.get("attack_stage", 1),
            exfiltration_timer=payload.get("exfiltration_timer", 0),
            false_isolations=payload.get("false_isolations", 0),
            task_level=payload.get("task_level", "medium"),
            topology_type=payload.get("topology_type", "mesh"),
            firewall_active=payload.get("firewall_active", False),
            firewall_steps_remaining=payload.get("firewall_steps_remaining", 0),
            total_reward=payload.get("total_reward", 0.0),
            business_impact_score=payload.get("business_impact_score", 0.0),
            containment_success=payload.get("containment_success", False),
        )

    # ── TYPED OVERRIDES (documentation / IDE support) ─────────────────────────

    async def reset(self, **kwargs: Any) -> StepResult[SOCObservation]:
        """
        Reset the environment and return the initial StepResult.

        Returns:
            StepResult[SOCObservation] with done=False and reward=0.0.
        """
        return await super().reset(**kwargs)

    async def step(self, action: SOCAction, **kwargs: Any) -> StepResult[SOCObservation]:
        """
        Apply a defender action and return the next StepResult.

        Args:
            action: SOCAction specifying action_type and target_node_id.

        Returns:
            StepResult[SOCObservation] with embedded reward and done flag.
        """
        return await super().step(action, **kwargs)

    async def state(self) -> SOCState:
        """
        Return the full ground-truth state (for grading / evaluation).

        This bypasses partial observability and exposes the complete internal
        state of the environment. Not available during normal training.

        Returns:
            SOCState with true_compromised list and all internal counters.
        """
        return await super().state()
