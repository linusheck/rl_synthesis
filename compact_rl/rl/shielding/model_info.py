"""Model info."""

from dataclasses import dataclass
import stormpy


def observation_index(model, state: int) -> int:
    """Return the feature-table row for a state in either an MDP or POMDP."""
    if hasattr(model, "get_observation"):
        return model.get_observation(state)
    return state


def observation_to_state_map(model) -> list[int]:
    """Build the fully-observable observation/state bijection used by shields."""
    nr_observations = getattr(model, "nr_observations", model.nr_states)
    mapping = [None] * nr_observations
    for state in range(model.nr_states):
        mapping[observation_index(model, state)] = state
    assert None not in mapping, "Some observations do not map to any state."
    return mapping

@dataclass
class ModelInfo:
    """Information about a model."""

    model: stormpy.storage.SparsePomdp
    observation_to_state: list[int]
    bad_state: str
    vmin: list[float]
    vmax: list[float]
