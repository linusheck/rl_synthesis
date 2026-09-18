"""State IDs need not equal observation IDs, even in an MDP."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from compact_rl.rl.shielding import risk_budget, risk_budget_training_env


def check_permuted_observations():
    # State features are [10,11], [20,21], [30,31], stored in observation order.
    model = SimpleNamespace(get_observation=lambda state: [2, 0, 1][state])
    features = np.array([[20, 21], [30, 31], [10, 11]], dtype=np.float32)
    info = SimpleNamespace(model=model, vmin=[0, 0.1, 0.2], vmax=[1, 0.9, 0.8])
    env = object.__new__(risk_budget_training_env.RiskBudgetTrainingEnv)
    env.model_info = info
    env._environment = SimpleNamespace(observation_valuations=features)
    env._pair_actions, env._pair_states, env._num_valid_pairs = [None], [None], [0]
    env.max_pairs, env.max_actions, env._state_feature_dim = 3, 1, 2
    obs = env._build_observation_for_lane(0, 0, [1.0])
    np.testing.assert_array_equal(obs["state_features"], [10, 11])
    np.testing.assert_array_equal(obs["pair_state_features"], [[20, 21], [30, 31], [0, 0]])

    class RecordingActor:
        max_pairs = 3
        input_tensor_spec = {"action_distribution": tf.TensorSpec([1], tf.float32)}

        def get_initial_state(self, batch_size):
            return ()

        def __call__(self, observation, step_type, network_state, training=False):
            self.observation = observation
            return SimpleNamespace(parameters={"loc": tf.zeros(tf.shape(observation["pair_mask"]))}), ()

    actor = RecordingActor()
    budget = risk_budget.NNRiskBudget(info, actor, ["a"], features)
    history = [0, [1.0], 0]
    budget(history, [1.0])  # Full history replay.
    np.testing.assert_array_equal(actor.observation["state_features"], [[[10, 11]]])
    np.testing.assert_array_equal(actor.observation["pair_state_features"][0, -1], obs["pair_state_features"])
    history.extend([1, [1.0], 0])
    budget(history, [1.0])  # Incremental cached inference.
    np.testing.assert_array_equal(actor.observation["state_features"], [[[20, 21]]])


class BudgetObservationTest(unittest.TestCase):
    def test_permuted_observation_ids_in_training_and_deployment(self):
        pairs = lambda *args: {(0, 1): 0.4, (0, 2): 0.6}
        with patch.object(risk_budget, "reachable_pair_probs", pairs), patch.object(
                risk_budget_training_env, "reachable_pair_probs", pairs):
            check_permuted_observations()


if __name__ == "__main__":
    unittest.main()
