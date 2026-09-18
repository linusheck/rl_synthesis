import unittest
import tempfile

import numpy as np
import tensorflow as tf

from compact_rl.rl.shielding.risk_budget_training_env import (
    allocation_choice_is_active,
)
from compact_rl.rl.shielding.train_risk_budget_shield import (
    compute_returns_and_mask,
    ppo_surrogate_loss,
    resolve_fixed_agent_folder,
)


class AllocationChoiceMaskTest(unittest.TestCase):
    def test_only_nontrivial_wasteless_choices_are_active(self):
        self.assertFalse(allocation_choice_is_active(0.0, [0.2, 0.3]))
        self.assertFalse(allocation_choice_is_active(0.2, [0.2, 0.0]))
        self.assertFalse(allocation_choice_is_active(0.5, [0.2, 0.3]))
        self.assertTrue(allocation_choice_is_active(0.4, [0.2, 0.3]))

    def test_inactive_decision_has_exactly_zero_ppo_gradient(self):
        new_log_probs = tf.Variable([0.0, 0.0])
        with tf.GradientTape() as tape:
            loss = ppo_surrogate_loss(
                new_log_probs=new_log_probs,
                old_log_probs=tf.zeros(2),
                advantages=tf.ones(2),
                outer_discount=tf.ones(2),
                mask=tf.constant([1.0, 0.0]),
            )
        gradient = tape.gradient(loss, new_log_probs).numpy()
        self.assertNotEqual(float(gradient[0]), 0.0)
        self.assertEqual(float(gradient[1]), 0.0)


class CompleteReturnTest(unittest.TestCase):
    def test_returns_stop_at_terminal_and_exclude_trailing_episode(self):
        rewards = np.array([[1.0, 2.0, 10.0, 20.0]], dtype=np.float32)
        discounts = np.array([[0.5, 0.0, 0.5, 0.5]], dtype=np.float32)
        returns, complete = compute_returns_and_mask(rewards, discounts)
        np.testing.assert_allclose(returns, [[2.0, 2.0, 20.0, 20.0]])
        np.testing.assert_array_equal(complete, [[True, True, False, False]])


class FixedAgentPathTest(unittest.TestCase):
    def test_missing_checkpoint_is_not_silently_accepted(self):
        with self.assertRaises(FileNotFoundError):
            resolve_fixed_agent_folder("models/example", "definitely-missing")

    def test_absolute_checkpoint_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = tf.train.Checkpoint(value=tf.Variable(1.0))
            tf.train.CheckpointManager(checkpoint, directory, max_to_keep=1).save()
            self.assertEqual(
                resolve_fixed_agent_folder("models/example", directory), directory
            )


if __name__ == "__main__":
    unittest.main()
