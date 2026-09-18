"""Train a neural risk-budget function with masked clipped PPO.

The fixed agent is the policy being shielded. The learned action is a vector
of risk-allocation logits. Collection uses complete Monte-Carlo returns and
the actor loss includes only decisions for which wasteless refinement leaves a
real allocation choice. Forced allocations still contribute value targets and
rewards to earlier returns.
"""
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import stormpy
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

from compact_rl.robust_rl.robust_rl_tools import load_sketch
from compact_rl.rl.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from compact_rl.rl.agents.tf_agents_modif import ppo_agent
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
from compact_rl.rl.shielding.model_info import ModelInfo, observation_to_state_map
from compact_rl.rl.shielding.risk_budget_networks import (
    RiskBudgetActorNetwork,
    RiskBudgetValueNetwork,
)
from compact_rl.rl.shielding.risk_budget_training_env import RiskBudgetTrainingEnv
from compact_rl.rl.tests.general_test_tools import init_args


def build_model_info(model, bad_state="bad"):
    assert model.nr_states == getattr(model, "nr_observations", model.nr_states)
    components = stormpy.SparseModelComponents(
        transition_matrix=model.transition_matrix,
        reward_models=model.reward_models,
        state_labeling=model.labeling,
    )
    components.choice_labeling = model.choice_labeling
    if model.has_state_valuations():
        components.state_valuations = model.state_valuations
    if model.has_choice_origins():
        components.choice_origins = model.choice_origins
    mdp = stormpy.storage.SparseMdp(components)
    min_result = stormpy.model_checking(
        mdp, stormpy.parse_properties(f'Pmin=? [ F "{bad_state}" ]')[0]
    )
    max_result = stormpy.model_checking(
        mdp, stormpy.parse_properties(f'Pmax=? [ F "{bad_state}" ]')[0]
    )
    return ModelInfo(
        model=model,
        observation_to_state=observation_to_state_map(model),
        bad_state=bad_state,
        vmin=min_result.get_values(),
        vmax=max_result.get_values(),
    )


def build_shielded_policy(environment, tf_env, args, agent_folder):
    """Load the fixed stochastic agent whose proposals will be shielded."""
    agent = Recurrent_PPO_agent(
        environment=environment,
        tf_environment=tf_env,
        args=args,
        load=True,
        agent_folder=agent_folder,
    )
    policy = agent.get_policy(False, True)
    policy.set_greedy(False)
    policy.set_policy_masker()
    policy.set_return_real_logits(True)
    return policy


def resolve_fixed_agent_folder(project_path, agent_folder):
    """Resolve a fixed-agent checkpoint and fail instead of silently randomising it."""
    candidates = [agent_folder]
    if not os.path.isabs(agent_folder):
        project_name = os.path.basename(os.path.normpath(project_path))
        candidates.append(os.path.join("trained_agents", project_name, agent_folder))
    for candidate in candidates:
        if os.path.isdir(candidate) and tf.train.latest_checkpoint(candidate):
            return candidate
    attempted = ", ".join(os.path.abspath(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find a fixed-agent checkpoint for --load-agent. Tried: {attempted}"
    )


def build_agent(train_env, tf_train_env, learning_rate=8.6e-4, num_epochs=3,
                normalize_rewards=True, entropy_regularization=0.0,
                actor_net=None, value_net=None):
    """Build the former TF-Agents PPO object for loading legacy checkpoints.

    New training uses :func:`build_masked_ppo`. Keeping this constructor makes
    checkpoints written before the masked trainer deployable through
    ``shielding.py --budget nn``.
    """
    state_feature_dim = train_env.observation_spec()["state_features"].shape[0]
    if actor_net is None:
        actor_net = RiskBudgetActorNetwork(
            tf_train_env.observation_spec(), train_env.max_pairs,
            train_env.max_actions, state_feature_dim,
        )
    if value_net is None:
        value_net = RiskBudgetValueNetwork(
            tf_train_env.observation_spec(), train_env.max_actions, state_feature_dim,
        )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.99, beta_2=0.99,
        weight_decay=0.0001,
    )
    agent = ppo_agent.PPOAgent(
        tf_train_env.time_step_spec(),
        tf_train_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=num_epochs,
        train_step_counter=tf.Variable(0),
        discount_factor=1.0,
        use_gae=True,
        lambda_value=0.95,
        entropy_regularization=entropy_regularization,
        normalize_observations=False,
        normalize_rewards=normalize_rewards,
        importance_ratio_clipping=0.2,
    )
    agent.initialize()
    return agent


def load_agent(agent, path):
    """Restore a legacy ``tf.train.Checkpoint(agent=...)`` checkpoint."""
    checkpoint = tf.train.Checkpoint(agent=agent)
    manager = tf.train.CheckpointManager(checkpoint, path, max_to_keep=5)
    if not manager.latest_checkpoint:
        raise FileNotFoundError(f"No checkpoint found at {path}")
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    print(f"Loaded legacy risk-budget agent from: {manager.latest_checkpoint}")


class RiskBudgetValueBaseline(tf.keras.Model):
    """Feed-forward pre-action baseline used only during PPO training."""

    def __init__(self, hidden_units=(64, 64)):
        super().__init__(name="RiskBudgetValueBaseline")
        self._hidden = [
            tf.keras.layers.Dense(units, activation="relu", name=f"value_fc_{i}")
            for i, units in enumerate(hidden_units)
        ]
        self._value = tf.keras.layers.Dense(1, name="value")

    def call(self, observations, remaining_risk, slack, training=False):
        # These are all pre-action quantities. The sampled allocation is never
        # an input, so the value remains a valid policy-gradient baseline.
        x = tf.concat([
            tf.cast(observations["state_features"], tf.float32),
            tf.cast(observations["action_distribution"], tf.float32),
            tf.cast(remaining_risk[..., tf.newaxis], tf.float32),
            tf.cast(slack[..., tf.newaxis], tf.float32),
        ], axis=-1)
        for layer in self._hidden:
            x = layer(x, training=training)
        return tf.squeeze(self._value(x, training=training), axis=-1)


@dataclass
class MaskedPPOTrainer:
    actor_net: RiskBudgetActorNetwork
    value_net: RiskBudgetValueBaseline
    actor_optimizer: tf.keras.optimizers.Optimizer
    value_optimizer: tf.keras.optimizers.Optimizer


def build_actor_network(train_env, tf_train_env, initial_std=0.35,
                        include_budget_features=True):
    state_feature_dim = train_env.observation_spec()["state_features"].shape[0]
    return RiskBudgetActorNetwork(
        tf_train_env.observation_spec(), train_env.max_pairs,
        train_env.max_actions, state_feature_dim,
        init_action_stddev=initial_std,
        include_budget_features=include_budget_features,
    )


def restore_masked_actor(actor, path):
    """Restore an actor-only checkpoint written by :func:`save_masked_ppo`."""
    latest = tf.train.latest_checkpoint(path)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found at {path}")
    tf.train.Checkpoint(actor_net=actor).restore(latest).expect_partial()
    print(f"Loaded masked-PPO actor from: {latest}")


def build_masked_ppo(train_env, tf_train_env, learning_rate=8.6e-4,
                     value_learning_rate=3e-3, initial_std=0.35):
    actor = build_actor_network(train_env, tf_train_env, initial_std)
    value = RiskBudgetValueBaseline()
    actor_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.99, beta_2=0.99,
        weight_decay=0.0001,
    )
    value_optimizer = tf.keras.optimizers.legacy.Adam(value_learning_rate)
    return MaskedPPOTrainer(actor, value, actor_optimizer, value_optimizer)


def collect_window(actor, train_env, tf_train_env, policy_state, window_steps,
                   episode_age=None):
    """Collect a recurrent-policy window plus pre-action mask/baseline data."""
    observations_list = []
    actions_list = []
    step_types_list = []
    rewards_list = []
    discounts_list = []
    episode_age_list = []
    active_list = []
    risk_list = []
    slack_list = []

    time_step = tf_train_env.current_time_step()
    initial_policy_state = policy_state
    num_envs = int(time_step.step_type.shape[0])
    current_age = (
        np.zeros(num_envs, dtype=np.float32)
        if episode_age is None else np.asarray(episode_age, dtype=np.float32).copy()
    )

    for _ in range(window_steps):
        is_last = time_step.is_last().numpy()
        active_list.append(np.asarray([
            (not is_last[i]) and train_env._allocation_active(
                i, train_env._last_states[i], train_env._last_distributions[i]
            )
            for i in range(num_envs)
        ], dtype=bool))
        # Keep the critic and actor on the exact same pre-action context.
        risk_list.append(time_step.observation["remaining_risk"].numpy().copy())
        slack_list.append(time_step.observation["allocation_slack"].numpy().copy())

        dist, policy_state = actor(
            time_step.observation, time_step.step_type, policy_state, training=False
        )
        action = dist.sample()
        observations_list.append(time_step.observation)
        actions_list.append(action)
        step_types_list.append(time_step.step_type)
        episode_age_list.append(current_age.copy())

        next_time_step = tf_train_env.step(action)
        rewards_list.append(next_time_step.reward)
        discounts_list.append(next_time_step.discount)
        current_age = np.where(
            next_time_step.step_type.numpy() == ts.StepType.FIRST,
            0.0,
            current_age + 1.0,
        )
        time_step = next_time_step

    observations = {
        key: tf.stack([observation[key] for observation in observations_list], axis=1)
        for key in observations_list[0]
    }
    return dict(
        observations=observations,
        actions=tf.stack(actions_list, axis=1),
        step_types=tf.stack(step_types_list, axis=1),
        rewards=tf.stack(rewards_list, axis=1).numpy(),
        discounts=tf.stack(discounts_list, axis=1).numpy(),
        episode_ages=np.stack(episode_age_list, axis=1),
        active=np.stack(active_list, axis=1),
        remaining_risk=np.stack(risk_list, axis=1),
        slack=np.stack(slack_list, axis=1),
        initial_policy_state=initial_policy_state,
        final_policy_state=policy_state,
        final_episode_age=current_age,
    )


def compute_returns_and_mask(rewards, discounts):
    """Exact return-to-go and completed-episode mask for ``[lane, time]`` arrays."""
    returns = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[0], dtype=np.float32)
    for index in reversed(range(rewards.shape[1])):
        running = rewards[:, index] + discounts[:, index] * running
        returns[:, index] = running
    is_episode_end = discounts == 0.0
    complete = np.flip(
        np.maximum.accumulate(np.flip(is_episode_end, axis=1), axis=1), axis=1
    )
    return returns, complete


def _masked_mean(values, mask):
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(tf.cast(values, tf.float32) * mask) \
        / tf.maximum(tf.reduce_sum(mask), 1.0)


def _normalize_advantage(advantage, mask, eps=1e-8):
    mean = _masked_mean(advantage, mask)
    centered = tf.cast(advantage, tf.float32) - mean
    variance = _masked_mean(tf.square(centered), mask)
    return centered / tf.sqrt(variance + eps), mean, tf.sqrt(variance + eps)


def _apply_gradients(optimizer, grads, variables, clip_norm):
    pairs = [(gradient, variable) for gradient, variable in zip(grads, variables)
             if gradient is not None]
    if not pairs:
        return 0.0
    gradients, variables = zip(*pairs)
    raw_norm = tf.linalg.global_norm(gradients)
    if clip_norm and clip_norm > 0:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
    optimizer.apply_gradients(zip(gradients, variables))
    return float(raw_norm.numpy())


def ppo_surrogate_loss(new_log_probs, old_log_probs, advantages, outer_discount,
                       mask, clip=0.2):
    """Masked clipped PPO loss, exposed separately for regression testing."""
    mask = tf.cast(mask, tf.float32)
    log_ratio = tf.clip_by_value(new_log_probs - old_log_probs, -20.0, 20.0)
    ratio = tf.exp(log_ratio)
    clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip, 1.0 + clip)
    surrogate = tf.minimum(ratio * advantages, clipped_ratio * advantages)
    return -tf.reduce_sum(outer_discount * surrogate * mask) \
        / tf.maximum(tf.reduce_sum(mask), 1.0)


def train_ppo_batch(trainer, batch, gamma, ppo_epochs=4, critic_epochs=3,
                    ppo_clip=0.2, grad_clip=5.0, normalize_advantages=True):
    returns, complete = compute_returns_and_mask(batch["rewards"], batch["discounts"])
    actor_mask = complete & batch["active"]
    value_mask = complete
    if not np.any(value_mask):
        return None

    actor_mask_tf = tf.constant(actor_mask, tf.float32)
    value_mask_tf = tf.constant(value_mask, tf.float32)
    returns_tf = tf.constant(returns, tf.float32)
    risk_tf = tf.constant(batch["remaining_risk"], tf.float32)
    slack_tf = tf.constant(batch["slack"], tf.float32)
    outer_discount = tf.constant(np.power(gamma, batch["episode_ages"]), tf.float32)

    old_dist, _ = trainer.actor_net(
        batch["observations"], batch["step_types"],
        batch["initial_policy_state"], training=False,
    )
    old_log_probs = tf.stop_gradient(old_dist.log_prob(batch["actions"]))
    old_values = tf.stop_gradient(trainer.value_net(
        batch["observations"], risk_tf, slack_tf, training=False
    ))
    raw_advantage = returns_tf - old_values
    if normalize_advantages and np.any(actor_mask):
        advantages, _, advantage_std = _normalize_advantage(
            raw_advantage, actor_mask_tf
        )
    else:
        advantages = raw_advantage
        centered = raw_advantage - _masked_mean(raw_advantage, actor_mask_tf)
        advantage_std = tf.sqrt(
            _masked_mean(tf.square(centered), actor_mask_tf) + 1e-8
        )
    advantages = tf.stop_gradient(advantages)

    actor_loss = tf.constant(0.0)
    actor_grad_norm = 0.0
    approx_kl = 0.0
    clip_fraction = 0.0
    if np.any(actor_mask):
        for _ in range(max(int(ppo_epochs), 1)):
            with tf.GradientTape() as tape:
                dist, _ = trainer.actor_net(
                    batch["observations"], batch["step_types"],
                    batch["initial_policy_state"], training=True,
                )
                new_log_probs = dist.log_prob(batch["actions"])
                actor_loss = ppo_surrogate_loss(
                    new_log_probs, old_log_probs, advantages, outer_discount,
                    actor_mask_tf, ppo_clip,
                )
            gradients = tape.gradient(actor_loss, trainer.actor_net.trainable_variables)
            actor_grad_norm = _apply_gradients(
                trainer.actor_optimizer, gradients,
                trainer.actor_net.trainable_variables, grad_clip,
            )
        new_dist, _ = trainer.actor_net(
            batch["observations"], batch["step_types"],
            batch["initial_policy_state"], training=False,
        )
        new_log_probs = new_dist.log_prob(batch["actions"])
        approx_kl = float(_masked_mean(
            old_log_probs - new_log_probs, actor_mask_tf
        ).numpy())
        changed = tf.cast(
            tf.abs(tf.exp(tf.clip_by_value(
                new_log_probs - old_log_probs, -20.0, 20.0
            )) - 1.0) > ppo_clip,
            tf.float32,
        )
        clip_fraction = float(_masked_mean(changed, actor_mask_tf).numpy())

    critic_loss = tf.constant(0.0)
    critic_grad_norm = 0.0
    for _ in range(max(int(critic_epochs), 1)):
        with tf.GradientTape() as tape:
            values = trainer.value_net(
                batch["observations"], risk_tf, slack_tf, training=True
            )
            critic_loss = _masked_mean(tf.square(values - returns_tf), value_mask_tf)
        gradients = tape.gradient(
            critic_loss, trainer.value_net.trainable_variables
        )
        critic_grad_norm = _apply_gradients(
            trainer.value_optimizer, gradients,
            trainer.value_net.trainable_variables, grad_clip,
        )

    # The optimizer's actor ``loss`` is a PPO surrogate and has no direct
    # interpretation as intervention cost. Report the actual Monte-Carlo
    # objective separately: -return at FIRST for episodes whose terminal step
    # is also present in this collection window.
    step_types = batch["step_types"].numpy()
    target_episode_mask = complete & (step_types == ts.StepType.FIRST)
    target_costs = -returns[target_episode_mask]

    return dict(
        loss=float(actor_loss.numpy()),
        target_cost_mean=float(target_costs.mean()) if target_costs.size else 0.0,
        target_cost_std=float(target_costs.std()) if target_costs.size else 0.0,
        target_episode_count=int(target_costs.size),
        baseline=float(_masked_mean(old_values, actor_mask_tf).numpy()),
        advantage_std=float(advantage_std.numpy()),
        grad_norm=actor_grad_norm,
        critic_loss=float(critic_loss.numpy()),
        critic_grad_norm=critic_grad_norm,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        active_fraction=float(batch["active"].mean()),
        active_count=int(actor_mask.sum()),
        complete_count=int(value_mask.sum()),
    )


def collect_and_train(trainer, train_env, tf_train_env, num_iterations,
                      window_steps, num_envs, gamma, ppo_epochs=4,
                      critic_epochs=3, ppo_clip=0.2, grad_clip=5.0,
                      normalize_advantages=True, log_every=1):
    tf_train_env.reset()
    policy_state = trainer.actor_net.get_initial_state(batch_size=num_envs)
    episode_age = np.zeros(num_envs, dtype=np.float32)
    metrics_history = []
    for iteration in range(num_iterations):
        batch = collect_window(
            trainer.actor_net, train_env, tf_train_env, policy_state,
            window_steps, episode_age,
        )
        policy_state = batch["final_policy_state"]
        episode_age = batch["final_episode_age"]
        metrics = train_ppo_batch(
            trainer, batch, gamma, ppo_epochs, critic_epochs,
            ppo_clip, grad_clip, normalize_advantages,
        )
        if metrics is None:
            if iteration % log_every == 0:
                print(json.dumps({
                    "iteration": iteration,
                    "method": "ppo",
                    "skipped": "no complete episode in collection window",
                }))
            continue
        metrics_history.append(metrics)
        if iteration % log_every == 0:
            print("BATCH", json.dumps({
                "iteration": iteration,
                "method": "ppo",
                "std": float(tf.nn.softplus(trainer.actor_net._std_bias)),
                **metrics,
            }), flush=True)
    return metrics_history


def save_masked_ppo(trainer, path):
    """Save a deployment checkpoint at ``path`` and resumable state below it."""
    os.makedirs(path, exist_ok=True)
    actor_checkpoint = tf.train.Checkpoint(actor_net=trainer.actor_net)
    actor_manager = tf.train.CheckpointManager(actor_checkpoint, path, max_to_keep=5)
    saved = actor_manager.save()

    training_checkpoint = tf.train.Checkpoint(
        actor_net=trainer.actor_net,
        value_net=trainer.value_net,
        actor_optimizer=trainer.actor_optimizer,
        value_optimizer=trainer.value_optimizer,
    )
    training_manager = tf.train.CheckpointManager(
        training_checkpoint, os.path.join(path, "training_state"), max_to_keep=2
    )
    training_manager.save()
    print(f"Saved masked-PPO actor to: {saved}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(
        "--load-agent",
        help="fixed-agent checkpoint path or name under trained_agents/<project>/",
    )
    policy_group.add_argument(
        "--uniform-random-policy", action="store_true",
        help="train the budget shield against a uniform distribution over enabled actions",
    )
    parser.add_argument(
        "--deterministic-agent", action="store_true",
        help="use the loaded agent's enabled-action argmax distribution",
    )
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument(
        "--bad-state-label", default="bad",
        help="model label defining unsafe states for the risk constraint",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-environments", type=int, default=16)
    parser.add_argument(
        "--trajectory-num-steps", "--window-steps", dest="window_steps",
        type=int, default=128,
        help="steps per lane and update; use more than the episode horizon",
    )
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--episode-length", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=8.6e-4)
    parser.add_argument("--value-learning-rate", type=float, default=3e-3)
    parser.add_argument("--initial-std", type=float, default=0.35)
    parser.add_argument("--save-agent", default=None,
                        help="path for the learned risk-budget actor checkpoint")
    parser.add_argument("--use-clamp", action="store_true", default=False)
    parser.add_argument("--num-epochs", "--ppo-epochs", dest="ppo_epochs",
                        type=int, default=4)
    parser.add_argument("--critic-epochs", type=int, default=3)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--normalize-advantages", dest="normalize_advantages",
                        action="store_true", default=True)
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages",
                        action="store_false")
    # Backward-compatible aliases for the old command line. The new trainer
    # standardizes advantages rather than maintaining a reward normalizer.
    parser.add_argument("--normalize-rewards", dest="normalize_advantages",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-normalize-rewards", dest="normalize_advantages",
                        action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--goal-rew", type=float, default=100.0)
    parser.add_argument("--fail-rew", type=float, default=-100.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    args_cli = parser.parse_args()

    if args_cli.uniform_random_policy and args_cli.deterministic_agent:
        parser.error("--deterministic-agent requires --load-agent, not --uniform-random-policy")

    if args_cli.seed is not None:
        tf.keras.utils.set_random_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

    prism_path = os.path.join(args_cli.project_path, "sketch.templ")
    properties_path = os.path.join(args_cli.project_path, "sketch.props")
    args = init_args(
        prism_path=prism_path,
        properties_path=properties_path,
        use_rnn_less=True,
        max_steps=args_cli.episode_length,
        seed=args_cli.seed,
        prefer_stochastic=True,
    )
    sketch = load_sketch(project_path=args_cli.project_path)
    model = sketch.pomdp if hasattr(sketch, "pomdp") else sketch.quotient_mdp
    args.num_environments = args_cli.num_environments
    environment = EnvironmentWrapperVec(
        model, args, num_envs=args_cli.num_environments,
        enforce_compilation=True, goal_value=args_cli.goal_rew,
        antigoal_value=args_cli.fail_rew,
    )
    model_info = build_model_info(model, bad_state=args_cli.bad_state_label)
    tf_env = TFPyEnvironment(environment)
    if args_cli.uniform_random_policy:
        from compact_rl.rl.shielding.custom_policy import create_uniform_random_policy
        print("Training against a uniform random policy.")
        policy = create_uniform_random_policy(environment)
    else:
        fixed_agent_folder = resolve_fixed_agent_folder(
            args_cli.project_path, args_cli.load_agent
        )
        print(f"Loading fixed agent from: {fixed_agent_folder}")
        policy = build_shielded_policy(environment, tf_env, args, fixed_agent_folder)
    train_env = RiskBudgetTrainingEnv(
        environment=environment,
        policy=policy,
        model_info=model_info,
        actions=environment.action_keywords,
        nu=args_cli.nu,
        gamma=args_cli.gamma,
        use_l1_projection=not args_cli.use_clamp,
        deterministic_policy=args_cli.deterministic_agent,
    )
    tf_train_env = TFPyEnvironment(train_env)
    trainer = build_masked_ppo(
        train_env, tf_train_env,
        learning_rate=args_cli.learning_rate,
        value_learning_rate=args_cli.value_learning_rate,
        initial_std=args_cli.initial_std,
    )
    collect_and_train(
        trainer, train_env, tf_train_env,
        args_cli.num_iterations, args_cli.window_steps,
        args_cli.num_environments, args_cli.gamma,
        ppo_epochs=args_cli.ppo_epochs,
        critic_epochs=args_cli.critic_epochs,
        ppo_clip=args_cli.ppo_clip,
        grad_clip=args_cli.grad_clip,
        normalize_advantages=args_cli.normalize_advantages,
        log_every=args_cli.log_every,
    )
    if args_cli.save_agent is not None:
        save_masked_ppo(trainer, args_cli.save_agent)


if __name__ == "__main__":
    main()
