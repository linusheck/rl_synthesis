"""Controlled corridor experiment using the repository's simulator and trainers.

Run from the repository root with MPLCONFIGDIR=/tmp/rl-synthesis-mpl:
  .venv/bin/python -m diagnostics.corridor_shield check
  .venv/bin/python -m diagnostics.corridor_shield train --iterations 100 --method reinforce
  .venv/bin/python -m diagnostics.corridor_shield train --iterations 100 --method normalized
  .venv/bin/python -m diagnostics.corridor_shield train --iterations 100 --method critic
  .venv/bin/python -m diagnostics.corridor_shield train --iterations 100 --method ppo
  .venv/bin/python -m diagnostics.corridor_shield evaluate --budget corridor-oracle

The evaluation subcommand invokes shielding.py's actual Click entry point,
substituting a fixed policy for its uniform-policy factory (no agent checkpoint).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from compact_rl.robust_rl.robust_rl_tools import load_sketch
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
from compact_rl.rl.shielding import custom_policy
from compact_rl.rl.shielding.risk_budget import BUDGET_FUNCTIONS, reachable_pair_probs
from compact_rl.rl.shielding.risk_budget_training_env import RiskBudgetTrainingEnv
from compact_rl.rl.shielding.train_risk_budget_shield import build_model_info
from compact_rl.rl.shielding.train_risk_budget_reinforce import (
    build_actor, collect_window, compute_returns_and_mask, train_step, save_actor,
)
from compact_rl.rl.tests.general_test_tools import init_args

PROJECT = "models/shielding/test-corridor"


def coordinates(model, state):
    valuation = json.loads(str(model.state_valuations.get_json(state)))
    return int(valuation["x"]), int(valuation["y"])


def retry_policy(environment):
    model = environment.stormpy_model
    table = {}
    for state in range(model.nr_states):
        action = "u" if coordinates(model, state) in ((0, 0), (0, 1)) else "c"
        probs = np.zeros(len(environment.action_keywords))
        probs[environment.action_keywords.index(action)] = 1
        if coordinates(model, state) == (0, 0):
            probs[environment.action_keywords.index("u")] = 0.5
            probs[environment.action_keywords.index("r")] = 0.5
        features = environment.observation_valuations[model.get_observation(state)]
        table[tuple(features)] = probs
    return custom_policy.create_custom_policy(environment, table)


class RetryBudget:
    """The exact zero-intervention witness for nu=1/16 and retry_policy."""
    def __init__(self, model_info):
        self.model_info = model_info

    def __call__(self, history, distribution):
        pairs = reachable_pair_probs(self.model_info, history[-3], distribution)
        weights = {pair: float(coordinates(self.model_info.model, pair[1]) in ((0, 0), (0, 1)))
                   for pair in pairs}
        total = sum(weights.values())
        return {p: w / total if total else 1 / len(pairs) for p, w in weights.items()}


class InteriorBudget(RetryBudget):
    """A strictly positive proposal with finite logits, also allowing this policy."""
    def __call__(self, history, distribution):
        state = history[-3]
        pairs = reachable_pair_probs(self.model_info, state, distribution)
        xy = coordinates(self.model_info.model, state)
        if xy == (0, 0):
            weights = {pair: 0.99 if coordinates(self.model_info.model, pair[1]) == (0, 1) else 0.01
                       for pair in pairs}
        elif xy == (0, 1):
            weights = {pair: {(0, 0): 0.35, (0, 1): 0.65}.get(coordinates(self.model_info.model, pair[1]), 1e-8)
                       for pair in pairs}
        else:
            weights = dict.fromkeys(pairs, 1.0)
        return {p: w / sum(weights.values()) for p, w in weights.items()}


def setup(args):
    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    settings = init_args(prism_path=f"{PROJECT}/sketch.templ", properties_path=f"{PROJECT}/sketch.props",
                         use_rnn_less=True, max_steps=args.horizon, seed=args.seed, prefer_stochastic=True)
    model = load_sketch(project_path=PROJECT).pomdp
    settings.num_environments = args.lanes
    environment = EnvironmentWrapperVec(model, settings, num_envs=args.lanes, enforce_compilation=True)
    info = build_model_info(model)
    env = RiskBudgetTrainingEnv(environment, retry_policy(environment), info,
                                environment.action_keywords, nu=0.0625, gamma=0.99)
    if args.legacy_features:
        original_build = env._build_observation_for_lane
        def build_observation(i, state, distribution):
            obs = original_build(i, state, distribution)
            obs["state_features"] = environment.observation_valuations[state].astype(np.float32)
            for k, s in enumerate(env._pair_states[i]):
                obs["pair_state_features"][k] = environment.observation_valuations[s]
            return obs
        env._build_observation_for_lane = build_observation
    return environment, info, env, TFPyEnvironment(env)

def allocation_active_mask(env, info):
    """Return which lanes currently have a non-forced wasteless allocation choice."""
    is_last = env.current_time_step().is_last().numpy()
    active = []
    for i, state in enumerate(env._last_states):
        d = env._last_distributions[i]
        slack = min(env._remaining_risk[i], env._shield._qmax(state, d)) - env._last_qmin_ds[i]
        capacities = [
            d[a] * p * (info.vmax[s] - info.vmin[s])
            for (a, s), p in reachable_pair_probs(info, state, d).items()
        ]
        active.append(
            slack > 1e-10
            and sum(c > 1e-10 for c in capacities) > 1
            and sum(capacities) > slack + 1e-10
            and not is_last[i]
        )
    return np.asarray(active, dtype=bool)


def allocation_slack(env):
    """Current distributable slack for every simulator lane, before sampling an allocation."""
    return np.asarray([
        min(env._remaining_risk[i], env._shield._qmax(state, env._last_distributions[i]))
        - env._last_qmin_ds[i]
        for i, state in enumerate(env._last_states)
    ], dtype=np.float32)


def install_active_recorder(env, info, active_history, risk_history=None, slack_history=None):
    """Record pre-action quantities that are legitimate action-independent baselines.

    `active_history` says whether the raw allocation can change successor budgets.
    The optional risk/slack histories are state information observed *before* the sampled
    allocation, so a critic may use them without depending on the current action.
    """
    original_step = env._step

    def recording_step(action):
        active_history.append(allocation_active_mask(env, info))
        if risk_history is not None:
            risk_history.append(np.asarray(env._remaining_risk, dtype=np.float32).copy())
        if slack_history is not None:
            slack_history.append(allocation_slack(env))
        return original_step(action)

    env._step = recording_step


def fixed_logits(env, budget):
    result = np.zeros((env.num_envs, env.max_pairs), np.float32)
    for i in range(env.num_envs):
        d = env._last_distributions[i]
        shares = budget([env._last_states[i], d, 0], d)
        for k, pair in enumerate(zip(env._pair_actions[i], env._pair_states[i])):
            result[i, k] = np.log(max(shares[pair], 1e-30))
    return tf.constant(result)


def evaluate_env(env, tf_env, actor=None, budget=None, stochastic=False, episodes=4):
    ts = tf_env.reset()
    state = actor.get_initial_state(env.num_envs) if actor else None
    costs = np.zeros(env.num_envs)
    totals = []
    ages = np.zeros(env.num_envs, dtype=int)
    for _ in range(episodes * (env._environment.args.max_steps + 1)):
        if actor:
            dist, state = actor(ts.observation, ts.step_type, state, training=False)
            action = dist.sample() if stochastic else dist.mean()
        else:
            action = fixed_logits(env, budget)
        ts = tf_env.step(action)
        costs -= np.power(env.gamma, ages) * ts.reward.numpy()
        ages += 1
        last = ts.is_last().numpy()
        totals.extend(costs[last])
        reset = ts.is_first().numpy()
        costs[reset] = 0
        ages[reset] = 0
    return float(np.mean(totals))


def check(args):
    environment, info, env, tf_env = setup(args)
    print("MODEL", [{"s": s, "obs": info.model.get_observation(s), "xy": coordinates(info.model, s),
                     "vmin": info.vmin[s], "vmax": info.vmax[s]}
                    for s in range(info.model.nr_states)], flush=True)
    # Independent Bellman iteration for eventual bad reachability of the fixed policy.
    values = np.array([float("bad" in info.model.labeling.get_labels_of_state(s))
                       for s in range(info.model.nr_states)])
    for _ in range(200):
        updated = values.copy()
        for state in range(info.model.nr_states):
            if "bad" in info.model.labeling.get_labels_of_state(state):
                continue
            xy = coordinates(info.model, state)
            proposed = {"u": 0.5, "r": 0.5} if xy == (0, 0) else {"u" if xy == (0, 1) else "c": 1.0}
            d = [proposed.get(label, 0) for label in env._choice_labels(state)]
            updated[state] = sum(d[a] * p * values[s]
                                 for (a, s), p in reachable_pair_probs(info, state, d).items())
        if np.max(np.abs(updated - values)) < 1e-14:
            values = updated
            break
        values = updated
    initial_state = info.model.initial_states[0]
    assert abs(values[initial_state] - 1 / 17) < 1e-12
    print("EXACT_RISK", values[initial_state], "threshold", env.nu, flush=True)
    # Each initial logit has independent Normal noise: their difference has sd sqrt(2)*sigma.
    for std in (0.35, 1.5):
        probability = 0.5 * math.erfc(math.log(16) / (2 * std))
        print("EXPLORATION", "std", std, "P(initial_risky_share>=16/17)", probability, flush=True)
    for name, factory in [("uniform", BUDGET_FUNCTIONS["uniform"]), ("oracle", RetryBudget),
                          ("interior", InteriorBudget)]:
        cost = evaluate_env(env, tf_env, budget=factory(info), episodes=8)
        print("CONTROL", name, "training_return_cost", cost, flush=True)
        if name in ("oracle", "interior"):
            assert cost < 1e-6, cost
    # Exact local budget update on all supported outcomes, including both retry branches.
    state = next(s for s in range(info.model.nr_states) if coordinates(info.model, s) == (0, 1))
    labels = env._choice_labels(state)
    d = [float(label == "u") for label in labels]
    slack = 0.125 - env._shield._qmin(state, d)
    for name, factory in [("uniform", BUDGET_FUNCTIONS["uniform"]), ("oracle", RetryBudget)]:
        allocation = env._shield._make_wasteless(factory(info)([state, d, 0], d), state, d, slack)
        print("ALLOCATION", name, [(coordinates(info.model, s), b,
              info.vmin[s] + slack * b / env._shield._transition_prob(state, d, a, s))
              for (a, s), b in allocation.items()], flush=True)


class CorridorValueBaseline(tf.keras.Model):
    """Small feed-forward value baseline for this diagnostic.

    It uses only pre-action information: the same global observation seen by the actor,
    plus current remaining risk and distributable slack. It never sees the sampled
    allocation, so it is a valid policy-gradient baseline. A feed-forward critic is
    intentional here: the corridor witness is memoryless, while a recurrent critic would
    require carrying its hidden state consistently across collection windows.
    """

    def __init__(self):
        super().__init__(name="CorridorValueBaseline")
        self._hidden_1 = tf.keras.layers.Dense(64, activation="relu")
        self._hidden_2 = tf.keras.layers.Dense(64, activation="relu")
        self._value = tf.keras.layers.Dense(1, activation=None)

    def call(self, observations, remaining_risk, slack, training=False):
        x = tf.concat([
            tf.cast(observations["state_features"], tf.float32),
            tf.cast(observations["action_distribution"], tf.float32),
            tf.cast(remaining_risk[..., tf.newaxis], tf.float32),
            tf.cast(slack[..., tf.newaxis], tf.float32),
        ], axis=-1)
        x = self._hidden_1(x, training=training)
        x = self._hidden_2(x, training=training)
        return tf.squeeze(self._value(x, training=training), axis=-1)


def _masked_mean(values, mask):
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(tf.cast(values, tf.float32) * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def _normalize_advantage(advantage, mask, eps=1e-8):
    """Standardize only over actor decisions that actually contribute to the gradient."""
    mask_tf = tf.cast(mask, tf.float32)
    mean = _masked_mean(advantage, mask_tf)
    centered = tf.cast(advantage, tf.float32) - mean
    variance = _masked_mean(tf.square(centered), mask_tf)
    return centered / tf.sqrt(variance + eps), mean, tf.sqrt(variance + eps)


def _apply_gradients(optimizer, grads, variables, clip_norm):
    pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
    if not pairs:
        return 0.0
    grad_list, var_list = zip(*pairs)
    raw_norm = tf.linalg.global_norm(grad_list)
    if clip_norm and clip_norm > 0:
        grad_list, _ = tf.clip_by_global_norm(grad_list, clip_norm)
    optimizer.apply_gradients(zip(grad_list, var_list))
    return float(raw_norm.numpy())


def train_normalized_reinforce(actor, optimizer, observations, step_types, actions, returns,
                               actor_mask, initial_policy_state, episode_ages, gamma, clip_norm):
    """REINFORCE with the exact active-action mask and whitened advantages."""
    mask = tf.constant(actor_mask, dtype=tf.float32)
    returns_tf = tf.constant(returns, dtype=tf.float32)
    baseline = _masked_mean(returns_tf, mask)
    advantage, _, advantage_std = _normalize_advantage(returns_tf - baseline, mask)
    outer_discount = tf.constant(np.power(gamma, episode_ages), dtype=tf.float32)

    with tf.GradientTape() as tape:
        dist, _ = actor(observations, step_types, initial_policy_state, training=True)
        log_probs = dist.log_prob(actions)
        loss = -tf.reduce_sum(outer_discount * tf.stop_gradient(advantage) * log_probs * mask) \
               / tf.maximum(tf.reduce_sum(mask), 1.0)
    grads = tape.gradient(loss, actor.trainable_variables)
    grad_norm = _apply_gradients(optimizer, grads, actor.trainable_variables, clip_norm)
    return dict(
        loss=float(loss.numpy()),
        baseline=float(baseline.numpy()),
        advantage_std=float(advantage_std.numpy()),
        grad_norm=grad_norm,
        critic_loss=float("nan"),
    )


def train_value_baseline(actor, actor_optimizer, critic, critic_optimizer, observations, step_types,
                         actions, returns, actor_mask, value_mask, initial_policy_state,
                         episode_ages, gamma, remaining_risk, slack, normalize_advantages,
                         clip_norm, critic_epochs):
    """REINFORCE with a learned pre-action V baseline and Monte-Carlo value targets.

    The actor still uses the exact complete return-to-go; the critic is only a control
    variate. The critic is trained on every complete-return step, including forced-action
    steps, because those are still valid state-value targets. Only the actor loss is masked
    to genuine allocation choices.
    """
    actor_mask_tf = tf.constant(actor_mask, dtype=tf.float32)
    value_mask_tf = tf.constant(value_mask, dtype=tf.float32)
    returns_tf = tf.constant(returns, dtype=tf.float32)
    risk_tf = tf.constant(remaining_risk, dtype=tf.float32)
    slack_tf = tf.constant(slack, dtype=tf.float32)
    outer_discount = tf.constant(np.power(gamma, episode_ages), dtype=tf.float32)

    # Use the critic *before* fitting it to this batch for the actor update.
    values_before = critic(observations, risk_tf, slack_tf, training=False)
    raw_advantage = returns_tf - tf.stop_gradient(values_before)
    if normalize_advantages:
        advantage, _, advantage_std = _normalize_advantage(raw_advantage, actor_mask_tf)
    else:
        advantage = raw_advantage
        centered = raw_advantage - _masked_mean(raw_advantage, actor_mask_tf)
        advantage_std = tf.sqrt(_masked_mean(tf.square(centered), actor_mask_tf) + 1e-8)

    with tf.GradientTape() as actor_tape:
        dist, _ = actor(observations, step_types, initial_policy_state, training=True)
        log_probs = dist.log_prob(actions)
        actor_loss = -tf.reduce_sum(
            outer_discount * tf.stop_gradient(advantage) * log_probs * actor_mask_tf
        ) / tf.maximum(tf.reduce_sum(actor_mask_tf), 1.0)
    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    actor_grad_norm = _apply_gradients(
        actor_optimizer, actor_grads, actor.trainable_variables, clip_norm
    )

    critic_loss = None
    critic_grad_norm = 0.0
    for _ in range(max(int(critic_epochs), 1)):
        with tf.GradientTape() as critic_tape:
            values = critic(observations, risk_tf, slack_tf, training=True)
            squared_error = tf.square(values - returns_tf)
            critic_loss = tf.reduce_sum(squared_error * value_mask_tf) \
                          / tf.maximum(tf.reduce_sum(value_mask_tf), 1.0)
        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
        critic_grad_norm = _apply_gradients(
            critic_optimizer, critic_grads, critic.trainable_variables, clip_norm
        )

    return dict(
        loss=float(actor_loss.numpy()),
        baseline=float(_masked_mean(values_before, actor_mask_tf).numpy()),
        advantage_std=float(advantage_std.numpy()),
        grad_norm=actor_grad_norm,
        critic_loss=float(critic_loss.numpy()),
        critic_grad_norm=critic_grad_norm,
    )



def train_ppo(actor, actor_optimizer, critic, critic_optimizer, observations, step_types,
              actions, returns, actor_mask, value_mask, initial_policy_state, episode_ages,
              gamma, remaining_risk, slack, normalize_advantages, clip_norm,
              critic_epochs, ppo_epochs, ppo_clip):
    """Clipped PPO on the same complete Monte-Carlo returns, with forced actions masked out.

    This deliberately keeps the corridor experiment simple: the critic supplies a baseline,
    while the exact complete return remains the target. No entropy bonus is used because the
    Gaussian logit entropy is unbounded in its standard deviation.
    """
    actor_mask_tf = tf.constant(actor_mask, dtype=tf.float32)
    value_mask_tf = tf.constant(value_mask, dtype=tf.float32)
    returns_tf = tf.constant(returns, dtype=tf.float32)
    risk_tf = tf.constant(remaining_risk, dtype=tf.float32)
    slack_tf = tf.constant(slack, dtype=tf.float32)
    outer_discount = tf.constant(np.power(gamma, episode_ages), dtype=tf.float32)

    old_dist, _ = actor(observations, step_types, initial_policy_state, training=False)
    old_log_probs = tf.stop_gradient(old_dist.log_prob(actions))
    old_values = tf.stop_gradient(critic(observations, risk_tf, slack_tf, training=False))
    raw_advantage = returns_tf - old_values
    if normalize_advantages:
        advantage, _, advantage_std = _normalize_advantage(raw_advantage, actor_mask_tf)
    else:
        advantage = raw_advantage
        centered = raw_advantage - _masked_mean(raw_advantage, actor_mask_tf)
        advantage_std = tf.sqrt(_masked_mean(tf.square(centered), actor_mask_tf) + 1e-8)
    advantage = tf.stop_gradient(advantage)

    actor_loss = None
    actor_grad_norm = 0.0
    approx_kl = 0.0
    clip_fraction = 0.0
    for _ in range(max(int(ppo_epochs), 1)):
        with tf.GradientTape() as actor_tape:
            dist, _ = actor(observations, step_types, initial_policy_state, training=True)
            log_probs = dist.log_prob(actions)
            log_ratio = tf.clip_by_value(log_probs - old_log_probs, -20.0, 20.0)
            ratio = tf.exp(log_ratio)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
            surrogate = tf.minimum(ratio * advantage, clipped_ratio * advantage)
            actor_loss = -tf.reduce_sum(outer_discount * surrogate * actor_mask_tf) \
                         / tf.maximum(tf.reduce_sum(actor_mask_tf), 1.0)
        actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
        actor_grad_norm = _apply_gradients(
            actor_optimizer, actor_grads, actor.trainable_variables, clip_norm
        )
        new_dist, _ = actor(observations, step_types, initial_policy_state, training=False)
        new_log_probs = new_dist.log_prob(actions)
        approx_kl = float(_masked_mean(old_log_probs - new_log_probs, actor_mask_tf).numpy())
        changed = tf.cast(tf.abs(tf.exp(tf.clip_by_value(new_log_probs - old_log_probs, -20.0, 20.0)) - 1.0)
                          > ppo_clip, tf.float32)
        clip_fraction = float(_masked_mean(changed, actor_mask_tf).numpy())

    critic_loss = None
    critic_grad_norm = 0.0
    for _ in range(max(int(critic_epochs), 1)):
        with tf.GradientTape() as critic_tape:
            values = critic(observations, risk_tf, slack_tf, training=True)
            squared_error = tf.square(values - returns_tf)
            critic_loss = tf.reduce_sum(squared_error * value_mask_tf) \
                          / tf.maximum(tf.reduce_sum(value_mask_tf), 1.0)
        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
        critic_grad_norm = _apply_gradients(
            critic_optimizer, critic_grads, critic.trainable_variables, clip_norm
        )

    return dict(
        loss=float(actor_loss.numpy()),
        baseline=float(_masked_mean(old_values, actor_mask_tf).numpy()),
        advantage_std=float(advantage_std.numpy()),
        grad_norm=actor_grad_norm,
        critic_loss=float(critic_loss.numpy()),
        critic_grad_norm=critic_grad_norm,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )

def train(args):
    environment, info, env, tf_env = setup(args)
    actor, optimizer = build_actor(env, tf_env, learning_rate=args.learning_rate)
    actor._std_bias.assign(np.log(np.expm1(args.initial_std)))

    critic = None
    critic_optimizer = None
    if args.method in ("critic", "ppo"):
        critic = CorridorValueBaseline()
        critic_optimizer = tf.keras.optimizers.legacy.Adam(args.value_learning_rate)

    active_history = []
    risk_history = []
    slack_history = []
    install_active_recorder(env, info, active_history, risk_history, slack_history)
    started = time.monotonic()
    records = []
    loss_history = []
    critic_loss_history = []

    for iteration in range(args.iterations + 1):
        if iteration % args.report_every == 0 or iteration == args.iterations:
            deterministic = evaluate_env(env, tf_env, actor=actor, episodes=4)
            stochastic = evaluate_env(env, tf_env, actor=actor, stochastic=True, episodes=4)
            record = dict(
                iteration=iteration,
                method=args.method,
                deterministic_cost=deterministic,
                stochastic_cost=stochastic,
                std=float(tf.nn.softplus(actor._std_bias)),
                seconds=time.monotonic() - started,
            )
            initial_ts = tf_env.reset()
            initial_dist, _ = actor(
                initial_ts.observation,
                initial_ts.step_type,
                actor.get_initial_state(env.num_envs),
            )
            record["initial_logits"] = initial_dist.mean().numpy()[0, :env._num_valid_pairs[0]].tolist()
            records.append(record)
            print("RESULT", json.dumps(record), flush=True)
            # Evaluation used this env, so explicitly start a fresh training window.
            tf_env.reset()
            policy_state = actor.get_initial_state(env.num_envs)
            age = np.zeros(env.num_envs)

        if iteration == args.iterations:
            break

        active_history.clear()
        risk_history.clear()
        slack_history.clear()
        (obs, actions, step_types, rewards, discounts, next_steps, ages,
         initial_state, policy_state, age) = collect_window(
            actor, tf_env, policy_state, args.window, age
        )
        returns, complete_valid = compute_returns_and_mask(rewards, discounts)
        active = np.asarray(active_history, dtype=bool).T
        remaining_risk = np.asarray(risk_history, dtype=np.float32).T
        slack = np.asarray(slack_history, dtype=np.float32).T

        actor_valid = complete_valid.copy()
        if args.active_only:
            # Keep every reward in every earlier return, but do not attach those
            # returns to raw allocations whose wasteless refinement is forced.
            actor_valid &= active
        if not np.any(actor_valid):
            raise AssertionError("No valid allocation decisions in training window")

        if args.method == "reinforce":
            loss, baseline, _ = train_step(
                actor, optimizer, obs, step_types, actions, returns, actor_valid,
                initial_state, ages, env.gamma
            )
            metrics = dict(
                loss=float(loss), baseline=float(baseline), advantage_std=float("nan"),
                grad_norm=float("nan"), critic_loss=float("nan")
            )
        elif args.method == "analytic":
            loss, baseline, c_star = train_step(
                actor, optimizer, obs, step_types, actions, returns, actor_valid,
                initial_state, ages, env.gamma, use_analytic_baseline=True
            )
            metrics = dict(
                loss=float(loss), baseline=float(baseline), advantage_std=float("nan"),
                grad_norm=float("nan"), critic_loss=float("nan"), c_star=float(c_star)
            )
        elif args.method == "normalized":
            metrics = train_normalized_reinforce(
                actor, optimizer, obs, step_types, actions, returns, actor_valid,
                initial_state, ages, env.gamma, args.grad_clip
            )
        elif args.method == "critic":
            metrics = train_value_baseline(
                actor, optimizer, critic, critic_optimizer, obs, step_types, actions,
                returns, actor_valid, complete_valid, initial_state, ages, env.gamma,
                remaining_risk, slack, args.normalize_advantages, args.grad_clip,
                args.critic_epochs,
            )
        elif args.method == "ppo":
            metrics = train_ppo(
                actor, optimizer, critic, critic_optimizer, obs, step_types, actions,
                returns, actor_valid, complete_valid, initial_state, ages, env.gamma,
                remaining_risk, slack, args.normalize_advantages, args.grad_clip,
                args.critic_epochs, args.ppo_epochs, args.ppo_clip,
            )
        else:
            raise ValueError(args.method)

        loss_value = float(metrics["loss"])
        loss_history.append((iteration, loss_value))
        if np.isfinite(metrics.get("critic_loss", np.nan)):
            critic_loss_history.append((iteration, metrics["critic_loss"]))

        if iteration % args.report_every == 0:
            batch_record = dict(
                iteration=iteration,
                method=args.method,
                active_fraction=float(active.mean()),
                active_count=int(actor_valid.sum()),
                complete_count=int(complete_valid.sum()),
                loss=loss_value,
                baseline=float(metrics.get("baseline", np.nan)),
                advantage_std=float(metrics.get("advantage_std", np.nan)),
                grad_norm=float(metrics.get("grad_norm", np.nan)),
                critic_loss=float(metrics.get("critic_loss", np.nan)),
            )
            if "critic_grad_norm" in metrics:
                batch_record["critic_grad_norm"] = float(metrics["critic_grad_norm"])
            if "c_star" in metrics:
                batch_record["c_star"] = float(metrics["c_star"])
            if "approx_kl" in metrics:
                batch_record["approx_kl"] = float(metrics["approx_kl"])
                batch_record["clip_fraction"] = float(metrics["clip_fraction"])
            print("BATCH", json.dumps(batch_record), flush=True)

        if not np.isfinite(loss_value):
            raise AssertionError(f"Nonfinite loss at {iteration}")

    plot_dir = Path(args.output) if args.output else Path(".")
    plot_dir.mkdir(parents=True, exist_ok=True)

    if loss_history and records:
        loss_iterations, losses = zip(*loss_history)
        report_iterations = [record["iteration"] for record in records]
        deterministic_costs = [record["deterministic_cost"] for record in records]
        stochastic_costs = [record["stochastic_cost"] for record in records]
        stds = [record["std"] for record in records]

        fig, loss_ax = plt.subplots(figsize=(9, 5.5))
        metric_ax = loss_ax.twinx()

        loss_line, = loss_ax.plot(loss_iterations, losses, label="Actor loss")
        deterministic_line, = metric_ax.plot(
            report_iterations, deterministic_costs, marker="o", label="Deterministic cost"
        )
        stochastic_line, = metric_ax.plot(
            report_iterations, stochastic_costs, marker="o", label="Stochastic cost"
        )
        std_line, = metric_ax.plot(
            report_iterations, stds, marker="o", label="Std"
        )

        loss_ax.set_xlabel("Iteration")
        loss_ax.set_ylabel("Actor loss")
        metric_ax.set_ylabel("Cost / std")
        loss_ax.set_title(f"Training metrics ({args.method})")
        loss_ax.grid(True, alpha=0.3)

        lines = [loss_line, deterministic_line, stochastic_line, std_line]
        loss_ax.legend(lines, [line.get_label() for line in lines], loc="best")

        fig.tight_layout()
        metrics_plot = plot_dir / "training_metrics.pdf"
        fig.savefig(metrics_plot, bbox_inches="tight")
        plt.close(fig)
        print("TRAINING_METRICS_PLOT", metrics_plot, flush=True)

    if critic_loss_history:
        critic_iterations, critic_losses = zip(*critic_loss_history)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(critic_iterations, critic_losses, label="Critic MSE")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Critic MSE")
        ax.set_title("Value-baseline fit")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        critic_plot = plot_dir / "critic_metrics.pdf"
        fig.savefig(critic_plot, bbox_inches="tight")
        plt.close(fig)
        print("CRITIC_METRICS_PLOT", critic_plot, flush=True)

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        save_actor(actor, args.output)
        if critic is not None:
            critic_ckpt = tf.train.Checkpoint(critic=critic)
            critic_manager = tf.train.CheckpointManager(
                critic_ckpt, str(Path(args.output) / "critic"), max_to_keep=5
            )
            critic_manager.save()
        Path(args.output, "results.json").write_text(json.dumps(records, indent=2) + "\n")


def evaluate(args):
    # Register before importing shielding: Click takes its budget choices at import time.
    BUDGET_FUNCTIONS["corridor-oracle"] = RetryBudget
    custom_policy.create_uniform_random_policy = retry_policy
    if args.legacy_features:
        from compact_rl.rl.shielding.risk_budget import NNRiskBudget
        NNRiskBudget._features = lambda self, state: np.asarray(self.state_features[state], dtype=np.float32)
    import shielding
    # shielding.py currently passes seed=None to init_args, despite --seed.
    original_init = shielding.init_args
    def seeded_init(**kwargs):
        kwargs["seed"] = args.seed
        return original_init(**kwargs)
    shielding.init_args = seeded_init
    argv = [PROJECT, "--shield", "budget", "--budget", args.budget,
            "--uniform-random-policy", "--nu", "0.0625", "--seed", str(args.seed),
            "--episode-length", str(args.horizon), "--num-environments", str(args.lanes),
            "--num-parallel-environments", str(args.lanes), "--min-episodes-per-environment", "8"]
    if args.output:
        argv += ["--budget-checkpoint", args.output]
    shielding.main.main(args=argv, standalone_mode=False)


def gradient(args):
    _, info, env, tf_env = setup(args)
    actor, optimizer = build_actor(env, tf_env)
    active_history = []
    install_active_recorder(env, info, active_history)
    ts = tf_env.reset()
    initial_state = actor.get_initial_state(env.num_envs)
    print("INITIAL_PAIRS", [(env._choice_labels(env._last_states[0])[a], coordinates(info.model, s))
                           for a, s in zip(env._pair_actions[0], env._pair_states[0])], flush=True)
    dist, _ = actor(ts.observation, ts.step_type, initial_state)
    print("VARIABLES", [(v.name, v.shape.as_list()) for v in actor.trainable_variables], flush=True)
    batch = collect_window(actor, tf_env, initial_state, args.window)
    obs, actions, steps, rewards, discounts, _, ages, initial_state, _, _ = batch
    returns, valid = compute_returns_and_mask(rewards, discounts)
    active = np.asarray(active_history).T
    print("ACTIVE_FRACTION", float(active.mean()), flush=True)
    if args.active_only:
        valid &= active
    if not np.any(valid):
        raise AssertionError("No valid active allocation decisions in gradient window")
    advantage = tf.constant(returns - returns[valid].mean())
    with tf.GradientTape() as tape:
        dist, _ = actor(obs, steps, initial_state, training=True)
        loss = -tf.reduce_sum(dist.log_prob(actions) * advantage * (0.99 ** ages) * valid) / valid.sum()
    grads = tape.gradient(loss, actor.trainable_variables)
    print("GRADIENTS", [(v.name, None if g is None else float(tf.linalg.norm(g)))
                        for v, g in zip(actor.trainable_variables, grads)], flush=True)
    before = [v.numpy().copy() for v in actor.trainable_variables]
    train_step(actor, optimizer, obs, steps, actions, returns, valid, initial_state, ages, 0.99)
    print("UPDATES", [(v.name, float(np.linalg.norm(v.numpy() - b)))
                      for v, b in zip(actor.trainable_variables, before)], flush=True)
    # A direct supervised target checks actor expressiveness without the RL estimator.
    initial = {k: v[:, 0] for k, v in obs.items()}
    opt = tf.keras.optimizers.legacy.Adam(0.003)
    for iteration in range(101):
        with tf.GradientTape() as tape:
            dist, _ = actor(initial, tf.zeros(env.num_envs, tf.int32), actor.get_initial_state(env.num_envs))
            gap = dist.mean()[:, 0] - dist.mean()[:, 1]
            loss = tf.reduce_mean((gap - np.log(99)) ** 2)
        if iteration in (0, 100):
            print("SUPERVISED", iteration, float(loss), float(tf.reduce_mean(gap)), flush=True)
        grads = tape.gradient(loss, actor.trainable_variables)
        opt.apply_gradients((g, v) for g, v in zip(grads, actor.trainable_variables) if g is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "train", "evaluate", "gradient"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--window", type=int, default=84)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=8.6e-4)
    parser.add_argument("--initial-std", type=float, default=0.35)
    parser.add_argument(
        "--method", choices=["reinforce", "analytic", "normalized", "critic", "ppo"],
        default="reinforce",
        help=(
            "Training estimator: plain masked REINFORCE; model-based analytic control "
            "variate; advantage-standardized REINFORCE; a learned pre-action value baseline; "
            "or clipped PPO with that value baseline."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="Global-norm clipping for normalized/critic methods; <=0 disables it.")
    parser.add_argument("--value-learning-rate", type=float, default=3e-3)
    parser.add_argument("--critic-epochs", type=int, default=3)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--normalize-advantages", dest="normalize_advantages", action="store_true", default=True)
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages", action="store_false")
    parser.add_argument("--budget", default="corridor-oracle")
    parser.add_argument("--output")
    parser.add_argument(
        "--active-only", dest="active_only", action="store_true", default=True,
        help="Mask forced allocation decisions out of the REINFORCE loss (default).",
    )
    parser.add_argument(
        "--all-actions", dest="active_only", action="store_false",
        help="Ablation: include forced/inactive allocation decisions in the REINFORCE loss.",
    )
    parser.add_argument("--legacy-features", action="store_true",
                        help="Reproduce the original state-ID/observation-ID indexing bug.")
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()