"""Controlled corridor experiment using the repository's simulator and trainers.

Run from the repository root with MPLCONFIGDIR=/tmp/rl-synthesis-mpl:
  .venv/bin/python -m diagnostics.corridor_shield check
  .venv/bin/python -m diagnostics.corridor_shield train --iterations 100
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


def install_active_recorder(env, info, active_history):
    """Record whether each sampled allocation can affect successor budgets."""
    original_step = env._step

    def recording_step(action):
        active_history.append(allocation_active_mask(env, info))
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


def train(args):
    environment, info, env, tf_env = setup(args)
    actor, optimizer = build_actor(env, tf_env, learning_rate=args.learning_rate)
    actor._std_bias.assign(np.log(np.expm1(args.initial_std)))
    active_history = []
    install_active_recorder(env, info, active_history)
    started = time.monotonic()
    records = []
    loss_history = []
    for iteration in range(args.iterations + 1):
        if iteration % args.report_every == 0 or iteration == args.iterations:
            deterministic = evaluate_env(env, tf_env, actor=actor, episodes=4)
            stochastic = evaluate_env(env, tf_env, actor=actor, stochastic=True, episodes=4)
            record = dict(iteration=iteration, deterministic_cost=deterministic, stochastic_cost=stochastic,
                          std=float(tf.nn.softplus(actor._std_bias)), seconds=time.monotonic() - started)
            initial_ts = tf_env.reset()
            initial_dist, _ = actor(initial_ts.observation, initial_ts.step_type,
                                    actor.get_initial_state(env.num_envs))
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
        (obs, actions, step_types, rewards, discounts, next_steps, ages,
         initial_state, policy_state, age) = collect_window(actor, tf_env, policy_state, args.window, age)
        returns, valid = compute_returns_and_mask(rewards, discounts)
        active = np.asarray(active_history).T
        if args.active_only:
            # Keep all rewards in the returns, but do not attach those returns to
            # allocations whose wasteless refinement is forced.
            valid &= active
        if not np.any(valid):
            raise AssertionError("No valid active allocation decisions in training window")
        loss, baseline, _ = train_step(actor, optimizer, obs, step_types, actions, returns, valid,
                                      initial_state, ages, env.gamma)
        loss_value = float(loss)
        loss_history.append((iteration, loss_value))
        if iteration % args.report_every == 0:
            print("BATCH", iteration, "active_fraction", float(active.mean()), "loss", loss, flush=True)
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

        loss_line, = loss_ax.plot(loss_iterations, losses, label="Loss")
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
        loss_ax.set_ylabel("Loss")
        metric_ax.set_ylabel("Cost / std")
        loss_ax.set_title("Training metrics")
        loss_ax.grid(True, alpha=0.3)

        lines = [loss_line, deterministic_line, stochastic_line, std_line]
        loss_ax.legend(lines, [line.get_label() for line in lines], loc="best")

        fig.tight_layout()
        metrics_plot = plot_dir / "training_metrics.pdf"
        fig.savefig(metrics_plot, bbox_inches="tight")
        plt.close(fig)
        print("TRAINING_METRICS_PLOT", metrics_plot, flush=True)

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        save_actor(actor, args.output)
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