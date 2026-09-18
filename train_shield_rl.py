#!/usr/bin/env python3
"""Train a risk-allocation shield with REINFORCE for two fixed agent behaviors.

Run: python3 train_shield.py       (Python standard library only)

The MDP is the example from the supplied draft. Reaching 'bad' is failure;
'good' and 'bad' are absorbing, so an episode stops on reaching either.
Q_min/Q_max are hardcoded reachability probabilities, not reward Q-values.

We use nu = 1/2 and total variation distance D. Both unshielded agents have
failure probability 1/2, but need different allocations to avoid intervention:
  A: alpha at s1, delta at s2.
  B: beta  at s1, gamma at s2.

RL formulation (the agent behavior and reachability Q-values remain fixed):
  State: (s, r, d_A), sufficient here because the agents are memoryless.
  Action: a risk allocation b(h, d), after projecting d_A to the safe choice d.
  Transition: apply MakeWasteless, sample (action, s'), and update the budget r.
  Reward: -D(d_A, d). Thus the expected return is -J_gamma.

In this MDP, only s0 has a nontrivial allocation decision. At s1, s2 is the
only successor with V_max > V_min, so all positive slack must go there.
The RL problem therefore reduces to a bandit with a full-episode return.

The five RL actions choose b(s0, epsilon) = (fraction, 1 - fraction) on
successors (s1, s2), for fraction in FRACTIONS. A categorical policy learns
their probabilities using softmax logits and REINFORCE with a running-mean
baseline. Each update uses one sampled allocation and one sampled episode.
The learner neither differentiates through the shield nor uses exact costs.

We report both the learned stochastic policy and greedy deployment (always
choose its most probable allocation). Every allocation is safe, including
during exploration: safety comes from projection and the budget updates,
not a penalty for visiting 'bad'. Exact evaluation is only used for reporting.

REINFORCE reference: Williams (1992), https://doi.org/10.1007/BF00992696

Capacity edge case: with positive slack, V_max(s') = V_min(s') gives capacity
zero. The support-indicator fallback is used only when slack is zero. This
corrects the draft's 'defined and positive' condition, which would otherwise
allow positive slack to be wasted on terminal states.
"""

import math
import random


nu = 0.5
gamma = 1.0  # Discount for intervention cost; episodes have at most three steps.
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)  # Fraction of initial slack sent to s1.

P = {
    "s0": {"epsilon": {"s1": 0.5, "s2": 0.5}},
    "s1": {
        "alpha": {"good": 1.0},
        "beta": {"s2": 0.5, "good": 0.25, "bad": 0.25},
    },
    "s2": {
        "gamma": {"good": 0.5, "bad": 0.5},
        "delta": {"bad": 1.0},
    },
}

V_min = {"s0": 0.25, "s1": 0.0, "s2": 0.5, "good": 0.0, "bad": 1.0}
V_max = {"s0": 0.875, "s1": 0.75, "s2": 1.0, "good": 0.0, "bad": 1.0}
Q_min = {
    "s0": {"epsilon": 0.25},
    "s1": {"alpha": 0.0, "beta": 0.5},
    "s2": {"gamma": 0.5, "delta": 1.0},
}
Q_max = {
    "s0": {"epsilon": 0.875},
    "s1": {"alpha": 0.0, "beta": 0.75},
    "s2": {"gamma": 0.5, "delta": 1.0},
}

AGENTS = {
    "A: alpha / delta": {
        "s0": {"epsilon": 1.0},
        "s1": {"alpha": 1.0, "beta": 0.0},
        "s2": {"gamma": 0.0, "delta": 1.0},
    },
    "B: beta / gamma": {
        "s0": {"epsilon": 1.0},
        "s1": {"alpha": 0.0, "beta": 1.0},
        "s2": {"gamma": 1.0, "delta": 0.0},
    },
}


def D(d_A, d):
    """Total variation distance between two action distributions."""
    return 0.5 * sum(abs(d_A[action] - d[action]) for action in d_A)


def shield(s, d_A, r):
    """Closest d to d_A subject to Q_min(s, d) <= r.

    With two actions, the feasible distributions form an interval, so the
    total variation projection simply clips the probability of the riskier
    action. Actions in P are ordered by increasing Q_min.
    """
    if len(P[s]) == 1:
        return d_A.copy()
    safer, riskier = P[s]
    p_max = (r - Q_min[s][safer]) / (Q_min[s][riskier] - Q_min[s][safer])
    p = min(d_A[riskier], max(0.0, min(1.0, p_max)))
    return {safer: 1.0 - p, riskier: p}


def make_wasteless(b, capacity):
    """Clip to capacity, then redistribute the missing mass (draft algorithm)."""
    b = {outcome: min(b[outcome], capacity[outcome]) for outcome in b}
    for outcome in b:
        missing = max(0.0, 1.0 - sum(b.values()))
        b[outcome] += min(missing, capacity[outcome] - b[outcome])
    assert abs(sum(b.values()) - 1.0) < 1e-12
    return b


def shield_step(s, d_A, r, fraction):
    """Return the shielded distribution and (probability, state, budget) outcomes.

    The raw allocation is memoryless; r carries the history-dependent budget.
    All allocation calculations use the shielded choice d.
    """
    d = shield(s, d_A, r)
    probability = {
        (action, s_next): d[action] * p
        for action in d
        for s_next, p in P[s][action].items()
        if d[action] * p > 0.0
    }
    Q_min_d = sum(d[action] * Q_min[s][action] for action in d)
    Q_max_d = sum(d[action] * Q_max[s][action] for action in d)
    assert Q_min_d <= r + 1e-12
    slack = max(0.0, min(r, Q_max_d) - Q_min_d)

    b = probability.copy()  # Default allocation on supported outcomes.
    if s == "s0":
        b = {("epsilon", "s1"): fraction, ("epsilon", "s2"): 1.0 - fraction}
    capacity = {
        (action, s_next): (
            p * (V_max[s_next] - V_min[s_next]) / slack if slack > 0.0 else 1.0
        )
        for (action, s_next), p in probability.items()
    }
    b = make_wasteless(b, capacity)

    outcomes = []
    for (action, s_next), p in probability.items():
        # Only supported, feasible outcomes are visited, so the guard holds.
        r_next = V_min[s_next] + slack * b[action, s_next] / p
        assert V_min[s_next] - 1e-12 <= r_next <= V_max[s_next] + 1e-12
        outcomes.append((p, s_next, r_next))
    assert sum(p * r_next for p, _, r_next in outcomes) <= r + 1e-12
    return d, outcomes


def episode(agent, fraction, rng):
    """Sample one shielded episode; its RL return is minus intervention cost."""
    s, r = "s0", min(nu, V_max["s0"])
    episode_return, discount = 0.0, 1.0
    while s in P:
        d_A = agent[s]
        d, outcomes = shield_step(s, d_A, r, fraction)
        reward = -D(d_A, d)
        episode_return += discount * reward
        # Sampling the joint outcome is equivalent to sampling action then state.
        _, s, r = rng.choices(outcomes, weights=[p for p, _, _ in outcomes])[0]
        discount *= gamma
    return episode_return


def softmax(logits):
    """Convert learned logits to allocation probabilities."""
    weights = [math.exp(value - max(logits)) for value in logits]
    return [weight / sum(weights) for weight in weights]


def train(agent, seed=0, episodes=10000, learning_rate=0.1):
    """REINFORCE with a baseline; return the learned allocation probabilities."""
    rng = random.Random(seed)
    logits = [0.0] * len(FRACTIONS)  # Initially choose all allocations equally.
    baseline = 0.0
    for _ in range(episodes):
        probabilities = softmax(logits)
        chosen = rng.choices(range(len(FRACTIONS)), weights=probabilities)[0]
        episode_return = episode(agent, FRACTIONS[chosen], rng)
        advantage = episode_return - baseline

        # Gradient ascent on expected return:
        # d log P(chosen) / d logits[i] = [i == chosen] - probabilities[i].
        for i in range(len(logits)):
            gradient = float(i == chosen) - probabilities[i]
            logits[i] += learning_rate * advantage * gradient

        # Update AFTER the policy, so the baseline used above is independent
        # of the current episode's sampled allocation and return.
        baseline += 0.05 * (episode_return - baseline)
    return softmax(logits)


def evaluate(agent, fraction, s="s0", r=None):
    """Exact (J_gamma, failure probability), summing the finite trajectory tree."""
    if s not in P:
        return 0.0, float(s == "bad")
    if r is None:
        r = min(nu, V_max["s0"])
    d, outcomes = shield_step(s, agent[s], r, fraction)
    cost, risk = D(agent[s], d), 0.0
    for p, s_next, r_next in outcomes:
        next_cost, next_risk = evaluate(agent, fraction, s_next, r_next)
        cost += gamma * p * next_cost
        risk += p * next_risk
    return cost, risk


def evaluate_policy(agent, probabilities):
    """Exact evaluation when an allocation is sampled once per episode."""
    cost, risk = 0.0, 0.0
    for probability, fraction in zip(probabilities, FRACTIONS):
        allocation_cost, allocation_risk = evaluate(agent, fraction)
        cost += probability * allocation_cost
        risk += probability * allocation_risk
    return cost, risk


def main():
    assert V_min["s0"] <= nu
    print(f"Risk threshold nu = {nu}; intervention discount gamma = {gamma}\n")
    print(f"Allocation choices (fraction of slack sent to s1): {FRACTIONS}\n")
    trained = {}
    for name, agent in AGENTS.items():
        probabilities = train(agent)
        fraction = FRACTIONS[probabilities.index(max(probabilities))]
        trained[name] = fraction
        uniform = [1.0 / len(FRACTIONS)] * len(FRACTIONS)
        before_cost, _ = evaluate_policy(agent, uniform)
        learned_cost, learned_risk = evaluate_policy(agent, probabilities)
        greedy_cost, greedy_risk = evaluate(agent, fraction)
        assert max(learned_risk, greedy_risk) <= nu + 1e-12
        print(name)
        print("  Learned probabilities: " + ", ".join(f"{p:.3f}" for p in probabilities))
        print(f"  Stochastic J_gamma: {before_cost:.4f} -> {learned_cost:.4f}; "
              f"P(bad) = {learned_risk:.4f}")
        print(f"  Greedy fraction = {fraction:.2f}; J_gamma = {greedy_cost:.4f}; "
              f"P(bad) = {greedy_risk:.4f}\n")

    print("Each greedy shield evaluated on both behaviors:")
    print(f"{'Trained for':<19} {'Agent':<19} {'J_gamma':>8} {'P(bad)':>8}")
    for trained_for, fraction in trained.items():
        for name, agent in AGENTS.items():
            cost, risk = evaluate(agent, fraction)
            assert risk <= nu + 1e-12
            print(f"{trained_for:<19} {name:<19} {cost:8.3f} {risk:8.3f}")


if __name__ == "__main__":
    main()
