# Corridor shield diagnosis

This experiment uses the real `test-corridor` model, PAYNT loader, vectorized
simulator, `RiskBudgetTrainingEnv`, actor, REINFORCE update, and deployment adapter.
It does not use the earlier transcript's independently implemented simulator.

## An attainable target

The fixed, stationary policy is:

* At `(0,0)`, choose up/right with probabilities `1/2, 1/2`.
* At `(0,1)`, choose up.
* Everywhere else, choose collect/stay (`c`).

Thus `(1,0)` is a safe branch. At `(0,1)`, up reaches goal with probability
`.7`, bad with `.1`, the start with `.1`, and itself with `.1`.
Writing `v` for its eventual bad probability at `(0,1)`,
`v = .1 + .1*v + .1*(v/2)`, so `v=2/17` and the initial risk is **1/17**.
Use **nu=1/16**, slightly above this risk.

The constructed budget sends all initial slack to the up branch. At `(0,1)`
it splits slack equally between `(0,0)` and `(0,1)` and gives zero to goal/bad.
On the first visit to `(0,1)`, remaining risk is `1/8`, Qmin is `.1`, and
slack is `.025`; both retry outcomes receive risk `.025*.5/.1 = .125`.
A retry through the start can only increase the risk available at `(0,1)`.
Capacity projection at larger budgets does not underfund these branches.
The fixed policy is therefore allowed on every reachable history: **J=0**.

There is also a strictly positive, finite-logit witness: send `.99/.01` to the
initial risky/safe branches and approximately `.35/.65` to the two retry
branches, with `1e-8` raw weight on goal/bad (removed by wasteless projection).
The diagnostic checks both witnesses in the actual training environment.

Observed control results (seed 7, 64 lanes, horizon 20): uniform budget cost
`8.75954`, constructed budget `0`, finite-logit witness `0`.
The constructed budget also produced zero cost through `shielding.py`'s
evaluation entry point; its observed bad rate was `.0569`.
Zero cost for the constructed budget has the analytic argument above; zero
cost for a learned network below means **zero in the sampled evaluation**.

## What the experiments establish

The original trainer is **not incapable of learning this corridor**. It learns
slowly with the defaults. With seed 7, 32 lanes, 84 steps/update, gamma `.99`,
and horizon 20, the original actor's deterministic evaluation cost went from
`6.39814` to `2.81172` in 100 updates.

One isolated change, learning rate `8.6e-4 -> 8.6e-3`, gave deterministic cost
`1.34528` after 20 updates and `0` after 40. At 60 updates both sampled and
deterministic evaluations reported `0`. This run retains the original feature
indexing and uses the original REINFORCE loss and architecture.

There is a concrete reason the action parameterization is harder than the
five-choice bandit in `train_shield_rl.py`. Let `b` be the initial share sent to
the risky branch. Its successor risk is `.125*b`:

* `b < .8` forces an immediate intervention at the slippery cell. L1 correction
  uses up the available slack, so later allocations cannot recover it.
* Following the entire fixed policy without intervention requires at least
  `.125*b >= 2/17`, or **b >= 16/17 = .941176**.
* Near-zero independent Gaussian logits with std `.35` sample that share with
  probability about **1.06e-8**. At std `1.5` it is about `.0956`.
* The standalone bandit explicitly samples boundary allocations with substantial
  probability. It also uses TV distance, whereas the repository uses L1 (a
  factor of two, not an explanation for failed learning).

The reward is not entirely flat below the threshold: the original network
does improve, and the faster optimizer crosses the threshold. The tail
calculation alone is **not proof that insufficient exploration is the sole
cause**. See `results.json` for the separate exploration ablation and second seed.
No claim here establishes the cause of an unspecified large-model/PPO run.

Other checks and ablations:

* The model bounds, budget update and wasteless projection admit the known
  zero-cost policy in both training and evaluation.
* Finite gradients reach every actor layer, and Adam changes every layer.
  Supervised fitting can move the initial logit difference to `log(99)`.
* A memoryless allocation suffices for this control, so missing budget/history
  inputs do not prevent a solution here.
* Masking decisions with no allocation effect (initially only about 6% have an
  effect) improved 100-update cost to `1.51122`. This is a useful variance
  ablation, not a demonstrated complete explanation. It is only in the diagnostic.
* Correcting the feature lookup alone gave cost `2.59678` at 100 updates. It
  does not explain the learning difficulty of this small control.
* The actual loaded goal is a sink with Vmax=0. Storm's builder receives the
  reach-goal specification. Looking only at outgoing commands in `sketch.templ`
  misses this. The transcript's contrary premise does not hold for this checkout.
* `shielding.py` evaluates budget networks; `--agent-training` trains the fixed
  agent, not the budget network.

## Confirmed indexing bug and fix

`EnvironmentWrapperVec.observation_valuations` is indexed by **observation ID**.
The budget environment and neural inference adapter indexed it by **state ID**.
Full observability only implies a bijection; it does not imply equal IDs.
In this corridor, state 0 has observation 7 and state 1 has observation 13.
Consequently the actor received another state's coordinates for both the
current state and its successors.

The patch uses `model.get_observation(state)` consistently in training and in
both full-history and cached inference. The regression test uses an explicit
permutation and verifies all three paths. Existing neural checkpoints trained
with the old indexing have changed feature semantics and should be retrained;
the diagnostic's `--legacy-features` flag reproduces their original inputs.

## Reproduction

From the repository root:

```sh
export MPLCONFIGDIR=/tmp/rl-synthesis-mpl
./.venv/bin/python -m diagnostics.corridor_shield check
./.venv/bin/python -m diagnostics.corridor_shield gradient --legacy-features --lanes 32
./.venv/bin/python -m diagnostics.corridor_shield train --legacy-features --lanes 32 --iterations 100 --output /tmp/corridor-default
./.venv/bin/python -m diagnostics.corridor_shield train --legacy-features --lanes 32 --iterations 100 --learning-rate .0086 --output /tmp/corridor-fast
./.venv/bin/python -m diagnostics.corridor_shield evaluate --budget corridor-oracle
./.venv/bin/python -m diagnostics.corridor_shield evaluate --legacy-features --budget nn-reinforce --output /tmp/corridor-fast --lanes 16
./.venv/bin/python -m unittest discover -s tests/shielding -v
```

`evaluate` invokes the actual `shielding.py` Click entry point. In this process
only, it substitutes the fixed policy for the uniform-policy factory, registers
the constructed budget, and forwards the simulator seed (the entry point
currently passes `seed=None` to `init_args` even when `--seed` is supplied).
Its existing "uniform random policy" log label therefore refers to the substituted
control policy. This avoids adding benchmark-specific flags to production.

Training reports the negated discounted environment return, whose intervention
reward is shifted one decision forward relative to the paper's indexing.
Use these numbers for within-training comparisons; the deployment metric uses
the shield's own episode accounting. They coincide at zero. Evaluations use
fresh sampled episodes, so small differences in nonzero costs are noisy.
The first baseline/active-mask runs predated one extra diagnostic reset added
for logging initial logits; their random streams are not precisely paired with
the later runs. All settings and intermediate results are retained in `results.json`.
