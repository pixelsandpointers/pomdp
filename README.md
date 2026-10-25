# pomdp

A Rust port of Julia's [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl): an interface and
toolkit for defining and solving Markov decision processes (MDPs) and partially observable MDPs
(POMDPs), with Python bindings.

> **Status:** early MVP. The core interface, discrete beliefs, a rollout simulator and two
> offline solvers are in place; the API will change.

## Overview

Like POMDPs.jl, the crate separates *problem definitions* from *solution methods*. A problem
implements a few traits; solvers, belief updaters and simulators are written against those traits
only, so any model works with any compatible tool.

| POMDPs.jl                                          | pomdp                                              |
|----------------------------------------------------|----------------------------------------------------|
| `transition`, `reward`, `discount`, `initialstate`, `isterminal` | `MDP` trait                          |
| `observation`                                      | `POMDP` trait                                      |
| `states`/`stateindex`, `actions`/`actionindex`     | `FiniteMDP` trait                                  |
| `observations`/`obsindex`                          | `FinitePOMDP` trait                                |
| `Deterministic`, `SparseCat`, `Uniform`            | `distributions` module                             |
| `action(policy, x)`                                | `Policy` trait                                     |
| `update(updater, b, a, o)`, `DiscreteUpdater`      | `Updater` trait, `DiscreteUpdater`                 |
| `RolloutSimulator` (POMDPTools.jl)                 | `RolloutSimulator`                                 |
| DiscreteValueIteration.jl                          | `ValueIterationSolver`                             |
| QMDP.jl                                            | `QMDPSolver` → `AlphaVectorPolicy`                 |
| `TigerPOMDP` (POMDPModels.jl)                      | `models::TigerPOMDP`                               |
| `SparseTabularMDP`/`SparseTabularPOMDP`            | `models::TabularMDP`/`TabularPOMDP`                |

## Rust

```sh
cargo add pomdp
```

### Solving a problem

```rust
use pomdp::models::{TigerAction, TigerPOMDP};
use pomdp::prelude::*;
use rand::{SeedableRng, rngs::StdRng};

let tiger = TigerPOMDP::default();
let policy = QMDPSolver::default().solve(&tiger);
let updater = DiscreteUpdater::new(&tiger);

let b0 = updater.initial_belief();
assert_eq!(policy.action(&b0), TigerAction::Listen);

let mut rng = StdRng::seed_from_u64(42);
let ret = RolloutSimulator::new(100).simulate_pomdp(&tiger, &policy, &updater, b0, &mut rng);
println!("discounted return: {ret:.2}");
```

Solvers, policies and updaters take the model by value. The traits are also implemented for
`&M`, `Box<M>`, `Rc<M>` and `Arc<M>`, so pass `&model` to borrow it or an `Arc` to share it.

### Defining a problem

Implement `MDP` (and `POMDP` for partial observability). Tabular solvers additionally need
`FiniteMDP`/`FinitePOMDP`, which enumerate the spaces and require discrete distributions.

```rust
use pomdp::prelude::*;

#[derive(Clone)]
struct Chain;

impl MDP for Chain {
    type State = usize;
    type Action = bool; // true = move right
    type TransitionDist = Deterministic<usize>;
    type InitialStateDist = Deterministic<usize>;

    fn transition(&self, s: &usize, a: &bool) -> Deterministic<usize> {
        Deterministic(if *a { (s + 1).min(9) } else { *s })
    }
    fn reward(&self, _s: &usize, _a: &bool, sp: &usize) -> f64 {
        if *sp == 9 { 1.0 } else { 0.0 }
    }
    fn discount(&self) -> f64 {
        0.95
    }
    fn initial_state(&self) -> Deterministic<usize> {
        Deterministic(0)
    }
    fn is_terminal(&self, s: &usize) -> bool {
        *s == 9
    }
}

impl FiniteMDP for Chain {
    fn states(&self) -> Vec<usize> { (0..10).collect() }
    fn actions(&self) -> Vec<bool> { vec![false, true] }
    fn state_index(&self, s: &usize) -> usize { *s }
    fn action_index(&self, a: &bool) -> usize { *a as usize }
}

let policy = ValueIterationSolver::default().solve(Chain);
assert!(policy.action(&0));
```

Any finite model can be converted to dense arrays with `TabularMDP::from_mdp` or
`TabularPOMDP::from_pomdp`.

## Python

The bindings are built with [maturin](https://www.maturin.rs/):

```sh
pip install maturin
maturin develop --release
```

Python models are tabular: states, actions and observations are integers, and the model is given
as nested lists (use `.tolist()` for NumPy arrays).

| argument          | shape / meaning                                     |
|-------------------|-----------------------------------------------------|
| `transition`      | `[s][a][sp]` → P(sp \| s, a)                        |
| `reward`          | `[s][a]` → R(s, a)                                  |
| `observation`     | `[a][sp][o]` → P(o \| a, sp)                        |
| `initial_state`   | `[s]`, defaults to uniform                          |
| `terminal`        | `[s]` booleans, defaults to all `False`             |

```python
import pomdp

m = pomdp.tiger()  # or pomdp.TabularPOMDP(transition, reward, observation, discount)
policy = pomdp.solve_qmdp(m)
updater = pomdp.DiscreteUpdater(m)

b = updater.initial_belief()      # [0.5, 0.5]
a = policy.action(b)              # 0 = listen
b = updater.update(b, a, 0)       # heard the tiger on the left: [0.85, 0.15]

ret = pomdp.simulate(m, policy, max_steps=100, seed=0)
```

For fully observable problems use `pomdp.TabularMDP(transition, reward, discount)` with
`pomdp.solve_value_iteration`. Malformed input raises `ValueError`.

## Examples and tests

```sh
cargo run --example tiger
cargo test

maturin develop && python examples/tiger.py
pytest tests/python
```

## Roadmap

- Generative interface (`gen`) for models that can only be sampled
- Online solvers: MCTS, POMCP
- Point-based offline solvers (PBVI, SARSOP-style)
- Particle filters
- Simulation histories
- Python models defined through callbacks, and type stubs

## License

Licensed under either of MIT or Apache-2.0, at your option.
