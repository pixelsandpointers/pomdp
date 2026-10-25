"""Solve the tiger problem from Python.

Build the extension first, e.g. `maturin develop --release`, then run `python examples/tiger.py`.
"""

import pomdp

LISTEN, OPEN_LEFT, OPEN_RIGHT = range(3)
HEAR_LEFT, HEAR_RIGHT = range(2)

m = pomdp.tiger()
policy = pomdp.solve_qmdp(m)
updater = pomdp.DiscreteUpdater(m)

b = updater.initial_belief()
for _ in range(3):
    print(f"belief {b} -> action {policy.action(b)} (value {policy.value(b):.2f})")
    b = updater.update(b, LISTEN, HEAR_LEFT)
print(f"belief {b} -> action {policy.action(b)}")

returns = [pomdp.simulate(m, policy, max_steps=100, seed=i) for i in range(1000)]
print(f"mean discounted return: {sum(returns) / len(returns):.2f}")
