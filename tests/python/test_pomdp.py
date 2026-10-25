"""Tests for the Python bindings. Run with `pytest tests/python` after `maturin develop`."""

import pytest

import pomdp


def chain():
    # Action 1 moves state 0 to the absorbing state 1, which pays 1 per step.
    return pomdp.TabularMDP(
        transition=[[[1, 0], [0, 1]], [[0, 1], [0, 1]]],
        reward=[[0, 0], [1, 1]],
        discount=0.9,
        initial_state=[1, 0],
    )


def test_value_iteration():
    policy = pomdp.solve_value_iteration(chain(), tolerance=1e-10, max_iterations=10_000)
    assert policy.value(1) == pytest.approx(10.0)
    assert policy.value(0) == pytest.approx(9.0)
    assert policy.action(0) == 1


def test_mdp_simulation_is_seeded():
    m = chain()
    policy = pomdp.solve_value_iteration(m)
    ret = pomdp.simulate(m, policy, max_steps=10, seed=0)
    assert ret == pomdp.simulate(m, policy, max_steps=10, seed=0)
    assert ret == pytest.approx(sum(0.9**t for t in range(1, 10)))


def test_tiger_qmdp():
    m = pomdp.tiger()
    assert (m.n_states, m.n_actions, m.n_observations) == (2, 3, 2)
    policy = pomdp.solve_qmdp(m)
    assert policy.action([0.5, 0.5]) == 0
    assert policy.action([0.99, 0.01]) == 2
    assert policy.action([0.01, 0.99]) == 1


def test_belief_update():
    up = pomdp.DiscreteUpdater(pomdp.tiger())
    b = up.update(up.initial_belief(), 0, 0)
    assert b == pytest.approx([0.85, 0.15])


def test_validation_errors():
    with pytest.raises(ValueError, match="transition"):
        pomdp.TabularMDP([[[0.5, 0.4]], [[0, 1]]], [[0], [0]], 0.9)
    policy = pomdp.solve_qmdp(pomdp.tiger())
    with pytest.raises(ValueError, match="length"):
        policy.action([1.0])
