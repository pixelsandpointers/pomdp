use crate::distributions::DiscreteDistribution;
use crate::model::FiniteMDP;
use crate::policy::Policy;

/// Tabular value iteration (DiscreteValueIteration.jl).
#[derive(Clone, Debug)]
pub struct ValueIterationSolver {
    pub max_iterations: usize,
    /// Stop once the Bellman residual (max change in value) falls below this.
    pub tolerance: f64,
}

impl Default for ValueIterationSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl ValueIterationSolver {
    pub fn solve<M: FiniteMDP>(&self, m: M) -> ValueIterationPolicy<M> {
        let sol = q_value_iteration(&m, self.max_iterations, self.tolerance);
        ValueIterationPolicy {
            actions: m.actions(),
            model: m,
            q: sol.q,
            iterations: sol.iterations,
            residual: sol.residual,
        }
    }
}

/// Greedy policy with respect to the Q-values found by [`ValueIterationSolver`].
#[derive(Clone, Debug)]
pub struct ValueIterationPolicy<M: FiniteMDP> {
    model: M,
    actions: Vec<M::Action>,
    /// `q[s][a]`, indexed by state and action index.
    pub q: Vec<Vec<f64>>,
    pub iterations: usize,
    pub residual: f64,
}

impl<M: FiniteMDP> ValueIterationPolicy<M> {
    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn q_values(&self, s: &M::State) -> &[f64] {
        &self.q[self.model.state_index(s)]
    }

    pub fn value(&self, s: &M::State) -> f64 {
        self.q_values(s)
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl<M: FiniteMDP> Policy<M::State> for ValueIterationPolicy<M> {
    type Action = M::Action;

    fn action(&self, s: &M::State) -> M::Action {
        self.actions[argmax(self.q_values(s))].clone()
    }
}

pub(crate) struct QSolution {
    pub q: Vec<Vec<f64>>,
    pub iterations: usize,
    pub residual: f64,
}

/// Runs value iteration and returns `Q[s][a]`; terminal states get all-zero rows.
pub(crate) fn q_value_iteration<M: FiniteMDP>(m: &M, max_iterations: usize, tol: f64) -> QSolution {
    let states = m.states();
    let actions = m.actions();
    let gamma = m.discount();

    // trans[s][a] = [(sp, probability, reward)], evaluated once up front.
    let terminal: Vec<bool> = states.iter().map(|s| m.is_terminal(s)).collect();
    let trans: Vec<Vec<Vec<(usize, f64, f64)>>> = states
        .iter()
        .zip(&terminal)
        .map(|(s, &term)| {
            actions
                .iter()
                .map(|a| {
                    if term {
                        return Vec::new();
                    }
                    m.transition(s, a)
                        .support()
                        .into_iter()
                        .map(|(sp, p)| (m.state_index(&sp), p, m.reward(s, a, &sp)))
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut v = vec![0.0; states.len()];
    let mut q = vec![vec![0.0; actions.len()]; states.len()];
    let mut residual = f64::INFINITY;
    let mut iterations = 0;
    while iterations < max_iterations && residual >= tol {
        iterations += 1;
        residual = 0.0;
        let mut v_new = vec![0.0; states.len()];
        for si in 0..states.len() {
            if terminal[si] {
                continue;
            }
            for (ai, outcomes) in trans[si].iter().enumerate() {
                q[si][ai] = outcomes
                    .iter()
                    .map(|&(spi, p, r)| p * (r + gamma * v[spi]))
                    .sum();
            }
            v_new[si] = q[si].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            residual = f64::max(residual, (v_new[si] - v[si]).abs());
        }
        v = v_new;
    }
    QSolution {
        q,
        iterations,
        residual,
    }
}

/// Index of the first maximum.
pub(crate) fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TabularMDP;

    /// Two states; action 1 moves 0 -> 1, state 1 is absorbing with reward 1 per step.
    fn chain() -> TabularMDP {
        TabularMDP::new(
            vec![
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            ],
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            0.9,
            vec![1.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn chain_values() {
        let policy = ValueIterationSolver {
            max_iterations: 10_000,
            tolerance: 1e-10,
        }
        .solve(chain());
        // V(1) = 1 / (1 - γ) = 10, V(0) = γ V(1) = 9.
        assert!((policy.value(&1) - 10.0).abs() < 1e-6);
        assert!((policy.value(&0) - 9.0).abs() < 1e-6);
        assert_eq!(policy.action(&0), 1);
    }

    #[test]
    fn terminal_states_have_zero_value() {
        let m = chain().with_terminals(vec![false, true]).unwrap();
        let policy = ValueIterationSolver::default().solve(&m);
        assert_eq!(policy.value(&1), 0.0);
        assert_eq!(policy.value(&0), 0.0);
    }
}
