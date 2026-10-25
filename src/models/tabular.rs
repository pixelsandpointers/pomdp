use std::fmt;

use crate::distributions::{DiscreteDistribution, SparseCat};
use crate::model::{FiniteMDP, FinitePOMDP, MDP, POMDP};

const PROB_TOL: f64 = 1e-6;

/// Returned when tabular model data is malformed.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ModelError> {
    Err(ModelError(msg.into()))
}

fn check_distribution(p: &[f64], what: &str) -> Result<(), ModelError> {
    if p.iter().any(|&x| !(0.0..=1.0 + PROB_TOL).contains(&x)) {
        return err(format!("{what} has a probability outside [0, 1]"));
    }
    let total: f64 = p.iter().sum();
    if (total - 1.0).abs() > PROB_TOL {
        return err(format!("{what} sums to {total}, not 1"));
    }
    Ok(())
}

fn sparse(row: &[f64]) -> SparseCat<usize> {
    let (vals, probs) = row.iter().enumerate().filter(|&(_, &p)| p > 0.0).unzip();
    SparseCat::new(vals, probs)
}

/// An MDP stored as dense arrays, with states and actions identified by their indices
/// (similar to `SparseTabularMDP` from POMDPTools.jl).
#[derive(Clone, Debug)]
pub struct TabularMDP {
    n_states: usize,
    n_actions: usize,
    /// `t[(s * n_actions + a) * n_states + sp]`
    t: Vec<f64>,
    /// `r[s * n_actions + a]`
    r: Vec<f64>,
    discount: f64,
    initial: Vec<f64>,
    terminal: Vec<bool>,
}

impl TabularMDP {
    /// Builds a model from `transition[s][a][sp]`, `reward[s][a]` and the initial state
    /// distribution.
    pub fn new(
        transition: Vec<Vec<Vec<f64>>>,
        reward: Vec<Vec<f64>>,
        discount: f64,
        initial_state: Vec<f64>,
    ) -> Result<Self, ModelError> {
        let n_states = transition.len();
        if n_states == 0 {
            return err("the model needs at least one state");
        }
        let n_actions = transition[0].len();
        if n_actions == 0 {
            return err("the model needs at least one action");
        }
        if !(0.0..=1.0).contains(&discount) {
            return err(format!("discount {discount} is outside [0, 1]"));
        }
        if reward.len() != n_states || reward.iter().any(|row| row.len() != n_actions) {
            return err(format!("reward must have shape ({n_states}, {n_actions})"));
        }
        if initial_state.len() != n_states {
            return err(format!(
                "initial state distribution must have length {n_states}"
            ));
        }
        check_distribution(&initial_state, "initial state distribution")?;

        let mut t = Vec::with_capacity(n_states * n_actions * n_states);
        for (s, rows) in transition.iter().enumerate() {
            if rows.len() != n_actions {
                return err(format!(
                    "transition must have shape ({n_states}, {n_actions}, {n_states})"
                ));
            }
            for (a, row) in rows.iter().enumerate() {
                if row.len() != n_states {
                    return err(format!(
                        "transition must have shape ({n_states}, {n_actions}, {n_states})"
                    ));
                }
                check_distribution(row, &format!("transition[{s}][{a}]"))?;
                t.extend_from_slice(row);
            }
        }

        Ok(Self {
            n_states,
            n_actions,
            t,
            r: reward.concat(),
            discount,
            initial: initial_state,
            terminal: vec![false; n_states],
        })
    }

    /// Marks states as terminal; `terminal[s]` must be given for every state.
    pub fn with_terminals(mut self, terminal: Vec<bool>) -> Result<Self, ModelError> {
        if terminal.len() != self.n_states {
            return err(format!("terminal must have length {}", self.n_states));
        }
        self.terminal = terminal;
        Ok(self)
    }

    /// Tabulates any finite MDP. Rewards that depend on the next state are replaced by their
    /// expectation over the transition, which leaves values and policies unchanged.
    pub fn from_mdp<M: FiniteMDP>(m: &M) -> Self {
        let states = m.states();
        let actions = m.actions();
        let (ns, na) = (states.len(), actions.len());
        let mut t = vec![0.0; ns * na * ns];
        let mut r = vec![0.0; ns * na];
        for (si, s) in states.iter().enumerate() {
            for (ai, a) in actions.iter().enumerate() {
                for (sp, p) in m.transition(s, a).support() {
                    t[(si * na + ai) * ns + m.state_index(&sp)] += p;
                    r[si * na + ai] += p * m.reward(s, a, &sp);
                }
            }
        }
        let mut initial = vec![0.0; ns];
        for (s, p) in m.initial_state().support() {
            initial[m.state_index(&s)] += p;
        }
        Self {
            n_states: ns,
            n_actions: na,
            t,
            r,
            discount: m.discount(),
            initial,
            terminal: states.iter().map(|s| m.is_terminal(s)).collect(),
        }
    }

    pub fn n_states(&self) -> usize {
        self.n_states
    }

    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    fn transition_row(&self, s: usize, a: usize) -> &[f64] {
        let start = (s * self.n_actions + a) * self.n_states;
        &self.t[start..start + self.n_states]
    }
}

impl MDP for TabularMDP {
    type State = usize;
    type Action = usize;
    type TransitionDist = SparseCat<usize>;
    type InitialStateDist = SparseCat<usize>;

    fn transition(&self, s: &usize, a: &usize) -> SparseCat<usize> {
        sparse(self.transition_row(*s, *a))
    }

    fn reward(&self, s: &usize, a: &usize, _sp: &usize) -> f64 {
        self.r[s * self.n_actions + a]
    }

    fn discount(&self) -> f64 {
        self.discount
    }

    fn initial_state(&self) -> SparseCat<usize> {
        sparse(&self.initial)
    }

    fn is_terminal(&self, s: &usize) -> bool {
        self.terminal[*s]
    }
}

impl FiniteMDP for TabularMDP {
    fn states(&self) -> Vec<usize> {
        (0..self.n_states).collect()
    }

    fn actions(&self) -> Vec<usize> {
        (0..self.n_actions).collect()
    }

    fn state_index(&self, s: &usize) -> usize {
        *s
    }

    fn action_index(&self, a: &usize) -> usize {
        *a
    }
}

/// A POMDP stored as dense arrays: a [`TabularMDP`] plus `observation[a][sp][o]`.
#[derive(Clone, Debug)]
pub struct TabularPOMDP {
    mdp: TabularMDP,
    n_observations: usize,
    /// `o[(a * n_states + sp) * n_observations + o]`
    o: Vec<f64>,
}

impl TabularPOMDP {
    pub fn new(mdp: TabularMDP, observation: Vec<Vec<Vec<f64>>>) -> Result<Self, ModelError> {
        let (ns, na) = (mdp.n_states, mdp.n_actions);
        let n_observations = observation
            .first()
            .and_then(|rows| rows.first())
            .map_or(0, Vec::len);
        if n_observations == 0 {
            return err("the model needs at least one observation");
        }
        let shape_err = || {
            err(format!(
                "observation must have shape ({na}, {ns}, {n_observations})"
            ))
        };
        if observation.len() != na {
            return shape_err();
        }
        let mut o = Vec::with_capacity(na * ns * n_observations);
        for (a, rows) in observation.iter().enumerate() {
            if rows.len() != ns {
                return shape_err();
            }
            for (sp, row) in rows.iter().enumerate() {
                if row.len() != n_observations {
                    return shape_err();
                }
                check_distribution(row, &format!("observation[{a}][{sp}]"))?;
                o.extend_from_slice(row);
            }
        }
        Ok(Self {
            mdp,
            n_observations,
            o,
        })
    }

    /// Tabulates any finite POMDP.
    pub fn from_pomdp<M: FinitePOMDP>(m: &M) -> Self {
        let mdp = TabularMDP::from_mdp(m);
        let states = m.states();
        let actions = m.actions();
        let n_observations = m.observations().len();
        let mut o = vec![0.0; actions.len() * states.len() * n_observations];
        for (ai, a) in actions.iter().enumerate() {
            for (spi, sp) in states.iter().enumerate() {
                for (obs, p) in m.observation(a, sp).support() {
                    o[(ai * states.len() + spi) * n_observations + m.observation_index(&obs)] += p;
                }
            }
        }
        Self {
            mdp,
            n_observations,
            o,
        }
    }

    pub fn mdp(&self) -> &TabularMDP {
        &self.mdp
    }

    pub fn n_states(&self) -> usize {
        self.mdp.n_states
    }

    pub fn n_actions(&self) -> usize {
        self.mdp.n_actions
    }

    pub fn n_observations(&self) -> usize {
        self.n_observations
    }
}

impl MDP for TabularPOMDP {
    type State = usize;
    type Action = usize;
    type TransitionDist = SparseCat<usize>;
    type InitialStateDist = SparseCat<usize>;

    fn transition(&self, s: &usize, a: &usize) -> SparseCat<usize> {
        self.mdp.transition(s, a)
    }

    fn reward(&self, s: &usize, a: &usize, sp: &usize) -> f64 {
        self.mdp.reward(s, a, sp)
    }

    fn discount(&self) -> f64 {
        self.mdp.discount
    }

    fn initial_state(&self) -> SparseCat<usize> {
        self.mdp.initial_state()
    }

    fn is_terminal(&self, s: &usize) -> bool {
        self.mdp.is_terminal(s)
    }
}

impl POMDP for TabularPOMDP {
    type Observation = usize;
    type ObservationDist = SparseCat<usize>;

    fn observation(&self, a: &usize, sp: &usize) -> SparseCat<usize> {
        let start = (a * self.mdp.n_states + sp) * self.n_observations;
        sparse(&self.o[start..start + self.n_observations])
    }
}

impl FiniteMDP for TabularPOMDP {
    fn states(&self) -> Vec<usize> {
        self.mdp.states()
    }

    fn actions(&self) -> Vec<usize> {
        self.mdp.actions()
    }

    fn state_index(&self, s: &usize) -> usize {
        *s
    }

    fn action_index(&self, a: &usize) -> usize {
        *a
    }
}

impl FinitePOMDP for TabularPOMDP {
    fn observations(&self) -> Vec<usize> {
        (0..self.n_observations).collect()
    }

    fn observation_index(&self, o: &usize) -> usize {
        *o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::DiscreteBelief;
    use crate::models::TigerPOMDP;
    use crate::policy::Policy;
    use crate::solvers::QMDPSolver;

    #[test]
    fn rejects_bad_rows() {
        let e = TabularMDP::new(
            vec![vec![vec![0.5, 0.4]], vec![vec![0.0, 1.0]]],
            vec![vec![0.0]; 2],
            0.9,
            vec![1.0, 0.0],
        )
        .unwrap_err();
        assert!(e.0.contains("transition[0][0]"), "{e}");
    }

    #[test]
    fn tabulated_tiger_matches_original() {
        let tiger = TigerPOMDP::default();
        let tab = TabularPOMDP::from_pomdp(&tiger);
        assert_eq!(
            (tab.n_states(), tab.n_actions(), tab.n_observations()),
            (2, 3, 2)
        );

        let a = QMDPSolver::default().solve(&tiger);
        let b = QMDPSolver::default().solve(&tab);
        let belief = DiscreteBelief::new(vec![0.7, 0.3]);
        assert!((a.value(&belief) - b.value(&belief)).abs() < 1e-9);
        assert_eq!(a.action(&belief) as usize, b.action(&belief));
    }
}
