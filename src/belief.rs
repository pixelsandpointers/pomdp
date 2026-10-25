//! Beliefs and belief updaters.

use crate::distributions::{DiscreteDistribution, Distribution};
use crate::model::FinitePOMDP;

/// Updates a belief after an action and an observation (`update(up, b, a, o)` in POMDPs.jl).
pub trait Updater {
    type Belief;
    type Action;
    type Observation;

    fn update(&self, b: &Self::Belief, a: &Self::Action, o: &Self::Observation) -> Self::Belief;
}

/// A probability vector over the states of a [`FinitePOMDP`], indexed by
/// [`FiniteMDP::state_index`](crate::FiniteMDP::state_index).
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteBelief {
    pub probs: Vec<f64>,
}

impl DiscreteBelief {
    pub fn new(probs: Vec<f64>) -> Self {
        Self { probs }
    }

    pub fn uniform(n: usize) -> Self {
        Self {
            probs: vec![1.0 / n as f64; n],
        }
    }

    /// Probability of the state with index `i`.
    pub fn pdf(&self, i: usize) -> f64 {
        self.probs[i]
    }
}

/// Exact Bayesian filter for finite POMDPs.
#[derive(Clone, Debug)]
pub struct DiscreteUpdater<M> {
    model: M,
}

impl<M: FinitePOMDP> DiscreteUpdater<M> {
    pub fn new(model: M) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    /// The belief matching the model's initial state distribution.
    pub fn initial_belief(&self) -> DiscreteBelief {
        let mut probs = vec![0.0; self.model.states().len()];
        for (s, p) in self.model.initial_state().support() {
            probs[self.model.state_index(&s)] += p;
        }
        DiscreteBelief { probs }
    }

    /// Bayes update, or `None` if `o` has zero probability under `b` and `a`.
    pub fn try_update(
        &self,
        b: &DiscreteBelief,
        a: &M::Action,
        o: &M::Observation,
    ) -> Option<DiscreteBelief> {
        let m = &self.model;
        let states = m.states();
        assert_eq!(
            b.probs.len(),
            states.len(),
            "belief has the wrong number of states"
        );

        let mut bp = vec![0.0; states.len()];
        for (s, &ps) in states.iter().zip(&b.probs) {
            if ps == 0.0 || m.is_terminal(s) {
                continue;
            }
            for (sp, pt) in m.transition(s, a).support() {
                bp[m.state_index(&sp)] += ps * pt;
            }
        }
        for (sp, p) in states.iter().zip(bp.iter_mut()) {
            if *p > 0.0 {
                *p *= m.observation(a, sp).pdf(o);
            }
        }

        let total: f64 = bp.iter().sum();
        if total <= 0.0 {
            return None;
        }
        bp.iter_mut().for_each(|p| *p /= total);
        Some(DiscreteBelief { probs: bp })
    }
}

impl<M: FinitePOMDP> Updater for DiscreteUpdater<M> {
    type Belief = DiscreteBelief;
    type Action = M::Action;
    type Observation = M::Observation;

    /// # Panics
    ///
    /// Panics if the observation is impossible; use [`DiscreteUpdater::try_update`] to handle that.
    fn update(&self, b: &DiscreteBelief, a: &M::Action, o: &M::Observation) -> DiscreteBelief {
        self.try_update(b, a, o)
            .expect("observation has zero probability under the current belief")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{TigerAction, TigerObservation, TigerPOMDP};

    #[test]
    fn tiger_listen_update() {
        let tiger = TigerPOMDP::default();
        let up = DiscreteUpdater::new(&tiger);
        let b0 = up.initial_belief();
        assert_eq!(b0.probs, vec![0.5, 0.5]);

        let b1 = up.update(&b0, &TigerAction::Listen, &TigerObservation::HearLeft);
        assert!((b1.probs[0] - 0.85).abs() < 1e-12);

        // Two consistent observations: 0.85² / (0.85² + 0.15²).
        let b2 = up.update(&b1, &TigerAction::Listen, &TigerObservation::HearLeft);
        let expected = 0.85f64.powi(2) / (0.85f64.powi(2) + 0.15f64.powi(2));
        assert!((b2.probs[0] - expected).abs() < 1e-12);
    }
}
