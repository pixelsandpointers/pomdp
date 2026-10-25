use super::value_iteration::{argmax, q_value_iteration};
use crate::belief::DiscreteBelief;
use crate::model::{FiniteMDP, FinitePOMDP};
use crate::policy::Policy;

/// The QMDP approximation (QMDP.jl): solve the fully observable MDP, then act as if all
/// uncertainty disappears after one step. Produces one alpha vector per action.
#[derive(Clone, Debug)]
pub struct QMDPSolver {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for QMDPSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl QMDPSolver {
    pub fn solve<M: FinitePOMDP>(&self, m: M) -> AlphaVectorPolicy<M> {
        let sol = q_value_iteration(&m, self.max_iterations, self.tolerance);
        let actions = m.actions();
        let alphas = (0..actions.len())
            .map(|ai| sol.q.iter().map(|row| row[ai]).collect())
            .collect();
        AlphaVectorPolicy::new(m, alphas, actions)
    }
}

/// A policy represented by alpha vectors, each tied to an action. The value of a belief is the
/// best dot product with any alpha vector.
#[derive(Clone, Debug)]
pub struct AlphaVectorPolicy<M: FiniteMDP> {
    model: M,
    /// `alphas[i][s]`, indexed by state index.
    pub alphas: Vec<Vec<f64>>,
    /// `action_map[i]` is the action for `alphas[i]`.
    pub action_map: Vec<M::Action>,
}

impl<M: FiniteMDP> AlphaVectorPolicy<M> {
    /// # Panics
    ///
    /// Panics if `alphas` and `action_map` differ in length.
    pub fn new(model: M, alphas: Vec<Vec<f64>>, action_map: Vec<M::Action>) -> Self {
        assert_eq!(
            alphas.len(),
            action_map.len(),
            "one action is needed per alpha vector"
        );
        Self {
            model,
            alphas,
            action_map,
        }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    fn scores(&self, b: &DiscreteBelief) -> Vec<f64> {
        self.alphas
            .iter()
            .map(|alpha| alpha.iter().zip(&b.probs).map(|(x, p)| x * p).sum())
            .collect()
    }

    pub fn value(&self, b: &DiscreteBelief) -> f64 {
        self.scores(b).into_iter().fold(f64::NEG_INFINITY, f64::max)
    }
}

impl<M: FiniteMDP> Policy<DiscreteBelief> for AlphaVectorPolicy<M> {
    type Action = M::Action;

    fn action(&self, b: &DiscreteBelief) -> M::Action {
        self.action_map[argmax(&self.scores(b))].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{TigerAction, TigerPOMDP};

    #[test]
    fn tiger_qmdp() {
        let policy = QMDPSolver::default().solve(TigerPOMDP::default());
        assert_eq!(
            policy.action(&DiscreteBelief::uniform(2)),
            TigerAction::Listen
        );
        // Tiger almost surely on the left: open the right door.
        assert_eq!(
            policy.action(&DiscreteBelief::new(vec![0.99, 0.01])),
            TigerAction::OpenRight
        );
        assert_eq!(
            policy.action(&DiscreteBelief::new(vec![0.01, 0.99])),
            TigerAction::OpenLeft
        );
    }
}
