//! Simulators that run policies on models.

use rand::Rng;

use crate::belief::Updater;
use crate::distributions::Distribution;
use crate::model::{MDP, POMDP};
use crate::policy::Policy;

/// Runs a single episode and returns the accumulated discounted reward
/// (`RolloutSimulator` from POMDPTools.jl).
#[derive(Clone, Debug)]
pub struct RolloutSimulator {
    /// Episodes stop after this many steps if no terminal state is reached.
    pub max_steps: usize,
}

impl RolloutSimulator {
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }

    /// Simulates a fully observable episode starting from a sample of the initial state.
    pub fn simulate_mdp<M, P, R>(&self, m: &M, policy: &P, rng: &mut R) -> f64
    where
        M: MDP,
        P: Policy<M::State, Action = M::Action>,
        R: Rng + ?Sized,
    {
        let mut s = m.initial_state().sample(rng);
        let mut ret = 0.0;
        let mut disc = 1.0;
        for _ in 0..self.max_steps {
            if m.is_terminal(&s) {
                break;
            }
            let a = policy.action(&s);
            let sp = m.transition(&s, &a).sample(rng);
            ret += disc * m.reward(&s, &a, &sp);
            disc *= m.discount();
            s = sp;
        }
        ret
    }

    /// Simulates a partially observable episode. The hidden state is sampled from the model's
    /// initial state distribution; the policy only ever sees beliefs starting from `b0`.
    pub fn simulate_pomdp<M, P, U, R>(
        &self,
        m: &M,
        policy: &P,
        updater: &U,
        b0: U::Belief,
        rng: &mut R,
    ) -> f64
    where
        M: POMDP,
        P: Policy<U::Belief, Action = M::Action>,
        U: Updater<Action = M::Action, Observation = M::Observation>,
        R: Rng + ?Sized,
    {
        let mut s = m.initial_state().sample(rng);
        let mut b = b0;
        let mut ret = 0.0;
        let mut disc = 1.0;
        for _ in 0..self.max_steps {
            if m.is_terminal(&s) {
                break;
            }
            let a = policy.action(&b);
            let sp = m.transition(&s, &a).sample(rng);
            let o = m.observation(&a, &sp).sample(rng);
            ret += disc * m.reward(&s, &a, &sp);
            disc *= m.discount();
            b = updater.update(&b, &a, &o);
            s = sp;
        }
        ret
    }
}
