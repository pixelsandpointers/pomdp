//! The problem-definition interface, following POMDPs.jl.
//!
//! | POMDPs.jl                  | this crate                        |
//! |----------------------------|-----------------------------------|
//! | `transition(m, s, a)`      | [`MDP::transition`]               |
//! | `reward(m, s, a, sp)`      | [`MDP::reward`]                   |
//! | `discount(m)`              | [`MDP::discount`]                 |
//! | `initialstate(m)`          | [`MDP::initial_state`]            |
//! | `isterminal(m, s)`         | [`MDP::is_terminal`]              |
//! | `observation(m, a, sp)`    | [`POMDP::observation`]            |
//! | `states(m)`, `stateindex`  | [`FiniteMDP::states`], [`FiniteMDP::state_index`] |
//! | `actions(m)`, `actionindex`| [`FiniteMDP::actions`], [`FiniteMDP::action_index`] |
//! | `observations(m)`, `obsindex` | [`FinitePOMDP::observations`], [`FinitePOMDP::observation_index`] |

use std::rc::Rc;
use std::sync::Arc;

use crate::distributions::{DiscreteDistribution, Distribution};

/// A Markov decision process given by its explicit distributions.
pub trait MDP {
    type State: Clone;
    type Action: Clone;
    type TransitionDist: Distribution<Self::State>;
    type InitialStateDist: Distribution<Self::State>;

    /// Distribution of the next state after taking `a` in `s`.
    fn transition(&self, s: &Self::State, a: &Self::Action) -> Self::TransitionDist;

    /// Reward for the transition `s --a--> sp`.
    fn reward(&self, s: &Self::State, a: &Self::Action, sp: &Self::State) -> f64;

    /// Discount factor in `[0, 1]`.
    fn discount(&self) -> f64;

    /// Distribution of the state at the start of an episode.
    fn initial_state(&self) -> Self::InitialStateDist;

    /// Whether `s` ends the episode. Terminal states have zero value.
    fn is_terminal(&self, _s: &Self::State) -> bool {
        false
    }
}

/// A partially observable MDP.
pub trait POMDP: MDP {
    type Observation: Clone;
    type ObservationDist: Distribution<Self::Observation>;

    /// Distribution of the observation received after taking `a` and landing in `sp`.
    fn observation(&self, a: &Self::Action, sp: &Self::State) -> Self::ObservationDist;
}

/// An MDP with enumerable state and action spaces and discrete distributions,
/// as required by tabular solvers.
pub trait FiniteMDP:
    MDP<
        TransitionDist: DiscreteDistribution<<Self as MDP>::State>,
        InitialStateDist: DiscreteDistribution<<Self as MDP>::State>,
    >
{
    fn states(&self) -> Vec<Self::State>;
    fn actions(&self) -> Vec<Self::Action>;
    /// Position of `s` in [`FiniteMDP::states`].
    fn state_index(&self, s: &Self::State) -> usize;
    /// Position of `a` in [`FiniteMDP::actions`].
    fn action_index(&self, a: &Self::Action) -> usize;
}

/// A POMDP with enumerable state, action and observation spaces.
pub trait FinitePOMDP:
    FiniteMDP + POMDP<ObservationDist: DiscreteDistribution<<Self as POMDP>::Observation>>
{
    fn observations(&self) -> Vec<Self::Observation>;
    /// Position of `o` in [`FinitePOMDP::observations`].
    fn observation_index(&self, o: &Self::Observation) -> usize;
}

// Forward the traits through references and shared pointers so solvers, policies and updaters
// can either borrow a model or share ownership of it.
macro_rules! forward_model_impls {
    ($($ptr:ty),*) => {$(
        impl<M: MDP + ?Sized> MDP for $ptr {
            type State = M::State;
            type Action = M::Action;
            type TransitionDist = M::TransitionDist;
            type InitialStateDist = M::InitialStateDist;

            fn transition(&self, s: &Self::State, a: &Self::Action) -> Self::TransitionDist {
                (**self).transition(s, a)
            }
            fn reward(&self, s: &Self::State, a: &Self::Action, sp: &Self::State) -> f64 {
                (**self).reward(s, a, sp)
            }
            fn discount(&self) -> f64 {
                (**self).discount()
            }
            fn initial_state(&self) -> Self::InitialStateDist {
                (**self).initial_state()
            }
            fn is_terminal(&self, s: &Self::State) -> bool {
                (**self).is_terminal(s)
            }
        }

        impl<M: POMDP + ?Sized> POMDP for $ptr {
            type Observation = M::Observation;
            type ObservationDist = M::ObservationDist;

            fn observation(&self, a: &Self::Action, sp: &Self::State) -> Self::ObservationDist {
                (**self).observation(a, sp)
            }
        }

        impl<M: FiniteMDP + ?Sized> FiniteMDP for $ptr {
            fn states(&self) -> Vec<Self::State> {
                (**self).states()
            }
            fn actions(&self) -> Vec<Self::Action> {
                (**self).actions()
            }
            fn state_index(&self, s: &Self::State) -> usize {
                (**self).state_index(s)
            }
            fn action_index(&self, a: &Self::Action) -> usize {
                (**self).action_index(a)
            }
        }

        impl<M: FinitePOMDP + ?Sized> FinitePOMDP for $ptr {
            fn observations(&self) -> Vec<Self::Observation> {
                (**self).observations()
            }
            fn observation_index(&self, o: &Self::Observation) -> usize {
                (**self).observation_index(o)
            }
        }
    )*};
}

forward_model_impls!(&M, Box<M>, Rc<M>, Arc<M>);
