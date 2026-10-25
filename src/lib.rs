//! A Rust port of Julia's [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl).
//!
//! The crate mirrors the POMDPs.jl design: problems are described through a small set of
//! traits ([`MDP`], [`POMDP`] and their finite counterparts), and solvers, belief updaters and
//! simulators are written against those traits only.
//!
//! ```
//! use pomdp::prelude::*;
//! use pomdp::models::{TigerAction, TigerPOMDP};
//! use rand::{SeedableRng, rngs::StdRng};
//!
//! let tiger = TigerPOMDP::default();
//! let policy = QMDPSolver::default().solve(&tiger);
//! let updater = DiscreteUpdater::new(&tiger);
//!
//! // With no information, listening is the best thing to do.
//! assert_eq!(policy.action(&updater.initial_belief()), TigerAction::Listen);
//!
//! let mut rng = StdRng::seed_from_u64(1);
//! let ret = RolloutSimulator::new(50).simulate_pomdp(
//!     &tiger,
//!     &policy,
//!     &updater,
//!     updater.initial_belief(),
//!     &mut rng,
//! );
//! assert!(ret.is_finite());
//! ```

pub mod belief;
pub mod distributions;
pub mod model;
pub mod models;
pub mod policy;
pub mod simulator;
pub mod solvers;

pub use model::{FiniteMDP, FinitePOMDP, MDP, POMDP};

/// Commonly used traits and types.
pub mod prelude {
    pub use crate::belief::{DiscreteBelief, DiscreteUpdater, Updater};
    pub use crate::distributions::{
        Deterministic, DiscreteDistribution, Distribution, SparseCat, Uniform,
    };
    pub use crate::model::{FiniteMDP, FinitePOMDP, MDP, POMDP};
    pub use crate::policy::Policy;
    pub use crate::simulator::RolloutSimulator;
    pub use crate::solvers::{
        AlphaVectorPolicy, QMDPSolver, ValueIterationPolicy, ValueIterationSolver,
    };
}
