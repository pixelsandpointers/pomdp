//! Offline solvers. Each solver's `solve` takes a model and returns a [`Policy`](crate::policy::Policy).

mod qmdp;
mod value_iteration;

pub use qmdp::{AlphaVectorPolicy, QMDPSolver};
pub use value_iteration::{ValueIterationPolicy, ValueIterationSolver};
