//! Ready-made problems (a small slice of POMDPModels.jl) and tabular model types.

mod tabular;
mod tiger;

pub use tabular::{ModelError, TabularMDP, TabularPOMDP};
pub use tiger::{TigerAction, TigerObservation, TigerPOMDP, TigerState};
