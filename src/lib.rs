//! A Rust port of Julia's [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl).
//!
//! The crate mirrors the POMDPs.jl design: problems are described through a small set of
//! traits ([`MDP`], [`POMDP`] and their finite counterparts), and solvers, belief updaters and
//! simulators are written against those traits only.

pub mod belief;
pub mod distributions;
pub mod model;
pub mod models;
pub mod policy;
pub mod simulator;

pub use model::{FiniteMDP, FinitePOMDP, MDP, POMDP};
