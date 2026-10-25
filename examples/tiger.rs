//! Solve the tiger problem with QMDP and evaluate the policy by simulation.
//!
//! Run with `cargo run --example tiger`.

use pomdp::models::{TigerAction, TigerObservation, TigerPOMDP};
use pomdp::prelude::*;
use rand::{SeedableRng, rngs::StdRng};

fn main() {
    let tiger = TigerPOMDP::default();
    let policy = QMDPSolver::default().solve(&tiger);
    let updater = DiscreteUpdater::new(&tiger);

    // Walk through a belief trajectory by hand.
    let mut b = updater.initial_belief();
    for _ in 0..3 {
        let a = policy.action(&b);
        println!(
            "belief {:.3?} -> {a:?} (value {:.2})",
            b.probs,
            policy.value(&b)
        );
        b = updater.update(&b, &TigerAction::Listen, &TigerObservation::HearLeft);
    }
    println!("belief {:.3?} -> {:?}", b.probs, policy.action(&b));

    // Monte Carlo estimate of the policy's discounted return.
    let sim = RolloutSimulator::new(100);
    let mut rng = StdRng::seed_from_u64(42);
    let n = 1000;
    let mean: f64 = (0..n)
        .map(|_| {
            sim.simulate_pomdp(
                &tiger,
                &policy,
                &updater,
                updater.initial_belief(),
                &mut rng,
            )
        })
        .sum::<f64>()
        / n as f64;
    println!("mean discounted return over {n} episodes: {mean:.2}");
}
