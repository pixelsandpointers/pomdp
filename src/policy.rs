//! Policies map what the agent knows (a state or a belief) to an action.

/// A decision rule. `X` is the policy input: a state for MDP policies, a belief for POMDP policies.
pub trait Policy<X> {
    type Action;

    fn action(&self, x: &X) -> Self::Action;
}

/// A policy backed by a closure, handy for baselines and tests.
pub struct FunctionPolicy<F>(pub F);

impl<X, A, F: Fn(&X) -> A> Policy<X> for FunctionPolicy<F> {
    type Action = A;

    fn action(&self, x: &X) -> A {
        (self.0)(x)
    }
}
