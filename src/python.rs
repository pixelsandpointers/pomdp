//! Python bindings (built with `--features python`, packaged with maturin).
//!
//! Python models are tabular: states, actions and observations are integers and the model is
//! given as nested lists (or anything convertible, such as NumPy arrays via `.tolist()`).

#[pyo3::pymodule(name = "pomdp")]
mod pomdp_module {
    use std::sync::Arc;

    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::belief::{DiscreteBelief, DiscreteUpdater};
    use crate::models::{ModelError, TabularMDP, TabularPOMDP, TigerPOMDP};
    use crate::policy::Policy;
    use crate::simulator::RolloutSimulator;
    use crate::solvers::{
        AlphaVectorPolicy, QMDPSolver, ValueIterationPolicy, ValueIterationSolver,
    };

    fn value_error(e: ModelError) -> PyErr {
        PyValueError::new_err(e.0)
    }

    fn check_index(i: usize, n: usize, what: &str) -> PyResult<()> {
        if i < n {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "{what} {i} is out of range (0..{n})"
            )))
        }
    }

    fn to_belief(probs: Vec<f64>, n_states: usize) -> PyResult<DiscreteBelief> {
        if probs.len() != n_states {
            return Err(PyValueError::new_err(format!(
                "belief has length {}, expected {n_states}",
                probs.len()
            )));
        }
        if probs.iter().any(|&p| p < 0.0) || (probs.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
            return Err(PyValueError::new_err("belief must be a probability vector"));
        }
        Ok(DiscreteBelief::new(probs))
    }

    fn build_mdp(
        transition: Vec<Vec<Vec<f64>>>,
        reward: Vec<Vec<f64>>,
        discount: f64,
        initial_state: Option<Vec<f64>>,
        terminal: Option<Vec<bool>>,
    ) -> PyResult<TabularMDP> {
        let n = transition.len();
        let initial = initial_state.unwrap_or_else(|| vec![1.0 / n as f64; n]);
        let mut m = TabularMDP::new(transition, reward, discount, initial).map_err(value_error)?;
        if let Some(terminal) = terminal {
            m = m.with_terminals(terminal).map_err(value_error)?;
        }
        Ok(m)
    }

    fn rng(seed: Option<u64>) -> StdRng {
        seed.map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64)
    }

    /// A finite MDP given by `transition[s][a][sp]` and `reward[s][a]`.
    ///
    /// `initial_state` defaults to uniform; `terminal[s]` marks absorbing, zero-value states.
    #[pyclass(name = "TabularMDP", frozen)]
    struct PyTabularMDP {
        inner: Arc<TabularMDP>,
    }

    #[pymethods]
    impl PyTabularMDP {
        #[new]
        #[pyo3(signature = (transition, reward, discount, initial_state=None, terminal=None))]
        fn new(
            transition: Vec<Vec<Vec<f64>>>,
            reward: Vec<Vec<f64>>,
            discount: f64,
            initial_state: Option<Vec<f64>>,
            terminal: Option<Vec<bool>>,
        ) -> PyResult<Self> {
            let m = build_mdp(transition, reward, discount, initial_state, terminal)?;
            Ok(Self { inner: Arc::new(m) })
        }

        #[getter]
        fn n_states(&self) -> usize {
            self.inner.n_states()
        }

        #[getter]
        fn n_actions(&self) -> usize {
            self.inner.n_actions()
        }

        fn __repr__(&self) -> String {
            format!(
                "TabularMDP(n_states={}, n_actions={})",
                self.inner.n_states(),
                self.inner.n_actions()
            )
        }
    }

    /// A finite POMDP: a `TabularMDP` plus `observation[a][sp][o]`.
    #[pyclass(name = "TabularPOMDP", frozen)]
    struct PyTabularPOMDP {
        inner: Arc<TabularPOMDP>,
    }

    #[pymethods]
    impl PyTabularPOMDP {
        #[new]
        #[pyo3(signature = (transition, reward, observation, discount, initial_state=None, terminal=None))]
        fn new(
            transition: Vec<Vec<Vec<f64>>>,
            reward: Vec<Vec<f64>>,
            observation: Vec<Vec<Vec<f64>>>,
            discount: f64,
            initial_state: Option<Vec<f64>>,
            terminal: Option<Vec<bool>>,
        ) -> PyResult<Self> {
            let mdp = build_mdp(transition, reward, discount, initial_state, terminal)?;
            let m = TabularPOMDP::new(mdp, observation).map_err(value_error)?;
            Ok(Self { inner: Arc::new(m) })
        }

        #[getter]
        fn n_states(&self) -> usize {
            self.inner.n_states()
        }

        #[getter]
        fn n_actions(&self) -> usize {
            self.inner.n_actions()
        }

        #[getter]
        fn n_observations(&self) -> usize {
            self.inner.n_observations()
        }

        fn __repr__(&self) -> String {
            format!(
                "TabularPOMDP(n_states={}, n_actions={}, n_observations={})",
                self.inner.n_states(),
                self.inner.n_actions(),
                self.inner.n_observations()
            )
        }
    }

    /// The classic tiger problem as a `TabularPOMDP`.
    ///
    /// States: 0 = tiger left, 1 = tiger right. Actions: 0 = listen, 1 = open left,
    /// 2 = open right. Observations: 0 = hear left, 1 = hear right.
    #[pyfunction]
    fn tiger() -> PyTabularPOMDP {
        PyTabularPOMDP {
            inner: Arc::new(TabularPOMDP::from_pomdp(&TigerPOMDP::default())),
        }
    }

    /// Greedy policy over the Q-values from value iteration.
    #[pyclass(name = "ValueIterationPolicy", frozen)]
    struct PyValueIterationPolicy {
        inner: ValueIterationPolicy<Arc<TabularMDP>>,
    }

    #[pymethods]
    impl PyValueIterationPolicy {
        fn action(&self, s: usize) -> PyResult<usize> {
            check_index(s, self.inner.model().n_states(), "state")?;
            Ok(self.inner.action(&s))
        }

        fn value(&self, s: usize) -> PyResult<f64> {
            check_index(s, self.inner.model().n_states(), "state")?;
            Ok(self.inner.value(&s))
        }

        /// `q[s][a]`
        #[getter]
        fn q(&self) -> Vec<Vec<f64>> {
            self.inner.q.clone()
        }

        #[getter]
        fn iterations(&self) -> usize {
            self.inner.iterations
        }

        #[getter]
        fn residual(&self) -> f64 {
            self.inner.residual
        }
    }

    /// A policy made of alpha vectors over states, one action per vector.
    #[pyclass(name = "AlphaVectorPolicy", frozen)]
    struct PyAlphaVectorPolicy {
        inner: AlphaVectorPolicy<Arc<TabularPOMDP>>,
    }

    #[pymethods]
    impl PyAlphaVectorPolicy {
        fn action(&self, belief: Vec<f64>) -> PyResult<usize> {
            Ok(self
                .inner
                .action(&to_belief(belief, self.inner.model().n_states())?))
        }

        fn value(&self, belief: Vec<f64>) -> PyResult<f64> {
            Ok(self
                .inner
                .value(&to_belief(belief, self.inner.model().n_states())?))
        }

        /// `alphas[i][s]`
        #[getter]
        fn alphas(&self) -> Vec<Vec<f64>> {
            self.inner.alphas.clone()
        }

        /// `action_map[i]` is the action of `alphas[i]`.
        #[getter]
        fn action_map(&self) -> Vec<usize> {
            self.inner.action_map.clone()
        }
    }

    /// Exact Bayesian belief updates for a `TabularPOMDP`.
    #[pyclass(name = "DiscreteUpdater", frozen)]
    struct PyDiscreteUpdater {
        inner: DiscreteUpdater<Arc<TabularPOMDP>>,
    }

    #[pymethods]
    impl PyDiscreteUpdater {
        #[new]
        fn new(pomdp: &PyTabularPOMDP) -> Self {
            Self {
                inner: DiscreteUpdater::new(pomdp.inner.clone()),
            }
        }

        fn initial_belief(&self) -> Vec<f64> {
            self.inner.initial_belief().probs
        }

        /// Returns the posterior belief; raises `ValueError` if `o` is impossible.
        fn update(&self, belief: Vec<f64>, a: usize, o: usize) -> PyResult<Vec<f64>> {
            let m = self.inner.model();
            let b = to_belief(belief, m.n_states())?;
            check_index(a, m.n_actions(), "action")?;
            check_index(o, m.n_observations(), "observation")?;
            self.inner
                .try_update(&b, &a, &o)
                .map(|b| b.probs)
                .ok_or_else(|| {
                    PyValueError::new_err("observation has zero probability under this belief")
                })
        }
    }

    /// Solves a `TabularMDP` with value iteration.
    #[pyfunction]
    #[pyo3(signature = (mdp, max_iterations=1000, tolerance=1e-6))]
    fn solve_value_iteration(
        mdp: &PyTabularMDP,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyValueIterationPolicy {
        let solver = ValueIterationSolver {
            max_iterations,
            tolerance,
        };
        PyValueIterationPolicy {
            inner: solver.solve(mdp.inner.clone()),
        }
    }

    /// Solves a `TabularPOMDP` with the QMDP approximation.
    #[pyfunction]
    #[pyo3(signature = (pomdp, max_iterations=1000, tolerance=1e-6))]
    fn solve_qmdp(
        pomdp: &PyTabularPOMDP,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyAlphaVectorPolicy {
        let solver = QMDPSolver {
            max_iterations,
            tolerance,
        };
        PyAlphaVectorPolicy {
            inner: solver.solve(pomdp.inner.clone()),
        }
    }

    /// Runs one episode and returns the discounted return.
    ///
    /// Accepts a `TabularMDP` with a `ValueIterationPolicy`, or a `TabularPOMDP` with an
    /// `AlphaVectorPolicy` (beliefs are tracked with a `DiscreteUpdater`).
    #[pyfunction]
    #[pyo3(signature = (model, policy, max_steps=100, seed=None))]
    fn simulate(
        model: &Bound<'_, PyAny>,
        policy: &Bound<'_, PyAny>,
        max_steps: usize,
        seed: Option<u64>,
    ) -> PyResult<f64> {
        let sim = RolloutSimulator::new(max_steps);
        let mut rng = rng(seed);
        if let (Ok(m), Ok(p)) = (
            model.extract::<PyRef<'_, PyTabularMDP>>(),
            policy.extract::<PyRef<'_, PyValueIterationPolicy>>(),
        ) {
            if m.inner.n_states() != p.inner.model().n_states() {
                return Err(PyValueError::new_err(
                    "policy was solved for a model with a different state space",
                ));
            }
            return Ok(sim.simulate_mdp(&m.inner, &p.inner, &mut rng));
        }
        if let (Ok(m), Ok(p)) = (
            model.extract::<PyRef<'_, PyTabularPOMDP>>(),
            policy.extract::<PyRef<'_, PyAlphaVectorPolicy>>(),
        ) {
            if m.inner.n_states() != p.inner.model().n_states() {
                return Err(PyValueError::new_err(
                    "policy was solved for a model with a different state space",
                ));
            }
            let updater = DiscreteUpdater::new(m.inner.clone());
            let b0 = updater.initial_belief();
            return Ok(sim.simulate_pomdp(&m.inner, &p.inner, &updater, b0, &mut rng));
        }
        Err(PyValueError::new_err(
            "expected (TabularMDP, ValueIterationPolicy) or (TabularPOMDP, AlphaVectorPolicy)",
        ))
    }
}
