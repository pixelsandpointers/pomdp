//! Probability distributions over states, actions and observations.

use rand::Rng;

/// A distribution that can be sampled from and evaluated.
pub trait Distribution<T> {
    /// Draws a value from the distribution.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T;

    /// Probability (mass or density) of `x`.
    fn pdf(&self, x: &T) -> f64;
}

/// A distribution with finite support that can be enumerated.
pub trait DiscreteDistribution<T>: Distribution<T> {
    /// Every outcome with nonzero probability, paired with its probability.
    fn support(&self) -> Vec<(T, f64)>;
}

/// A distribution that always yields the same value.
#[derive(Clone, Debug, PartialEq)]
pub struct Deterministic<T>(pub T);

impl<T: Clone + PartialEq> Distribution<T> for Deterministic<T> {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
        self.0.clone()
    }

    fn pdf(&self, x: &T) -> f64 {
        if *x == self.0 { 1.0 } else { 0.0 }
    }
}

impl<T: Clone + PartialEq> DiscreteDistribution<T> for Deterministic<T> {
    fn support(&self) -> Vec<(T, f64)> {
        vec![(self.0.clone(), 1.0)]
    }
}

/// A sparse categorical distribution: explicit values with their probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseCat<T> {
    vals: Vec<T>,
    probs: Vec<f64>,
}

impl<T> SparseCat<T> {
    /// Creates the distribution. `probs` should sum to one.
    ///
    /// # Panics
    ///
    /// Panics if `vals` and `probs` differ in length or are empty.
    pub fn new(vals: Vec<T>, probs: Vec<f64>) -> Self {
        assert_eq!(
            vals.len(),
            probs.len(),
            "SparseCat: values and probabilities differ in length"
        );
        assert!(!vals.is_empty(), "SparseCat: empty support");
        Self { vals, probs }
    }

    pub fn values(&self) -> &[T] {
        &self.vals
    }

    pub fn probabilities(&self) -> &[f64] {
        &self.probs
    }
}

impl<T: Clone + PartialEq> Distribution<T> for SparseCat<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let mut u = rng.random::<f64>() * self.probs.iter().sum::<f64>();
        for (v, &p) in self.vals.iter().zip(&self.probs) {
            if u < p {
                return v.clone();
            }
            u -= p;
        }
        // Only reachable through floating-point round-off.
        self.vals.last().unwrap().clone()
    }

    fn pdf(&self, x: &T) -> f64 {
        self.vals
            .iter()
            .zip(&self.probs)
            .filter(|(v, _)| *v == x)
            .map(|(_, p)| p)
            .sum()
    }
}

impl<T: Clone + PartialEq> DiscreteDistribution<T> for SparseCat<T> {
    fn support(&self) -> Vec<(T, f64)> {
        self.vals
            .iter()
            .cloned()
            .zip(self.probs.iter().copied())
            .filter(|&(_, p)| p > 0.0)
            .collect()
    }
}

/// A uniform distribution over a finite set of values.
#[derive(Clone, Debug, PartialEq)]
pub struct Uniform<T> {
    vals: Vec<T>,
}

impl<T> Uniform<T> {
    /// # Panics
    ///
    /// Panics if `vals` is empty.
    pub fn new(vals: Vec<T>) -> Self {
        assert!(!vals.is_empty(), "Uniform: empty support");
        Self { vals }
    }
}

impl<T: Clone + PartialEq> Distribution<T> for Uniform<T> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        self.vals[rng.random_range(0..self.vals.len())].clone()
    }

    fn pdf(&self, x: &T) -> f64 {
        self.vals.iter().filter(|v| *v == x).count() as f64 / self.vals.len() as f64
    }
}

impl<T: Clone + PartialEq> DiscreteDistribution<T> for Uniform<T> {
    fn support(&self) -> Vec<(T, f64)> {
        let p = 1.0 / self.vals.len() as f64;
        self.vals.iter().map(|v| (v.clone(), p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn sparse_cat_pdf_and_support() {
        let d = SparseCat::new(vec!['a', 'b', 'c'], vec![0.2, 0.0, 0.8]);
        assert_eq!(d.pdf(&'a'), 0.2);
        assert_eq!(d.pdf(&'z'), 0.0);
        assert_eq!(d.support(), vec![('a', 0.2), ('c', 0.8)]);
    }

    #[test]
    fn sparse_cat_sampling_matches_probabilities() {
        let d = SparseCat::new(vec![0, 1], vec![0.25, 0.75]);
        let mut rng = StdRng::seed_from_u64(0);
        let n = 20_000;
        let ones = (0..n).filter(|_| d.sample(&mut rng) == 1).count();
        assert!((ones as f64 / n as f64 - 0.75).abs() < 0.02);
    }

    #[test]
    fn uniform_and_deterministic() {
        let u = Uniform::new(vec![1, 2, 3, 4]);
        assert_eq!(u.pdf(&3), 0.25);
        let d = Deterministic(7);
        assert_eq!(d.pdf(&7), 1.0);
        assert_eq!(d.sample(&mut StdRng::seed_from_u64(0)), 7);
    }
}
