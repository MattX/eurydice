use std::{borrow::Cow, collections::BTreeMap};

use miette::Diagnostic;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Outcome(pub Vec<i32>);

impl From<Vec<i32>> for Outcome {
    fn from(v: Vec<i32>) -> Self {
        Outcome(v)
    }
}

impl From<&[i32]> for Outcome {
    fn from(v: &[i32]) -> Self {
        Outcome(v.to_vec())
    }
}

impl From<Outcome> for Vec<i32> {
    fn from(outcome: Outcome) -> Self {
        outcome.0
    }
}

/// Represents a probability distribution over sequences of integers.
/// 
/// Most functions only work if all outcomes in a distribution have the same length.
#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    pub probabilities: BTreeMap<Outcome, f64>,
}

impl Distribution {
    const EPSILON: f64 = f64::EPSILON;

    /// Returns the size of outcomes in this distribution, or an error if the distribution is empty or
    /// has inconsistent outcome sizes.
    pub fn outcome_size(&self) -> Result<usize, ProbabilityError> {
        let first_outcome = self
            .probabilities
            .keys()
            .next()
            .ok_or(ProbabilityError::NoOutcomes)?;
        for outcome in self.probabilities.keys() {
            if outcome.0.len() != first_outcome.0.len() {
                return Err(ProbabilityError::InconsistentOutcomeSize {
                    first_outcome: first_outcome.clone(),
                    second_outcome: outcome.clone(),
                });
            }
        }
        Ok(first_outcome.0.len())
    }

    pub fn normalize(&mut self) {
        let sum: f64 = self.probabilities.values().sum();
        if sum == 0.0 {
            return;
        }
        for value in self.probabilities.values_mut() {
            *value /= sum;
        }
    }

    pub fn is_normalized(&self) -> bool {
        let sum: f64 = self.probabilities.values().sum();
        (sum - 1.0).abs() < Self::EPSILON
    }

    fn remove_zero_entries(&mut self) {
        self.probabilities
            .retain(|_, &mut prob| prob.abs() > Self::EPSILON);
    }

    fn clean_up(&mut self) {
        self.remove_zero_entries();
        // Avoid doing tons of divisions if we can avoid it
        if !self.is_normalized() {
            self.normalize();
        }
    }

    /// Generates a uniform integer distribution between `start` and `end` (inclusive).
    pub fn uniform(start: i32, end: i32) -> Self {
        let mut probabilities = BTreeMap::new();
        let count = (end - start + 1) as f64;
        let probability = 1.0 / count;

        for i in start..=end {
            probabilities.insert(Outcome(vec![i]), probability);
        }

        Distribution { probabilities }
    }

    /// Applies a function to each outcome in the distribution, producing a new distribution
    pub fn try_map<F, E>(&self, f: F) -> Result<Self, E>
    where
        F: Fn(&Outcome) -> Result<Outcome, E>,
    {
        let mut new_probabilities = BTreeMap::new();

        for (outcome, probability) in &self.probabilities {
            let new_outcome = f(outcome)?;
            *new_probabilities.entry(new_outcome).or_insert(0.0) += probability;
        }

        let mut ret = Distribution {
            probabilities: new_probabilities,
        };
        ret.clean_up();
        Ok(ret)
    }

    /// Like [`Distribution::try_map`], but with a mapper that cannot fail.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&Outcome) -> Outcome,
    {
        self.try_map(|o| Ok::<Outcome, ()>(f(o))).expect("unwrapping Result<_, !>")
    }

    /// Take a number of independent distributions, and return a new distribution over the cross product of the distributions.
    ///
    /// The size of outcomes in this new distribution is the sum of outcome sizes in all input distributions.
    /// The result can be very large, and. If the next step is to map over the new distribution, consider using
    /// [`Distribution::reduce`] when that's possible.
    pub fn cross_product<'a, I>(distributions: I) -> Self
    where
        I: IntoIterator<Item = &'a Distribution>,
    {
        let mut distributions = distributions.into_iter();

        let mut result = match distributions.next() {
            Some(dist) => dist.clone(),
            None => panic!("cross_product called with no distributions"),
        };

        for dist in distributions {
            let mut new_probabilities = BTreeMap::new();

            for (outcome1, prob1) in result.probabilities {
                for (outcome2, prob2) in &dist.probabilities {
                    let mut new_outcome = outcome1.0.clone();
                    new_outcome.extend(outcome2.0.clone());
                    let new_prob = prob1 * prob2;
                    new_probabilities.insert(Outcome(new_outcome), new_prob);
                }
            }

            result = Distribution {
                probabilities: new_probabilities,
            };
        }

        result.clean_up();
        result
    }

    /// Marginalizes a distribution, returning a vector of distributions, each of dimension 1.
    /// 
    /// This is the opposite of [`Distribution::cross_product`] in the case where all arguments to
    /// `cross_product` are of dimension 1.
    pub fn marginalize(&self) -> Result<Vec<Self>, ProbabilityError> {
        let outcome_size = self.outcome_size()?;
        let mut distributions = vec![BTreeMap::new(); outcome_size];
        for (outcome, prob) in &self.probabilities {
            for (index, item) in outcome.0.iter().enumerate() {
                let marginalized_outcome = vec![*item];
                *distributions[index].entry(Outcome(marginalized_outcome)).or_insert(0.0) += prob;
            }
        }
        Ok(distributions.into_iter().map(|probabilities| Distribution { probabilities }).collect())
    }

    /// Takes a number of independent probability distribution, and reduces it by applying a function to each pair of outcomes.
    /// 
    /// This is conceptually close to a [`Distribution::cross_product`] followed by a [`Distribution::map`], but is much
    /// more efficient if the mapping function can be described as a repeated associative operation.
    /// 
    /// For instance, use this function to compute the sum or the maximum of distributions.
    pub fn try_reduce<'a, I, F, E>(distributions: I, f: F) -> Result<Self, E>
    where
        I: IntoIterator<Item = &'a Distribution>,
        F: Fn(&Outcome, &Outcome) -> Result<Outcome, E>,
    {
        let ret = distributions
            .into_iter()
            // `acc` in reduce starts out as an &Distribution. This Cow allows us to make the `acc`
            // either an &Distribution or a Distribution, depending on whether it's the first iteration.
            //
            // Additionally, we have to wrap the accumulator in a Result because to allow us to return
            // a result from the reducer.
            .map(|d| Ok(Cow::<Distribution>::Borrowed(d)))
            .reduce(|acc, dist| {
                let acc = acc?;
                let dist = dist?;
                let mut new_probabilities = BTreeMap::new();

                for (outcome1, prob1) in &(*acc).probabilities {
                    for (outcome2, prob2) in &dist.probabilities {
                        let new_outcome = f(&outcome1, outcome2)?;
                        let new_prob = prob1 * prob2;
                        *new_probabilities.entry(new_outcome).or_insert(0.0) += new_prob;
                    }
                }

                Ok(Cow::Owned(Distribution {
                    probabilities: new_probabilities,
                }))
            }).expect("try_reduce called with no distributions")?;
        let mut ret = ret.into_owned();
        ret.clean_up();
        Ok(ret)
    }

    /// Like [`Distribution::try_reduce`], but with a reducer that cannot fail.
    pub fn reduce<'a, I, F>(distributions: I, f: F) -> Self
    where
        I: IntoIterator<Item = &'a Distribution>,
        F: Fn(&Outcome, &Outcome) -> Outcome,
    {
        Self::try_reduce(distributions, |a, b| Ok::<Outcome, ()>(f(a, b))).expect("unwrapping Result<_, !>")
    }

    /// Takes a number of independent distributions which must all have the same dimension, and
    /// applies a function to each pair of elements corresponding to the outcomes.
    /// 
    /// For instance, if you have two distributions over two die rolls, you can use this function to compute
    /// the distribution over two sums, one corresponding to the first die and the second to the second die.
    pub fn reduce_pairwise<F>(
        distributions: &[Distribution],
        f: F,
    ) -> Result<Self, ProbabilityError>
    where
        F: Fn(i32, i32) -> i32,
    {
        let distribution_size = distributions[0].outcome_size()?;
        for (i, d) in distributions.iter().enumerate() {
            let size = d.outcome_size()?;
            if size != distribution_size {
                return Err(ProbabilityError::InconsistentOutcomeCounts {
                    first_example_size: distribution_size,
                    second_example_index: i,
                    second_example_size: size,
                });
            }
        }
        Ok(Self::reduce(distributions, |a, b| {
            Outcome(a.0.iter().zip(b.0.iter()).map(|(&x, &y)| f(x, y)).collect())
        }))
    }

    /// Computes the mean vector for a distribution. All outcomes in the distribution must have the same dimension.
    pub fn mean(&self) -> Result<Vec<f64>, ProbabilityError> {
        let outcome_size = self.outcome_size()?;
        let mut mean = vec![0.0; outcome_size];
        for (outcome, prob) in &self.probabilities {
            for (i, &value) in outcome.0.iter().enumerate() {
                mean[i] += value as f64 * prob;
            }
        }
        Ok(mean)
    }

    /// Computes the standard deviation vector for a distribution. All outcomes in the distribution must have the same dimension.
    pub fn stddev(&self) -> Result<Vec<f64>, ProbabilityError> {
        let outcome_size = self.outcome_size()?;
        let mean = self.mean()?;
        let mut variance = vec![0.0; outcome_size];
        for (outcome, prob) in &self.probabilities {
            for (i, &value) in outcome.0.iter().enumerate() {
                variance[i] += (value as f64 - mean[i]).powi(2) * prob;
            }
        }
        let stddev = variance.iter().map(|&v| v.sqrt()).collect();
        Ok(stddev)
    }
}

/// A distribution over multiple variables, some of which can be independent.
/// 
/// Each group of dependent variables is represented by a [`Distribution`].
#[derive(Debug, Clone, Serialize)]
pub struct JointDistribution(Vec<Distribution>);

impl From<Distribution> for JointDistribution {
    fn from(d: Distribution) -> Self {
        JointDistribution(vec![d])
    }
}

impl JointDistribution {
    /// The outcome sizes for each distribution in the joint distribution.
    pub fn outcome_sizes(&self) -> Result<Vec<usize>, ProbabilityError> {
        self.0.iter().map(|d| d.outcome_size()).collect()
    }

    /// The size of the logical outcomes in this distribution.
    /// 
    /// This is equal to the sum of the vector returned by [`JointDistribution::outcome_sizes`].
    pub fn outcome_size(&self) -> Result<usize, ProbabilityError> {
        Ok(self.outcome_sizes()?.into_iter().sum())
    }

    pub fn into_inner(self) -> Vec<Distribution> {
        self.0
    }

    pub fn inner(&self) -> &[Distribution] {
        &self.0
    }

    /// Removes independence of underlying distributions. This can be a very expensive operation.
    pub fn cross_product(&self) -> JointDistribution {
        JointDistribution(vec![Distribution::cross_product(&self.0)])
    }

    /// Individually maps f over each distribution in the joint distribution.
    pub fn map_each<F>(&self, f: F) -> JointDistribution
    where
        F: Fn(&Outcome) -> Outcome,
    {
        JointDistribution(self.0.iter().map(|d| d.map(&f)).collect())
    }

    /// Map across the entire cross distribution.
    /// 
    /// This calls [`JointDistribution::cross_product`] and is potentially quite expensive.
    pub fn map<F>(&self, f: F) -> JointDistribution
    where
        F: Fn(&Outcome) -> Outcome,
    {
        JointDistribution(vec![self.cross_product().0[0].map(&f)])
    }

    /// Efficiently maps an outcome to a 1 dimensional outcome using an (i32, i32) -> i32 reducer.
    /// 
    /// For this to work, each sub-distribution must have a consistent outcome size.
    /// 
    /// For instance, use this to compute the distribution of sums over all the underlying variables.
    pub fn reduce<F>(&self, f: F) -> Result<Distribution, ProbabilityError>
    where
        F: Fn(i32, i32) -> i32,
    {
        let outcome_sizes = self.outcome_sizes()?;
        let mut non_empty_dists = outcome_sizes.into_iter().zip(self.0.iter()).filter_map(|(size, dist)| {
            if size > 0 {
                Some(dist)
            } else {
                None
            }
        }).collect::<Vec<_>>();
        if non_empty_dists.is_empty() {
            return Err(ProbabilityError::NoDistributions);
        }
        // Unwrapping here is OK, because the only None condition is if the size is 0, which is filtered out.
        let first = non_empty_dists[0].map(|outcome| Outcome(vec![outcome.0.iter().cloned().reduce(|acc, x| f(acc, x)).unwrap()]));
        non_empty_dists[0] = &first;
        Ok(Distribution::reduce(non_empty_dists, |a, b| {
            // The accumulator is always a distribution with outcome dimension = 1
            debug_assert_eq!(a.0.len(), 1);
            Outcome(vec![b.0.iter().cloned().fold(a.0[0], |acc, x| f(acc, x))])
        }))
    }
}

#[derive(Debug, Diagnostic, Error)]
pub enum ProbabilityError {
    #[error("Empty input")]
    NoDistributions,

    #[error("This distribution has no outcomes. This is generally an error.")]
    NoOutcomes,

    #[error("This distribution does not have a consistent outcome size")]
    InconsistentOutcomeSize {
        first_outcome: Outcome,
        second_outcome: Outcome,
    },

    #[error("Some distributions have different outcome sizes")]
    InconsistentOutcomeCounts {
        first_example_size: usize,
        second_example_index: usize,
        second_example_size: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_distribution() {
        let dist = Distribution::uniform(1, 3);
        assert_eq!(dist.probabilities.len(), 3);
        assert_relative_eq!(dist.probabilities[&Outcome(vec![1])], 1.0 / 3.0);
        assert_relative_eq!(dist.probabilities[&Outcome(vec![2])], 1.0 / 3.0);
        assert_relative_eq!(dist.probabilities[&Outcome(vec![3])], 1.0 / 3.0);
    }

    #[test]
    fn test_map() {
        let dist = Distribution::uniform(-3, 3);
        let mapped = dist.map(|outcome| Outcome(vec![outcome.0[0].abs()]));
        assert_eq!(mapped.probabilities.len(), 4);
        assert_relative_eq!(mapped.probabilities[&Outcome(vec![0])], 1.0 / 7.0);
        assert_relative_eq!(mapped.probabilities[&Outcome(vec![1])], 2.0 / 7.0);
        assert_relative_eq!(mapped.probabilities[&Outcome(vec![2])], 2.0 / 7.0);
        assert_relative_eq!(mapped.probabilities[&Outcome(vec![3])], 2.0 / 7.0);
    }

    #[test]
    fn test_cross_product() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let cross = Distribution::cross_product(&[dist1, dist2]);
        assert_eq!(cross.probabilities.len(), 4);
        assert_relative_eq!(cross.probabilities[&Outcome(vec![1, 3])], 0.25);
        assert_relative_eq!(cross.probabilities[&Outcome(vec![1, 4])], 0.25);
        assert_relative_eq!(cross.probabilities[&Outcome(vec![2, 3])], 0.25);
        assert_relative_eq!(cross.probabilities[&Outcome(vec![2, 4])], 0.25);
    }

    #[test]
    fn test_reduce() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let reduced = Distribution::reduce(&[dist1, dist2], |a, b| Outcome(vec![a.0[0] + b.0[0]]));
        assert_eq!(reduced.probabilities.len(), 3);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![4])], 0.25);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![5])], 0.5);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![6])], 0.25);
    }

    #[test]
    fn test_reduce_pairwise() {
        let dist1 = Distribution {
            probabilities: vec![(Outcome(vec![1, 2]), 0.5), (Outcome(vec![3, 4]), 0.5)].into_iter().collect(),
        };
        let dist2 = Distribution {
            probabilities: vec![(Outcome(vec![5, 6]), 0.5), (Outcome(vec![7, 8]), 0.5)].into_iter().collect(),
        };
        let reduced = Distribution::reduce_pairwise(&[dist1, dist2], |a, b| a + b).unwrap();
        assert_eq!(reduced.probabilities.len(), 3);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![6, 8])], 0.25);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![8, 10])], 0.5);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![10, 12])], 0.25);
    }

    #[test]
    fn test_mean_and_stddev() {
        let dist = Distribution {
            probabilities: vec![(Outcome(vec![1, 2]), 0.5), (Outcome(vec![3, 4]), 0.5)].into_iter().collect(),
        };
        let mean = dist.mean().unwrap();
        assert_relative_eq!(mean[0], 2.0);
        assert_relative_eq!(mean[1], 3.0);

        let stddev = dist.stddev().unwrap();
        assert_relative_eq!(stddev[0], 1.0);
        assert_relative_eq!(stddev[1], 1.0);
    }

    #[test]
    fn test_normalize_and_is_normalized() {
        let mut dist = Distribution {
            probabilities: vec![(Outcome(vec![1]), 2.0), (Outcome(vec![2]), 2.0)].into_iter().collect(),
        };
        assert!(!dist.is_normalized());
        dist.normalize();
        assert!(dist.is_normalized());
        assert_relative_eq!(dist.probabilities[&Outcome(vec![1])], 0.5);
        assert_relative_eq!(dist.probabilities[&Outcome(vec![2])], 0.5);
    }

    #[test]
    fn test_remove_zero_entries() {
        let mut dist = Distribution {
            probabilities: vec![
                (Outcome(vec![1]), 0.5),
                (Outcome(vec![2]), 0.0),
                (Outcome(vec![3]), 0.5),
            ].into_iter().collect(),
        };
        dist.remove_zero_entries();
        assert_eq!(dist.probabilities.len(), 2);
        assert!(!dist.probabilities.contains_key(&Outcome(vec![2])));
    }

    #[test]
    fn test_clean_up() {
        let mut dist = Distribution {
            probabilities: vec![
                (Outcome(vec![1]), 1.0),
                (Outcome(vec![2]), 0.0),
                (Outcome(vec![3]), 1.0),
            ].into_iter().collect(),
        };
        dist.clean_up();
        assert_eq!(dist.probabilities.len(), 2);
        assert!(dist.is_normalized());
        assert_relative_eq!(dist.probabilities[&Outcome(vec![1])], 0.5);
        assert_relative_eq!(dist.probabilities[&Outcome(vec![3])], 0.5);
    }


    #[test]
    fn test_outcome_size_empty_distribution() {
        let dist = Distribution {
            probabilities: BTreeMap::new(),
        };
        assert!(matches!(
            dist.outcome_size(),
            Err(ProbabilityError::NoOutcomes)
        ));
    }

    #[test]
    fn test_outcome_size_inconsistent() {
        let dist = Distribution {
            probabilities: vec![
                (Outcome(vec![1, 2]), 0.5),
                (Outcome(vec![3]), 0.5),
            ].into_iter().collect(),
        };
        assert!(matches!(
            dist.outcome_size(),
            Err(ProbabilityError::InconsistentOutcomeSize { .. })
        ));
    }

    #[test]
    #[should_panic]
    fn test_reduce_empty_input() {
        Distribution::reduce(&[], |a, b| Outcome(vec![a.0[0] + b.0[0]]));
    }

    #[test]
    fn test_reduce_pairwise_inconsistent_sizes() {
        let dist1 = Distribution {
            probabilities: vec![(Outcome(vec![1]), 1.0)].into_iter().collect(),
        };
        let dist2 = Distribution {
            probabilities: vec![(Outcome(vec![1, 2]), 1.0)].into_iter().collect(),
        };
        let result = Distribution::reduce_pairwise(&[dist1, dist2], |a, b| a + b);
        assert!(matches!(
            result,
            Err(ProbabilityError::InconsistentOutcomeCounts { .. })
        ));
    }

    #[test]
    fn test_mean_inconsistent_sizes() {
        let dist = Distribution {
            probabilities: vec![
                (Outcome(vec![1]), 0.5),
                (Outcome(vec![2, 3]), 0.5),
            ].into_iter().collect(),
        };
        assert!(matches!(
            dist.mean(),
            Err(ProbabilityError::InconsistentOutcomeSize { .. })
        ));
    }

    #[test]
    fn test_stddev_inconsistent_sizes() {
        let dist = Distribution {
            probabilities: vec![
                (Outcome(vec![1]), 0.5),
                (Outcome(vec![2, 3]), 0.5),
            ].into_iter().collect(),
        };
        assert!(matches!(
            dist.stddev(),
            Err(ProbabilityError::InconsistentOutcomeSize { .. })
        ));
    }


    #[test]
    fn test_marginalize() {
        let dist = Distribution {
            probabilities: vec![
                (Outcome(vec![1, 2]), 0.3),
                (Outcome(vec![2, 1]), 0.2),
                (Outcome(vec![3, 3]), 0.5),
            ].into_iter().collect(),
        };
        let marginalized = dist.marginalize().unwrap();
        assert_eq!(marginalized.len(), 2);
        
        // First marginal distribution
        assert_relative_eq!(marginalized[0].probabilities[&Outcome(vec![1])], 0.3);
        assert_relative_eq!(marginalized[0].probabilities[&Outcome(vec![2])], 0.2);
        assert_relative_eq!(marginalized[0].probabilities[&Outcome(vec![3])], 0.5);
        
        // Second marginal distribution
        assert_relative_eq!(marginalized[1].probabilities[&Outcome(vec![1])], 0.2);
        assert_relative_eq!(marginalized[1].probabilities[&Outcome(vec![2])], 0.3);
        assert_relative_eq!(marginalized[1].probabilities[&Outcome(vec![3])], 0.5);
    }

    #[test]
    fn test_try_map() {
        let dist = Distribution::uniform(1, 3);
        let mapped = dist.try_map(|outcome| Ok::<Outcome, &str>(Outcome(vec![outcome.0[0] * 2]))).unwrap();
        assert_eq!(mapped.probabilities.len(), 3);
        assert_relative_eq!(mapped.probabilities[&Outcome(vec![2])], 1.0 / 3.0);
        assert_relative_eq!(mapped.probabilities[&Outcome(vec![4])], 1.0 / 3.0);
        assert_relative_eq!(mapped.probabilities[&Outcome(vec![6])], 1.0 / 3.0);
    }

    #[test]
    fn test_try_map_error() {
        let dist = Distribution::uniform(1, 3);
        let result = dist.try_map(|outcome| {
            if outcome.0[0] == 2 {
                Err("Error on 2")
            } else {
                Ok(Outcome(vec![outcome.0[0] * 2]))
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_try_reduce() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let reduced = Distribution::try_reduce(&[dist1, dist2], |a, b| Ok::<Outcome, &str>(Outcome(vec![a.0[0] + b.0[0]]))).unwrap();
        assert_eq!(reduced.probabilities.len(), 3);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![4])], 0.25);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![5])], 0.5);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![6])], 0.25);
    }

    #[test]
    fn test_try_reduce_error() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let result = Distribution::try_reduce(&[dist1, dist2], |a, b| {
            if a.0[0] + b.0[0] > 5 {
                Err("Sum too large")
            } else {
                Ok(Outcome(vec![a.0[0] + b.0[0]]))
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_joint_distribution_outcome_sizes() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let joint = JointDistribution(vec![dist1, dist2]);
        let sizes = joint.outcome_sizes().unwrap();
        assert_eq!(sizes, vec![1, 1]);
    }

    #[test]
    fn test_joint_distribution_outcome_size() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let joint = JointDistribution(vec![dist1, dist2]);
        let size = joint.outcome_size().unwrap();
        assert_eq!(size, 2);
    }

    #[test]
    fn test_joint_distribution_cross_product() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let joint = JointDistribution(vec![dist1, dist2]);
        let crossed = joint.cross_product();
        assert_eq!(crossed.0.len(), 1);
        assert_eq!(crossed.0[0].probabilities.len(), 4);
    }

    #[test]
    fn test_joint_distribution_map_each() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let joint = JointDistribution(vec![dist1, dist2]);
        let mapped = joint.map_each(|outcome| Outcome(vec![outcome.0[0] * 2]));
        assert_eq!(mapped.0.len(), 2);
        assert_eq!(mapped.0[0].probabilities.len(), 2);
        assert_eq!(mapped.0[1].probabilities.len(), 2);
    }

    #[test]
    fn test_joint_distribution_map() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let joint = JointDistribution(vec![dist1, dist2]);
        let mapped = joint.map(|outcome| Outcome(vec![outcome.0[0] + outcome.0[1]]));
        assert_eq!(mapped.0.len(), 1);
        assert_eq!(mapped.0[0].probabilities.len(), 3);
    }

    #[test]
    fn test_joint_distribution_reduce() {
        let dist1 = Distribution::uniform(1, 2);
        let dist2 = Distribution::uniform(3, 4);
        let joint = JointDistribution(vec![dist1, dist2]);
        let reduced = joint.reduce(|a, b| a + b).unwrap();
        assert_eq!(reduced.probabilities.len(), 3);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![4])], 0.25);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![5])], 0.5);
        assert_relative_eq!(reduced.probabilities[&Outcome(vec![6])], 0.25);
    }

    #[test]
    fn test_joint_distribution_reduce_empty() {
        let joint = JointDistribution(vec![]);
        let result = joint.reduce(|a, b| a + b);
        assert!(matches!(result, Err(ProbabilityError::NoDistributions)));
    }
}
