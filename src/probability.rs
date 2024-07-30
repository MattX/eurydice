use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl Into<Vec<i32>> for Outcome {
    fn into(self) -> Vec<i32> {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct Distribution {
    pub probabilities: BTreeMap<Outcome, f64>
}

impl Distribution {
    const EPSILON: f64 = f64::EPSILON;

    pub fn outcome_size(&self) -> Option<usize> {
        let first_outcome = self.probabilities.keys().next()?.0.len();
        if self.probabilities.keys().all(|outcome| outcome.0.len() == first_outcome) {
            Some(first_outcome)
        } else {
            None
        }
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

    pub fn remove_zero_entries(&mut self) {
        self.probabilities.retain(|_, &mut prob| prob.abs() > Self::EPSILON);
    }

    pub fn uniform(start: i32, end: i32) -> Self {
        let mut probabilities = BTreeMap::new();
        let count = (end - start + 1) as f64;
        let probability = 1.0 / count;

        for i in start..=end {
            probabilities.insert(Outcome(vec![i]), probability);
        }

        Distribution { probabilities }
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&Outcome) -> Outcome,
    {
        let mut new_probabilities = BTreeMap::new();

        for (outcome, probability) in &self.probabilities {
            let new_outcome = f(outcome);
            *new_probabilities.entry(new_outcome).or_insert(0.0) += probability;
        }

        Distribution { probabilities: new_probabilities }
    }

    pub fn cross_product(distributions: Vec<Distribution>) -> Self {
        if distributions.is_empty() {
            return Distribution { probabilities: BTreeMap::new() };
        }

        let mut result = distributions[0].clone();

        for dist in distributions.into_iter().skip(1) {
            let mut new_probabilities = BTreeMap::new();

            for (outcome1, prob1) in result.probabilities {
                for (outcome2, prob2) in &dist.probabilities {
                    let mut new_outcome = outcome1.0.clone();
                    new_outcome.extend(outcome2.0.clone());
                    let new_prob = prob1 * prob2;
                    new_probabilities.insert(Outcome(new_outcome), new_prob);
                }
            }

            result = Distribution { probabilities: new_probabilities };
        }

        result
    }

    pub fn reduce<F>(distributions: Vec<Distribution>, f: F) -> Option<Self>
    where
        F: Fn(&Outcome, &Outcome) -> Outcome,
    {
        distributions.into_iter().reduce(|acc, dist| {
            let mut new_probabilities = BTreeMap::new();

            for (outcome1, prob1) in acc.probabilities {
                for (outcome2, prob2) in &dist.probabilities {
                    let new_outcome = f(&outcome1, outcome2);
                    let new_prob = prob1 * prob2;
                    *new_probabilities.entry(new_outcome).or_insert(0.0) += new_prob;
                }
            }

            Distribution { probabilities: new_probabilities }
        })
    }

    pub fn sum(distributions: Vec<Distribution>) -> Option<Self> {
        Self::reduce(distributions, |a, b| {
            Outcome(a.0.iter().zip(b.0.iter()).map(|(&x, &y)| x + y).collect())
        })
    }

    pub fn max(distributions: Vec<Distribution>) -> Option<Self> {
        Self::reduce(distributions, |a, b| {
            Outcome(a.0.iter().zip(b.0.iter()).map(|(&x, &y)| x.max(y)).collect())
        })
    }
}

#[derive(Debug, Clone)]
pub struct ConditionalDistribution {
    conditioned_on: Vec<usize>,
    probabilities: BTreeMap<Outcome, Distribution>
}

impl From<Distribution> for ConditionalDistribution {
    fn from(distribution: Distribution) -> Self {
        ConditionalDistribution {
            conditioned_on: Vec::new(),
            probabilities: vec![(Outcome(Vec::new()), distribution)].into_iter().collect()
        }
    }
}
