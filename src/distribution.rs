use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;

use malachite::Natural;

use crate::dice::{Pool, SUM_MAPPER};

/// A fully denormalized distribution over a type `T`.
#[derive(Debug, Clone)]
pub struct Distribution<T: Eq + PartialEq + Ord + PartialOrd + Debug> {
    values: BTreeMap<T, Natural>,
}

impl Distribution<i32> {
    fn uniform(min: i32, max: i32) -> Self {
        let mut values = BTreeMap::new();
        for i in min..=max {
            values.insert(i, 1u8.into());
        }
        Self { values }
    }
}

impl<T: Eq + PartialEq + Ord + PartialOrd + Debug> FromIterator<T> for Distribution<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut values = BTreeMap::new();
        for i in iter {
            values.insert(i, 1u8.into());
        }
        Self { values }
    }
}

impl<T: Eq + PartialEq + Ord + PartialOrd + Debug> FromIterator<(T, Natural)> for Distribution<T> {
    fn from_iter<I: IntoIterator<Item = (T, Natural)>>(iter: I) -> Self {
        let mut values = BTreeMap::new();
        for (k, v) in iter {
            values.insert(k, v);
        }
        Self { values }
    }
}

impl<T: Eq + PartialEq + Ord + PartialOrd + Debug> From<BTreeMap<T, Natural>> for Distribution<T> {
    fn from(values: BTreeMap<T, Natural>) -> Self {
        Self { values }
    }
}

impl<T: Eq + PartialEq + Ord + PartialOrd + Debug> From<HashMap<T, Natural>> for Distribution<T> {
    fn from(values: HashMap<T, Natural>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }
}

impl<T: Eq + PartialEq + Ord + PartialOrd + Debug> From<Vec<(T, Natural)>> for Distribution<T> {
    fn from(values: Vec<(T, Natural)>) -> Self {
        let mut map = BTreeMap::new();
        for (k, v) in values {
            map.insert(k, v);
        }
        Self { values: map }
    }
}

impl<T: Eq + PartialEq + Ord + PartialOrd + Debug> Distribution<T> {
    pub fn inner(&self) -> &BTreeMap<T, Natural> {
        &self.values
    }

    pub fn into_inner(self) -> BTreeMap<T, Natural> {
        self.values
    }

    pub fn map_values<U: Eq + PartialEq + Ord + PartialOrd + Debug, F: Fn(T) -> U>(
        self,
        f: F,
    ) -> Distribution<U> {
        self.into_inner()
            .into_iter()
            .map(|(outcome, prob)| (f(outcome), prob.clone()))
            .collect()
    }

    pub fn flat_map_values<
        U: Eq + PartialEq + Ord + PartialOrd + Debug,
        F: Fn(T) -> Distribution<U>,
    >(
        self,
        f: F,
    ) -> Distribution<U> {
        self.into_inner()
            .into_iter()
            .flat_map(|(outcome, prob)| {
                f(outcome)
                    .into_inner()
                    .into_iter()
                    .map(move |(mapped_outcome, mapped_prob)| {
                        (mapped_outcome, mapped_prob * prob.clone())
                    })
            })
            .collect()
    }

    pub fn map_weights(self, f: impl Fn(Natural) -> Natural) -> Self {
        self.into_inner()
            .into_iter()
            .map(|(outcome, prob)| (outcome, f(prob)))
            .collect()
    }
}

impl<T: Eq + PartialEq + Ord + PartialOrd + Debug + Clone> Distribution<T> {
    pub fn map_cross_product<
        U: Eq + PartialEq + Ord + PartialOrd + Debug + Clone,
        V: Eq + PartialEq + Ord + PartialOrd + Debug,
        F: Fn(T, U) -> V,
    >(
        self,
        other: Distribution<U>,
        f: F,
    ) -> Distribution<V> {
        let self_values = self.into_inner();
        let other_values = other.into_inner();
        let mut result_values = BTreeMap::new();
        for (self_outcome, self_prob) in self_values {
            for (other_outcome, other_prob) in other_values.clone() {
                let outcome = f(self_outcome.clone(), other_outcome.clone());
                let prob = self_prob.clone() * other_prob.clone();
                result_values.insert(outcome, prob);
            }
        }
        Distribution {
            values: result_values,
        }
    }
}

impl Distribution<i32> {
    pub fn d_operator(self, left: Distribution<i32>) -> Distribution<i32> {
        left.flat_map_values(|left_outcome| {
            self.clone()
                .map_values(move |right_outcome| left_outcome * right_outcome)
        })
    }
}

/// The type used at runtime by the interpreter.
#[derive(Debug, Clone)]
pub enum PoolOrDistribution {
    Pool(Pool),
    Distribution(Distribution<i32>),
}

impl From<Pool> for PoolOrDistribution {
    fn from(pool: Pool) -> Self {
        PoolOrDistribution::Pool(pool)
    }
}

impl From<Distribution<i32>> for PoolOrDistribution {
    fn from(distribution: Distribution<i32>) -> Self {
        PoolOrDistribution::Distribution(distribution)
    }
}

impl PoolOrDistribution {
    /// If this is a pool, returns a distribution over the sum of the dice. Has no effect otherwise.
    pub fn to_sum_distribution(self) -> Distribution<i32> {
        match self {
            PoolOrDistribution::Pool(pool) => pool.apply(SUM_MAPPER).into(),
            PoolOrDistribution::Distribution(distribution) => distribution,
        }
    }

    /// Creates a pool of `num` dice with `sides` sides.
    pub fn pool(num: u32, sides: u32) -> Self {
        PoolOrDistribution::Pool(Pool::ndn(num, sides))
    }

    /// Creates a pool of dice with the given sides.
    pub fn pool_from_list(num: u32, sides: Vec<i32>) -> Self {
        PoolOrDistribution::Pool(Pool::from_list(num, sides))
    }
}
