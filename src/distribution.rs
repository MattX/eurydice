use std::collections::BTreeMap;
use std::fmt::Debug;

use malachite::Natural;

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
