use std::collections::BTreeMap;
use std::fmt::Debug;

use malachite::Natural;

/// A fully denormalized distribution over a type `T`.
#[derive(Debug, Clone)]
struct Distribution<T: Eq + PartialEq + Ord + PartialOrd + Debug> {
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

impl<T: Eq + PartialEq + Ord + PartialOrd + Debug> Distribution<T> {
    fn inner(&self) -> &BTreeMap<T, Natural> {
        &self.values
    }

    fn into_inner(self) -> BTreeMap<T, Natural> {
        self.values
    }

    fn map_values<U: Eq + PartialEq + Ord + PartialOrd + Debug, F: Fn(T) -> U>(
        self,
        f: F,
    ) -> Distribution<U> {
        self.into_inner()
            .into_iter()
            .map(|(outcome, prob)| (f(outcome), prob.clone()))
            .collect()
    }

    fn flat_map_values<
        U: Eq + PartialEq + Ord + PartialOrd + Debug,
        F: Fn(T) -> Distribution<U>,
    >(
        self,
        f: F,
    ) -> Distribution<U> {
        self.into_inner().into_iter().flat_map(|(outcome, prob)| {
            f(outcome)
                .into_inner()
                .into_iter()
                .map(|(mapped_outcome, mapped_prob)| (mapped_outcome, mapped_prob * prob.clone()))
                .collect()
        })
    }
}

impl<i32> Distribution<i32> {
    fn at_operator(self, left: Distribution<i32>) -> Distribution<i32> {
        left.flat_map_values(|left_outcome| {
            self.map_values(move |right_outcome| left_outcome + right_outcome)
        })
    }
}
