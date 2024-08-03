//! Module for calculating probabilities of dice pools.
//!
//! This is an implementation of the Icepool algorithm[^icepool]. It allows for
//! efficient computation of mapped distributions, where the mapping function
//! can be expressed as an iterated function from `(state, outcome, count)` to
//! `distribution(state)`, with a small state.
//!
//! In this module, distributions are represented as counts of states, not floating-point
//! probabilities.
//!
//! [^icepool]: Liu, A. J. (2022). Icepool: Efficient Computation of Dice Pool Probabilities.
//! _Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital
//! Entertainment_, 18(1), 258-265. https://doi.org/10.1609/aiide.v18i1.21971

use lazy_static::lazy_static;
use malachite::{Natural, Rational};
use std::{collections::HashMap, hash::Hash, sync::RwLock};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Pool {
    n: i32,
    sides: i32,
}

impl std::fmt::Display for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}d{}", self.n, self.sides)
    }
}

impl Pool {
    pub fn apply<S, F>(&self, mapper: StateMapper<S, F>) -> HashMap<S, Natural>
    where
        S: Clone + Hash + Eq,
        F: Fn(&S, i32, i32) -> S,
    {
        let mut cache = HashMap::new();
        self.apply_inner(&mut cache, &mapper)
    }

    pub fn apply_inner<S, F>(
        &self,
        cache: &mut HashMap<Pool, HashMap<S, Natural>>,
        mapper: &StateMapper<S, F>,
    ) -> HashMap<S, Natural>
    where
        S: Clone + Hash + Eq,
        F: Fn(&S, i32, i32) -> S,
    {
        if let Some(value) = cache.get(self) {
            return value.clone();
        }
        let (outcome, new_max_outcome) = (self.sides, self.sides - 1);
        if new_max_outcome == 0 {
            return [(
                (mapper.f)(&mapper.initial_state, outcome, self.n),
                1usize.into(),
            )]
            .into();
        }
        let mut result = HashMap::new();
        for num_with_outcome in 0..=self.n {
            let sub_pool = Pool {
                n: self.n - num_with_outcome,
                sides: new_max_outcome,
            };
            let inner_result = sub_pool.apply_inner(cache, mapper);
            for (state, count) in inner_result {
                let inner_state = (mapper.f)(&state, outcome, num_with_outcome);
                *result.entry(inner_state).or_default() +=
                    count * binom(self.n as usize, num_with_outcome as usize)
            }
        }
        cache.insert(*self, result.clone());
        result
    }
}

pub struct StateMapper<S, F>
where
    S: Clone + Hash + Eq,
    F: Fn(&S, i32, i32) -> S,
{
    initial_state: S,
    f: F,
}

/// Converts a distribution represented as counts into a distribution represented as probabilities.
pub fn counter_to_probability<S>(counter: &HashMap<S, Natural>) -> HashMap<S, f64>
where
    S: Clone + Hash + Eq,
{
    let total = counter.values().sum();
    counter
        .iter()
        .map(|(state, count)| {
            (
                state.clone(),
                Rational::from_naturals_ref(count, &total)
                    .approx_log()
                    .exp(),
            )
        })
        .collect()
}

lazy_static! {
    /// Cache for binomial coefficients. Rows are either missing or fully calculated.
    static ref BINOM_CACHE: RwLock<Vec<Vec<Natural>>> = RwLock::new(vec![vec![1usize.into()]]);
    static ref ONE: Natural = 1usize.into();
}

fn sum_mapper(state: &i32, outcome: i32, count: i32) -> i32 {
    state + outcome * count
}

/// Mapper that sums the outcomes.
pub const SUM_MAPPER: StateMapper<i32, fn(&i32, i32, i32) -> i32> = StateMapper {
    initial_state: 0,
    f: sum_mapper,
};

fn product_mapper(state: &i32, outcome: i32, count: i32) -> i32 {
    state * outcome.pow(count as u32)
}

/// Mapper that multiplies the outcomes.
pub const PRODUCT_MAPPER: StateMapper<i32, fn(&i32, i32, i32) -> i32> = StateMapper {
    initial_state: 1,
    f: product_mapper,
};

fn max_mapper(state: &i32, outcome: i32, count: i32) -> i32 {
    (*state).max(outcome)
}

/// Mapper that takes the maximum of the outcomes.
pub const MAX_MAPPER: StateMapper<i32, fn(&i32, i32, i32) -> i32> = StateMapper {
    initial_state: i32::MIN,
    f: max_mapper,
};

fn min_mapper(state: &i32, outcome: i32, count: i32) -> i32 {
    (*state).min(outcome)
}

/// Mapper that takes the minimum of the outcomes.
pub const MIN_MAPPER: StateMapper<i32, fn(&i32, i32, i32) -> i32> = StateMapper {
    initial_state: i32::MAX,
    f: min_mapper,
};

/// The first value represents the current sum, or None if the target is already reached.
/// The second value represents the number of rolls made so far.
type MaxDiceToReachState = (Option<i32>, i32);

/// Returns a mapper that calculates the minimum number of dice to sum, starting from the
/// lowest dice, to reach the target.
pub fn make_max_dice_to_reach_mapper(
    target: i32,
) -> StateMapper<MaxDiceToReachState, impl Fn(&MaxDiceToReachState, i32, i32) -> MaxDiceToReachState>
{
    StateMapper {
        initial_state: (Some(0), 0),
        f: move |state, outcome, count| {
            let (sum, rolls) = state;
            let sum = match sum {
                Some(sum) => sum,
                None => {
                    return *state;
                }
            };
            let count_needed = (target - sum + outcome - 1) / outcome;
            if count_needed <= count {
                (None, rolls + count_needed)
            } else {
                (Some(sum + count * outcome), rolls + count)
            }
        },
    }
}

/// Calculate binomial coefficient n choose k, with value caching.
///
/// Panics if k > n.
fn binom(n: usize, k: usize) -> Natural {
    if n == k {
        return ONE.clone();
    }
    if k == 0 {
        return ONE.clone();
    }
    let mut binom_cache = BINOM_CACHE.write().unwrap();
    let max_row = binom_cache.len();
    if n < max_row {
        // This may panic if k > n, but that's fine - caller should have known better.
        return binom_cache[n][k].clone();
    }
    for row_idx in max_row..=n {
        let mut row: Vec<Natural> = vec![1usize.into(); row_idx + 1];
        for i in 1..row_idx {
            // The first and last elements are always 1, so we can skip them.
            row[i] = &binom_cache[row_idx - 1][i - 1] + &binom_cache[row_idx - 1][i];
        }
        binom_cache.push(row);
    }
    binom_cache[n][k].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binom_base_cases() {
        assert_eq!(binom(0, 0), 1);
        assert_eq!(binom(1, 0), 1);
        assert_eq!(binom(1, 1), 1);
    }

    #[test]
    fn test_binom_small_values() {
        assert_eq!(binom(2, 1), 2);
        assert_eq!(binom(3, 1), 3);
        assert_eq!(binom(3, 2), 3);
        assert_eq!(binom(4, 2), 6);
    }

    #[test]
    fn test_binom_larger_values() {
        assert_eq!(binom(10, 5), 252);
        assert_eq!(binom(20, 10), 184756);
    }

    #[test]
    fn test_binom_edge_cases() {
        assert_eq!(binom(100, 0), 1);
        assert_eq!(binom(100, 100), 1);
    }

    fn to_counter<T: Hash + Eq>(v: Vec<(T, usize)>) -> HashMap<T, Natural> {
        v.into_iter().map(|(k, v)| (k, Natural::from(v))).collect()
    }

    #[test]
    fn test_sum_10d20() {
        let pool = Pool { n: 10, sides: 20 };
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(result.len(), 191);
    }

    #[test]
    fn test_sum_1d6() {
        let pool = Pool { n: 1, sides: 6 };
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(
            result,
            to_counter(vec![(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)])
        );
    }

    #[test]
    fn test_sum_2d6() {
        let pool = Pool { n: 2, sides: 6 };
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(
            result,
            to_counter(vec![
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 5),
                (9, 4),
                (10, 3),
                (11, 2),
                (12, 1)
            ])
        );
    }

    #[test]
    fn test_make_max_dice_to_reach_mapper() {
        let mapper = make_max_dice_to_reach_mapper(10);
        let result = Pool { n: 3, sides: 6 }.apply(mapper);
        let mut keep_count_only = HashMap::new();
        for (k, v) in result {
            // If the sum is None, we've already reached the target. Replace with a number of rolls.
            // Otherwise, return None as we haven't reached the target.
            let key = if k.0.is_some() { None } else { Some(k.1) };
            *keep_count_only.entry(key).or_default() += v;
        }
        assert_eq!(
            keep_count_only,
            to_counter(vec![
                (None, 81usize.into()),
                (Some(2), 11usize.into()),
                (Some(3), 124usize.into())
            ])
        )
    }
}
