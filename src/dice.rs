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
// TODO some functions in this file take ownership (or a mutable reference), but don't really need to.

use lazy_static::lazy_static;
use malachite::num::arithmetic::traits::{DivExact, Factorial, Lcm, Pow};
use malachite::num::basic::traits::{One, Zero};
use malachite::{Natural, Rational};
use std::collections::BTreeMap;
use std::{collections::HashMap, fmt::Debug, hash::Hash, sync::RwLock};
use tinyvec::TinyVec;

/// Represents a pool of identical independent dice.
#[derive(Debug, Clone)]
pub struct Pool {
    n: u32,
    // Outcomes are ordered by their face value. The tuple represents (value, weight / count).
    // Outcomes must be unique and have nonzero weight.
    ordered_outcomes: Vec<(i32, Natural)>,
    // This vector must be of size `n`. A false element at position `n` means that the n-th
    // lowest dice should be ignored.
    keep_list: Vec<bool>,
}

impl std::fmt::Display for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO format nicely if this is a regular dn.
        write!(
            f,
            "{}d[{}]",
            self.n,
            self.ordered_outcomes
                .iter()
                .map(|(outcome, weight)| format!("{}:{}", outcome, weight))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Cache key for the Icepool algorithm. `n` is the number of dice remaining, and
/// `remaining_count` is the number of outcomes remaining.
///
/// The outcomes that remain in consideration are the `remaining_count` smallest outcomes.`
#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
struct SubPool {
    n: u32,
    remaining_outcomes: usize,
}

impl SubPool {
    fn initial(pool: &Pool) -> Self {
        Self {
            n: pool.n,
            remaining_outcomes: pool.ordered_outcomes.len(),
        }
    }
}

impl Pool {
    /// Creates a new pool of `n` dice, each with `sides` sides.
    pub fn ndn(n: u32, sides: u32) -> Self {
        Self {
            n,
            ordered_outcomes: (1..=sides)
                .map(|side| (side as i32, 1usize.into()))
                .collect::<Vec<_>>(),
            keep_list: vec![true; n as usize],
        }
    }

    /// Creates a new pool from a list of outcomes. Repeats are allowed and will count as multiple weights.
    pub fn from_list(n: u32, outcomes: Vec<i32>) -> Self {
        let mut outcomes_map = BTreeMap::new();
        for outcome in outcomes {
            *outcomes_map.entry(outcome).or_insert(Natural::from(0usize)) += Natural::from(1usize);
        }
        let ordered_outcomes = outcomes_map.into_iter().collect::<Vec<_>>();
        Self {
            n,
            ordered_outcomes,
            keep_list: vec![true; n as usize],
        }
    }

    /// Sets the number of dice in the pool, resetting the keep list.
    pub fn set_n(&mut self, n: u32) {
        self.n = n;
        self.keep_list = vec![true; n as usize];
    }

    /// Gets the number of dice in the pool.
    pub fn get_n(&self) -> u32 {
        self.n
    }

    /// Maps the outcomes of the pool using the given function. The function can be non-injective,
    /// in which case the weights of the outcomes are summed.
    pub fn map_outcomes(self, f: impl Fn(i32) -> i32) -> Self {
        let mut new_outcomes = BTreeMap::new();
        for (outcome, weight) in self.ordered_outcomes.into_iter() {
            *new_outcomes
                .entry(f(outcome))
                .or_insert(Natural::from(0usize)) += weight;
        }
        Self {
            ordered_outcomes: new_outcomes.into_iter().collect(),
            ..self
        }
    }

    /// Maps the weights of the pool using the given function.
    pub fn map_weights(self, f: impl Fn(Natural) -> Natural) -> Self {
        Self {
            ordered_outcomes: self
                .ordered_outcomes
                .into_iter()
                .map(|(outcome, weight)| (outcome, f(weight)))
                .collect(),
            ..self
        }
    }

    fn from_weights(outcomes: impl Iterator<Item = (i32, Natural)>) -> Self {
        let mut ordered_outcomes = outcomes.collect::<Vec<_>>();
        ordered_outcomes.sort();
        Self {
            n: 1,
            ordered_outcomes,
            keep_list: vec![true],
        }
    }

    pub fn set_keep_list(&mut self, keep_list: Vec<bool>) {
        assert_eq!(
            keep_list.len(),
            self.n as usize,
            "keep list length must be the number of dice in the pool"
        );
        self.keep_list = keep_list;
    }

    pub fn apply<S, F>(&self, mapper: StateMapper<S, F>) -> HashMap<S, Natural>
    where
        S: Clone + Hash + Eq,
        F: Fn(&S, i32, u32) -> S,
    {
        let mut cache = HashMap::new();
        self.apply_inner(SubPool::initial(self), &mut cache, &mapper)
    }

    fn apply_inner<S, F>(
        &self,
        sub_pool: SubPool,
        cache: &mut HashMap<SubPool, HashMap<S, Natural>>,
        mapper: &StateMapper<S, F>,
    ) -> HashMap<S, Natural>
    where
        S: Clone + Hash + Eq,
        F: Fn(&S, i32, u32) -> S,
    {
        if let Some(value) = cache.get(&sub_pool) {
            return value.clone();
        }
        let new_remaining_outcomes = sub_pool.remaining_outcomes - 1;
        let (outcome, weight) = &self.ordered_outcomes[new_remaining_outcomes];
        let result = if new_remaining_outcomes == 0 {
            let num_kept = self.num_kept(sub_pool, sub_pool.n);
            [(
                (mapper.f)(&mapper.initial_state, *outcome, num_kept),
                weight.pow(sub_pool.n as u64),
            )]
            .into()
        } else {
            let mut result = HashMap::new();
            for num_with_outcome in 0..=sub_pool.n {
                // Replace num_with_outcome with the actual number of dice to keep in the considered range.
                // Ignore anything in the keep list above index `sub_pool.n`, and below `sub_pool.n - num_with_outcome`.
                let num_kept = self.num_kept(sub_pool, num_with_outcome);

                let sub_sub_pool = SubPool {
                    n: sub_pool.n - num_with_outcome,
                    remaining_outcomes: new_remaining_outcomes,
                };
                let sub_sub_pool_result = self.apply_inner(sub_sub_pool, cache, mapper);
                for (state, count) in sub_sub_pool_result {
                    let inner_state = (mapper.f)(&state, *outcome, num_kept);
                    // There were binom(self.n, num_with_outcome) ways to get this outcome,
                    // times weight^num_with_outcome if the weight is >1.
                    *result.entry(inner_state).or_default() += count
                        * binom(sub_pool.n as usize, num_with_outcome as usize)
                        * weight.pow(num_with_outcome as u64);
                }
            }
            result
        };
        cache.insert(sub_pool, result.clone());
        result
    }

    fn num_kept(&self, sub_pool: SubPool, num_with_outcome: u32) -> u32 {
        self.keep_list[(sub_pool.n - num_with_outcome) as usize..sub_pool.n as usize]
            .iter()
            .filter(|&&keep| keep)
            .count() as u32
    }

    /// Sums the distribution; the resulting pool is guaranteed to have n=1.
    pub fn sum(&self) -> Pool {
        self.apply(SUM_MAPPER).into_iter().collect()
    }

    pub fn into_iter(self) -> impl Iterator<Item = (i32, Natural)> {
        self.ordered_outcomes.into_iter()
    }

    pub fn ordered_outcomes(&self) -> &[(i32, Natural)] {
        &self.ordered_outcomes
    }

    pub fn multiset_iterator(&self) -> PoolMultisetIterator {
        PoolMultisetIterator::new(self)
    }

    pub fn outcome_iterator(&self) -> PoolIterator {
        PoolIterator::new(self)
    }

    /// This functions call `f` with each multiset outcome from the pool. The distributions returned
    /// by `f` are flatmapped together to create a new distribution, stored as a size-1 pool.
    pub fn flat_map<F>(&self, f: F) -> Self
    where
        F: Fn(TinyVec<[i32; 6]>) -> BTreeMap<i32, Natural>,
    {
        // Positions will take all values between [0, 0, ..., 0] and [self.n - 1, self.n - 1, ..., self.n - 1],
        // in lexicographic order.
        let mut new_outcomes = BTreeMap::new();
        let mut outcome_sum_lcm = Natural::from(1usize);

        for (outcome, ways) in self.multiset_iterator() {
            // Map the outcome. The weights returned by the outcome distribution should still sum to
            // |weight|, but instead, the sum can be an arbitrary value.
            // To solve this, we store each weight in |new_outcomes| as a fraction (divided by the total
            // weight returned by f), and multiply everything by the LCM at the end to go back
            // to naturals.
            let f_outcome = f(outcome);
            let f_outcome_sum = f_outcome.values().sum();
            outcome_sum_lcm = outcome_sum_lcm.lcm(&f_outcome_sum);
            for (new_outcome, sub_weight) in f_outcome {
                *new_outcomes.entry(new_outcome).or_insert(Rational::ZERO) +=
                    Rational::from_naturals(&ways * sub_weight, f_outcome_sum.clone());
            }
        }
        let new_outcomes = new_outcomes
            .into_iter()
            .map(|(outcome, weight)| {
                let (numerator, denominator) =
                    (weight * Rational::from(&outcome_sum_lcm)).into_numerator_and_denominator();
                debug_assert_eq!(denominator, Natural::ONE);
                (outcome, numerator)
            })
            .collect::<BTreeMap<_, _>>();
        Pool::from_weights(new_outcomes.into_iter())
    }

    /// Maps multiset outcomes to a single value each.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(TinyVec<[i32; 6]>) -> i32,
    {
        self.flat_map(|outcome| BTreeMap::from([(f(outcome), 1usize.into())]))
    }

    pub fn add(&self, other: &Pool) -> Pool {
        let other_summed = other.sum();
        self.sum().flat_map(|outcome| {
            other_summed
                .map(|other_outcome| outcome[0] + other_outcome[0])
                .into()
        })
    }
}

/// Iterator over multisets of outcomes in a pool, with their weight.
pub struct PoolMultisetIterator<'a> {
    pool: &'a Pool,
    positions: Vec<usize>,
    // Factorial of the number of outcomes in the dice pool.
    factorial: Natural,
    // Whether we're finished; if true, then |positions| may contain invalid indices.
    done: bool,
}

impl<'a> PoolMultisetIterator<'a> {
    fn new(pool: &'a Pool) -> Self {
        Self {
            pool,
            positions: vec![0; pool.n as usize],
            factorial: Natural::factorial(pool.n as u64),
            done: false,
        }
    }

    fn advance_position(&mut self) {
        let pool_size = self.pool.n as isize;
        let mut position_index = pool_size - 1;
        while position_index >= 0 {
            self.positions[position_index as usize] += 1;
            if self.positions[position_index as usize] == self.pool.ordered_outcomes.len() {
                if position_index == 0 {
                    self.done = true;
                    return;
                }
                position_index -= 1;
            } else {
                break;
            }
        }
        // Now go back and fix up the position indices
        for i in position_index + 1..pool_size {
            self.positions[i as usize] = self.positions[i as usize - 1];
        }
    }
}

impl<'a> Iterator for PoolMultisetIterator<'a> {
    type Item = (TinyVec<[i32; 6]>, Natural);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let outcome = self
            .positions
            .iter()
            .map(|&i| self.pool.ordered_outcomes[i].0)
            .collect::<TinyVec<[i32; 6]>>();

        // Compute the number of ways to get this outcome, and multiply by the weight of all
        // elements that make it up.
        let weight: Natural = self
            .positions
            .iter()
            .map(|&i| &self.pool.ordered_outcomes[i].1)
            .product();
        let permutations = item_factorials(&outcome);
        let ways = weight * (&self.factorial).div_exact(permutations);

        self.advance_position();
        Some((outcome, ways))
    }
}

/// For each group of consecutive equal values in the outcomes, this computes
/// factorial(numer of same outcomes). The result is the product of all these
/// factorials.
fn item_factorials(outcome: &TinyVec<[i32; 6]>) -> Natural {
    let mut product = Natural::ONE;
    let mut count = 1;
    for i in 1..outcome.len() {
        let prev = outcome.get(i - 1);
        if prev.is_some() && *prev.unwrap() == outcome[i] {
            count += 1;
        } else {
            product *= Natural::factorial(count as u64);
            count = 1;
        }
    }
    product *= Natural::factorial(count as u64);
    product
}

/// Iterator over all possible outcomes in a pool.
pub struct PoolIterator<'a> {
    pool: &'a Pool,
    positions: Vec<usize>,
    /// Whether we're finished; if true, then |positions| may contain invalid indices.
    done: bool,
}

impl<'a> PoolIterator<'a> {
    fn new(pool: &'a Pool) -> Self {
        Self { pool, positions: vec![0; pool.n as usize], done: false }
    }
}

impl<'a> Iterator for PoolIterator<'a> {
    type Item = (TinyVec<[i32; 6]>, Natural);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let outcome = self
            .positions
            .iter()
            .map(|&i| self.pool.ordered_outcomes[i].0)
            .collect::<TinyVec<[i32; 6]>>();
        let weight: Natural = self
            .positions
            .iter()
            .map(|&i| &self.pool.ordered_outcomes[i].1)
            .product();
        let mut position = self.pool.n as isize - 1;
        while position >= 0 {
            self.positions[position as usize] += 1;
            if self.positions[position as usize] == self.pool.ordered_outcomes.len() {
                if position == 0 {
                    self.done = true;
                    break;
                }
                self.positions[position as usize] = 0;
                position -= 1;
            } else {
                break;
            }
        }
        Some((outcome, weight))
    }
}

impl FromIterator<(i32, Natural)> for Pool {
    fn from_iter<I: IntoIterator<Item = (i32, Natural)>>(iter: I) -> Self {
        let mut ordered_outcomes = iter.into_iter().collect::<Vec<_>>();
        ordered_outcomes.sort();
        Self {
            n: 1,
            ordered_outcomes,
            keep_list: vec![true],
        }
    }
}

impl From<Vec<(i32, Natural)>> for Pool {
    fn from(ordered_outcomes: Vec<(i32, Natural)>) -> Self {
        Self {
            n: 1,
            ordered_outcomes,
            keep_list: vec![true],
        }
    }
}

impl From<Pool> for BTreeMap<i32, Natural> {
    fn from(pool: Pool) -> Self {
        pool.into_iter().collect()
    }
}

pub fn explode(die: Vec<(i32, Natural)>, on: &[i32], depth: usize) -> Vec<(i32, Natural)> {
    if depth == 0 {
        return die;
    }

    let inner_explode: Pool = explode(die.clone(), on, depth - 1).into();
    let die_dist: Pool = die.into_iter().collect::<Pool>().flat_map(|outcome| {
        let exploder = outcome[0];
        if on.contains(&exploder) {
            inner_explode
                .clone()
                .map(|exploded_outcome| exploder + exploded_outcome[0])
                .into()
        } else {
            vec![(exploder, Natural::ONE)].into_iter().collect()
        }
    });
    die_dist.ordered_outcomes
}

pub struct StateMapper<S, F>
where
    S: Clone + Hash + Eq,
    F: Fn(&S, i32, u32) -> S,
{
    initial_state: S,
    f: F,
}

fn sum_mapper(state: &i32, outcome: i32, count: u32) -> i32 {
    state + outcome * (count as i32)
}

/// Mapper that sums the outcomes.
pub const SUM_MAPPER: StateMapper<i32, fn(&i32, i32, u32) -> i32> = StateMapper {
    initial_state: 0,
    f: sum_mapper,
};

fn product_mapper(state: &i32, outcome: i32, count: u32) -> i32 {
    state * outcome.pow(count)
}

/// Mapper that multiplies the outcomes.
pub const PRODUCT_MAPPER: StateMapper<i32, fn(&i32, i32, u32) -> i32> = StateMapper {
    initial_state: 1,
    f: product_mapper,
};

fn max_mapper(state: &i32, outcome: i32, _count: u32) -> i32 {
    (*state).max(outcome)
}

/// Mapper that takes the maximum of the outcomes.
pub const MAX_MAPPER: StateMapper<i32, fn(&i32, i32, u32) -> i32> = StateMapper {
    initial_state: i32::MIN,
    f: max_mapper,
};

fn min_mapper(state: &i32, outcome: i32, _count: u32) -> i32 {
    (*state).min(outcome)
}

/// Mapper that takes the minimum of the outcomes.
pub const MIN_MAPPER: StateMapper<i32, fn(&i32, i32, u32) -> i32> = StateMapper {
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
) -> StateMapper<MaxDiceToReachState, impl Fn(&MaxDiceToReachState, i32, u32) -> MaxDiceToReachState>
{
    StateMapper {
        initial_state: (Some(0), 0),
        f: move |state, outcome, count| {
            let count = count as i32;
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

lazy_static! {
    /// Cache for binomial coefficients. Rows are either missing or fully calculated.
    static ref BINOM_CACHE: RwLock<Vec<Vec<Natural>>> = RwLock::new(vec![vec![1usize.into()]]);
    static ref ONE: Natural = 1usize.into();
}

/// Calculate binomial coefficient n choose k, with value caching.
///
/// Panics if k > n.
// TODO - this can likely be made faster / smaller by only caching factorials.
#[allow(clippy::needless_range_loop)]
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
        let pool = Pool::ndn(10, 20);
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(result.len(), 191);
        assert_eq!(result[&133], Natural::from(70942066700u64));
    }

    #[test]
    fn test_sum_1d6() {
        let pool = Pool::ndn(1, 6);
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(
            result,
            to_counter(vec![(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)])
        );
    }

    #[test]
    fn test_sum_2d6() {
        let pool = Pool::ndn(2, 6);
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
    fn test_sum_2d3() {
        let pool = Pool::ndn(2, 3);
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(
            result,
            to_counter(vec![(2, 1), (3, 2), (4, 3), (5, 2), (6, 1),])
        );
    }

    #[test]
    fn test_make_max_dice_to_reach_mapper() {
        let mapper = make_max_dice_to_reach_mapper(10);
        let result = Pool::ndn(3, 6).apply(mapper);
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
                (None, 81usize),
                (Some(2), 11usize),
                (Some(3), 124usize)
            ])
        )
    }

    #[test]
    fn test_sum_non_continuous() {
        let pool = Pool::from_list(3, vec![-2, 0, 1, 5]);
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(
            result,
            to_counter(vec![
                (-6, 1),
                (-4, 3),
                (-3, 3),
                (-2, 3),
                (-1, 6),
                (0, 4),
                (1, 6),
                (2, 3),
                (3, 7),
                (4, 6),
                (5, 3),
                (6, 6),
                (7, 3),
                (8, 3),
                (10, 3),
                (11, 3),
                (15, 1)
            ])
        );
    }

    #[test]
    fn test_sum_weighted() {
        let pool = Pool::from_list(3, vec![-1, -1, 0, 1, 1, 1]);
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(
            result,
            to_counter(vec![
                (-3, 8),
                (-2, 12),
                (-1, 42),
                (0, 37),
                (1, 63),
                (2, 27),
                (3, 27)
            ])
        );
    }

    #[test]
    fn test_sum_6d10_keep_3() {
        let mut pool = Pool::ndn(6, 10);
        pool.set_keep_list(vec![false, false, false, true, true, true]);
        let result = pool.apply(SUM_MAPPER);
        assert_eq!(result.len(), 28);
        assert_eq!(result[&15], Natural::from(16617u64));
    }

    #[test]
    fn test_explode_d5_on_3() {
        let die = vec![1, 2, 3, 4, 5]
            .into_iter()
            .map(|i| (i, 1usize.into()))
            .collect();
        let result = explode(die, &[3], 2);
        assert_eq!(
            result,
            vec![
                (1, 25),
                (2, 25),
                (4, 30),
                (5, 30),
                (7, 6),
                (8, 6),
                (9, 1),
                (10, 1),
                (11, 1)
            ]
            .into_iter()
            .map(|(i, w)| (i, Natural::from(w as u32)))
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_explode_d10() {
        let die = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            .into_iter()
            .map(|i| (i, 1usize.into()))
            .collect();
        let result = explode(die, &[2, 8], 3);
        let map = result.into_iter().collect::<HashMap<_, _>>();
        let expected = [
            (1, 1000),
            (3, 1100),
            (4, 1000),
            (5, 1110),
            (6, 1100),
            (7, 1111),
            (8, 111),
            (9, 1211),
            (10, 1011),
            (11, 231),
            (12, 201),
            (13, 134),
            (14, 134),
            (15, 124),
            (16, 24),
            (17, 133),
            (18, 103),
            (19, 36),
            (20, 36),
            (21, 16),
            (22, 16),
            (23, 13),
            (24, 3),
            (25, 14),
            (26, 14),
            (27, 4),
            (28, 4),
            (29, 1),
            (30, 1),
            (31, 1),
            (32, 1),
            (33, 1),
            (34, 1),
        ]
        .into_iter()
        .map(|(i, w)| (i, Natural::from(w as u32)))
        .collect::<HashMap<_, _>>();
        assert_eq!(map, expected);
    }

    #[test]
    fn test_explode_weighted() {
        let pool = Pool::from_list(1, vec![1, 1, 3, 4, 4, 4, 5, 5]);
        let die = pool.ordered_outcomes;
        let result = explode(die, &[1, 4], 3);
        let map = result.into_iter().collect::<HashMap<_, _>>();
        let expected = [
            (3, 512),
            (4, 144),
            (5, 1056),
            (6, 264),
            (7, 352),
            (8, 112),
            (9, 420),
            (10, 408),
            (11, 144),
            (12, 54),
            (13, 360),
            (14, 108),
            (15, 27),
            (16, 81),
            (17, 54),
        ]
        .into_iter()
        .map(|(i, w)| (i, Natural::from(w as u32)))
        .collect::<HashMap<_, _>>();
        assert_eq!(map, expected);
    }

    #[test]
    fn test_map() {
        let pool = Pool::from_list(5, vec![1, 2, 3, 4, 5]);
        fn multiset_to_int(multiset: TinyVec<[i32; 6]>) -> i32 {
            let mut total = 0;
            for (idx, item) in multiset.iter().rev().enumerate() {
                total += (item - 1) * 5i32.pow(idx as u32);
            }
            total
        }

        let result = pool.map(multiset_to_int);

        let map = result.into_iter().collect::<HashMap<_, _>>();
        let expected = [
            (0, 1),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (6, 10),
            (7, 20),
            (8, 20),
            (9, 20),
            (12, 10),
            (13, 20),
            (14, 20),
            (18, 10),
            (19, 20),
            (24, 10),
            (31, 10),
            (32, 30),
            (33, 30),
            (34, 30),
            (37, 30),
            (38, 60),
            (39, 60),
            (43, 30),
            (44, 60),
            (49, 30),
            (62, 10),
            (63, 30),
            (64, 30),
            (68, 30),
            (69, 60),
            (74, 30),
            (93, 10),
            (94, 30),
            (99, 30),
            (124, 10),
            (156, 5),
            (157, 20),
            (158, 20),
            (159, 20),
            (162, 30),
            (163, 60),
            (164, 60),
            (168, 30),
            (169, 60),
            (174, 30),
            (187, 20),
            (188, 60),
            (189, 60),
            (193, 60),
            (194, 120),
            (199, 60),
            (218, 20),
            (219, 60),
            (224, 60),
            (249, 20),
            (312, 5),
            (313, 20),
            (314, 20),
            (318, 30),
            (319, 60),
            (324, 30),
            (343, 20),
            (344, 60),
            (349, 60),
            (374, 20),
            (468, 5),
            (469, 20),
            (474, 30),
            (499, 20),
            (624, 5),
            (781, 1),
            (782, 5),
            (783, 5),
            (784, 5),
            (787, 10),
            (788, 20),
            (789, 20),
            (793, 10),
            (794, 20),
            (799, 10),
            (812, 10),
            (813, 30),
            (814, 30),
            (818, 30),
            (819, 60),
            (824, 30),
            (843, 10),
            (844, 30),
            (849, 30),
            (874, 10),
            (937, 5),
            (938, 20),
            (939, 20),
            (943, 30),
            (944, 60),
            (949, 30),
            (968, 20),
            (969, 60),
            (974, 60),
            (999, 20),
            (1093, 5),
            (1094, 20),
            (1099, 30),
            (1124, 20),
            (1249, 5),
            (1562, 1),
            (1563, 5),
            (1564, 5),
            (1568, 10),
            (1569, 20),
            (1574, 10),
            (1593, 10),
            (1594, 30),
            (1599, 30),
            (1624, 10),
            (1718, 5),
            (1719, 20),
            (1724, 30),
            (1749, 20),
            (1874, 5),
            (2343, 1),
            (2344, 5),
            (2349, 10),
            (2374, 10),
            (2499, 5),
            (3124, 1),
        ]
        .into_iter()
        .map(|(i, w)| (i, Natural::from(w as u32)))
        .collect::<HashMap<_, _>>();
        assert_eq!(map, expected);
    }

    #[test]
    // Computes (d3 @ d4) in AnyDice or Icepool notation.
    fn test_flat_map() {
        let pool1 = Pool::from_list(1, vec![1, 2, 3]);
        let pool2 = Pool::from_list(1, vec![1, 2, 3, 4]);
        let result = pool1.flat_map(|outcome| {
            let mut summed_pool = pool2.clone();
            summed_pool.set_n(outcome[0] as u32);
            summed_pool.sum().into_iter().collect::<BTreeMap<_, _>>()
        });
        let map = result.into_iter().collect::<HashMap<_, _>>();
        let expected = [
            (1, 16),
            (2, 20),
            (3, 25),
            (4, 31),
            (5, 22),
            (6, 22),
            (7, 20),
            (8, 16),
            (9, 10),
            (10, 6),
            (11, 3),
            (12, 1),
        ]
        .into_iter()
        .map(|(i, w)| (i, Natural::from(w as u32)))
        .collect::<HashMap<_, _>>();
        assert_eq!(map, expected);
    }

    #[test]
    fn test_multiset_iterator() {
        let pool = Pool::from_list(2, vec![1, 2, 3]);
        let mut iter = pool.multiset_iterator();
        assert_eq!(iter.next(), Some((vec![1, 1].as_slice().into(), 1usize.into())));
        assert_eq!(iter.next(), Some((vec![1, 2].as_slice().into(), 2usize.into())));
        assert_eq!(iter.next(), Some((vec![1, 3].as_slice().into(), 2usize.into())));
        assert_eq!(iter.next(), Some((vec![2, 2].as_slice().into(), 1usize.into())));
        assert_eq!(iter.next(), Some((vec![2, 3].as_slice().into(), 2usize.into())));
        assert_eq!(iter.next(), Some((vec![3, 3].as_slice().into(), 1usize.into())));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_multiset_iterator_weighted() {
        let pool = Pool::from_list(2, vec![1, 1, 2, 3, 3, 3]);
        let mut iter = pool.multiset_iterator();
        assert_eq!(iter.next(), Some((vec![1, 1].as_slice().into(), 4usize.into())));
        assert_eq!(iter.next(), Some((vec![1, 2].as_slice().into(), 4usize.into())));
        assert_eq!(iter.next(), Some((vec![1, 3].as_slice().into(), 12usize.into())));
        assert_eq!(iter.next(), Some((vec![2, 2].as_slice().into(), 1usize.into())));
        assert_eq!(iter.next(), Some((vec![2, 3].as_slice().into(), 6usize.into())));
        assert_eq!(iter.next(), Some((vec![3, 3].as_slice().into(), 9usize.into())));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_pool_iterator() {
        let pool = Pool::from_list(2, vec![1, 2, 2, 3]);
        let mut iter = pool.outcome_iterator();
        assert_eq!(iter.next(), Some((vec![1, 1].as_slice().into(), 1usize.into())));
        assert_eq!(iter.next(), Some((vec![1, 2].as_slice().into(), 2usize.into())));
        assert_eq!(iter.next(), Some((vec![1, 3].as_slice().into(), 1usize.into())));
        assert_eq!(iter.next(), Some((vec![2, 1].as_slice().into(), 2usize.into())));
        assert_eq!(iter.next(), Some((vec![2, 2].as_slice().into(), 4usize.into())));
        assert_eq!(iter.next(), Some((vec![2, 3].as_slice().into(), 2usize.into())));
        assert_eq!(iter.next(), Some((vec![3, 1].as_slice().into(), 1usize.into())));
        assert_eq!(iter.next(), Some((vec![3, 2].as_slice().into(), 2usize.into())));
        assert_eq!(iter.next(), Some((vec![3, 3].as_slice().into(), 1usize.into())));
        assert_eq!(iter.next(), None);
    }
}
