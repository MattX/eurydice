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

/// Represents a pool of identical independent dice.
#[derive(Debug, Clone)]
pub struct Pool {
    dimension: u32,
    // Outcomes are ordered by their face value. The tuple represents (value, weight / count).
    // Outcomes must be unique and have nonzero weight.
    ordered_outcomes: Vec<(i32, Natural)>,
}

impl std::fmt::Display for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dimension != 1 {
            write!(f, "{}", self.dimension)?;
        }
        if self
            .ordered_outcomes
            .iter()
            .enumerate()
            .all(|(i, (outcome, weight))| {
                usize::try_from(*outcome).ok() == Some(i + 1) && *weight == Natural::ONE
            })
        {
            // This is a standard dn with no repeats.
            write!(f, "d{}", self.ordered_outcomes.len())
        } else {
            write!(
                f,
                "d{{{}}}",
                self.ordered_outcomes
                    .iter()
                    .map(|(outcome, weight)| if weight == &Natural::ONE {
                        outcome.to_string()
                    } else {
                        format!("{}:{}", outcome, weight)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

/// Cache key for the Icepool algorithm. `n` is the number of dice remaining, and
/// `remaining_count` is the number of outcomes remaining.
///
/// The outcomes that remain in consideration are the `remaining_count` smallest outcomes.`
#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
struct SubPool {
    dimension: u32,
    remaining_outcomes: usize,
}

impl SubPool {
    fn initial(pool: &Pool) -> Self {
        Self {
            dimension: pool.dimension,
            remaining_outcomes: pool.ordered_outcomes.len(),
        }
    }
}

impl Pool {
    /// Creates a new pool of `n` dice, each with `sides` sides.
    pub fn ndn(dimension: u32, sides: u32) -> Self {
        Self {
            dimension,
            ordered_outcomes: (1..=sides)
                .map(|side| (i32::try_from(side).expect("side is > 1"), 1usize.into()))
                .collect::<Vec<_>>(),
        }
    }

    /// Creates a new pool from a list of outcomes. Repeats are allowed and will count as multiple weights.
    pub fn from_list(dimension: u32, outcomes: Vec<i32>) -> Self {
        let mut outcomes_map = BTreeMap::new();
        for outcome in outcomes {
            *outcomes_map.entry(outcome).or_insert(Natural::ZERO) += Natural::ONE;
        }
        let ordered_outcomes = outcomes_map.into_iter().collect::<Vec<_>>();
        Self {
            dimension,
            ordered_outcomes,
        }
    }

    /// Sets the number of dice in the pool, resetting the keep list.
    ///
    /// Panics if `n` is negative.
    pub fn set_dimension(&mut self, dimension: u32) {
        self.dimension = dimension;
    }

    /// Gets the number of dice in the pool.
    pub fn dimension(&self) -> u32 {
        self.dimension
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
        ordered_outcomes.sort_unstable();
        Self {
            dimension: 1,
            ordered_outcomes,
        }
    }

    pub fn apply<S, F>(&self, mapper: StateMapper<S, F>, keep_list: &[bool]) -> HashMap<S, Natural>
    where
        S: Clone + Hash + Eq,
        F: Fn(&S, i32, u32) -> S,
    {
        debug_assert_eq!(
            keep_list.len(),
            self.dimension as usize,
            "`apply` called with keep list of incorrect length"
        );
        if self.ordered_outcomes.is_empty() {
            return [(mapper.initial_state.clone(), Natural::ONE)].into();
        }
        let mut cache = HashMap::new();
        self.apply_inner(SubPool::initial(self), &mut cache, &mapper, keep_list)
    }

    fn apply_inner<S, F>(
        &self,
        sub_pool: SubPool,
        cache: &mut HashMap<SubPool, HashMap<S, Natural>>,
        mapper: &StateMapper<S, F>,
        keep_list: &[bool],
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
            let num_kept = self.num_kept(keep_list, sub_pool, sub_pool.dimension);
            [(
                (mapper.f)(&mapper.initial_state, *outcome, num_kept),
                weight.pow(sub_pool.dimension as u64),
            )]
            .into()
        } else {
            let mut result = HashMap::new();
            for num_with_outcome in 0..=sub_pool.dimension {
                // Replace num_with_outcome with the actual number of dice to keep in the considered range.
                // Ignore anything in the keep list above index `sub_pool.n`, and below `sub_pool.n - num_with_outcome`.
                let num_kept = self.num_kept(keep_list, sub_pool, num_with_outcome);

                let sub_sub_pool = SubPool {
                    dimension: sub_pool.dimension - num_with_outcome,
                    remaining_outcomes: new_remaining_outcomes,
                };
                let sub_sub_pool_result = self.apply_inner(sub_sub_pool, cache, mapper, keep_list);
                for (state, count) in sub_sub_pool_result {
                    let inner_state = (mapper.f)(&state, *outcome, num_kept);
                    // There were binom(self.n, num_with_outcome) ways to get this outcome,
                    // times weight^num_with_outcome if the weight is >1.
                    *result.entry(inner_state).or_default() += count
                        * binom(sub_pool.dimension as usize, num_with_outcome as usize)
                        * weight.pow(num_with_outcome as u64);
                }
            }
            result
        };
        cache.insert(sub_pool, result.clone());
        result
    }

    fn num_kept(&self, keep_list: &[bool], sub_pool: SubPool, num_with_outcome: u32) -> u32 {
        let count = keep_list
            [(sub_pool.dimension - num_with_outcome) as usize..sub_pool.dimension as usize]
            .iter()
            .filter(|&&keep| keep)
            .count();
        u32::try_from(count).expect("count greater than max u32")
    }

    /// Sums the distribution; the resulting pool is guaranteed to have dimension 1.
    pub fn sum(&self) -> Pool {
        if self.dimension == 0 || self.ordered_outcomes.is_empty() {
            return Pool {
                dimension: 1,
                ordered_outcomes: vec![(0, Natural::ONE)],
            };
        } else if self.dimension == 1 {
            return self.clone();
        }
        let keep_list = vec![true; self.dimension as usize];
        self.apply(SUM_MAPPER, &keep_list).into_iter().collect()
    }

    pub fn sum_with_keep_list(&self, keep_list: &[bool]) -> Pool {
        self.apply(SUM_MAPPER, keep_list).into_iter().collect()
    }

    pub fn into_die_iter(self) -> impl Iterator<Item = (i32, Natural)> {
        self.ordered_outcomes.into_iter()
    }

    pub fn ordered_outcomes(&self) -> &[(i32, Natural)] {
        &self.ordered_outcomes
    }

    pub fn multiset_iterator(&self) -> PoolMultisetIterator {
        PoolMultisetIterator::new(self)
    }

    /// This functions call `f` with each multiset outcome from the pool. The distributions returned
    /// by `f` are flatmapped together to create a new distribution, stored as a size-1 pool.
    pub fn flat_map<F>(&self, f: F) -> Self
    where
        F: Fn(&[i32]) -> BTreeMap<i32, Natural>,
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
            let f_outcome = f(&outcome);
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
        F: Fn(&[i32]) -> i32,
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
            positions: vec![0; pool.dimension as usize],
            factorial: Natural::factorial(pool.dimension as u64),
            done: false,
        }
    }

    pub fn reset(&mut self) {
        self.positions = vec![0; self.pool.dimension as usize];
        self.done = false;
    }

    fn advance_position(&mut self) {
        let pool_size = isize::try_from(self.pool.dimension).expect("pool dimension fits in isize");
        let mut position_index = pool_size - 1;
        while position_index >= 0 {
            let position_index_u =
                usize::try_from(position_index).expect("position_index is a positive isize");
            self.positions[position_index_u] += 1;
            if self.positions[position_index_u] == self.pool.ordered_outcomes.len() {
                if position_index_u == 0 {
                    self.done = true;
                    return;
                }
                position_index -= 1;
            } else {
                break;
            }
        }
        let first_position_index =
            usize::try_from(position_index + 1).expect("position_index is >= -1");
        let pool_size = usize::try_from(pool_size).expect("pool_size is positive");
        // Now go back and fix up the position indices
        for i in first_position_index..pool_size {
            self.positions[i] = self.positions[i - 1];
        }
    }
}

impl<'a> Iterator for PoolMultisetIterator<'a> {
    type Item = (Vec<i32>, Natural);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pool.ordered_outcomes.is_empty() || self.pool.dimension == 0 {
            return None;
        }
        let outcome = self
            .positions
            .iter()
            .map(|&i| self.pool.ordered_outcomes[i].0)
            .collect::<Vec<i32>>();

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

/// An interator over the cross product of several PoolMultisetIterators.
pub struct MultisetCrossProductIterator<'a> {
    sub_iterators: Vec<PoolMultisetIterator<'a>>,
    started: bool,
    /// None if the iterator is finished. If Some, the outer vector has the same
    /// length as `sub_iterators`.
    values: Option<Vec<(Vec<i32>, Natural)>>,
}

impl<'a> MultisetCrossProductIterator<'a> {
    pub fn new(sub_iterators: Vec<PoolMultisetIterator<'a>>) -> Self {
        let mut result = Self {
            sub_iterators,
            started: false,
            values: None,
        };
        let values: Option<Vec<_>> = result
            .sub_iterators
            .iter_mut()
            .map(|iter| iter.next())
            .collect();
        let values = match values {
            Some(values) => values,
            None => return result,
        };
        result.values = Some(values);
        result
    }

    fn advance_values(&mut self) {
        if self.values.is_none() {
            return;
        }
        let values = self.values.as_mut().unwrap();
        let mut stopped = false;
        for (idx, iterator) in self.sub_iterators.iter_mut().enumerate() {
            let (new_val, cont) = match iterator.next() {
                Some(val) => (val, false),
                None => {
                    iterator.reset();
                    // OK to unwrap here: if any iterators were empty after reset, we have gotten here
                    (iterator.next().unwrap(), true)
                }
            };
            values[idx] = new_val;
            if !cont {
                stopped = true;
                break;
            }
        }
        if !stopped {
            // If *every* iterator had to be reset, we've already gone through every permutation.
            self.values = None;
        }
    }
}

impl<'a> Iterator for MultisetCrossProductIterator<'a> {
    type Item = (Vec<Vec<i32>>, Natural);

    fn next(&mut self) -> Option<(Vec<Vec<i32>>, Natural)> {
        if !self.started {
            self.started = true;
        } else {
            self.advance_values();
        }
        match self.values {
            Some(ref values) => {
                let ways = values.iter().map(|(_, w)| w).product();
                let outcomes = values.iter().map(|(o, _)| o.clone()).collect();
                Some((outcomes, ways))
            }
            None => None,
        }
    }
}

/// For each group of consecutive equal values in the outcomes, this computes
/// factorial(numer of same outcomes). The result is the product of all these
/// factorials.
fn item_factorials(outcome: &[i32]) -> Natural {
    let mut product = Natural::ONE;
    let mut count = 1u64;
    for i in 1..outcome.len() {
        let prev = outcome.get(i - 1);
        if prev.is_some() && *prev.unwrap() == outcome[i] {
            count += 1;
        } else {
            product *= Natural::factorial(count);
            count = 1;
        }
    }
    product *= Natural::factorial(count);
    product
}

impl FromIterator<(i32, Natural)> for Pool {
    fn from_iter<I: IntoIterator<Item = (i32, Natural)>>(iter: I) -> Self {
        let mut ordered_outcomes = iter.into_iter().collect::<Vec<_>>();
        ordered_outcomes.sort_unstable();
        Self {
            dimension: 1,
            ordered_outcomes,
        }
    }
}

impl From<Vec<(i32, Natural)>> for Pool {
    fn from(ordered_outcomes: Vec<(i32, Natural)>) -> Self {
        Self {
            dimension: 1,
            ordered_outcomes,
        }
    }
}

impl From<Pool> for BTreeMap<i32, Natural> {
    fn from(pool: Pool) -> Self {
        pool.into_die_iter().collect()
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
#[allow(clippy::type_complexity)]
pub const SUM_MAPPER: StateMapper<i32, fn(&i32, i32, u32) -> i32> = StateMapper {
    initial_state: 0,
    f: sum_mapper,
};

lazy_static! {
    /// Cache for binomial coefficients. Rows are either missing or fully calculated.
    static ref BINOM_CACHE: RwLock<Vec<Vec<Natural>>> = RwLock::new(vec![vec![Natural::ONE]]);
}

/// Calculate binomial coefficient n choose k, with value caching.
///
/// Panics if k > n.
// TODO - this can likely be made faster / smaller by only caching factorials.
#[allow(clippy::needless_range_loop)]
fn binom(n: usize, k: usize) -> Natural {
    if n == k {
        return Natural::ONE;
    }
    if k == 0 {
        return Natural::ONE;
    }
    let mut binom_cache = BINOM_CACHE.write().unwrap();
    let max_row = binom_cache.len();
    if n < max_row {
        // This may panic if k > n, but that's fine - caller should have known better.
        return binom_cache[n][k].clone();
    }
    for row_idx in max_row..=n {
        let mut row: Vec<Natural> = vec![Natural::ONE; row_idx + 1];
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
        let keep_list = vec![true; 10];
        let result = pool.apply(SUM_MAPPER, &keep_list);
        assert_eq!(result.len(), 191);
        assert_eq!(result[&133], Natural::from(70942066700u64));
    }

    #[test]
    fn test_sum_1d6() {
        let pool = Pool::ndn(1, 6);
        let keep_list = vec![true; 1];
        let result = pool.apply(SUM_MAPPER, &keep_list);
        assert_eq!(
            result,
            to_counter(vec![(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)])
        );
    }

    #[test]
    fn test_sum_2d6() {
        let pool = Pool::ndn(2, 6);
        let keep_list = vec![true; 2];
        let result = pool.apply(SUM_MAPPER, &keep_list);
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
        let keep_list = vec![true; 2];
        let result = pool.apply(SUM_MAPPER, &keep_list);
        assert_eq!(
            result,
            to_counter(vec![(2, 1), (3, 2), (4, 3), (5, 2), (6, 1),])
        );
    }

    /// The first value represents the current sum, or None if the target is already reached.
    /// The second value represents the number of rolls made so far.
    type MaxDiceToReachState = (Option<i32>, i32);

    /// Returns a mapper that calculates the minimum number of dice to sum, starting from the
    /// lowest dice, to reach the target.
    #[allow(clippy::type_complexity)]
    pub fn make_max_dice_to_reach_mapper(
        target: i32,
    ) -> StateMapper<
        MaxDiceToReachState,
        impl Fn(&MaxDiceToReachState, i32, u32) -> MaxDiceToReachState,
    > {
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

    #[test]
    fn test_make_max_dice_to_reach_mapper() {
        let mapper = make_max_dice_to_reach_mapper(10);
        let keep_list = vec![true; 3];
        let result = Pool::ndn(3, 6).apply(mapper, &keep_list);
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
        let keep_list = vec![true; 3];
        let result = pool.apply(SUM_MAPPER, &keep_list);
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
        let keep_list = vec![true; 3];
        let result = pool.apply(SUM_MAPPER, &keep_list);
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
        let pool = Pool::ndn(6, 10);
        let keep_list = vec![false, false, false, true, true, true];
        let result = pool.apply(SUM_MAPPER, &keep_list);
        assert_eq!(result.len(), 28);
        assert_eq!(result[&15], Natural::from(16617u64));
    }

    #[test]
    fn test_sum_no_outcomes() {
        let pool = Pool::from_list(10, vec![]);
        let keep_list = vec![true; 10];
        let result = pool.apply(SUM_MAPPER, &keep_list);
        assert_eq!(
            result.into_iter().collect::<Vec<_>>(),
            vec![(0, Natural::ONE)]
        );
    }

    #[test]
    fn test_sum_0_dim() {
        let pool = Pool::from_list(0, vec![1, 2, 3]);
        let keep_list = vec![];
        let result = pool.apply(SUM_MAPPER, &keep_list);
        assert_eq!(
            result.into_iter().collect::<Vec<_>>(),
            vec![(0, Natural::ONE)]
        );
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
        fn multiset_to_int(multiset: &[i32]) -> i32 {
            let mut total = 0;
            for (idx, item) in multiset.iter().rev().enumerate() {
                total += (item - 1) * 5i32.pow(idx as u32);
            }
            total
        }

        let result = pool.map(multiset_to_int);

        let map = result.into_die_iter().collect::<HashMap<_, _>>();
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
            summed_pool.set_dimension(outcome[0] as u32);
            summed_pool
                .sum()
                .into_die_iter()
                .collect::<BTreeMap<_, _>>()
        });
        let map = result.into_die_iter().collect::<HashMap<_, _>>();
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
        assert_eq!(
            iter.next(),
            Some((vec![1, 1].as_slice().into(), 1usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![1, 2].as_slice().into(), 2usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![1, 3].as_slice().into(), 2usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![2, 2].as_slice().into(), 1usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![2, 3].as_slice().into(), 2usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![3, 3].as_slice().into(), 1usize.into()))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_multiset_iterator_weighted() {
        let pool = Pool::from_list(2, vec![1, 1, 2, 3, 3, 3]);
        let mut iter = pool.multiset_iterator();
        assert_eq!(
            iter.next(),
            Some((vec![1, 1].as_slice().into(), 4usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![1, 2].as_slice().into(), 4usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![1, 3].as_slice().into(), 12usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![2, 2].as_slice().into(), 1usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![2, 3].as_slice().into(), 6usize.into()))
        );
        assert_eq!(
            iter.next(),
            Some((vec![3, 3].as_slice().into(), 9usize.into()))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_multiset_iterator_0_dim() {
        let pool = Pool::from_list(0, vec![1, 2, 3]);
        let mut iter = pool.multiset_iterator();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_multiset_iterator_no_outcomes() {
        let pool = Pool::from_list(2, vec![]);
        let mut iter = pool.multiset_iterator();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_multiset_cross_product_dim1() {
        let pools = [
            Pool::from_list(1, vec![1, 2, 3]),
            Pool::from_list(1, vec![4, 5, 6]),
        ];
        let iter = MultisetCrossProductIterator::new(
            pools.iter().map(|pool| pool.multiset_iterator()).collect(),
        );
        let values = iter.collect::<Vec<_>>();
        let values_only = values.iter().map(|(v, _)| v).cloned().collect::<Vec<_>>();
        assert_eq!(
            values_only,
            vec![
                vec![vec![1], vec![4]],
                vec![vec![2], vec![4]],
                vec![vec![3], vec![4]],
                vec![vec![1], vec![5]],
                vec![vec![2], vec![5]],
                vec![vec![3], vec![5]],
                vec![vec![1], vec![6]],
                vec![vec![2], vec![6]],
                vec![vec![3], vec![6]],
            ]
        );
        let weights_only = values.iter().map(|(_, w)| w).collect::<Vec<_>>();
        assert_eq!(weights_only, vec![&Natural::ONE; 9])
    }

    #[test]
    fn test_multiset_cross_product_higher_dim() {
        let pools = [
            Pool::from_list(2, vec![1, 2]),
            Pool::from_list(3, vec![4, 5]),
        ];
        let iter = MultisetCrossProductIterator::new(
            pools.iter().map(|pool| pool.multiset_iterator()).collect(),
        );
        let values = iter.collect::<Vec<_>>();
        let values_only = values.iter().map(|(v, _)| v.clone()).collect::<Vec<_>>();
        assert_eq!(
            values_only,
            vec![
                vec![vec![1, 1], vec![4, 4, 4]],
                vec![vec![1, 2], vec![4, 4, 4]],
                vec![vec![2, 2], vec![4, 4, 4]],
                vec![vec![1, 1], vec![4, 4, 5]],
                vec![vec![1, 2], vec![4, 4, 5]],
                vec![vec![2, 2], vec![4, 4, 5]],
                vec![vec![1, 1], vec![4, 5, 5]],
                vec![vec![1, 2], vec![4, 5, 5]],
                vec![vec![2, 2], vec![4, 5, 5]],
                vec![vec![1, 1], vec![5, 5, 5]],
                vec![vec![1, 2], vec![5, 5, 5]],
                vec![vec![2, 2], vec![5, 5, 5]],
            ]
        );
        let weights_only = values.iter().map(|(_, w)| w).cloned().collect::<Vec<_>>();
        assert_eq!(
            weights_only,
            vec![
                Natural::from(1usize),
                2usize.into(),
                1usize.into(),
                3usize.into(),
                6usize.into(),
                3usize.into(),
                3usize.into(),
                6usize.into(),
                3usize.into(),
                1usize.into(),
                2usize.into(),
                1usize.into()
            ]
        );
    }

    #[test]
    fn test_multiset_cross_product_one_empy() {
        let pools = [
            Pool::from_list(2, vec![1, 2]),
            Pool::from_list(0, vec![4, 5, 6]),
        ];
        let mut iter = MultisetCrossProductIterator::new(
            pools.iter().map(|pool| pool.multiset_iterator()).collect(),
        );
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }
}
