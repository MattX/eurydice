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
use malachite::Natural;
use malachite::base::num::arithmetic::traits::{DivExact, Factorial, Lcm, Pow};
use malachite::base::num::basic::traits::{One, Zero};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::rc::Rc;
use std::{collections::HashMap, fmt::Debug, hash::Hash, sync::RwLock};

/// Represents a pool of identical independent dice whose faces have type `T`.
///
/// The Icepool algorithm only needs outcomes to have a stable total order and
/// equality. Numeric summation is provided separately for `Pool<i32>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Pool<T = i32> {
    dimension: u32,
    // Outcomes are ordered by their face value. The tuple represents (value, weight / count).
    // Outcomes must be unique and have nonzero weight.
    ordered_outcomes: Vec<(T, Natural)>,
}

/// Cache key for the Icepool algorithm. `n` is the number of dice remaining, and
/// `remaining_count` is the number of outcomes remaining.
///
/// The outcomes that remain in consideration are the `remaining_count` smallest outcomes.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
struct SubPool {
    dimension: u32,
    remaining_outcomes: usize,
}

impl SubPool {
    fn initial<T>(pool: &Pool<T>) -> Self {
        Self {
            dimension: pool.dimension,
            remaining_outcomes: pool.ordered_outcomes.len(),
        }
    }
}

impl Pool<i32> {
    /// Creates a new pool of `n` dice, each with `sides` sides.
    pub fn ndn(dimension: u32, sides: u32) -> Self {
        Self {
            dimension,
            ordered_outcomes: (1..=sides)
                .map(|side| (i32::try_from(side).expect("side is > 1"), 1usize.into()))
                .collect::<Vec<_>>(),
        }
    }

    /// Sums the distribution; the resulting pool is guaranteed to have dimension 1.
    pub fn sum(&self) -> Pool<i32> {
        if self.dimension == 0 {
            return self.clone();
        }
        if self.dimension == 1 && !self.ordered_outcomes.is_empty() {
            return self.clone();
        }
        self.sum_by(0, sum_mapper)
    }

    pub fn sum_with_keep_list(&self, keep_list: &[bool]) -> Pool<i32> {
        self.sum_with_keep_list_by(keep_list, 0, sum_mapper)
    }
}

impl<T> Pool<T>
where
    T: Clone + Ord,
{
    /// Creates a new pool from a list of outcomes. Repeats are allowed and will count as multiple weights.
    pub fn from_list(dimension: u32, outcomes: Vec<T>) -> Self {
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

    /// Sets the number of dice in the pool.
    pub fn set_dimension(&mut self, dimension: u32) {
        self.dimension = dimension;
    }

    /// Gets the number of dice in the pool.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Maps the outcomes of the pool using the given function. The function can be non-injective,
    /// in which case the weights of the outcomes are summed.
    pub fn map_outcomes<U>(self, f: impl Fn(T) -> U) -> Pool<U>
    where
        U: Clone + Ord,
    {
        let mut new_outcomes = BTreeMap::new();
        for (outcome, weight) in self.ordered_outcomes.into_iter() {
            *new_outcomes
                .entry(f(outcome))
                .or_insert(Natural::from(0usize)) += weight;
        }
        Pool {
            ordered_outcomes: new_outcomes.into_iter().collect(),
            dimension: self.dimension,
        }
    }

    /// Fallible counterpart to `map_outcomes`.
    pub fn try_map_outcomes<U, E>(self, f: impl Fn(T) -> Result<U, E>) -> Result<Pool<U>, E>
    where
        U: Clone + Ord,
    {
        let mut new_outcomes = BTreeMap::new();
        for (outcome, weight) in self.ordered_outcomes {
            *new_outcomes.entry(f(outcome)?).or_insert(Natural::ZERO) += weight;
        }
        Ok(Pool {
            ordered_outcomes: new_outcomes.into_iter().collect(),
            dimension: self.dimension,
        })
    }

    /// Combines weighted component distributions without allowing a
    /// component's arbitrary total weight to bias the mixture.
    ///
    /// Empty components are ignored. Each component must already represent a
    /// distribution (conventionally a dimension-one pool).
    pub(crate) fn from_mixture(components: impl IntoIterator<Item = (Natural, Pool<T>)>) -> Self {
        let components = components
            .into_iter()
            .filter_map(|(outer_weight, distribution)| {
                let total = distribution
                    .ordered_outcomes
                    .iter()
                    .map(|(_, weight)| weight)
                    .sum::<Natural>();
                (outer_weight != 0 && total != 0).then_some((outer_weight, distribution, total))
            })
            .collect::<Vec<_>>();
        let common_denominator = components
            .iter()
            .fold(Natural::ONE, |lcm, (_, _, total)| lcm.lcm(total));
        components
            .into_iter()
            .flat_map(|(outer_weight, distribution, total)| {
                let scale = common_denominator.clone().div_exact(total);
                distribution
                    .ordered_outcomes
                    .into_iter()
                    .map(move |(outcome, weight)| (outcome, &outer_weight * weight * &scale))
            })
            .collect()
    }

    pub fn apply<S, F>(&self, mapper: StateMapper<S, F>, keep_list: &[bool]) -> HashMap<S, Natural>
    where
        S: Clone + Hash + Eq,
        F: Fn(&S, &T, u32) -> S,
    {
        let StateMapper { initial_state, f } = mapper;
        let mapper = StateMapper {
            initial_state,
            f: |state: &S, outcome: &T, count| Ok::<_, Infallible>(f(state, outcome, count)),
        };
        match self.try_apply(mapper, keep_list) {
            Ok(states) => states,
            Err(never) => match never {},
        }
    }

    /// [`Self::apply`] with a state function that can fail.
    ///
    /// The first failure abandons the walk, so the error names the outcome that
    /// caused it rather than a state some way downstream of it.
    pub fn try_apply<S, F, E>(
        &self,
        mapper: StateMapper<S, F>,
        keep_list: &[bool],
    ) -> Result<HashMap<S, Natural>, E>
    where
        S: Clone + Hash + Eq,
        F: Fn(&S, &T, u32) -> Result<S, E>,
    {
        debug_assert_eq!(
            keep_list.len(),
            self.dimension as usize,
            "`apply` called with keep list of incorrect length"
        );
        if self.is_empty() {
            return Ok([(mapper.initial_state.clone(), Natural::ONE)].into());
        }
        let mut cache = HashMap::new();
        self.apply_inner(SubPool::initial(self), &mut cache, &mapper, keep_list)
    }

    fn apply_inner<S, F, E>(
        &self,
        sub_pool: SubPool,
        cache: &mut HashMap<SubPool, HashMap<S, Natural>>,
        mapper: &StateMapper<S, F>,
        keep_list: &[bool],
    ) -> Result<HashMap<S, Natural>, E>
    where
        S: Clone + Hash + Eq,
        F: Fn(&S, &T, u32) -> Result<S, E>,
    {
        if let Some(value) = cache.get(&sub_pool) {
            return Ok(value.clone());
        }
        let new_remaining_outcomes = sub_pool.remaining_outcomes - 1;
        let (outcome, weight) = &self.ordered_outcomes[new_remaining_outcomes];
        let result = if new_remaining_outcomes == 0 {
            let num_kept = self.num_kept(keep_list, sub_pool, sub_pool.dimension);
            [(
                (mapper.f)(&mapper.initial_state, outcome, num_kept)?,
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
                let sub_sub_pool_result =
                    self.apply_inner(sub_sub_pool, cache, mapper, keep_list)?;
                for (state, count) in sub_sub_pool_result {
                    let inner_state = (mapper.f)(&state, outcome, num_kept)?;
                    // There were binom(self.n, num_with_outcome) ways to get this outcome,
                    // times weight^num_with_outcome if the weight is >1.
                    *result.entry(inner_state).or_default() += count
                        * binom(sub_pool.dimension as usize, num_with_outcome as usize)
                        * weight.pow(num_with_outcome as u64);
                }
            }
            result
        };
        // Only successful sub-walks are cached; a failure abandons the whole
        // walk, so there is nothing for a later lookup to reuse.
        cache.insert(sub_pool, result.clone());
        Ok(result)
    }

    fn num_kept(&self, keep_list: &[bool], sub_pool: SubPool, num_with_outcome: u32) -> u32 {
        let count = keep_list
            [(sub_pool.dimension - num_with_outcome) as usize..sub_pool.dimension as usize]
            .iter()
            .filter(|&&keep| keep)
            .count();
        u32::try_from(count).expect("count greater than max u32")
    }

    pub fn ordered_outcomes(&self) -> &[(T, Natural)] {
        &self.ordered_outcomes
    }

    /// Tests whether any die has the given outcome using the Icepool algorithm.
    ///
    /// The resulting pool contains only zero and one and is guaranteed to have
    /// dimension 1.
    pub fn contains(&self, needle: &T) -> Pool<i32> {
        let keep_list = vec![true; self.dimension as usize];
        self.apply(
            StateMapper {
                initial_state: 0,
                f: |found: &i32, outcome: &T, count| {
                    found | i32::from(count > 0 && outcome == needle)
                },
            },
            &keep_list,
        )
        .into_iter()
        .collect()
    }

    /// Counts dice whose outcomes occur in `needles` using the Icepool algorithm.
    ///
    /// Repeated needles count repeatedly. The resulting pool is guaranteed to
    /// have dimension 1.
    pub fn count(&self, needles: &[T]) -> Pool<i32> {
        let mut needle_counts = BTreeMap::new();
        for needle in needles {
            *needle_counts.entry(needle.clone()).or_insert(0i32) += 1;
        }
        let keep_list = vec![true; self.dimension as usize];
        self.apply(
            StateMapper {
                initial_state: 0,
                f: move |state: &i32, outcome: &T, count| {
                    let count = i32::try_from(count).expect("pool dimension fits in i32");
                    state + needle_counts.get(outcome).copied().unwrap_or(0) * count
                },
            },
            &keep_list,
        )
        .into_iter()
        .collect()
    }

    /// Sums every die in this pool using an arbitrary additive state.
    ///
    /// `add_scaled` receives the current state, one face, and the number of
    /// dice showing that face. This keeps the Icepool summation path generic
    /// without requiring the face type itself to implement numeric traits.
    pub(crate) fn sum_by<S, F>(&self, identity: S, add_scaled: F) -> Pool<S>
    where
        S: Clone + Eq + Hash + Ord,
        F: Fn(&S, &T, u32) -> S,
    {
        if self.is_empty() {
            return Pool::from_list(1, vec![identity]);
        }
        let keep_list = vec![true; self.dimension as usize];
        self.sum_with_keep_list_by(&keep_list, identity, add_scaled)
    }

    /// [`Self::sum_by`] with an addition that can fail.
    pub(crate) fn try_sum_by<S, F, E>(&self, identity: S, add_scaled: F) -> Result<Pool<S>, E>
    where
        S: Clone + Eq + Hash + Ord,
        F: Fn(&S, &T, u32) -> Result<S, E>,
    {
        if self.is_empty() {
            return Ok(Pool::from_list(1, vec![identity]));
        }
        let keep_list = vec![true; self.dimension as usize];
        self.try_sum_with_keep_list_by(&keep_list, identity, add_scaled)
    }

    /// The generic counterpart of `sum_with_keep_list`.
    pub(crate) fn sum_with_keep_list_by<S, F>(
        &self,
        keep_list: &[bool],
        identity: S,
        add_scaled: F,
    ) -> Pool<S>
    where
        S: Clone + Eq + Hash + Ord,
        F: Fn(&S, &T, u32) -> S,
    {
        if self.is_empty() {
            return Pool::from_list(1, vec![identity]);
        }
        self.apply(
            StateMapper {
                initial_state: identity,
                f: add_scaled,
            },
            keep_list,
        )
        .into_iter()
        .collect()
    }

    /// [`Self::sum_with_keep_list_by`] with an addition that can fail.
    pub(crate) fn try_sum_with_keep_list_by<S, F, E>(
        &self,
        keep_list: &[bool],
        identity: S,
        add_scaled: F,
    ) -> Result<Pool<S>, E>
    where
        S: Clone + Eq + Hash + Ord,
        F: Fn(&S, &T, u32) -> Result<S, E>,
    {
        if self.is_empty() {
            return Ok(Pool::from_list(1, vec![identity]));
        }
        Ok(self
            .try_apply(
                StateMapper {
                    initial_state: identity,
                    f: add_scaled,
                },
                keep_list,
            )?
            .into_iter()
            .collect())
    }

    pub fn multiset_iterator(&self) -> PoolMultisetIterator<'_, T> {
        PoolMultisetIterator::new(self)
    }

    /// Calls `f` with every multiset outcome and combines the returned
    /// distributions without bias from their relative weight scales.
    pub fn flat_map<U, F>(&self, f: F) -> Pool<U>
    where
        U: Clone + Ord,
        F: Fn(&[T]) -> BTreeMap<U, Natural>,
    {
        let components = self
            .multiset_iterator()
            .map(|(outcome, ways)| (ways, f(&outcome).into_iter().collect()));
        Pool::from_mixture(components)
    }

    /// [`Self::flat_map`] with a mapping that can fail.
    pub fn try_flat_map<U, F, E>(&self, f: F) -> Result<Pool<U>, E>
    where
        U: Clone + Ord,
        F: Fn(&[T]) -> Result<BTreeMap<U, Natural>, E>,
    {
        let components = self
            .multiset_iterator()
            .map(|(outcome, ways)| Ok((ways, f(&outcome)?.into_iter().collect())))
            .collect::<Result<Vec<_>, E>>()?;
        Ok(Pool::from_mixture(components))
    }

    /// Maps multiset outcomes to a single value each.
    pub fn map<U, F>(&self, f: F) -> Pool<U>
    where
        U: Clone + Ord,
        F: Fn(&[T]) -> U,
    {
        self.flat_map(|outcome| BTreeMap::from([(f(outcome), 1usize.into())]))
    }

    pub fn is_empty(&self) -> bool {
        self.ordered_outcomes.is_empty() || self.dimension == 0
    }
}

/// Iterator over multisets of outcomes in a pool, with their weight.
pub struct PoolMultisetIterator<'a, T = i32> {
    pool: &'a Pool<T>,
    positions: Vec<usize>,
    // Factorial of the number of outcomes in the dice pool.
    factorial: Natural,
    // Whether we're finished; if true, then |positions| may contain invalid indices.
    done: bool,
}

impl<'a, T> PoolMultisetIterator<'a, T> {
    fn new(pool: &'a Pool<T>) -> Self {
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

impl<T> Iterator for PoolMultisetIterator<'_, T>
where
    T: Clone + Eq,
{
    type Item = (Vec<T>, Natural);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pool.ordered_outcomes.is_empty() || self.pool.dimension == 0 {
            return None;
        }
        let outcome = self
            .positions
            .iter()
            .map(|&i| self.pool.ordered_outcomes[i].0.clone())
            .collect::<Vec<T>>();

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
pub struct MultisetCrossProductIterator<'a, T = i32> {
    sub_iterators: Vec<PoolMultisetIterator<'a, T>>,
    started: bool,
    /// None if the iterator is finished. If Some, the outer vector has the same
    /// length as `sub_iterators`.
    values: Option<Vec<(Rc<Vec<T>>, Natural)>>,
}

impl<'a, T> MultisetCrossProductIterator<'a, T>
where
    T: Clone + Eq,
{
    pub fn new(sub_iterators: Vec<PoolMultisetIterator<'a, T>>) -> Self {
        let mut result = Self {
            sub_iterators,
            started: false,
            values: None,
        };
        let values: Option<Vec<_>> = result
            .sub_iterators
            .iter_mut()
            .map(|iter| iter.next().map(|(outcome, ways)| (Rc::new(outcome), ways)))
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
        let Some(values) = self.values.as_mut() else {
            return;
        };
        let mut stopped = false;
        for (idx, iterator) in self.sub_iterators.iter_mut().enumerate() {
            let (new_val, cont) = match iterator.next() {
                Some((outcome, ways)) => ((Rc::new(outcome), ways), false),
                None => {
                    iterator.reset();
                    let Some((outcome, ways)) = iterator.next() else {
                        self.values = None;
                        return;
                    };
                    ((Rc::new(outcome), ways), true)
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

impl<T> Iterator for MultisetCrossProductIterator<'_, T>
where
    T: Clone + Eq,
{
    type Item = (Vec<Rc<Vec<T>>>, Natural);

    fn next(&mut self) -> Option<Self::Item> {
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
/// factorial(number of same outcomes). The result is the product of all these
/// factorials.
fn item_factorials<T: Eq>(outcome: &[T]) -> Natural {
    let mut product = Natural::ONE;
    let mut count = 1u64;
    for pair in outcome.windows(2) {
        if pair[0] == pair[1] {
            count += 1;
        } else {
            product *= Natural::factorial(count);
            count = 1;
        }
    }
    product *= Natural::factorial(count);
    product
}

impl<T: Ord> FromIterator<(T, Natural)> for Pool<T> {
    fn from_iter<I: IntoIterator<Item = (T, Natural)>>(iter: I) -> Self {
        let mut outcomes = BTreeMap::new();
        for (outcome, weight) in iter {
            if weight != 0 {
                *outcomes.entry(outcome).or_insert(Natural::ZERO) += weight;
            }
        }
        Self {
            dimension: 1,
            ordered_outcomes: outcomes.into_iter().collect(),
        }
    }
}

impl<T: Ord> From<Vec<(T, Natural)>> for Pool<T> {
    fn from(ordered_outcomes: Vec<(T, Natural)>) -> Self {
        ordered_outcomes.into_iter().collect()
    }
}

impl<T: Ord> From<Pool<T>> for BTreeMap<T, Natural> {
    fn from(pool: Pool<T>) -> Self {
        pool.ordered_outcomes.into_iter().collect()
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

pub fn reroll(die: Vec<(i32, Natural)>, on: &[i32], depth: usize) -> Vec<(i32, Natural)> {
    if depth == 0 {
        return die;
    }

    let inner_reroll: Pool = reroll(die.clone(), on, depth - 1).into();
    let die_dist: Pool = die.into_iter().collect::<Pool>().flat_map(|outcome| {
        let roller = outcome[0];
        if on.contains(&roller) {
            inner_reroll.clone().into()
        } else {
            vec![(roller, Natural::ONE)].into_iter().collect()
        }
    });
    die_dist.ordered_outcomes
}

pub struct StateMapper<S, F> {
    initial_state: S,
    f: F,
}

fn sum_mapper(state: &i32, outcome: &i32, count: u32) -> i32 {
    state + outcome * i32::try_from(count).expect("count fits in i32")
}

lazy_static! {
    /// Cache for binomial coefficients. Rows are either missing or fully calculated.
    static ref BINOM_CACHE: RwLock<Vec<Vec<Natural>>> = RwLock::new(vec![vec![Natural::ONE]]);
}

/// Calculate binomial coefficient n choose k, with value caching.
///
/// Panics if k > n.
// Pascal rows use more memory than cached factorials, but make repeated
// coefficient lookups addition-only and avoid large-integer division.
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

    /// Mapper that sums the outcomes. Production summation goes through
    /// [`Pool::sum`] and its generic counterparts; this is the bare mapper the
    /// tests below drive [`Pool::apply`] with directly.
    #[allow(clippy::type_complexity)]
    const SUM_MAPPER: StateMapper<i32, fn(&i32, &i32, u32) -> i32> = StateMapper {
        initial_state: 0,
        f: sum_mapper,
    };

    #[test]
    fn test_binom() {
        // Degenerate rows and columns, then values large enough to catch an
        // off-by-one in the recurrence.
        for (n, k, expected) in [
            (0, 0, 1u64),
            (1, 0, 1),
            (1, 1, 1),
            (2, 1, 2),
            (3, 1, 3),
            (3, 2, 3),
            (4, 2, 6),
            (10, 5, 252),
            (20, 10, 184756),
            (100, 0, 1),
            (100, 100, 1),
        ] {
            assert_eq!(binom(n, k), expected, "binom({n}, {k})");
        }
    }

    fn to_counter<T: Hash + Eq>(v: Vec<(T, usize)>) -> HashMap<T, Natural> {
        v.into_iter().map(|(k, v)| (k, Natural::from(v))).collect()
    }

    #[test]
    fn weighted_pool_construction_canonicalizes_outcomes() {
        let pool = Pool::from(vec![
            (2, Natural::from(3u32)),
            (1, Natural::ZERO),
            (2, Natural::from(4u32)),
            (1, Natural::from(2u32)),
        ]);

        assert_eq!(
            pool.ordered_outcomes(),
            &[(1, Natural::from(2u32)), (2, Natural::from(7u32)),]
        );
    }

    #[test]
    fn weighted_mixture_normalizes_each_component() {
        let result = Pool::from_mixture([
            (Natural::ONE, Pool::from_list(1, vec![1, 2])),
            (Natural::ONE, Pool::from_list(1, vec![1, 2, 3, 4])),
            (Natural::ONE, Pool::from_list(1, Vec::<i32>::new())),
        ]);

        assert_eq!(
            result.ordered_outcomes(),
            &[
                (1, Natural::from(3u32)),
                (2, Natural::from(3u32)),
                (3, Natural::ONE),
                (4, Natural::ONE),
            ]
        );
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
        impl Fn(&MaxDiceToReachState, &i32, u32) -> MaxDiceToReachState,
    > {
        StateMapper {
            initial_state: (Some(0), 0),
            f: move |state: &MaxDiceToReachState, outcome: &i32, count| {
                let count = i32::try_from(count).expect("pool dimension fits in i32");
                let (sum, rolls) = *state;
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
        assert!(pool.is_empty());
        assert_eq!(pool.dimension(), 10);
        assert_eq!(pool.multiset_iterator().next(), None);
        assert_eq!(pool.sum().ordered_outcomes(), &[(0, Natural::ONE)]);
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
        assert!(pool.is_empty());
        assert_eq!(pool.dimension(), 0);
        assert_eq!(pool.ordered_outcomes().len(), 3);
        assert_eq!(pool.multiset_iterator().next(), None);
        assert_eq!(pool.sum(), pool);
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
            .map(|(i, w)| (i, Natural::from(u32::try_from(w).unwrap())))
            .collect::<Vec<_>>()
        );
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
        .map(|(i, w)| (i, Natural::from(u32::try_from(w).unwrap())))
        .collect::<HashMap<_, _>>();
        assert_eq!(map, expected);
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum SymbolicOutcome {
        Miss,
        Hit,
    }

    #[test]
    fn test_generic_pool_icepool_application() {
        let pool = Pool::from_list(
            2,
            vec![
                SymbolicOutcome::Miss,
                SymbolicOutcome::Hit,
                SymbolicOutcome::Hit,
            ],
        );
        let mapper = StateMapper {
            initial_state: 0,
            f: |hits: &u32, outcome: &SymbolicOutcome, count| {
                hits + u32::from(*outcome == SymbolicOutcome::Hit) * count
            },
        };

        let result = pool.apply(mapper, &[true, true]);

        assert_eq!(
            result,
            [
                (0, Natural::from(1u32)),
                (1, Natural::from(4u32)),
                (2, Natural::from(4u32)),
            ]
            .into()
        );
    }

    #[test]
    fn test_pool_count_uses_needle_multiplicity() {
        let pool = Pool::from_list(
            2,
            vec![
                SymbolicOutcome::Miss,
                SymbolicOutcome::Hit,
                SymbolicOutcome::Hit,
            ],
        );

        assert_eq!(
            pool.count(&[SymbolicOutcome::Hit, SymbolicOutcome::Hit])
                .ordered_outcomes(),
            &[
                (0, Natural::from(1u32)),
                (2, Natural::from(4u32)),
                (4, Natural::from(4u32)),
            ]
        );
        assert_eq!(
            pool.count(&[]).ordered_outcomes(),
            &[(0, Natural::from(9u32))]
        );
        assert_eq!(
            Pool::from_list(0, vec![SymbolicOutcome::Hit])
                .count(&[SymbolicOutcome::Hit])
                .ordered_outcomes(),
            &[(0, Natural::ONE)]
        );
    }

    #[test]
    fn test_generic_pool_maps_to_tuple_outcomes() {
        let pool = Pool::from_list(
            1,
            vec![
                SymbolicOutcome::Miss,
                SymbolicOutcome::Hit,
                SymbolicOutcome::Hit,
            ],
        );

        let result: Pool<(u8, u8)> = pool.map(|outcomes| match outcomes[0] {
            SymbolicOutcome::Miss => (1, 0),
            SymbolicOutcome::Hit => (0, 1),
        });

        assert_eq!(
            result.ordered_outcomes(),
            &[((0, 1), Natural::from(2u32)), ((1, 0), Natural::from(1u32)),]
        );
    }

    #[test]
    // Computes (d3 @ d4) in AnyDice or Icepool notation.
    fn test_flat_map() {
        let pool1 = Pool::from_list(1, vec![1, 2, 3]);
        let pool2 = Pool::from_list(1, vec![1, 2, 3, 4]);
        let result = pool1.flat_map(|outcome| {
            let mut summed_pool = pool2.clone();
            summed_pool.set_dimension(u32::try_from(outcome[0]).unwrap());
            summed_pool.sum().into()
        });
        let map = result
            .ordered_outcomes
            .into_iter()
            .collect::<HashMap<_, _>>();
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
        .map(|(i, w)| (i, Natural::from(u32::try_from(w).unwrap())))
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
    fn test_multiset_cross_product_dim1() {
        let pools = [
            Pool::from_list(1, vec![1, 2, 3]),
            Pool::from_list(1, vec![4, 5, 6]),
        ];
        let iter = MultisetCrossProductIterator::new(
            pools.iter().map(|pool| pool.multiset_iterator()).collect(),
        );
        let values = iter.collect::<Vec<_>>();
        let values_only = values
            .iter()
            .map(|(v, _)| v.iter().map(|o| (**o).clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
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
        let values_only = values
            .iter()
            .map(|(v, _)| v.iter().map(|o| (**o).clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
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
