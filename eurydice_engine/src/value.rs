//! Runtime values and their element-type behavior.

use std::{fmt::Write, rc::Rc};

use malachite::{Natural, base::num::basic::traits::One};

use crate::ast;
use crate::dice::Pool;
use crate::error::{NonAdditiveSubject, NonAdditiveSumError, RuntimeError};

/// One `enum:` declaration: a named domain of symbols.
///
/// Sets exist for display only. They carry no typing power: every symbol has
/// the same element type whatever set it was declared in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolSet {
    pub name: String,
    /// Global index of this set's first member. A set's members are contiguous.
    first: u32,
    len: u32,
}

/// Every symbol declared so far, in declaration order.
///
/// Symbol values are bare indices into this table, so the table is the only
/// place that knows a symbol's name and the set it came from. Keeping it out of
/// the values means an outcome is a plain `u32`: no reference counting in the
/// Icepool inner loops, and a derived `Ord` that already sorts symbols in
/// declaration order.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    names: Vec<String>,
    /// Parallel to `names`: which set each symbol belongs to.
    sets_by_symbol: Vec<u32>,
    sets: Vec<SymbolSet>,
}

impl SymbolTable {
    /// Declares a set, returning the global index of each member in order.
    pub(crate) fn define_set(&mut self, name: String, members: &[String]) -> Vec<u32> {
        let first = u32::try_from(self.names.len()).expect("symbol count fits in u32");
        let set = u32::try_from(self.sets.len()).expect("set count fits in u32");
        self.sets.push(SymbolSet {
            name,
            first,
            len: u32::try_from(members.len()).expect("member count fits in u32"),
        });
        for member in members {
            self.names.push(member.clone());
            self.sets_by_symbol.push(set);
        }
        (first..first + u32::try_from(members.len()).expect("member count fits in u32")).collect()
    }

    pub fn name(&self, symbol: u32) -> &str {
        self.names
            .get(usize::try_from(symbol).expect("symbol index fits in usize"))
            .map(String::as_str)
            .unwrap_or("<?>")
    }

    pub fn set_of(&self, symbol: u32) -> &SymbolSet {
        &self.sets[self.set_index(symbol)]
    }

    /// Which declared set a symbol belongs to. Sets are numbered in
    /// declaration order, so this doubles as their display order.
    pub fn set_index(&self, symbol: u32) -> usize {
        let set = self.sets_by_symbol[usize::try_from(symbol).expect("symbol index fits in usize")];
        usize::try_from(set).expect("set index fits in usize")
    }

    pub fn set(&self, index: usize) -> &SymbolSet {
        &self.sets[index]
    }

    /// The names of every member of `set`, in declaration order.
    pub fn members(&self, set: &SymbolSet) -> &[String] {
        let first = usize::try_from(set.first).expect("symbol index fits in usize");
        let len = usize::try_from(set.len).expect("member count fits in usize");
        &self.names[first..first + len]
    }

    /// The ordinal of `symbol` within its own set, which is what an output's
    /// field values are expressed in.
    pub fn ordinal(&self, symbol: u32) -> i32 {
        let set = self.set_of(symbol);
        i32::try_from(symbol - set.first).expect("member count fits in i32")
    }
}

/// Types that can be elements in collections.
///
/// A collection tracks, per scalar position, only whether every value there is
/// known to be an `int`. That one bit is what `is_additive` and argument
/// coercion need; anything finer — a symbol's name, its set, whether a
/// collection happens to be all symbols — is read back off the values, which
/// are always available when there are any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElementType {
    /// An empty collection with no evidence about its outcome type.
    Uninhabited,
    /// The polymorphic identity produced by summing an uninhabited collection.
    AdditiveIdentity,
    /// Every value is an `int`.
    Int,
    /// At least one value is not an `int`, so the outcomes cannot be added.
    /// Not every value need be non-numeric: one symbol among the ints is enough.
    NotAllInt,
    Tuple(Rc<[ElementType]>),
}

impl ElementType {
    /// The join: the most precise type describing values of both types.
    ///
    /// `None` means the two cannot appear in one collection at all, which is
    /// the case only across shapes — a scalar with a tuple, or tuples of
    /// different arity. Scalars always join, to `NotAllInt` if they disagree.
    pub(crate) fn merged_with(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (ElementType::Uninhabited, other) | (other, ElementType::Uninhabited) => {
                Some(other.clone())
            }
            (ElementType::AdditiveIdentity, other) if other.is_additive() => Some(other.clone()),
            (other, ElementType::AdditiveIdentity) if other.is_additive() => Some(other.clone()),
            (ElementType::Int, ElementType::Int) => Some(ElementType::Int),
            (
                ElementType::Int | ElementType::NotAllInt,
                ElementType::Int | ElementType::NotAllInt,
            ) => Some(ElementType::NotAllInt),
            (ElementType::Tuple(left), ElementType::Tuple(right)) if left.len() == right.len() => {
                left.iter()
                    .zip(right.iter())
                    .map(|(left, right)| left.merged_with(right))
                    .collect::<Option<Vec<_>>>()
                    .map(|fields| ElementType::Tuple(fields.into()))
            }
            _ => None,
        }
    }

    /// Whether every value of this type is also a value of `required`.
    ///
    /// This is the check the join cannot do: now that mismatched scalars join
    /// to `NotAllInt` rather than failing, "these two can coexist" no longer
    /// implies "this one will do where that one is demanded".
    pub(crate) fn satisfies(&self, required: &Self) -> bool {
        match (self, required) {
            (ElementType::Uninhabited, _) => true,
            (ElementType::AdditiveIdentity, required) => required.is_additive(),
            (ElementType::Tuple(left), ElementType::Tuple(right)) if left.len() == right.len() => {
                left.iter()
                    .zip(right.iter())
                    .all(|(left, right)| left.satisfies(right))
            }
            (left, right) => left == right,
        }
    }

    pub(crate) fn display_name(&self) -> String {
        match self {
            ElementType::Uninhabited => "empty".to_string(),
            ElementType::AdditiveIdentity => "additive identity".to_string(),
            ElementType::Int => "int".to_string(),
            ElementType::NotAllInt => "non-numeric".to_string(),
            ElementType::Tuple(fields) => format!(
                "tuple({})",
                fields
                    .iter()
                    .map(ElementType::display_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    /// Whether this type is still the empty sum: a collection with no evidence
    /// about its outcomes, or the polymorphic identity they summed to. Both
    /// stand in for a value of whatever type they meet, which is why so much of
    /// the engine treats them alike. They part ways only in [`Self::merged_with`],
    /// [`Self::satisfies`], [`Self::summed_type`] and
    /// [`RuntimeValue::flattened_outcome_type`], where the difference between
    /// "no evidence" and "must be additive" is the whole point.
    ///
    /// Use this where the code asks a yes-or-no question. Where it is one arm of
    /// a `match` over the variants, spell `Uninhabited | AdditiveIdentity` out
    /// instead: the compiler then checks that a new variant is handled, which a
    /// guard on this predicate would silently skip.
    pub(crate) fn is_empty_sum(&self) -> bool {
        matches!(
            self,
            ElementType::Uninhabited | ElementType::AdditiveIdentity
        )
    }

    pub(crate) fn additive_identity(&self) -> Option<ElementValue> {
        match self {
            // The empty sum; see `is_empty_sum`.
            ElementType::Uninhabited | ElementType::AdditiveIdentity => {
                Some(ElementValue::AdditiveIdentity)
            }
            ElementType::Int => Some(ElementValue::Int(0)),
            ElementType::Tuple(fields)
                if fields.iter().all(|field| matches!(field, ElementType::Int)) =>
            {
                Some(ElementValue::Tuple(
                    fields
                        .iter()
                        .map(|_| ElementValue::Int(0))
                        .collect::<Vec<_>>()
                        .into(),
                ))
            }
            ElementType::NotAllInt | ElementType::Tuple(_) => None,
        }
    }

    pub(crate) fn is_additive(&self) -> bool {
        self.is_empty_sum()
            || matches!(self, ElementType::Int)
            || matches!(self, ElementType::Tuple(fields) if fields.iter().all(|field| matches!(field, ElementType::Int)))
    }

    pub(crate) fn summed_type(&self) -> Self {
        match self {
            ElementType::Uninhabited => ElementType::AdditiveIdentity,
            other => other.clone(),
        }
    }

    pub(crate) fn defaulted(&self) -> Self {
        match self {
            ElementType::Uninhabited | ElementType::AdditiveIdentity => ElementType::Int,
            other => other.clone(),
        }
    }
}

/// Variant order is the iteration order: ints, then symbols in global
/// declaration order, then tuples lexicographically. Symbol indices are handed
/// out in declaration order, so the derived `Ord` gets that for free.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementValue {
    AdditiveIdentity,
    Int(i32),
    Symbol(u32),
    Tuple(Rc<[ElementValue]>),
}

impl ElementValue {
    pub fn element_type(&self) -> ElementType {
        match self {
            ElementValue::AdditiveIdentity => ElementType::AdditiveIdentity,
            ElementValue::Int(_) => ElementType::Int,
            ElementValue::Symbol(_) => ElementType::NotAllInt,
            ElementValue::Tuple(fields) => ElementType::Tuple(
                fields
                    .iter()
                    .map(ElementValue::element_type)
                    .collect::<Vec<_>>()
                    .into(),
            ),
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        match self {
            ElementValue::Int(value) => Some(*value),
            ElementValue::AdditiveIdentity | ElementValue::Symbol(_) | ElementValue::Tuple(_) => {
                None
            }
        }
    }

    pub(crate) fn add_scaled(&self, other: &Self, count: u32) -> Self {
        match (self, other) {
            (ElementValue::AdditiveIdentity, ElementValue::AdditiveIdentity) => {
                ElementValue::AdditiveIdentity
            }
            (ElementValue::AdditiveIdentity, other) => other
                .element_type()
                .additive_identity()
                .expect("identity can only be added to an additive value")
                .add_scaled(other, count),
            (other, ElementValue::AdditiveIdentity) => other.clone(),
            (ElementValue::Int(left), ElementValue::Int(right)) => {
                let count = i32::try_from(count).expect("pool dimension fits in i32");
                ElementValue::Int(left + right * count)
            }
            (ElementValue::Tuple(left), ElementValue::Tuple(right)) => ElementValue::Tuple(
                left.iter()
                    .zip(right.iter())
                    .map(|(left, right)| left.add_scaled(right, count))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            _ => unreachable!("additive values have matching element types"),
        }
    }

    pub(crate) fn try_map_ints(
        &self,
        f: &impl Fn(i32) -> Result<i32, String>,
    ) -> Result<Self, String> {
        match self {
            ElementValue::AdditiveIdentity => Ok(ElementValue::AdditiveIdentity),
            ElementValue::Int(value) => Ok(ElementValue::Int(f(*value)?)),
            ElementValue::Tuple(fields) => Ok(ElementValue::Tuple(
                fields
                    .iter()
                    .map(|field| field.try_map_ints(f))
                    .collect::<Result<Vec<_>, _>>()?
                    .into(),
            )),
            // Reachable through `checked_neg`, which a negative dice count applies to
            // every face. `make_d` rejects that case first; this keeps the branch total.
            ElementValue::Symbol(_) => Err("Cannot negate a symbol".to_string()),
        }
    }

    pub(crate) fn try_zip_ints(
        &self,
        other: &Self,
        f: &impl Fn(i32, i32) -> Result<i32, String>,
    ) -> Result<Self, String> {
        match (self, other) {
            (ElementValue::AdditiveIdentity, ElementValue::AdditiveIdentity) => {
                Ok(ElementValue::AdditiveIdentity)
            }
            (ElementValue::Int(left), ElementValue::Int(right)) => {
                Ok(ElementValue::Int(f(*left, *right)?))
            }
            (ElementValue::Tuple(left), ElementValue::Tuple(right)) => Ok(ElementValue::Tuple(
                left.iter()
                    .zip(right.iter())
                    .map(|(left, right)| left.try_zip_ints(right, f))
                    .collect::<Result<Vec<_>, _>>()?
                    .into(),
            )),
            _ => unreachable!("tuple arithmetic operands have matching types"),
        }
    }

    pub(crate) fn checked_neg(&self) -> Result<Self, String> {
        self.try_map_ints(&|value| {
            value
                .checked_neg()
                .ok_or_else(|| format!("Negation overflow: -{value}"))
        })
    }

    pub(crate) fn materialize_identity(&self, outcome_type: &ElementType) -> Self {
        match self {
            ElementValue::AdditiveIdentity => outcome_type
                .additive_identity()
                .filter(|value| !matches!(value, ElementValue::AdditiveIdentity))
                .expect("identity must be materialized as a concrete additive type"),
            other => other.clone(),
        }
    }
}

/// A value paired with the table that can name its symbols.
///
/// Symbol values are bare indices, so rendering one needs the [`SymbolTable`];
/// this wrapper carries it to `Display` rather than putting it back into every
/// value. Obtain one with [`ElementValue::display`] or [`RuntimeValue::display`].
pub struct Displayed<'a, T> {
    value: &'a T,
    symbols: &'a SymbolTable,
}

impl ElementValue {
    pub fn display<'a>(&'a self, symbols: &'a SymbolTable) -> Displayed<'a, Self> {
        Displayed {
            value: self,
            symbols,
        }
    }
}

impl std::fmt::Display for Displayed<'_, ElementValue> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value {
            // Use italic 𝑒 for additive identity
            ElementValue::AdditiveIdentity => write!(f, "\u{1d452}"),
            ElementValue::Int(value) => write!(f, "{value}"),
            ElementValue::Symbol(symbol) => write!(f, "{}", self.symbols.name(*symbol)),
            ElementValue::Tuple(fields) => write!(
                f,
                "({})",
                fields
                    .iter()
                    .map(|field| field.display(self.symbols).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    Element(ElementValue),
    List(Rc<Vec<ElementValue>>, ElementType),
    Pool(Rc<Pool<ElementValue>>, ElementType),
}

impl RuntimeValue {
    pub fn display<'a>(&'a self, symbols: &'a SymbolTable) -> Displayed<'a, Self> {
        Displayed {
            value: self,
            symbols,
        }
    }
}

impl std::fmt::Display for Displayed<'_, RuntimeValue> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.value {
            RuntimeValue::Element(value) => write!(f, "{}", value.display(self.symbols)),
            RuntimeValue::List(list, _) => write!(
                f,
                "{{{}}}",
                list.iter()
                    .map(|value| value.display(self.symbols).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            RuntimeValue::Pool(pool, outcome_type) => {
                write!(f, "{}", display_pool(pool, outcome_type, self.symbols))
            }
        }
    }
}

fn display_pool(
    pool: &Pool<ElementValue>,
    outcome_type: &ElementType,
    symbols: &SymbolTable,
) -> String {
    if pool.ordered_outcomes().is_empty() {
        return "d{}".to_string();
    }
    let mut f = String::new();
    if pool.dimension() != 1 {
        write!(f, "{}", pool.dimension()).expect("write to string");
    }
    if pool.ordered_outcomes() == [(ElementValue::Int(0), Natural::ONE)] {
        write!(f, "d0").expect("write to string");
        return f;
    }
    if outcome_type == &ElementType::Int
        && pool
            .ordered_outcomes()
            .iter()
            .enumerate()
            .all(|(i, (outcome, weight))| {
                usize::try_from(outcome.as_int().expect("int outcome in int pool")).ok()
                    == Some(i + 1)
                    && *weight == Natural::ONE
            })
    {
        // This is a standard dn with no repeats.
        write!(f, "d{}", pool.ordered_outcomes().len()).expect("write to string");
    } else {
        write!(
            f,
            "d{{{}}}",
            pool.ordered_outcomes()
                .iter()
                .map(|(outcome, weight)| if weight == &Natural::ONE {
                    outcome.display(symbols).to_string()
                } else {
                    format!("{}:{}", outcome.display(symbols), weight)
                })
                .collect::<Vec<_>>()
                .join(", ")
        )
        .expect("write to string");
    }
    f
}

impl RuntimeValue {
    /// The empty list returned by a call cut short by the maximum function
    /// depth. Its outcome type is uninhabited rather than `int`, so that
    /// summing it yields the polymorphic additive identity: a recursion that
    /// accumulates tuples still truncates cleanly instead of failing to add a
    /// tuple to `0`.
    pub(crate) fn empty_list() -> Self {
        RuntimeValue::List(Rc::new(Vec::new()), ElementType::Uninhabited)
    }

    /// The empty distribution returned by a function with no `result`
    /// statement. Unlike [`Self::empty_list`], this is a dimension-one pool
    /// with no outcomes, so it contributes nothing to the caller rather than
    /// summing to zero; AnyDice distinguishes the two.
    pub(crate) fn empty_pool() -> Self {
        RuntimeValue::Pool(
            Rc::new(Pool::from_list(1, Vec::new())),
            ElementType::Uninhabited,
        )
    }

    pub(crate) fn outcome_type(&self) -> ElementType {
        match self {
            RuntimeValue::Element(value) => value.element_type(),
            RuntimeValue::List(_, outcome_type) | RuntimeValue::Pool(_, outcome_type) => {
                outcome_type.clone()
            }
        }
    }

    /// Whether every outcome will do where an `int` is demanded.
    ///
    /// This is `satisfies(&Int)` under a name that reads better at the operator
    /// call sites; defining it that way rather than restating the variant list
    /// keeps the two from drifting apart.
    pub(crate) fn is_numeric_compatible(&self) -> bool {
        self.outcome_type().satisfies(&ElementType::Int)
    }

    pub(crate) fn is_additive(&self) -> bool {
        self.outcome_type().is_additive()
    }

    pub(crate) fn flattened_outcome_type(&self) -> ElementType {
        match self {
            RuntimeValue::Pool(_, ElementType::Uninhabited) => ElementType::AdditiveIdentity,
            _ => self.outcome_type(),
        }
    }

    pub(crate) fn materialize_identities(&self, outcome_type: &ElementType) -> Self {
        if outcome_type.is_empty_sum() {
            return self.clone();
        }
        let materialize = |value: &ElementValue| value.materialize_identity(outcome_type);
        match self {
            RuntimeValue::Element(value) => RuntimeValue::Element(materialize(value)),
            // A collection with no values has nothing to materialize, so adopting
            // `outcome_type` would only invent evidence it never carried. That
            // matters for an empty die: told it produces `int`s, it displays as
            // `0` instead of as nothing, and `[explode d{}]` stops agreeing with
            // `d{}`.
            RuntimeValue::List(values, _) if values.is_empty() => self.clone(),
            RuntimeValue::Pool(pool, _) if pool.ordered_outcomes().is_empty() => self.clone(),
            RuntimeValue::List(values, _) => RuntimeValue::List(
                Rc::new(values.iter().map(materialize).collect()),
                outcome_type.clone(),
            ),
            RuntimeValue::Pool(pool, _) => RuntimeValue::Pool(
                Rc::new((**pool).clone().map_outcomes(|value| materialize(&value))),
                outcome_type.clone(),
            ),
        }
    }

    pub(crate) fn to_list(&self, repeat: usize) -> Result<Vec<ElementValue>, NonAdditiveSum> {
        match self {
            RuntimeValue::Element(value) => Ok(vec![value.clone(); repeat]),
            RuntimeValue::List(list, _) => {
                Ok((0..repeat).flat_map(|_| list.iter().cloned()).collect())
            }
            RuntimeValue::Pool(pool, _) => {
                let outcomes = sum_pool(pool, &self.outcome_type())?
                    .ordered_outcomes()
                    .iter()
                    .map(|(outcome, _)| outcome)
                    .cloned()
                    .collect::<Vec<_>>();
                Ok((0..repeat).flat_map(|_| outcomes.iter().cloned()).collect())
            }
        }
    }

    pub(crate) fn map_numeric_outcomes(&self, f: impl Fn(i32) -> i32 + Copy) -> Self {
        match self {
            RuntimeValue::Element(ElementValue::AdditiveIdentity) => f(0).into(),
            RuntimeValue::Element(ElementValue::Int(value)) => f(*value).into(),
            RuntimeValue::List(_, ElementType::Uninhabited | ElementType::AdditiveIdentity) => {
                f(0).into()
            }
            RuntimeValue::List(list, _) => f(list.iter().map(expect_int).sum()).into(),
            RuntimeValue::Pool(
                pool,
                outcome_type @ (ElementType::Uninhabited | ElementType::AdditiveIdentity),
            ) => RuntimeValue::Pool(
                Rc::new(
                    sum_pool(pool, outcome_type)
                        .expect("an empty pool is additive")
                        .map_outcomes(|outcome| {
                            ElementValue::Int(f(expect_int(
                                &outcome.materialize_identity(&ElementType::Int),
                            )))
                        }),
                ),
                ElementType::Int,
            ),
            RuntimeValue::Pool(pool, _) => RuntimeValue::Pool(
                Rc::new(
                    (**pool)
                        .clone()
                        .map_outcomes(|outcome| ElementValue::Int(f(expect_int(&outcome)))),
                ),
                ElementType::Int,
            ),
            RuntimeValue::Element(ElementValue::Symbol(_) | ElementValue::Tuple(_)) => {
                unreachable!("non-numeric values are rejected before numeric mapping")
            }
        }
    }

    pub(crate) fn to_pool(&self) -> Pool<ElementValue> {
        match self {
            RuntimeValue::Element(value) => Pool::from_list(1, vec![value.clone()]),
            RuntimeValue::List(list, outcome_type) if self.is_additive() => {
                Pool::from_list(1, vec![sum_elements(list, outcome_type)])
            }
            RuntimeValue::List(list, _) => Pool::from_list(1, (**list).clone()),
            RuntimeValue::Pool(pool, _) => (**pool).clone(),
        }
    }
}

/// Puts two values on comparable footing for the equality-based operations.
///
/// `=`, `!=`, `[_ contains _]` and `[count _ in _]` are total: values of
/// different kinds are simply never equal, so mismatched outcome types count
/// zero matches rather than failing. The join is consulted only to give the
/// empty sum a concrete type, so that `{} = 0` still holds; where there is no
/// join there is nothing to reconcile, and the comparison is false whatever the
/// values.
pub(crate) fn materialize_comparable_pair(
    left: &RuntimeValue,
    right: &RuntimeValue,
) -> (RuntimeValue, RuntimeValue) {
    match left.outcome_type().merged_with(&right.outcome_type()) {
        Some(outcome_type) => {
            let outcome_type = outcome_type.summed_type();
            (
                left.materialize_identities(&outcome_type),
                right.materialize_identities(&outcome_type),
            )
        }
        None => (left.clone(), right.clone()),
    }
}

pub(crate) fn expect_int(value: &ElementValue) -> i32 {
    value
        .as_int()
        .expect("numeric operation received a non-numeric element")
}

pub(crate) fn sum_elements(values: &[ElementValue], outcome_type: &ElementType) -> ElementValue {
    values.iter().fold(
        outcome_type
            .additive_identity()
            .expect("only additive sequences can be summed"),
        |sum, value| sum.add_scaled(value, 1),
    )
}

/// A pool that cannot be summed because its outcomes cannot be added together.
///
/// Carries a `witness`: one outcome that cannot be added, so the diagnostic can
/// name a value the user wrote rather than only its type. The range belongs to
/// whichever call site forced the sum, so it is supplied by
/// [`NonAdditiveSum::into_error`].
#[derive(Debug, Clone)]
pub(crate) struct NonAdditiveSum {
    dimension: u32,
    witness: Option<ElementValue>,
    field: Option<usize>,
}

impl NonAdditiveSum {
    /// `action` names what forced the sum; it completes "<action> requires summing ...".
    pub(crate) fn into_error(self, range: ast::Range, action: &'static str) -> RuntimeError {
        RuntimeError::NonAdditiveSum(Box::new(NonAdditiveSumError {
            range: range.into(),
            action,
            subject: NonAdditiveSubject::Pool(self.dimension),
            witness: self.witness,
            field: self.field,
        }))
    }
}

/// The one-based field that stops a tuple from being added, if it is a tuple
/// and one of its fields is not a number.
pub(crate) fn non_numeric_field(value: &ElementValue) -> Option<usize> {
    match value {
        ElementValue::Tuple(fields) => fields
            .iter()
            .position(|field| field.as_int().is_none())
            .map(|index| index + 1),
        _ => None,
    }
}

/// The first outcome that stands in the way of summing, and which of its
/// fields is at fault when it is a tuple.
fn non_additive_witness(pool: &Pool<ElementValue>) -> (Option<ElementValue>, Option<usize>) {
    let witness = pool
        .ordered_outcomes()
        .iter()
        .map(|(outcome, _)| outcome)
        .find(|outcome| !outcome.element_type().is_additive());
    (witness.cloned(), witness.and_then(non_numeric_field))
}

/// Whether displaying a pool means summing it first.
///
/// A die with no faces is the exception, and its outcome type has no say in it.
/// There is no distribution over an empty outcome set, so an empty die displays
/// as no outcomes rather than as its additive identity — summing one would
/// invent an outcome that cannot occur. This is the distinction
/// [`RuntimeValue::empty_pool`] draws: an empty die contributes nothing, where
/// an empty *sequence* legitimately sums to zero.
///
/// Note this asks about faces, not dice. `0d6` has six faces and no dice, so it
/// sums, to the `0` that rolling no dice yields.
///
/// The evaluator applies this when it stores an output and [`crate::output`]
/// applies it when it renders one. Stating it in one place is what keeps those
/// two answering the same way.
pub(crate) fn display_requires_summing(pool: &Pool<ElementValue>) -> bool {
    !pool.ordered_outcomes().is_empty()
}

pub(crate) fn sum_pool(
    pool: &Pool<ElementValue>,
    outcome_type: &ElementType,
) -> Result<Pool<ElementValue>, NonAdditiveSum> {
    if pool.dimension() == 1 && !pool.ordered_outcomes().is_empty() {
        return Ok(pool.clone());
    }
    let Some(identity) = outcome_type.additive_identity() else {
        // A single die is already its own sum, and a pool with no outcomes has nothing
        // to add whatever its dimension.
        if pool.dimension() == 1 || pool.ordered_outcomes().is_empty() {
            return Ok(pool.clone());
        }
        let (witness, field) = non_additive_witness(pool);
        return Err(NonAdditiveSum {
            dimension: pool.dimension(),
            witness,
            field,
        });
    };
    Ok(pool.sum_by(identity, ElementValue::add_scaled))
}

impl From<i32> for RuntimeValue {
    fn from(value: i32) -> Self {
        RuntimeValue::Element(ElementValue::Int(value))
    }
}

impl From<Rc<Vec<i32>>> for RuntimeValue {
    fn from(value: Rc<Vec<i32>>) -> Self {
        RuntimeValue::List(
            Rc::new(value.iter().copied().map(ElementValue::Int).collect()),
            ElementType::Int,
        )
    }
}

impl From<Vec<i32>> for RuntimeValue {
    fn from(value: Vec<i32>) -> Self {
        RuntimeValue::List(
            Rc::new(value.into_iter().map(ElementValue::Int).collect()),
            ElementType::Int,
        )
    }
}

impl From<Pool> for RuntimeValue {
    fn from(value: Pool) -> Self {
        RuntimeValue::Pool(
            Rc::new(value.map_outcomes(ElementValue::Int)),
            ElementType::Int,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table holding one declared set, `ATTACK_RESULT { MISS, HIT }`.
    pub(crate) fn attack_result() -> SymbolTable {
        let mut symbols = SymbolTable::default();
        symbols.define_set(
            "ATTACK_RESULT".to_string(),
            &["MISS".to_string(), "HIT".to_string()],
        );
        symbols
    }

    fn enum_value(index: u32) -> ElementValue {
        ElementValue::Symbol(index)
    }

    fn enum_pool(dimension: u32, members: &[u32]) -> Pool<ElementValue> {
        Pool::from_list(
            dimension,
            members.iter().copied().map(enum_value).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn summing_a_multidimensional_non_additive_pool_is_an_error() {
        let error = sum_pool(&enum_pool(2, &[0, 1]), &ElementType::NotAllInt)
            .expect_err("two enum dice cannot be added together");
        assert_eq!(error.dimension, 2);

        // Dimension 0 with outcomes is just as unsummable as dimension 2.
        assert!(sum_pool(&enum_pool(0, &[0, 1]), &ElementType::NotAllInt).is_err());
    }

    #[test]
    fn non_additive_pools_that_need_no_addition_are_returned_unchanged() {
        // A single die is already its own sum.
        let one_die = enum_pool(1, &[0, 1]);
        assert_eq!(
            sum_pool(&one_die, &ElementType::NotAllInt).unwrap(),
            one_die
        );

        // A pool with no outcomes has nothing to add, whatever its dimension.
        for dimension in [0, 1, 2] {
            let empty = enum_pool(dimension, &[]);
            assert_eq!(sum_pool(&empty, &ElementType::NotAllInt).unwrap(), empty);
        }
    }

    #[test]
    fn summing_additive_pools_is_unaffected() {
        let ints = Pool::from_list(2, vec![ElementValue::Int(1), ElementValue::Int(2)]);
        let summed = sum_pool(&ints, &ElementType::Int).unwrap();
        assert_eq!(summed.dimension(), 1);
        assert_eq!(
            summed
                .ordered_outcomes()
                .iter()
                .map(|(outcome, _)| outcome.clone())
                .collect::<Vec<_>>(),
            vec![
                ElementValue::Int(2),
                ElementValue::Int(3),
                ElementValue::Int(4)
            ]
        );

        let pair =
            |a, b| ElementValue::Tuple(vec![ElementValue::Int(a), ElementValue::Int(b)].into());
        let tuples = Pool::from_list(2, vec![pair(1, 10), pair(2, 20)]);
        let tuple_type = ElementType::Tuple(vec![ElementType::Int, ElementType::Int].into());
        let summed = sum_pool(&tuples, &tuple_type).unwrap();
        assert_eq!(summed.dimension(), 1);
        assert_eq!(
            summed
                .ordered_outcomes()
                .iter()
                .map(|(outcome, _)| outcome.clone())
                .collect::<Vec<_>>(),
            vec![pair(2, 20), pair(3, 30), pair(4, 40)]
        );
    }

    /// Multiset iteration sorts by `ElementValue`'s `Ord`, which is what gives
    /// symbols their declaration order: indices are handed out in declaration
    /// order, so the derived `Ord` sorts by it. Pinning it here means changing
    /// how indices are assigned breaks a test rather than the semantics.
    #[test]
    fn enum_multisets_are_ordered_by_declaration() {
        let multisets = enum_pool(2, &[0, 1])
            .multiset_iterator()
            .collect::<Vec<_>>();
        assert_eq!(
            multisets,
            vec![
                (vec![enum_value(0), enum_value(0)], Natural::ONE),
                (vec![enum_value(0), enum_value(1)], Natural::from(2u32)),
                (vec![enum_value(1), enum_value(1)], Natural::ONE),
            ]
        );
    }

    /// The error names an outcome the user can recognize, not just a type.
    #[test]
    fn a_non_additive_sum_error_carries_an_offending_outcome() {
        let error = sum_pool(&enum_pool(2, &[0, 1]), &ElementType::NotAllInt)
            .unwrap_err()
            .into_error((0, 1).into(), "displaying a pool");
        let RuntimeError::NonAdditiveSum(error) = error else {
            panic!("expected a non-additive sum error");
        };
        assert!(matches!(error.subject, NonAdditiveSubject::Pool(2)));
        assert_eq!(error.witness, Some(ElementValue::Symbol(0)));
        assert_eq!(error.field, None);
    }

    /// For a tuple it also names which field is at fault.
    #[test]
    fn a_non_additive_tuple_sum_names_the_offending_field() {
        let outcome = ElementValue::Tuple(
            vec![
                ElementValue::Int(1),
                ElementValue::Int(2),
                ElementValue::Symbol(0),
            ]
            .into(),
        );
        let pool = Pool::from_list(2, vec![outcome.clone()]);
        let outcome_type = ElementType::Tuple(
            vec![ElementType::Int, ElementType::Int, ElementType::NotAllInt].into(),
        );
        let error = sum_pool(&pool, &outcome_type)
            .unwrap_err()
            .into_error((0, 1).into(), "displaying a pool");
        let RuntimeError::NonAdditiveSum(error) = error else {
            panic!("expected a non-additive sum error");
        };
        assert_eq!(error.witness, Some(outcome));
        assert_eq!(error.field, Some(3));
    }

    /// Symbols join with ints instead of failing, but the join stops at scalars:
    /// nothing reconciles a scalar with a tuple, or two tuple arities.
    #[test]
    fn scalars_join_but_shapes_do_not() {
        let pair = ElementType::Tuple(vec![ElementType::Int, ElementType::Int].into());
        let triple =
            ElementType::Tuple(vec![ElementType::Int, ElementType::Int, ElementType::Int].into());
        let mixed_pair = ElementType::Tuple(vec![ElementType::NotAllInt, ElementType::Int].into());

        assert_eq!(
            ElementType::Int.merged_with(&ElementType::NotAllInt),
            Some(ElementType::NotAllInt)
        );
        assert_eq!(
            ElementType::Int.merged_with(&ElementType::Int),
            Some(ElementType::Int)
        );
        // A field-wise join is how `[tuple DIE 1]` over a mixed die stays typed.
        assert_eq!(pair.merged_with(&mixed_pair), Some(mixed_pair.clone()));
        assert_eq!(pair.merged_with(&triple), None);
        assert_eq!(ElementType::Int.merged_with(&pair), None);
        assert_eq!(ElementType::NotAllInt.merged_with(&pair), None);
    }

    /// The join says two values can share a collection; `satisfies` says one
    /// will do where the other is demanded. Mixed outcomes satisfy nothing.
    #[test]
    fn mixed_outcomes_do_not_satisfy_an_int_requirement() {
        assert!(ElementType::Int.satisfies(&ElementType::Int));
        assert!(!ElementType::NotAllInt.satisfies(&ElementType::Int));
        assert!(ElementType::Uninhabited.satisfies(&ElementType::Int));
        assert!(ElementType::AdditiveIdentity.satisfies(&ElementType::Int));

        let pair = ElementType::Tuple(vec![ElementType::Int, ElementType::Int].into());
        let mixed_pair = ElementType::Tuple(vec![ElementType::NotAllInt, ElementType::Int].into());
        assert!(pair.satisfies(&pair));
        assert!(!mixed_pair.satisfies(&pair));
    }

    /// A symbol carries only its index; the table is what names it.
    #[test]
    fn symbols_are_named_by_the_table() {
        let symbols = attack_result();
        assert_eq!(enum_value(1).display(&symbols).to_string(), "HIT");
        assert_eq!(symbols.set_of(1).name, "ATTACK_RESULT");
        assert_eq!(symbols.ordinal(1), 1);
    }
}
