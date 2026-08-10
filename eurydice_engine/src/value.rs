//! Runtime values and their element-type behavior.

use std::{collections::HashSet, fmt::Write, rc::Rc};

use malachite::{Natural, base::num::basic::traits::One};

use crate::ast;
use crate::dice::Pool;
use crate::error::{NonAdditiveSubject, NonAdditiveSumError, RuntimeError, ShapeMismatchError};

/// Every symbol declared so far, in declaration order.
///
/// Symbol values are bare indices into this table. Keeping names out of values
/// means an outcome is a plain `u32`: no reference counting in the Icepool
/// inner loops, and a derived `Ord` that already sorts symbols in declaration
/// order.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    names: Vec<String>,
    declared: HashSet<String>,
}

impl SymbolTable {
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.declared.contains(name)
    }

    /// Declares one symbol and returns its global index in declaration order.
    pub(crate) fn define(&mut self, name: String) -> u32 {
        let symbol = u32::try_from(self.names.len()).expect("symbol count fits in u32");
        assert!(
            self.declared.insert(name.clone()),
            "symbol is declared once"
        );
        self.names.push(name);
        symbol
    }

    pub fn name(&self, symbol: u32) -> &str {
        self.names
            .get(usize::try_from(symbol).expect("symbol index fits in usize"))
            .map(String::as_str)
            .unwrap_or("<?>")
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
    pub fn as_int(&self) -> Option<i32> {
        match self {
            ElementValue::Int(value) => Some(*value),
            ElementValue::AdditiveIdentity | ElementValue::Symbol(_) | ElementValue::Tuple(_) => {
                None
            }
        }
    }

    /// The integer this value denotes, counting the empty sum as `0`.
    ///
    /// The operators that are only defined on integers accept the empty sum
    /// too, since it stands in for whatever it meets and here that is a number.
    pub(crate) fn as_int_or_identity(&self) -> Option<i32> {
        match self {
            ElementValue::AdditiveIdentity => Some(0),
            other => other.as_int(),
        }
    }

    /// Whether this value can take part in addition at all.
    pub(crate) fn is_additive(&self) -> bool {
        match self {
            ElementValue::AdditiveIdentity | ElementValue::Int(_) => true,
            ElementValue::Symbol(_) => false,
            ElementValue::Tuple(fields) => fields.iter().all(ElementValue::is_additive),
        }
    }

    /// The zero of this value's own shape: `0` for a number, a tuple of zeros
    /// for a tuple, and the polymorphic identity for the identity itself.
    ///
    /// `None` when the value cannot be added, and so has no zero to speak of.
    pub(crate) fn additive_identity(&self) -> Option<ElementValue> {
        match self {
            ElementValue::AdditiveIdentity => Some(ElementValue::AdditiveIdentity),
            ElementValue::Int(_) => Some(ElementValue::Int(0)),
            ElementValue::Symbol(_) => None,
            ElementValue::Tuple(fields) => fields
                .iter()
                .map(ElementValue::additive_identity)
                .collect::<Option<Vec<_>>>()
                .map(|fields| ElementValue::Tuple(fields.into())),
        }
    }

    /// Adds `other` to this value `count` times.
    ///
    /// Both operands must be additive and the same shape. Neither is checked
    /// anywhere earlier — this is where a sequence or pool discovers that its
    /// values cannot be added, at the value that says so.
    ///
    /// This runs once per die per state in the Icepool fold, so the cases that
    /// succeed are matched first and nothing asks whether a value is additive
    /// until one of them has failed to apply.
    pub(crate) fn add_scaled(&self, other: &Self, count: u32) -> Result<Self, ElementMismatch> {
        match (self, other) {
            // The identity stands in for whatever it meets, in either position.
            (ElementValue::AdditiveIdentity, ElementValue::AdditiveIdentity) => {
                Ok(ElementValue::AdditiveIdentity)
            }
            (ElementValue::AdditiveIdentity, other) => other
                .additive_identity()
                .ok_or_else(|| ElementMismatch::not_numeric(other))?
                .add_scaled(other, count),
            (other, ElementValue::AdditiveIdentity) => Ok(other.clone()),
            (ElementValue::Int(left), ElementValue::Int(right)) => {
                let count = i32::try_from(count).expect("pool dimension fits in i32");
                Ok(ElementValue::Int(left + right * count))
            }
            (ElementValue::Tuple(left), ElementValue::Tuple(right))
                if left.len() == right.len() =>
            {
                left.iter()
                    .zip(right.iter())
                    .map(|(left, right)| left.add_scaled(right, count))
                    .collect::<Result<Vec<_>, _>>()
                    .map(|fields| ElementValue::Tuple(fields.into()))
                    // A field is not a value the user wrote; the tuple holding
                    // it is. Re-blame it so the diagnostic can name the whole
                    // outcome and which of its fields is at fault.
                    .map_err(|inner| match inner.kind() {
                        Mismatch::NotNumeric { .. } if !other.is_additive() => {
                            ElementMismatch::not_numeric(other)
                        }
                        Mismatch::NotNumeric { .. } => ElementMismatch::not_numeric(self),
                        Mismatch::Shapes { .. } => inner,
                    })
            }
            (_, right) if !right.is_additive() => Err(ElementMismatch::not_numeric(right)),
            (left, _) if !left.is_additive() => Err(ElementMismatch::not_numeric(left)),
            (left, right) => Err(ElementMismatch::shapes(left, right)),
        }
    }

    pub(crate) fn try_map_ints(
        &self,
        f: &impl Fn(i32) -> Result<i32, String>,
    ) -> Result<Self, ElementOpError> {
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
            ElementValue::Symbol(_) => Err(ElementMismatch::not_numeric(self).into()),
        }
    }

    pub(crate) fn try_zip_ints(
        &self,
        other: &Self,
        f: &impl Fn(i32, i32) -> Result<i32, String>,
    ) -> Result<Self, ElementOpError> {
        match (self, other) {
            (ElementValue::AdditiveIdentity, ElementValue::AdditiveIdentity) => {
                Ok(ElementValue::AdditiveIdentity)
            }
            (ElementValue::Int(left), ElementValue::Int(right)) => {
                Ok(ElementValue::Int(f(*left, *right)?))
            }
            (ElementValue::Tuple(left), ElementValue::Tuple(right))
                if left.len() == right.len() =>
            {
                Ok(ElementValue::Tuple(
                    left.iter()
                        .zip(right.iter())
                        .map(|(left, right)| left.try_zip_ints(right, f))
                        .collect::<Result<Vec<_>, _>>()?
                        .into(),
                ))
            }
            (_, right) if !right.is_additive() => Err(ElementMismatch::not_numeric(right).into()),
            (left, _) if !left.is_additive() => Err(ElementMismatch::not_numeric(left).into()),
            (left, right) => Err(ElementMismatch::shapes(left, right).into()),
        }
    }

    pub(crate) fn checked_neg(&self) -> Result<Self, ElementOpError> {
        self.try_map_ints(&|value| {
            value
                .checked_neg()
                .ok_or_else(|| format!("Negation overflow: -{value}"))
        })
    }

    /// This value with the empty sum replaced by the zero of `like`'s shape.
    ///
    /// The empty sum has no shape of its own, so where it has to line up with
    /// real values — to be displayed beside them, say — it takes theirs.
    pub(crate) fn materialize_identity_like(&self, like: &ElementValue) -> ElementValue {
        match self {
            ElementValue::AdditiveIdentity => like
                .additive_identity()
                .unwrap_or(ElementValue::AdditiveIdentity),
            other => other.clone(),
        }
    }

    /// Equality that lets the empty sum stand for the zero of whatever it
    /// meets, so `{} = 0` holds and so does `{} = [tuple 0 0]`.
    pub(crate) fn equals_with_identity(&self, other: &Self) -> bool {
        match (self, other) {
            (ElementValue::AdditiveIdentity, other) | (other, ElementValue::AdditiveIdentity) => {
                other.additive_identity().as_ref() == Some(other)
                    || matches!(other, ElementValue::AdditiveIdentity)
            }
            (left, right) => left == right,
        }
    }
}

/// Two values an element operation could not be applied to.
///
/// Nothing checks for this ahead of time: a sequence or a pool holds whatever
/// values it was built from, and the operation that needs them to be numbers,
/// or to be the same shape, is what finds out otherwise. Carrying the offending
/// value means the diagnostic can name something the user wrote.
///
/// The payload is boxed because this is the error half of the `Result` the
/// Icepool fold returns for every die of every state. Behind one pointer it
/// fits in the niche of the value half, so the successful path — which is
/// almost all of them — carries nothing extra.
#[derive(Debug, Clone)]
pub(crate) struct ElementMismatch(Box<Mismatch>);

#[derive(Debug, Clone)]
pub(crate) enum Mismatch {
    /// A value that is not a number, with the one-based tuple field at fault
    /// when the value is a tuple.
    NotNumeric {
        witness: ElementValue,
        field: Option<usize>,
    },
    /// Two values whose shapes do not line up: a scalar with a tuple, or two
    /// tuples of different arity.
    Shapes {
        left: ElementValue,
        right: ElementValue,
    },
}

impl ElementMismatch {
    pub(crate) fn not_numeric(witness: &ElementValue) -> Self {
        ElementMismatch(Box::new(Mismatch::NotNumeric {
            witness: witness.clone(),
            field: non_numeric_field(witness),
        }))
    }

    pub(crate) fn shapes(left: &ElementValue, right: &ElementValue) -> Self {
        ElementMismatch(Box::new(Mismatch::Shapes {
            left: left.clone(),
            right: right.clone(),
        }))
    }

    pub(crate) fn kind(&self) -> &Mismatch {
        &self.0
    }

    /// `action` names what needed the values to line up; it completes
    /// "<action> requires summing ...".
    pub(crate) fn into_error(
        self,
        range: ast::ByteRange,
        action: &'static str,
        subject: NonAdditiveSubject,
    ) -> RuntimeError {
        match *self.0 {
            Mismatch::NotNumeric { witness, field } => {
                RuntimeError::NonAdditiveSum(Box::new(NonAdditiveSumError {
                    range,
                    action,
                    subject,
                    witness: Some(witness),
                    field,
                }))
            }
            Mismatch::Shapes { left, right } => {
                RuntimeError::ShapeMismatch(Box::new(ShapeMismatchError {
                    range,
                    action,
                    first: left,
                    second: right,
                }))
            }
        }
    }
}

/// An element operation that failed, either on the arithmetic or on the values.
#[derive(Debug, Clone)]
pub(crate) enum ElementOpError {
    /// Arithmetic that overflowed, or a division by zero.
    Math(String),
    /// Values the operation is not defined for.
    Mismatch(ElementMismatch),
}

impl From<String> for ElementOpError {
    fn from(message: String) -> Self {
        ElementOpError::Math(message)
    }
}

impl From<ElementMismatch> for ElementOpError {
    fn from(mismatch: ElementMismatch) -> Self {
        ElementOpError::Mismatch(mismatch)
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
    List(Rc<Vec<ElementValue>>),
    Pool(Rc<Pool<ElementValue>>),
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
            RuntimeValue::List(list) => write!(
                f,
                "{{{}}}",
                list.iter()
                    .map(|value| value.display(self.symbols).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            RuntimeValue::Pool(pool) => {
                write!(f, "{}", display_pool(pool, self.symbols))
            }
        }
    }
}

fn display_pool(pool: &Pool<ElementValue>, symbols: &SymbolTable) -> String {
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
    // A standard `dn` is one whose faces are exactly 1..n, each once. Reading
    // that off the faces is the same walk the old outcome-type check needed
    // anyway, so asking whether they are integers costs nothing extra.
    if pool
        .ordered_outcomes()
        .iter()
        .enumerate()
        .all(|(i, (outcome, weight))| {
            outcome
                .as_int()
                .and_then(|outcome| usize::try_from(outcome).ok())
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
    /// depth. It is empty rather than zero, so that summing it yields the
    /// polymorphic additive identity: a recursion that accumulates tuples still
    /// truncates cleanly instead of failing to add a tuple to `0`.
    pub(crate) fn empty_list() -> Self {
        RuntimeValue::List(Rc::new(Vec::new()))
    }

    /// The empty distribution returned by a function with no `result`
    /// statement. Unlike [`Self::empty_list`], this is a dimension-one pool
    /// with no outcomes, so it contributes nothing to the caller rather than
    /// summing to zero; AnyDice distinguishes the two.
    pub(crate) fn empty_pool() -> Self {
        RuntimeValue::Pool(Rc::new(Pool::from_list(1, Vec::new())))
    }

    /// Every value this holds, in order. A single element counts as one.
    ///
    /// Nothing here is cached: where the old outcome type was consulted, the
    /// values themselves are now walked, and this is how.
    pub(crate) fn elements(&self) -> &[ElementValue] {
        match self {
            RuntimeValue::Element(value) => std::slice::from_ref(value),
            RuntimeValue::List(list) => list,
            RuntimeValue::Pool(_) => &[],
        }
    }

    /// The first value that is not a number, if there is one.
    ///
    /// For a pool this reads the faces rather than the sums, which is what the
    /// operators that reject non-numeric operands want: a pool whose faces are
    /// symbols has no numeric sum either.
    pub(crate) fn non_numeric(&self) -> Option<&ElementValue> {
        match self {
            RuntimeValue::Element(value) => (value.as_int_or_identity().is_none()).then_some(value),
            RuntimeValue::List(list) => list
                .iter()
                .find(|value| value.as_int_or_identity().is_none()),
            RuntimeValue::Pool(pool) => pool
                .ordered_outcomes()
                .iter()
                .map(|(outcome, _)| outcome)
                .find(|outcome| outcome.as_int_or_identity().is_none()),
        }
    }

    pub(crate) fn to_list(&self, repeat: usize) -> Result<Vec<ElementValue>, ElementMismatch> {
        match self {
            RuntimeValue::Element(value) => Ok(vec![value.clone(); repeat]),
            RuntimeValue::List(list) => {
                Ok((0..repeat).flat_map(|_| list.iter().cloned()).collect())
            }
            // A die contributes its faces, so a die with no faces contributes
            // nothing: `{d2}` is `{1, 2}` and `{d{}}` is `{}`. Summing it first
            // would invent the single `𝑒` face that no distribution over an empty
            // outcome set has — and that a sequence of symbols cannot materialize.
            RuntimeValue::Pool(pool) if pool.ordered_outcomes().is_empty() => Ok(Vec::new()),
            RuntimeValue::Pool(pool) => {
                let outcomes = sum_pool(pool)
                    .map_err(|failure| failure.mismatch)?
                    .ordered_outcomes()
                    .iter()
                    .map(|(outcome, _)| outcome)
                    .cloned()
                    .collect::<Vec<_>>();
                Ok((0..repeat).flat_map(|_| outcomes.iter().cloned()).collect())
            }
        }
    }

    /// Applies `f` to every number this holds, summing a sequence first.
    ///
    /// Fails at the first value that is not a number, naming it. A sequence is
    /// summed rather than mapped, so an empty one contributes the `0` its empty
    /// sum stands for.
    pub(crate) fn map_numeric_outcomes(
        &self,
        f: impl Fn(i32) -> i32 + Copy,
    ) -> Result<Self, ElementMismatch> {
        let number = |value: &ElementValue| {
            value
                .as_int_or_identity()
                .ok_or_else(|| ElementMismatch::not_numeric(value))
        };
        match self {
            RuntimeValue::Element(value) => Ok(f(number(value)?).into()),
            RuntimeValue::List(list) => {
                let sum: i32 = list
                    .iter()
                    .map(number)
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .sum();
                Ok(f(sum).into())
            }
            RuntimeValue::Pool(pool) => Ok(RuntimeValue::Pool(Rc::new(
                sum_pool(pool)
                    .map_err(|failure| failure.mismatch)?
                    .try_map_outcomes(|outcome| {
                        number(&outcome).map(|value| ElementValue::Int(f(value)))
                    })?,
            ))),
        }
    }

    pub(crate) fn to_pool(&self) -> Result<Pool<ElementValue>, ElementMismatch> {
        match self {
            RuntimeValue::Element(value) => Ok(Pool::from_list(1, vec![value.clone()])),
            // A sequence is one value where it meets a pool, so it is summed;
            // a sequence that cannot be summed says so here.
            RuntimeValue::List(list) => Ok(Pool::from_list(1, vec![sum_elements(list)?])),
            RuntimeValue::Pool(pool) => Ok((**pool).clone()),
        }
    }
}

/// Puts two values on comparable footing for the equality-based primitives.
///
/// `[_ contains _]` and `[count _ in _]` compare with `==` — the pool versions
/// through Icepool, which has no room for a smarter comparison — and plain `==`
/// would not match the empty sum against the `0` it stands for. Giving `𝑒` the
/// shape of some value from the other side is enough; where the other side has
/// no such value there was nothing it could have matched anyway.
pub(crate) fn materialize_comparable_pair(
    left: &RuntimeValue,
    right: &RuntimeValue,
) -> (RuntimeValue, RuntimeValue) {
    let sample = |value: &RuntimeValue| match value {
        RuntimeValue::Pool(pool) => pool
            .ordered_outcomes()
            .iter()
            .map(|(outcome, _)| outcome)
            .find(|outcome| !matches!(outcome, ElementValue::AdditiveIdentity))
            .cloned(),
        value => value
            .elements()
            .iter()
            .find(|value| !matches!(value, ElementValue::AdditiveIdentity))
            .cloned(),
    };
    let Some(zero) = sample(left)
        .or_else(|| sample(right))
        .and_then(|value| value.additive_identity())
        .filter(|zero| !matches!(zero, ElementValue::AdditiveIdentity))
    else {
        return (left.clone(), right.clone());
    };
    let materialize = |value: &RuntimeValue| {
        let swap = |element: &ElementValue| match element {
            ElementValue::AdditiveIdentity => zero.clone(),
            other => other.clone(),
        };
        match value {
            RuntimeValue::Element(element) => RuntimeValue::Element(swap(element)),
            RuntimeValue::List(list) => {
                RuntimeValue::List(Rc::new(list.iter().map(swap).collect()))
            }
            RuntimeValue::Pool(pool) => RuntimeValue::Pool(Rc::new(
                (**pool).clone().map_outcomes(|outcome| swap(&outcome)),
            )),
        }
    };
    (materialize(left), materialize(right))
}

pub(crate) fn sum_elements(values: &[ElementValue]) -> Result<ElementValue, ElementMismatch> {
    values
        .iter()
        .try_fold(ElementValue::AdditiveIdentity, |sum, value| {
            sum.add_scaled(value, 1)
        })
}

/// A pool that could not be summed, and the number of dice it was summing.
///
/// The dimension belongs to the pool, the rest to the value that stopped the
/// sum, so this pairs the two rather than restating either.
#[derive(Debug, Clone)]
pub(crate) struct PoolSumFailure {
    dimension: u32,
    mismatch: ElementMismatch,
}

impl PoolSumFailure {
    /// `action` names what forced the sum; it completes "<action> requires summing ...".
    pub(crate) fn into_error(self, range: ast::ByteRange, action: &'static str) -> RuntimeError {
        self.mismatch
            .into_error(range, action, NonAdditiveSubject::Pool(self.dimension))
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

/// Whether displaying a pool means summing it first.
///
/// A die with no faces is the exception. There is no distribution over an empty
/// outcome set, so an empty die displays
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

/// Sums a pool's dice, or names the outcome that stopped it.
///
/// A pool of one die is already its own sum, so it never looks at its faces at
/// all — which is what lets a single die of symbols be displayed. Everything
/// else folds through the Icepool state machine, seeded with the polymorphic
/// identity so the first outcome added settles the shape.
///
/// A pool with nothing to add — no faces, or no dice — sums to that identity
/// rather than to nothing, which is what makes `2d{} + 5` equal `5`. Whether an
/// empty die *displays* as that identity is a separate question, and
/// [`display_requires_summing`] is where it is answered.
pub(crate) fn sum_pool(pool: &Pool<ElementValue>) -> Result<Pool<ElementValue>, PoolSumFailure> {
    if pool.dimension() == 1 && !pool.ordered_outcomes().is_empty() {
        return Ok(pool.clone());
    }
    // Seed with the zero of the faces' own shape where they have one, so that
    // rolling *no* tuple dice still yields a tuple of zeros: the faces say what
    // shape the sum would have had even when none of them is ever added. Faces
    // that have no zero — or no faces at all — leave the polymorphic identity,
    // which either meets something that gives it a shape or stays the empty sum.
    let identity = pool
        .ordered_outcomes()
        .first()
        .and_then(|(outcome, _)| outcome.additive_identity())
        .unwrap_or(ElementValue::AdditiveIdentity);
    pool.try_sum_by(identity, ElementValue::add_scaled)
        .map_err(|mismatch| PoolSumFailure {
            dimension: pool.dimension(),
            mismatch,
        })
}

impl From<i32> for RuntimeValue {
    fn from(value: i32) -> Self {
        RuntimeValue::Element(ElementValue::Int(value))
    }
}

impl From<Rc<Vec<i32>>> for RuntimeValue {
    fn from(value: Rc<Vec<i32>>) -> Self {
        RuntimeValue::List(Rc::new(
            value.iter().copied().map(ElementValue::Int).collect(),
        ))
    }
}

impl From<Vec<i32>> for RuntimeValue {
    fn from(value: Vec<i32>) -> Self {
        RuntimeValue::List(Rc::new(value.into_iter().map(ElementValue::Int).collect()))
    }
}

impl From<Pool> for RuntimeValue {
    fn from(value: Pool) -> Self {
        RuntimeValue::Pool(Rc::new(value.map_outcomes(ElementValue::Int)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table holding the declared symbols `MISS` and `HIT`.
    pub(crate) fn attack_result() -> SymbolTable {
        let mut symbols = SymbolTable::default();
        symbols.define("MISS".to_string());
        symbols.define("HIT".to_string());
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
        let error =
            sum_pool(&enum_pool(2, &[0, 1])).expect_err("two enum dice cannot be added together");
        assert_eq!(error.dimension, 2);
    }

    /// Rolling no dice yields the empty sum whatever the faces are, because
    /// there is nothing to add: `0d{A, B}` is `0`, just as `0d6` is.
    #[test]
    fn rolling_no_dice_sums_to_the_empty_sum() {
        let summed = sum_pool(&enum_pool(0, &[0, 1])).expect("no dice, nothing to add");
        assert_eq!(
            summed.ordered_outcomes(),
            [(ElementValue::AdditiveIdentity, Natural::ONE)]
        );
    }

    /// A single die is already its own sum, so its faces are never added and
    /// never have to be addable.
    #[test]
    fn a_single_non_additive_die_is_returned_unchanged() {
        let one_die = enum_pool(1, &[0, 1]);
        assert_eq!(sum_pool(&one_die).unwrap(), one_die);
    }

    /// A pool with no faces has nothing to add whatever its dimension, so it
    /// sums to the empty sum rather than to nothing. That is what makes
    /// `2d{} + 5` equal `5`; whether such a die *displays* as anything is
    /// [`display_requires_summing`]'s question, not this one.
    #[test]
    fn a_pool_with_no_faces_sums_to_the_empty_sum() {
        for dimension in [0, 1, 2] {
            assert_eq!(
                sum_pool(&enum_pool(dimension, &[]))
                    .unwrap()
                    .ordered_outcomes(),
                [(ElementValue::AdditiveIdentity, Natural::ONE)],
                "dimension {dimension}"
            );
        }
    }

    #[test]
    fn summing_additive_pools_is_unaffected() {
        let ints = Pool::from_list(2, vec![ElementValue::Int(1), ElementValue::Int(2)]);
        let summed = sum_pool(&ints).unwrap();
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
        let summed = sum_pool(&tuples).unwrap();
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
        let error = sum_pool(&enum_pool(2, &[0, 1]))
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
        let error = sum_pool(&pool)
            .unwrap_err()
            .into_error((0, 1).into(), "displaying a pool");
        let RuntimeError::NonAdditiveSum(error) = error else {
            panic!("expected a non-additive sum error");
        };
        assert_eq!(error.witness, Some(outcome));
        assert_eq!(error.field, Some(3));
    }

    fn pair(a: i32, b: i32) -> ElementValue {
        ElementValue::Tuple(vec![ElementValue::Int(a), ElementValue::Int(b)].into())
    }

    /// Adding is defined between values of the same shape. Nothing reconciles a
    /// scalar with a tuple, or two tuples of different arity, and the failure
    /// names both sides rather than a type.
    #[test]
    fn adding_values_of_different_shapes_reports_both() {
        let triple = ElementValue::Tuple(
            vec![
                ElementValue::Int(1),
                ElementValue::Int(2),
                ElementValue::Int(3),
            ]
            .into(),
        );
        for (left, right) in [
            (pair(1, 2), triple.clone()),
            (pair(1, 2), ElementValue::Int(7)),
            (ElementValue::Int(7), pair(1, 2)),
        ] {
            let failure = left
                .add_scaled(&right, 1)
                .expect_err("shapes that do not line up cannot be added");
            let Mismatch::Shapes { .. } = failure.kind() else {
                panic!("expected a shape mismatch, got {failure:?}");
            };
        }

        assert_eq!(pair(1, 2).add_scaled(&pair(3, 4), 1).unwrap(), pair(4, 6));
    }

    /// A symbol anywhere is what stops a value being added, and the failure
    /// names the value the user wrote rather than the symbol buried in it.
    #[test]
    fn adding_a_non_numeric_value_reports_it() {
        let mixed = ElementValue::Tuple(vec![ElementValue::Int(1), ElementValue::Symbol(0)].into());
        assert!(!mixed.is_additive());
        assert_eq!(mixed.additive_identity(), None);

        let failure = pair(1, 2)
            .add_scaled(&mixed, 1)
            .expect_err("a tuple holding a symbol cannot be added");
        let Mismatch::NotNumeric { witness, field } = failure.kind() else {
            panic!("expected a non-numeric mismatch, got {failure:?}");
        };
        assert_eq!(witness, &mixed);
        assert_eq!(field, &Some(2));
    }

    /// The empty sum takes the shape of whatever it meets, in a sum and in a
    /// comparison alike, which is what makes `{} = 0` and `{} = [tuple 0 0]`
    /// both hold without anything tracking a type.
    #[test]
    fn the_empty_sum_takes_the_shape_it_meets() {
        assert_eq!(sum_elements(&[]).unwrap(), ElementValue::AdditiveIdentity);
        assert_eq!(sum_elements(&[pair(1, 2), pair(3, 4)]).unwrap(), pair(4, 6));

        let identity = ElementValue::AdditiveIdentity;
        assert!(identity.equals_with_identity(&ElementValue::Int(0)));
        assert!(identity.equals_with_identity(&pair(0, 0)));
        assert!(!identity.equals_with_identity(&ElementValue::Int(1)));
        assert!(!identity.equals_with_identity(&pair(0, 1)));
        assert!(!identity.equals_with_identity(&ElementValue::Symbol(0)));
    }

    /// A symbol carries only its index; the table is what names it.
    #[test]
    fn symbols_are_named_by_the_table() {
        let symbols = attack_result();
        assert_eq!(enum_value(1).display(&symbols).to_string(), "HIT");
        assert_eq!(symbols.name(0), "MISS");
    }
}
