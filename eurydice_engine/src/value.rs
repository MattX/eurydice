//! Runtime values and their element-type behavior.

use std::{fmt::Write, rc::Rc};

use malachite::{Natural, base::num::basic::traits::One};

use crate::ast;
use crate::dice::Pool;
use crate::error::{RuntimeError, SemanticErrorKind};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnumType {
    pub name: String,
    pub members: Vec<String>,
}

impl EnumType {
    pub fn member_name(&self, value: i32) -> Option<&str> {
        usize::try_from(value)
            .ok()
            .and_then(|index| self.members.get(index))
            .map(String::as_str)
    }
}

/// Types that can be elements in collections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElementType {
    /// An empty collection with no evidence about its outcome type.
    Uninhabited,
    /// The polymorphic identity produced by summing an uninhabited collection.
    AdditiveIdentity,
    Int,
    Enum(Rc<EnumType>),
    Tuple(Rc<[ElementType]>),
}

impl ElementType {
    pub(crate) fn merged_with(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (ElementType::Uninhabited, other) | (other, ElementType::Uninhabited) => {
                Some(other.clone())
            }
            (ElementType::AdditiveIdentity, other) if other.is_additive() => Some(other.clone()),
            (other, ElementType::AdditiveIdentity) if other.is_additive() => Some(other.clone()),
            (left, right) if left == right => Some(left.clone()),
            _ => None,
        }
    }

    pub(crate) fn display_name(&self) -> String {
        match self {
            ElementType::Uninhabited => "empty".to_string(),
            ElementType::AdditiveIdentity => "additive identity".to_string(),
            ElementType::Int => "int".to_string(),
            ElementType::Enum(ty) => ty.name.clone(),
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

    pub(crate) fn additive_identity(&self) -> Option<ElementValue> {
        match self {
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
            ElementType::Enum(_) | ElementType::Tuple(_) => None,
        }
    }

    pub(crate) fn is_additive(&self) -> bool {
        matches!(
            self,
            ElementType::Uninhabited | ElementType::AdditiveIdentity | ElementType::Int
        ) || matches!(self, ElementType::Tuple(fields) if fields.iter().all(|field| matches!(field, ElementType::Int)))
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementValue {
    AdditiveIdentity,
    Int(i32),
    Enum { value: i32, ty: Rc<EnumType> },
    Tuple(Rc<[ElementValue]>),
}

impl ElementValue {
    pub fn element_type(&self) -> ElementType {
        match self {
            ElementValue::AdditiveIdentity => ElementType::AdditiveIdentity,
            ElementValue::Int(_) => ElementType::Int,
            ElementValue::Enum { ty, .. } => ElementType::Enum(Rc::clone(ty)),
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
            ElementValue::AdditiveIdentity | ElementValue::Enum { .. } | ElementValue::Tuple(_) => {
                None
            }
        }
    }

    pub fn enum_type(&self) -> Option<Rc<EnumType>> {
        match self {
            ElementValue::Enum { ty, .. } => Some(Rc::clone(ty)),
            ElementValue::AdditiveIdentity | ElementValue::Int(_) | ElementValue::Tuple(_) => None,
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
            ElementValue::Enum { .. } => Err(format!("Cannot negate {self}")),
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

impl std::fmt::Display for ElementValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Use italic 𝑒 for additive identity
            ElementValue::AdditiveIdentity => write!(f, "\u{1d452}"),
            ElementValue::Int(value) => write!(f, "{value}"),
            ElementValue::Enum { value, ty } => {
                write!(f, "{}", ty.member_name(*value).unwrap_or("<?>"))
            }
            ElementValue::Tuple(fields) => write!(
                f,
                "({})",
                fields
                    .iter()
                    .map(ToString::to_string)
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

impl std::fmt::Display for RuntimeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeValue::Element(value) => write!(f, "{value}"),
            RuntimeValue::List(list, _) => write!(
                f,
                "{{{}}}",
                list.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            RuntimeValue::Pool(pool, outcome_type) => {
                write!(f, "{}", display_pool(pool, outcome_type))
            }
        }
    }
}

fn display_pool(pool: &Pool<ElementValue>, outcome_type: &ElementType) -> String {
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
                    outcome.to_string()
                } else {
                    format!("{}:{}", outcome, weight)
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

    pub fn enum_type(&self) -> Option<Rc<EnumType>> {
        match self.outcome_type() {
            ElementType::Enum(ty) => Some(ty),
            ElementType::Uninhabited
            | ElementType::AdditiveIdentity
            | ElementType::Int
            | ElementType::Tuple(_) => None,
        }
    }

    pub(crate) fn merged_outcome_type(&self, other: &Self) -> Option<ElementType> {
        self.outcome_type().merged_with(&other.outcome_type())
    }

    pub(crate) fn is_numeric_compatible(&self) -> bool {
        matches!(
            self.outcome_type(),
            ElementType::Uninhabited | ElementType::AdditiveIdentity | ElementType::Int
        )
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
        if matches!(
            outcome_type,
            ElementType::Uninhabited | ElementType::AdditiveIdentity
        ) {
            return self.clone();
        }
        let materialize = |value: &ElementValue| value.materialize_identity(outcome_type);
        match self {
            RuntimeValue::Element(value) => RuntimeValue::Element(materialize(value)),
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
            RuntimeValue::Element(ElementValue::Enum { .. } | ElementValue::Tuple(_)) => {
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
/// Carries just enough to build the error message; the range belongs to whichever
/// call site forced the sum, so it is supplied by [`NonAdditiveSum::into_error`].
#[derive(Debug, Clone)]
pub(crate) struct NonAdditiveSum {
    outcome_type: ElementType,
    dimension: u32,
}

impl NonAdditiveSum {
    /// `action` names what forced the sum; it completes "<action> requires summing ...".
    pub(crate) fn into_error(self, range: ast::Range, action: &str) -> RuntimeError {
        RuntimeError::Semantic {
            kind: SemanticErrorKind::NonAdditiveValue,
            range: range.into(),
            message: format!(
                "{action} requires summing a pool of {} dice, and {} outcomes cannot be added \
                 together",
                self.dimension,
                self.outcome_type.display_name(),
            ),
        }
    }
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
        return Err(NonAdditiveSum {
            outcome_type: outcome_type.clone(),
            dimension: pool.dimension(),
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

    fn attack_result() -> Rc<EnumType> {
        Rc::new(EnumType {
            name: "ATTACK_RESULT".to_string(),
            members: vec!["MISS".to_string(), "HIT".to_string()],
        })
    }

    fn enum_value(index: i32) -> ElementValue {
        ElementValue::Enum {
            value: index,
            ty: attack_result(),
        }
    }

    fn enum_pool(dimension: u32, members: &[i32]) -> Pool<ElementValue> {
        Pool::from_list(
            dimension,
            members.iter().copied().map(enum_value).collect::<Vec<_>>(),
        )
    }

    fn enum_type() -> ElementType {
        ElementType::Enum(attack_result())
    }

    #[test]
    fn summing_a_multidimensional_non_additive_pool_is_an_error() {
        let error = sum_pool(&enum_pool(2, &[0, 1]), &enum_type())
            .expect_err("two enum dice cannot be added together");
        assert_eq!(error.dimension, 2);

        // Dimension 0 with outcomes is just as unsummable as dimension 2.
        assert!(sum_pool(&enum_pool(0, &[0, 1]), &enum_type()).is_err());
    }

    #[test]
    fn non_additive_pools_that_need_no_addition_are_returned_unchanged() {
        // A single die is already its own sum.
        let one_die = enum_pool(1, &[0, 1]);
        assert_eq!(sum_pool(&one_die, &enum_type()).unwrap(), one_die);

        // A pool with no outcomes has nothing to add, whatever its dimension.
        for dimension in [0, 1, 2] {
            let empty = enum_pool(dimension, &[]);
            assert_eq!(sum_pool(&empty, &enum_type()).unwrap(), empty);
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
    /// enum members their declaration order. Pinning it here means reordering the
    /// fields of `ElementValue::Enum` breaks a test rather than the semantics.
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

    #[test]
    fn a_non_additive_sum_error_uses_the_stable_diagnostic_kind() {
        let error = sum_pool(&enum_pool(2, &[0, 1]), &enum_type())
            .unwrap_err()
            .into_error((0, 1).into(), "displaying a pool");
        let RuntimeError::Semantic { kind, message, .. } = error else {
            panic!("expected a semantic error");
        };
        assert_eq!(kind, SemanticErrorKind::NonAdditiveValue);
        assert!(message.contains("ATTACK_RESULT"), "{message}");
    }
}
