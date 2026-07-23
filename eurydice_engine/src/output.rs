use malachite::base::num::conversion::traits::RoundingFrom;
use malachite::{base::rounding_modes::RoundingMode, rational::Rational, Natural};
use serde::Serialize;
use std::fmt::Write;

use crate::dice::Pool;
use crate::eval::{sum_pool, ElementType, ElementValue, RuntimeValue};

#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    pub fields: Vec<FieldSchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_names: Option<Vec<String>>,
    pub probabilities: Vec<(Vec<i32>, f64)>,
}

/// Describes a single output field, shared by every outcome of a distribution.
///
/// Field values in the outcomes themselves are always raw `i32`s; for an enum
/// field the value is the member's ordinal, and the labels here map it back to
/// a display name. Hoisting the schema up here keeps the (potentially large)
/// list of outcomes free of repeated enum metadata.
#[derive(Debug, Clone, Serialize)]
pub enum FieldSchema {
    Int,
    Enum {
        enum_name: String,
        labels: Vec<String>,
    },
}

impl From<RuntimeValue> for Distribution {
    fn from(value: RuntimeValue) -> Self {
        Self::from_runtime(value, None)
    }
}

impl Distribution {
    /// Converts an evaluated value into its serialized display representation,
    /// attaching validated tuple field names when the output supplied them.
    pub fn from_runtime(value: RuntimeValue, field_names: Option<Vec<String>>) -> Self {
        match value {
            RuntimeValue::Element(value) => {
                let outcome_type = value.element_type();
                pool_output(
                    &Pool::from_list(1, vec![value]),
                    &outcome_type,
                    false,
                    field_names,
                )
            }
            RuntimeValue::List(values, outcome_type) => pool_output(
                &Pool::from_list(1, values.to_vec()),
                &outcome_type,
                false,
                field_names,
            ),
            RuntimeValue::Pool(pool, outcome_type) => {
                pool_output(&pool, &outcome_type, true, field_names)
            }
        }
    }
}

/// Flattens any output outcome to raw `i32`s. Scalar outcomes have one value;
/// tuples have one per field, and enum values are represented by their ordinal.
fn output_values(value: &ElementValue) -> Vec<i32> {
    match value {
        ElementValue::AdditiveIdentity => {
            unreachable!("additive identities are materialized before output")
        }
        ElementValue::Int(value) | ElementValue::Enum { value, .. } => vec![*value],
        ElementValue::Tuple(fields) => fields
            .iter()
            .map(|field| match field {
                ElementValue::AdditiveIdentity => 0,
                ElementValue::Int(value) | ElementValue::Enum { value, .. } => *value,
                ElementValue::Tuple(_) => {
                    unreachable!("nested tuples are rejected by the evaluator")
                }
            })
            .collect(),
    }
}

fn pool_output(
    pool: &Pool<ElementValue>,
    outcome_type: &ElementType,
    sum: bool,
    field_names: Option<Vec<String>>,
) -> Distribution {
    let pool = if sum {
        if pool.ordered_outcomes().is_empty() && matches!(outcome_type, ElementType::Uninhabited) {
            pool.clone()
        } else {
            sum_pool(pool, outcome_type)
        }
    } else {
        pool.clone()
    };
    let unresolved = matches!(
        outcome_type,
        ElementType::Uninhabited | ElementType::AdditiveIdentity
    );
    let outcome_type = outcome_type.defaulted();
    let pool = if unresolved {
        pool.map_outcomes(|value| value.materialize_identity(&outcome_type))
    } else {
        pool
    };
    let fields = match &outcome_type {
        ElementType::Uninhabited | ElementType::AdditiveIdentity => {
            unreachable!("defaulted outcome types are concrete")
        }
        ElementType::Int => vec![FieldSchema::Int],
        ElementType::Enum(ty) => vec![FieldSchema::Enum {
            enum_name: ty.name.clone(),
            labels: ty.members.clone(),
        }],
        ElementType::Tuple(field_types) => field_types
            .iter()
            .map(|ty| match ty {
                ElementType::Uninhabited | ElementType::AdditiveIdentity => {
                    unreachable!("tuple fields always have concrete types")
                }
                ElementType::Int => FieldSchema::Int,
                ElementType::Enum(ty) => FieldSchema::Enum {
                    enum_name: ty.name.clone(),
                    labels: ty.members.clone(),
                },
                ElementType::Tuple(_) => {
                    unreachable!("nested tuples are rejected by the evaluator")
                }
            })
            .collect(),
    };
    Distribution {
        fields,
        field_names: if matches!(outcome_type, ElementType::Tuple(_)) {
            field_names
        } else {
            None
        },
        probabilities: to_probabilities_generic(pool.ordered_outcomes())
            .into_iter()
            .map(|(value, probability)| (output_values(&value), probability))
            .collect(),
    }
}

pub fn to_probabilities_generic<T: Clone>(ordered_outcomes: &[(T, Natural)]) -> Vec<(T, f64)> {
    let total: Natural = ordered_outcomes.iter().map(|(_, count)| count).sum();
    ordered_outcomes
        .iter()
        .map(|(outcome, count)| {
            (
                outcome.clone(),
                f64::rounding_from(
                    Rational::from_naturals(count.clone(), total.clone()),
                    RoundingMode::Nearest,
                )
                .0,
            )
        })
        .collect()
}

pub fn export_anydice_format(name: &str, pool: &Pool) -> String {
    let pool = pool_for_output(pool);
    let probabilities = to_probabilities(pool.ordered_outcomes());
    let mean = mean(&probabilities);
    let stddev = stddev(&probabilities, mean);
    let (min, max) = if probabilities.is_empty() {
        (0, 0)
    } else {
        min_and_max(&probabilities)
    };

    let mut string = String::new();
    writeln!(string, "\"{}\",{},{},{},{}", name, mean, stddev, min, max).unwrap();
    writeln!(string, "#,%").unwrap();
    for (outcome, prob) in probabilities {
        writeln!(string, "{},{}", outcome, prob * 100.0).unwrap();
    }
    string
}

fn pool_for_output(pool: &Pool) -> Pool {
    if pool.is_empty() {
        pool.clone()
    } else {
        pool.sum()
    }
}

pub fn to_probabilities(ordered_outcomes: &[(i32, Natural)]) -> Vec<(i32, f64)> {
    let total: Natural = ordered_outcomes.iter().map(|(_, count)| count).sum();
    ordered_outcomes
        .iter()
        .map(|(outcome, count)| {
            (
                *outcome,
                f64::rounding_from(
                    Rational::from_naturals(count.clone(), total.clone()),
                    RoundingMode::Nearest,
                )
                .0,
            )
        })
        .collect()
}

pub fn mean(probabilities: &[(i32, f64)]) -> f64 {
    probabilities
        .iter()
        .map(|(outcome, prob)| *outcome as f64 * *prob)
        .sum()
}

pub fn stddev(probabilities: &[(i32, f64)], mean: f64) -> f64 {
    let variance: f64 = probabilities
        .iter()
        .map(|(outcome, prob)| (*outcome as f64 - mean).powi(2) * *prob)
        .sum();
    variance.sqrt()
}

pub fn min_and_max(probabilities: &[(i32, f64)]) -> (i32, i32) {
    let min = probabilities.iter().map(|(outcome, _)| *outcome).min();
    let max = probabilities.iter().map(|(outcome, _)| *outcome).max();
    (min.unwrap(), max.unwrap())
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;

    #[test]
    fn converts_int_to_distribution() {
        let distribution = Distribution::from(RuntimeValue::from(5));

        assert_eq!(distribution.probabilities, vec![(vec![5], 1.0)]);
    }

    #[test]
    fn converts_list_to_distribution_and_combines_duplicates() {
        let value = RuntimeValue::from(Rc::new(vec![
            5, -1, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 5,
        ]));

        let distribution = Distribution::from(value);

        assert_eq!(
            distribution.probabilities,
            vec![
                (vec![-1], 0.0625),
                (vec![0], 0.0625),
                (vec![1], 0.0625),
                (vec![2], 0.0625),
                (vec![3], 0.0625),
                (vec![4], 0.0625),
                (vec![5], 0.625),
            ]
        );
    }

    #[test]
    fn converts_pool_to_distribution_by_summing_dice() {
        let value = RuntimeValue::from(Pool::from_list(2, vec![1, 2]));

        let distribution = Distribution::from(value);

        assert_eq!(
            distribution.probabilities,
            vec![(vec![2], 0.25), (vec![3], 0.5), (vec![4], 0.25)]
        );
    }
}
