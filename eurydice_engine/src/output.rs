use malachite::base::num::conversion::traits::RoundingFrom;
use malachite::{base::rounding_modes::RoundingMode, rational::Rational, Natural};
use serde::Serialize;
use std::fmt::Write;

use crate::dice::Pool;
use crate::eval::{sum_pool, RuntimeValue, ScalarType, ScalarValue};

#[derive(Debug, Clone, Serialize)]
pub enum OutputValue {
    Distribution(Distribution),
    TupleDistribution(TupleDistribution),
}

#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    pub probabilities: Vec<(i32, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
}

/// Describes a single tuple field, shared by every outcome of a tuple output.
///
/// Field values in the outcomes themselves are always raw `i32`s; for an enum
/// field the value is the member's ordinal, and the labels here map it back to
/// a display name. Hoisting the schema up here keeps the (potentially large)
/// list of outcomes free of repeated enum metadata.
#[derive(Debug, Clone, Serialize)]
pub enum TupleFieldSchema {
    Int,
    Enum {
        enum_name: String,
        labels: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct TupleDistribution {
    pub fields: Vec<TupleFieldSchema>,
    pub probabilities: Vec<(Vec<i32>, f64)>,
}

impl From<RuntimeValue> for OutputValue {
    fn from(value: RuntimeValue) -> Self {
        match value {
            RuntimeValue::Scalar(value) => {
                let outcome_type = value.scalar_type();
                pool_output(&Pool::from_list(1, vec![value]), &outcome_type, false)
            }
            RuntimeValue::List(values, outcome_type) => {
                pool_output(&Pool::from_list(1, values.to_vec()), &outcome_type, false)
            }
            RuntimeValue::Pool(pool, outcome_type) => pool_output(&pool, &outcome_type, true),
        }
    }
}

/// Flattens a tuple's fields to raw `i32`s; enum fields become their ordinal.
fn tuple_values(values: &[ScalarValue]) -> Vec<i32> {
    values
        .iter()
        .map(|value| match value {
            ScalarValue::Int(value) => *value,
            ScalarValue::Enum { value, .. } => *value,
            ScalarValue::Tuple(_) => unreachable!("nested tuples are rejected by the evaluator"),
        })
        .collect()
}

fn pool_output(pool: &Pool<ScalarValue>, outcome_type: &ScalarType, sum: bool) -> OutputValue {
    let pool = if sum { sum_pool(pool) } else { pool.clone() };
    match outcome_type {
        ScalarType::Int => OutputValue::Distribution(Distribution {
            probabilities: to_probabilities(
                &pool
                    .ordered_outcomes()
                    .iter()
                    .map(|(value, weight)| {
                        (
                            value.as_int().expect("homogeneous numeric pool"),
                            weight.clone(),
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            enum_name: None,
            labels: None,
        }),
        ScalarType::Enum(ty) => OutputValue::Distribution(Distribution {
            probabilities: to_probabilities(
                &pool
                    .ordered_outcomes()
                    .iter()
                    .map(|(value, weight)| match value {
                        ScalarValue::Enum { value, .. } => (*value, weight.clone()),
                        _ => unreachable!("homogeneous enum pool"),
                    })
                    .collect::<Vec<_>>(),
            ),
            enum_name: Some(ty.name.clone()),
            labels: Some(ty.members.clone()),
        }),
        ScalarType::Tuple(field_types) => OutputValue::TupleDistribution(TupleDistribution {
            fields: field_types
                .iter()
                .map(|ty| match ty {
                    ScalarType::Int => TupleFieldSchema::Int,
                    ScalarType::Enum(ty) => TupleFieldSchema::Enum {
                        enum_name: ty.name.clone(),
                        labels: ty.members.clone(),
                    },
                    ScalarType::Tuple(_) => {
                        unreachable!("nested tuples are rejected by the evaluator")
                    }
                })
                .collect(),
            probabilities: to_probabilities_generic(pool.ordered_outcomes())
                .into_iter()
                .map(|(value, probability)| match value {
                    ScalarValue::Tuple(fields) => (tuple_values(&fields), probability),
                    _ => unreachable!("homogeneous tuple pool"),
                })
                .collect(),
        }),
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
    let probabilities = to_probabilities(pool.sum().ordered_outcomes());
    let mean = mean(&probabilities);
    let stddev = stddev(&probabilities, mean);
    let (min, max) = min_and_max(&probabilities);

    let mut string = String::new();
    writeln!(string, "\"{}\",{},{},{},{}", name, mean, stddev, min, max).unwrap();
    writeln!(string, "#,%").unwrap();
    for (outcome, prob) in probabilities {
        writeln!(string, "{},{}", outcome, prob * 100.0).unwrap();
    }
    string
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
        let OutputValue::Distribution(distribution) = OutputValue::from(RuntimeValue::from(5))
        else {
            panic!("expected numeric distribution");
        };

        assert_eq!(distribution.probabilities, vec![(5, 1.0)]);
    }

    #[test]
    fn converts_list_to_distribution_and_combines_duplicates() {
        let value = RuntimeValue::from(Rc::new(vec![
            5, -1, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 5,
        ]));

        let OutputValue::Distribution(distribution) = OutputValue::from(value) else {
            panic!("expected numeric distribution");
        };

        assert_eq!(
            distribution.probabilities,
            vec![
                (-1, 0.0625),
                (0, 0.0625),
                (1, 0.0625),
                (2, 0.0625),
                (3, 0.0625),
                (4, 0.0625),
                (5, 0.625),
            ]
        );
    }

    #[test]
    fn converts_pool_to_distribution_by_summing_dice() {
        let value = RuntimeValue::from(Pool::from_list(2, vec![1, 2]));

        let OutputValue::Distribution(distribution) = OutputValue::from(value) else {
            panic!("expected numeric distribution");
        };

        assert_eq!(
            distribution.probabilities,
            vec![(2, 0.25), (3, 0.5), (4, 0.25)]
        );
    }
}
