use malachite::base::num::conversion::traits::RoundingFrom;
use malachite::{base::rounding_modes::RoundingMode, rational::Rational, Natural};
use serde::Serialize;
use std::fmt::Write;

use crate::dice::Pool;
use crate::eval::{sum_pool, RuntimeValue, ScalarValue};

#[derive(Debug, Clone, Serialize)]
pub enum OutputValue {
    Int(i32),
    List(Vec<i32>),
    EnumInt(EnumScalar),
    EnumList(EnumSequence),
    Distribution(Distribution),
    Tuple(TupleScalar),
    TupleList(TupleSequence),
    TupleDistribution(TupleDistribution),
}

#[derive(Debug, Clone, Serialize)]
pub struct EnumScalar {
    pub value: i32,
    pub enum_name: String,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnumSequence {
    pub values: Vec<i32>,
    pub enum_name: String,
    pub labels: Vec<String>,
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
pub struct TupleScalar {
    pub fields: Vec<TupleFieldSchema>,
    pub values: Vec<i32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TupleSequence {
    pub fields: Vec<TupleFieldSchema>,
    pub values: Vec<Vec<i32>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TupleDistribution {
    pub fields: Vec<TupleFieldSchema>,
    pub probabilities: Vec<(Vec<i32>, f64)>,
}

impl From<RuntimeValue> for OutputValue {
    fn from(value: RuntimeValue) -> Self {
        match value {
            RuntimeValue::Scalar(ScalarValue::Int(i)) => OutputValue::Int(i),
            RuntimeValue::Scalar(ScalarValue::Enum { value, ty }) => {
                OutputValue::EnumInt(EnumScalar {
                    value,
                    enum_name: ty.name.clone(),
                    labels: ty.members.clone(),
                })
            }
            RuntimeValue::Scalar(ScalarValue::Tuple(values)) => {
                OutputValue::Tuple(TupleScalar {
                    fields: tuple_schema(&values),
                    values: tuple_values(&values),
                })
            }
            RuntimeValue::List(values, _) => list_output(&values),
            RuntimeValue::Pool(pool, _) => pool_output(&pool),
        }
    }
}

/// Derives the shared per-field schema from a representative tuple. Since tuple
/// pools and lists are homogeneous, any single outcome describes the whole set.
fn tuple_schema(values: &[ScalarValue]) -> Vec<TupleFieldSchema> {
    values
        .iter()
        .map(|value| match value {
            ScalarValue::Int(_) => TupleFieldSchema::Int,
            ScalarValue::Enum { ty, .. } => TupleFieldSchema::Enum {
                enum_name: ty.name.clone(),
                labels: ty.members.clone(),
            },
            ScalarValue::Tuple(_) => unreachable!("nested tuples are rejected by the evaluator"),
        })
        .collect()
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

fn list_output(values: &[ScalarValue]) -> OutputValue {
    match values.first() {
        None | Some(ScalarValue::Int(_)) => OutputValue::List(
            values
                .iter()
                .map(|value| value.as_int().expect("homogeneous numeric sequence"))
                .collect(),
        ),
        Some(ScalarValue::Enum { ty, .. }) => OutputValue::EnumList(EnumSequence {
            values: values
                .iter()
                .map(|value| match value {
                    ScalarValue::Enum { value, .. } => *value,
                    _ => unreachable!("homogeneous enum sequence"),
                })
                .collect(),
            enum_name: ty.name.clone(),
            labels: ty.members.clone(),
        }),
        Some(ScalarValue::Tuple(first)) => OutputValue::TupleList(TupleSequence {
            fields: tuple_schema(first),
            values: values
                .iter()
                .map(|value| match value {
                    ScalarValue::Tuple(fields) => tuple_values(fields),
                    _ => unreachable!("homogeneous tuple sequence"),
                })
                .collect(),
        }),
    }
}

fn pool_output(pool: &Pool<ScalarValue>) -> OutputValue {
    let pool = sum_pool(pool);
    match pool.ordered_outcomes().first().map(|(value, _)| value) {
        None | Some(ScalarValue::Int(_)) => OutputValue::Distribution(Distribution {
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
        Some(ScalarValue::Enum { ty, .. }) => OutputValue::Distribution(Distribution {
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
        Some(ScalarValue::Tuple(first)) => OutputValue::TupleDistribution(TupleDistribution {
            fields: tuple_schema(first),
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
