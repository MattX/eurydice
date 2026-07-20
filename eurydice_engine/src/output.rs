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
    TupleList(Vec<TupleScalar>),
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

#[derive(Debug, Clone, Serialize)]
pub struct TupleScalar {
    pub values: Vec<TupleField>,
}

#[derive(Debug, Clone, Serialize)]
pub enum TupleField {
    Int(i32),
    Enum(EnumScalar),
}

#[derive(Debug, Clone, Serialize)]
pub struct TupleDistribution {
    pub probabilities: Vec<(TupleScalar, f64)>,
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
                OutputValue::Tuple(tuple_output(&values))
            }
            RuntimeValue::List(values, _) => list_output(&values),
            RuntimeValue::Pool(pool, _) => pool_output(&pool),
        }
    }
}

fn enum_output(value: i32, ty: &crate::eval::EnumType) -> EnumScalar {
    EnumScalar {
        value,
        enum_name: ty.name.clone(),
        labels: ty.members.clone(),
    }
}

fn tuple_output(values: &[ScalarValue]) -> TupleScalar {
    TupleScalar {
        values: values
            .iter()
            .map(|value| match value {
                ScalarValue::Int(value) => TupleField::Int(*value),
                ScalarValue::Enum { value, ty } => TupleField::Enum(enum_output(*value, ty)),
                ScalarValue::Tuple(_) => {
                    unreachable!("nested tuples are rejected by the evaluator")
                }
            })
            .collect(),
    }
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
        Some(ScalarValue::Tuple(_)) => OutputValue::TupleList(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Tuple(fields) => tuple_output(fields),
                    _ => unreachable!("homogeneous tuple sequence"),
                })
                .collect(),
        ),
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
        Some(ScalarValue::Tuple(_)) => OutputValue::TupleDistribution(TupleDistribution {
            probabilities: to_probabilities_generic(pool.ordered_outcomes())
                .into_iter()
                .map(|(value, probability)| match value {
                    ScalarValue::Tuple(fields) => (tuple_output(&fields), probability),
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
