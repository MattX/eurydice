use malachite::num::conversion::traits::RoundingFrom;
use malachite::{rounding_modes::RoundingMode, Natural, Rational};
use serde::Serialize;
use std::fmt::Write;

use crate::dice::Pool;
use crate::eval::RuntimeValue;

#[derive(Debug, Clone, Serialize)]
pub enum OutputValue {
    Int(i32),
    List(Vec<i32>),
    Distribution(Distribution),
}

#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    pub probabilities: Vec<(i32, f64)>,
}

impl From<RuntimeValue> for OutputValue {
    fn from(value: RuntimeValue) -> Self {
        match value {
            RuntimeValue::Int(i) => OutputValue::Int(i),
            RuntimeValue::List(is) => OutputValue::List(is.to_vec()),
            RuntimeValue::Pool(d) => {
                let probabilities = to_probabilities(d.sum().ordered_outcomes());
                OutputValue::Distribution(Distribution { probabilities })
            }
        }
    }
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
