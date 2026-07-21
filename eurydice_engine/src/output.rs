use malachite::base::num::conversion::traits::RoundingFrom;
use malachite::{base::rounding_modes::RoundingMode, rational::Rational, Natural};
use serde::Serialize;
use std::fmt::Write;

use crate::dice::Pool;
use crate::eval::RuntimeValue;

#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    pub probabilities: Vec<(i32, f64)>,
}

impl From<RuntimeValue> for Distribution {
    fn from(value: RuntimeValue) -> Self {
        let pool = match value {
            RuntimeValue::Int(i) => Pool::from_list(1, vec![i]),
            RuntimeValue::List(values) => Pool::from_list(1, values.to_vec()),
            RuntimeValue::Pool(pool) => pool.sum(),
        };
        let probabilities = to_probabilities(pool.ordered_outcomes());
        Distribution { probabilities }
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

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;

    #[test]
    fn converts_int_to_distribution() {
        let distribution = Distribution::from(RuntimeValue::Int(5));

        assert_eq!(distribution.probabilities, vec![(5, 1.0)]);
    }

    #[test]
    fn converts_list_to_distribution_and_combines_duplicates() {
        let value = RuntimeValue::List(Rc::new(vec![
            5, -1, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 5,
        ]));

        let distribution = Distribution::from(value);

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

        let distribution = Distribution::from(value);

        assert_eq!(
            distribution.probabilities,
            vec![(2, 0.25), (3, 0.5), (4, 0.25)]
        );
    }
}
