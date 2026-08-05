use malachite::base::num::conversion::traits::RoundingFrom;
use malachite::{Natural, base::rounding_modes::RoundingMode, rational::Rational};
use serde::Serialize;
use std::fmt::Write;

use crate::dice::Pool;
use crate::eval::{ElementType, ElementValue, RuntimeValue, SymbolTable, sum_pool};
use crate::value::display_requires_summing;

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

impl Distribution {
    /// Converts an evaluated value into its serialized display representation,
    /// attaching validated tuple field names when the output supplied them.
    pub fn from_runtime(
        value: RuntimeValue,
        field_names: Option<Vec<String>>,
        symbols: &SymbolTable,
    ) -> Self {
        match value {
            RuntimeValue::Element(value) => {
                let outcome_type = value.element_type();
                pool_output(
                    &Pool::from_list(1, vec![value]),
                    &outcome_type,
                    false,
                    field_names,
                    symbols,
                )
            }
            RuntimeValue::List(values, outcome_type) => pool_output(
                &Pool::from_list(1, values.to_vec()),
                &outcome_type,
                false,
                field_names,
                symbols,
            ),
            RuntimeValue::Pool(pool, outcome_type) => {
                pool_output(&pool, &outcome_type, true, field_names, symbols)
            }
        }
    }
}

/// How one field position is rendered, derived from the values at that
/// position rather than from the outcome type.
///
/// The type only records whether a field is all `int`s, which is not enough to
/// label a symbol field: the names and the declared sets live on the values.
enum FieldRender {
    Int,
    /// The declared sets present at this position, in declaration order. Their
    /// members are concatenated to form the field's labels, so a symbol's wire
    /// value is its ordinal plus its set's offset in that concatenation.
    Symbols(Vec<usize>),
}

impl FieldRender {
    /// The wire value for one field of one outcome.
    fn ordinal(&self, field: &ElementValue, symbols: &SymbolTable) -> i32 {
        match (self, field) {
            (_, ElementValue::AdditiveIdentity) => 0,
            (_, ElementValue::Int(value)) => *value,
            (FieldRender::Symbols(sets), ElementValue::Symbol(symbol)) => {
                let set = symbols.set_index(*symbol);
                let offset: usize = sets
                    .iter()
                    .take_while(|present| **present != set)
                    .map(|present| symbols.members(symbols.set(*present)).len())
                    .sum();
                i32::try_from(offset).expect("label count fits in i32") + symbols.ordinal(*symbol)
            }
            (FieldRender::Int, ElementValue::Symbol(_)) => {
                unreachable!("a field mixing ints and symbols is rejected by the evaluator")
            }
            (_, ElementValue::Tuple(_)) => {
                unreachable!("nested tuples are rejected by the evaluator")
            }
        }
    }

    fn schema(&self, symbols: &SymbolTable) -> FieldSchema {
        match self {
            FieldRender::Int => FieldSchema::Int,
            FieldRender::Symbols(sets) => FieldSchema::Enum {
                // Several sets in one field is rare, and joining their names
                // keeps the wire format — and the frontend's grouping of
                // outputs by enum name — unchanged for the common single-set
                // case.
                enum_name: sets
                    .iter()
                    .map(|set| symbols.set(*set).name.as_str())
                    .collect::<Vec<_>>()
                    .join(" | "),
                labels: sets
                    .iter()
                    .flat_map(|set| symbols.members(symbols.set(*set)).iter().cloned())
                    .collect(),
            },
        }
    }
}

/// Derives each field position's rendering by scanning that position's values.
///
/// `arity` is fixed for a distribution, so every outcome contributes one value
/// per position; a position with no symbols renders as `int`, which is also
/// what an empty distribution falls back to.
fn field_renders(
    outcomes: &[(ElementValue, Natural)],
    arity: usize,
    symbols: &SymbolTable,
) -> Vec<FieldRender> {
    let mut sets_by_field: Vec<Vec<usize>> = vec![Vec::new(); arity];
    for (outcome, _) in outcomes {
        let fields: &[ElementValue] = match outcome {
            ElementValue::Tuple(fields) => fields,
            other => std::slice::from_ref(other),
        };
        for (position, field) in fields.iter().enumerate() {
            if let ElementValue::Symbol(symbol) = field {
                let set = symbols.set_index(*symbol);
                let sets = &mut sets_by_field[position];
                if let Err(index) = sets.binary_search(&set) {
                    sets.insert(index, set);
                }
            }
        }
    }
    sets_by_field
        .into_iter()
        .map(|sets| {
            if sets.is_empty() {
                FieldRender::Int
            } else {
                FieldRender::Symbols(sets)
            }
        })
        .collect()
}

fn pool_output(
    pool: &Pool<ElementValue>,
    outcome_type: &ElementType,
    sum: bool,
    field_names: Option<Vec<String>>,
    symbols: &SymbolTable,
) -> Distribution {
    let pool = if sum && display_requires_summing(pool) {
        sum_pool(pool, outcome_type).expect("output values are checked by the evaluator")
    } else {
        pool.clone()
    };
    let unresolved = outcome_type.is_empty_sum();
    let outcome_type = outcome_type.defaulted();
    let pool = if unresolved {
        pool.map_outcomes(|value| value.materialize_identity(&outcome_type))
    } else {
        pool
    };
    let arity = match &outcome_type {
        ElementType::Tuple(field_types) => field_types.len(),
        _ => 1,
    };
    let renders = field_renders(pool.ordered_outcomes(), arity, symbols);
    Distribution {
        fields: renders
            .iter()
            .map(|render| render.schema(symbols))
            .collect(),
        field_names: if matches!(outcome_type, ElementType::Tuple(_)) {
            field_names
        } else {
            None
        },
        probabilities: to_probabilities_generic(pool.ordered_outcomes())
            .into_iter()
            .map(|(value, probability)| {
                let fields: &[ElementValue] = match &value {
                    ElementValue::Tuple(fields) => fields,
                    other => std::slice::from_ref(other),
                };
                (
                    fields
                        .iter()
                        .zip(renders.iter())
                        .map(|(field, render)| render.ordinal(field, symbols))
                        .collect(),
                    probability,
                )
            })
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
    let (min, max) = min_and_max(&probabilities);

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
    to_probabilities_generic(ordered_outcomes)
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
    let mut outcomes = probabilities.iter().map(|(outcome, _)| *outcome);
    let Some(first) = outcomes.next() else {
        return (0, 0);
    };
    outcomes.fold((first, first), |(min, max), outcome| {
        (min.min(outcome), max.max(outcome))
    })
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;

    /// Most outputs contain no symbols, so they need nothing from the table.
    fn distribution(value: RuntimeValue) -> Distribution {
        Distribution::from_runtime(value, None, &SymbolTable::default())
    }

    #[test]
    fn converts_int_to_distribution() {
        let distribution = distribution(RuntimeValue::from(5));

        assert_eq!(distribution.probabilities, vec![(vec![5], 1.0)]);
    }

    #[test]
    fn converts_list_to_distribution_and_combines_duplicates() {
        let value = RuntimeValue::from(Rc::new(vec![
            5, -1, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 5,
        ]));

        let distribution = distribution(value);

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

        let distribution = distribution(value);

        assert_eq!(
            distribution.probabilities,
            vec![(vec![2], 0.25), (vec![3], 0.5), (vec![4], 0.25)]
        );
    }

    #[test]
    fn min_and_max_handles_empty_and_nonempty_distributions() {
        assert_eq!(min_and_max(&[]), (0, 0));
        assert_eq!(min_and_max(&[(3, 0.25), (-2, 0.5), (1, 0.25)]), (-2, 3));
    }
}
