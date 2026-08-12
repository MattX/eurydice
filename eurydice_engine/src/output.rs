use malachite::{
    Natural,
    base::{num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode},
    rational::Rational,
};
#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{
    dice::Pool,
    eval::{ElementValue, RuntimeValue, SymbolTable, sum_pool},
    value::display_requires_summing,
};

/// A distribution of outcomes, each with a probability.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct Distribution {
    /// Describes the type and name of each field in the outcomes.
    ///
    /// In the simple case of a scalar integer field, this slice will have a
    /// single entry with [`FieldSchema::Int`].
    fields: Box<[Field]>,

    /// The outcomes, concatenated in row-major order.
    ///
    /// Each consecutive `fields.len()` values form one outcome. If a given
    /// field's schema is [`FieldSchema::Categorical`], its value is a
    /// non-negative ordinal index into the schema's labels.
    values: Box<[i32]>,

    /// One probability for each outcome in `values`.
    probabilities: Box<[f64]>,
}

/// Information on how to render an output field.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct Field {
    /// The name the `labeled` clause gave this field, if any.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub name: Option<String>,
    /// The schema for this field, describing how its values should be
    /// displayed.
    pub schema: FieldSchema,
}

/// Describes how one output field's values are displayed, shared by every
/// outcome of a distribution.
///
/// Field values in the outcomes themselves are always raw `i32`s; for a
/// categorical field the value is an ordinal, and the labels here map it back
/// to a display name. This includes all-symbol fields as well as fields that
/// mix numbers and symbols. Hoisting the schema up here keeps the (potentially
/// large) list of outcomes free of repeated metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
#[non_exhaustive]
pub enum FieldSchema {
    /// An integer field.
    Int,
    /// A field whose values are drawn from a fixed set of symbols. Each field
    /// in an outcome is an index into this array.
    #[non_exhaustive]
    Categorical {
        /// The symbols that can appear in this field.
        labels: Vec<String>,
    },
}

impl Distribution {
    /// Describes the type and name of each field in the outcomes.
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Iterates over each outcome and its probability.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&[i32], f64)> + '_ {
        let arity = self.fields.len();
        debug_assert_ne!(arity, 0);
        debug_assert_eq!(self.values.len(), arity * self.probabilities.len());
        self.values
            .chunks_exact(arity)
            .zip(self.probabilities.iter().copied())
    }

    /// Converts an evaluated value into its serialized display representation,
    /// attaching validated tuple field names when the output supplied them.
    pub(crate) fn from_runtime(
        value: RuntimeValue,
        field_names: Option<Vec<String>>,
        symbols: &SymbolTable,
    ) -> Self {
        match value {
            RuntimeValue::Element(value) => pool_output(
                &Pool::from_list(1, vec![value]),
                false,
                field_names,
                symbols,
            ),
            RuntimeValue::List(values) => pool_output(
                &Pool::from_list(1, values.to_vec()),
                false,
                field_names,
                symbols,
            ),
            RuntimeValue::Pool(pool) => pool_output(&pool, true, field_names, symbols),
        }
    }
}

/// How one field position is rendered, derived from the values at that
/// position rather than from the outcome type.
///
/// The type only records whether a field is all `int`s, which is not enough to
/// label a symbol field: symbol names live in the evaluator's symbol table.
enum FieldRender {
    Int,
    /// The symbols observed at this position, in declaration order. Their
    /// position in this vector is the field's serialized value.
    Symbols(Vec<u32>),
    /// A field containing both numbers and symbols is rendered categorically.
    /// Keeping the original values here gives every outcome a distinct wire
    /// ordinal even when, for example, the integer `0` and the first symbol
    /// would otherwise both be encoded as zero.
    Mixed(Vec<ElementValue>),
}

impl FieldRender {
    /// The wire value for one field of one outcome.
    fn ordinal(&self, field: &ElementValue) -> i32 {
        match (self, field) {
            (FieldRender::Mixed(present), field) => i32::try_from(
                present
                    .binary_search(field)
                    .expect("every rendered mixed value was collected from the outcomes"),
            )
            .expect("label count fits in i32"),
            (_, ElementValue::AdditiveIdentity) => 0,
            (_, ElementValue::Int(value)) => *value,
            (FieldRender::Symbols(present), ElementValue::Symbol(symbol)) => i32::try_from(
                present
                    .binary_search(symbol)
                    .expect("every rendered symbol was collected from the outcomes"),
            )
            .expect("label count fits in i32"),
            (FieldRender::Int, ElementValue::Symbol(_)) => {
                unreachable!("a field containing symbols gets a categorical renderer")
            }
            (_, ElementValue::Tuple(_)) => {
                unreachable!("nested tuples are rejected by the evaluator")
            }
        }
    }

    fn schema(&self, symbols: &SymbolTable) -> FieldSchema {
        match self {
            FieldRender::Int => FieldSchema::Int,
            FieldRender::Symbols(present) => FieldSchema::Categorical {
                labels: present
                    .iter()
                    .map(|symbol| symbols.name(*symbol).to_string())
                    .collect(),
            },
            FieldRender::Mixed(present) => FieldSchema::Categorical {
                labels: present
                    .iter()
                    .map(|value| match value {
                        ElementValue::AdditiveIdentity => "0".to_string(),
                        other => other.display(symbols).to_string(),
                    })
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
fn field_renders(outcomes: &[(ElementValue, Natural)], arity: usize) -> Vec<FieldRender> {
    let mut symbols_by_field: Vec<Vec<u32>> = vec![Vec::new(); arity];
    let mut has_non_symbols = vec![false; arity];
    for (outcome, _) in outcomes {
        let fields: &[ElementValue] = match outcome {
            ElementValue::Tuple(fields) => fields,
            other => std::slice::from_ref(other),
        };
        for (position, field) in fields.iter().enumerate() {
            if let ElementValue::Symbol(symbol) = field {
                let present = &mut symbols_by_field[position];
                if let Err(index) = present.binary_search(symbol) {
                    present.insert(index, *symbol);
                }
            } else {
                has_non_symbols[position] = true;
            }
        }
    }
    let mut renders = symbols_by_field
        .into_iter()
        .zip(has_non_symbols)
        .map(|(present, has_non_symbols)| {
            if present.is_empty() {
                FieldRender::Int
            } else if has_non_symbols {
                FieldRender::Mixed(Vec::new())
            } else {
                FieldRender::Symbols(present)
            }
        })
        .collect::<Vec<_>>();

    // Mixed fields are uncommon. Populate their categorical dictionaries in a
    // second pass so the ordinary all-int and all-symbol paths keep their small
    // representations and avoid cloning every outcome.
    if renders
        .iter()
        .any(|render| matches!(render, FieldRender::Mixed(_)))
    {
        for (outcome, _) in outcomes {
            let fields: &[ElementValue] = match outcome {
                ElementValue::Tuple(fields) => fields,
                other => std::slice::from_ref(other),
            };
            for (field, render) in fields.iter().zip(&mut renders) {
                if let FieldRender::Mixed(present) = render
                    && let Err(index) = present.binary_search(field)
                {
                    present.insert(index, field.clone());
                }
            }
        }
    }
    renders
}

/// Renders a distribution, whose outcomes the evaluator has already checked
/// share one shape — see `reject_mixed_output_shapes`. That is what lets the
/// first outcome speak for the arity, and what keeps `FieldRender` total.
fn pool_output(
    pool: &Pool<ElementValue>,
    sum: bool,
    field_names: Option<Vec<String>>,
    symbols: &SymbolTable,
) -> Distribution {
    let pool = if sum && display_requires_summing(pool) {
        sum_pool(pool).expect("output values are checked by the evaluator")
    } else {
        pool.clone()
    };
    // The empty sum has no shape, so the shape comes from an outcome that does;
    // the evaluator has already checked they all agree on it. Giving the
    // identity that shape here is what lets one die of tuples and one truncated
    // recursion share a distribution.
    let mut shape = None;
    let mut has_identity = false;
    for (outcome, _) in pool.ordered_outcomes() {
        if matches!(outcome, ElementValue::AdditiveIdentity) {
            has_identity = true;
        } else if shape.is_none() {
            shape = Some(outcome.clone());
        }
        if has_identity && shape.is_some() {
            break;
        }
    }
    // Every outcome that is not the identity maps to itself, so rebuilding the
    // pool only pays for itself when there is an identity to replace. Skipping
    // it keeps a long list off a rebuild it would not change.
    let pool = match (&shape, has_identity) {
        (Some(shape), true) => {
            pool.map_outcomes(|outcome| outcome.materialize_identity_like(shape))
        }
        _ => pool,
    };
    let is_tuple = matches!(shape, Some(ElementValue::Tuple(_)));
    let arity = match &shape {
        Some(ElementValue::Tuple(fields)) => fields.len(),
        // Nothing but empty sums, or no outcomes at all: a single int field.
        _ => 1,
    };
    let renders = field_renders(pool.ordered_outcomes(), arity);
    // The evaluator only accepts labels on tuple outputs, and checks that it
    // got one per field, so the names line up with the renders by position.
    let field_names = if is_tuple { field_names } else { None };
    let fields = renders
        .iter()
        .enumerate()
        .map(|(position, render)| Field {
            name: field_names
                .as_ref()
                .and_then(|names| names.get(position).cloned()),
            schema: render.schema(symbols),
        })
        .collect::<Box<_>>();
    let outcomes = pool.ordered_outcomes();
    let total: Natural = outcomes.iter().map(|(_, count)| count).sum();
    let mut values = Vec::with_capacity(outcomes.len().saturating_mul(arity));
    let mut probabilities = Vec::with_capacity(outcomes.len());
    for (value, count) in outcomes {
        let outcome_fields: &[ElementValue] = match value {
            ElementValue::Tuple(fields) => fields,
            other => std::slice::from_ref(other),
        };
        values.extend(
            outcome_fields
                .iter()
                .zip(&renders)
                .map(|(field, render)| render.ordinal(field)),
        );
        probabilities.push(
            f64::rounding_from(
                Rational::from_naturals(count.clone(), total.clone()),
                RoundingMode::Nearest,
            )
            .0,
        );
    }
    Distribution {
        fields,
        values: values.into_boxed_slice(),
        probabilities: probabilities.into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// Most outputs contain no symbols, so they need nothing from the table.
    fn distribution(value: RuntimeValue) -> Distribution {
        Distribution::from_runtime(value, None, &SymbolTable::default())
    }

    /// This exact representation is the frontend wire contract. If the private
    /// storage changes, a custom `Serialize` implementation must keep this test
    /// passing.
    #[cfg(feature = "serde")]
    #[test]
    fn distribution_wire_format_is_stable() {
        let distribution = Distribution {
            fields: vec![
                Field {
                    name: Some("Roll".into()),
                    schema: FieldSchema::Int,
                },
                Field {
                    name: None,
                    schema: FieldSchema::Categorical {
                        labels: vec!["MISS".into(), "HIT".into()],
                    },
                },
            ]
            .into_boxed_slice(),
            values: vec![20, 1].into_boxed_slice(),
            probabilities: vec![1.0].into_boxed_slice(),
        };

        assert_eq!(
            serde_lexpr::to_string(&distribution).unwrap(),
            "((fields ((name \"Roll\") (schema (kind . \"int\"))) \
             ((schema (kind . \"categorical\") (labels \"MISS\" \"HIT\")))) \
             (values 20 1) (probabilities 1.0))"
        );
    }

    #[test]
    fn converts_int_to_distribution() {
        let distribution = distribution(RuntimeValue::from(5));

        assert_eq!(
            distribution.entries().collect::<Vec<_>>(),
            [(&[5][..], 1.0)]
        );
    }

    #[test]
    fn converts_list_to_distribution_and_combines_duplicates() {
        let value = RuntimeValue::from(Arc::new(vec![
            5, -1, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 5, 5,
        ]));

        let distribution = distribution(value);

        assert_eq!(
            distribution.entries().collect::<Vec<_>>(),
            vec![
                (&[-1][..], 0.0625),
                (&[0][..], 0.0625),
                (&[1][..], 0.0625),
                (&[2][..], 0.0625),
                (&[3][..], 0.0625),
                (&[4][..], 0.0625),
                (&[5][..], 0.625),
            ]
        );
    }

    #[test]
    fn converts_pool_to_distribution_by_summing_dice() {
        let value = RuntimeValue::from(Pool::from_list(2, vec![1, 2]));

        let distribution = distribution(value);

        assert_eq!(
            distribution.entries().collect::<Vec<_>>(),
            vec![(&[2][..], 0.25), (&[3][..], 0.5), (&[4][..], 0.25)]
        );
    }
}
