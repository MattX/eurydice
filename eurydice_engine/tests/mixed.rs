//! Collections whose outcomes are not all of one kind.
//!
//! The motivating shape is a die with mostly numeric faces plus a symbolic
//! one, such as `d{0:2, 1:2, 2, TIMES_TWO}`.

use eurydice_engine::{
    DiagnosticCode, Engine, EngineDiagnostic, LabelStyle,
    eval::{EvaluatedOutput, Evaluator, SymbolTable},
    grammar,
    output::{Distribution, FieldSchema},
};

/// Runs a program, returning its outputs together with the symbol table needed
/// to render them: a symbol value is an index, and the evaluator that assigned
/// it does not outlive this call.
fn run(program: &str) -> Result<(Vec<EvaluatedOutput>, SymbolTable), String> {
    let statements = grammar::BodyParser::new()
        .parse(program)
        .map_err(|error| format!("{error:?}"))?;
    let mut evaluator = Evaluator::new();
    for statement in statements {
        evaluator
            .execute(&statement)
            .map_err(|error| format!("{error:?}"))?;
    }
    let symbols = evaluator.symbols().clone();
    Ok((evaluator.take_outputs(), symbols))
}

/// The probabilities of every output of a program that is expected to run.
fn probabilities(program: &str) -> Vec<Vec<(Vec<i32>, f64)>> {
    let (outputs, symbols) = run(program).unwrap_or_else(|error| panic!("{program}: {error}"));
    outputs
        .into_iter()
        .map(|output| {
            Distribution::from_runtime(output.value, output.field_names, &symbols).probabilities
        })
        .collect()
}

fn error(program: &str) -> EngineDiagnostic {
    Engine::new()
        .run_with_diagnostics(program)
        .error()
        .unwrap_or_else(|| panic!("expected an error for {program}"))
        .clone()
}

/// The die this feature exists for, declared as a list and as a pool.
const DIE: &str = "enum { TIMES_TWO } DIE: d{0:2, 1:2, 2, TIMES_TWO}";

/// Nothing inspects a sequence until an operation needs to. A thousand-element
/// sequence with one symbol near the end is built without complaint, and it is
/// summing it that fails — naming the value that could not be added, not the
/// sequence that held it.
#[test]
fn a_long_sequence_fails_at_the_value_that_cannot_be_added() {
    let program = "enum { ODD_ONE_OUT } SEQ: {1..998, ODD_ONE_OUT, 1000}";

    // Building and inspecting it is fine, and it really is 1000 long.
    assert_eq!(
        probabilities(&format!("{program} output #SEQ")),
        vec![vec![(vec![1000], 1.0f64)]]
    );

    // Summing it is not, and the error names the element at fault rather than
    // stopping at the literal that holds it.
    for summing in [
        format!("{program} output SEQ + 0"),
        format!("{program} function: total S:n {{ result: S }} output [total SEQ]"),
    ] {
        let diagnostic = error(&summing);
        assert_eq!(
            diagnostic.code,
            DiagnosticCode::NonAdditiveValue,
            "{summing}"
        );
        assert!(
            diagnostic.summary.contains("a sequence of 1000 values"),
            "{}",
            diagnostic.summary
        );
        assert_eq!(
            diagnostic.labels[0].message.as_deref(),
            Some("values like `ODD_ONE_OUT` are not numbers"),
            "{summing}"
        );
    }
}

#[test]
fn lists_and_pools_can_mix_numbers_and_symbols() {
    // Six equally likely faces, three of which contribute nothing to a sum.
    assert_eq!(
        probabilities(&format!("{DIE} output [sum integers in DIE]")),
        vec![vec![
            (vec![0], 3.0 / 6.0),
            (vec![1], 2.0 / 6.0),
            (vec![2], 1.0 / 6.0),
        ]]
    );
    // A mixed list is a list like any other; `#` counts its elements.
    assert_eq!(
        probabilities("enum { A } output #{1, A, 2}"),
        vec![vec![(vec![3], 1.0)]]
    );
}

/// A value pulled out of a mixed pool has no memory of the union it came from,
/// so equality has to be total for the central idiom to work at all.
#[test]
fn equality_discriminates_symbols_inside_a_mixed_pool() {
    assert_eq!(
        probabilities(&format!(
            "{DIE} function: doubler X:n {{ result: X = TIMES_TWO }} output [doubler DIE]"
        )),
        vec![vec![(vec![0], 5.0 / 6.0), (vec![1], 1.0 / 6.0)]]
    );
}

#[test]
fn integers_are_summed_and_counted_across_a_whole_pool() {
    // Four dice, each an integer with probability 5/6.
    let all_ints = (5.0f64 / 6.0).powi(4);
    let counts = probabilities(&format!("{DIE} output [count integers in 4dDIE]"));
    assert_eq!(counts[0].len(), 5);
    let (outcome, probability) = counts[0].last().unwrap();
    assert_eq!(outcome, &vec![4]);
    assert!((probability - all_ints).abs() < 1e-9, "{probability}");

    // Summing ignores the symbols rather than failing on them, at any dimension.
    let sums = probabilities(&format!("{DIE} output [sum integers in 4dDIE]"));
    assert_eq!(sums[0].first().unwrap().0, vec![0]);
    assert_eq!(sums[0].last().unwrap().0, vec![8]);
    let total: f64 = sums[0].iter().map(|(_, probability)| probability).sum();
    assert!((total - 1.0).abs() < 1e-9, "{total}");

    // Sequences take the same route as pools.
    assert_eq!(
        probabilities("enum { A } output [sum integers in {1, A, 2}]"),
        vec![vec![(vec![3], 1.0)]]
    );
    assert_eq!(
        probabilities("enum { A } output [count integers in {1, A, 2}]"),
        vec![vec![(vec![2], 1.0)]]
    );
}

#[test]
fn is_integer_distinguishes_the_faces_of_a_mixed_die() {
    assert_eq!(
        probabilities(&format!("{DIE} output [DIE is integer]")),
        vec![vec![(vec![0], 1.0 / 6.0), (vec![1], 5.0 / 6.0)]]
    );
    assert_eq!(
        probabilities("enum { A } output [A is integer] output [1 is integer]"),
        vec![vec![(vec![0], 1.0)], vec![(vec![1], 1.0)]]
    );
}

/// The worked example: total the numeric faces, then double once per special
/// face. `POOL:s` keeps the sum and the count correlated within each multiset.
///
/// Checked against an independent enumeration of all 6^4 face combinations.
#[test]
fn the_motivating_program_matches_a_brute_force_enumeration() {
    let program = format!(
        r#"
        {DIE}
        function: score POOL:s {{
          result: [sum integers in POOL] * (2 ^ [count {{TIMES_TWO}} in POOL])
        }}
        output [score 4dDIE]
        "#
    );
    let expected: Vec<(Vec<i32>, f64)> = [
        (0, 81),
        (1, 64),
        (2, 224),
        (3, 160),
        (4, 328),
        (5, 80),
        (6, 160),
        (7, 8),
        (8, 129),
        (10, 24),
        (12, 28),
        (16, 10),
    ]
    .into_iter()
    .map(|(outcome, weight)| (vec![outcome], f64::from(weight) / 1296.0))
    .collect();
    let actual = probabilities(&program).remove(0);
    assert_eq!(actual.len(), expected.len());
    for ((actual_outcome, actual_p), (expected_outcome, expected_p)) in
        actual.iter().zip(expected.iter())
    {
        assert_eq!(actual_outcome, expected_outcome);
        assert!(
            (actual_p - expected_p).abs() < 1e-9,
            "{actual_outcome:?}: {actual_p} != {expected_p}"
        );
    }
}

/// A tuple built over a mixed die has a mixed *field*, which is the one kind of
/// union that arises without anyone asking for it. It can be displayed
/// categorically or projected back out as an ordinary mixed scalar.
#[test]
fn tuples_carry_a_mixed_field_through_construction_and_projection() {
    let program = format!("{DIE} PAIR: [tuple DIE 1] output [sum integers in [field 1 of PAIR]]");
    assert_eq!(
        probabilities(&program),
        vec![vec![
            (vec![0], 3.0 / 6.0),
            (vec![1], 2.0 / 6.0),
            (vec![2], 1.0 / 6.0),
        ]]
    );

    let (outputs, symbols) = run(&format!("{DIE} output [tuple DIE 1]")).expect("runs");
    let distribution = Distribution::from_runtime(outputs[0].value.clone(), None, &symbols);
    assert_eq!(
        distribution.fields,
        vec![
            FieldSchema::Enum {
                labels: vec!["0".into(), "1".into(), "2".into(), "TIMES_TWO".into()],
            },
            FieldSchema::Int,
        ]
    );
    assert_eq!(
        distribution.probabilities,
        vec![
            (vec![0, 1], 2.0 / 6.0),
            (vec![1, 1], 2.0 / 6.0),
            (vec![2, 1], 1.0 / 6.0),
            (vec![3, 1], 1.0 / 6.0),
        ]
    );
}

/// A sequence holds whatever it was written with. Mixing a scalar and a tuple
/// is only a problem for an operation that has to combine them — and the two
/// that always do are summing and displaying.
#[test]
fn shapes_mix_freely_until_something_has_to_combine_them() {
    // Building one is fine, and `print` shows it as it is.
    for program in [
        "X: {1, [tuple 1 2]} print X",
        "X: {[tuple 1 2], [tuple 1 2 3]} print X",
        "enum { A } X: {A, [tuple 1 2]} print X",
    ] {
        assert!(run(program).is_ok(), "{program}");
    }

    for program in [
        "output {1, [tuple 1 2]}",
        "output {[tuple 1 2], [tuple 1 2 3]}",
        "enum { A } output {A, [tuple 1 2]}",
        "output d{1, [tuple 1 2]}",
        "function: f X:n { if X { result: 1 } result: [tuple 1 2] } output [f d{0, 1}]",
    ] {
        // The wording depends on what had to combine them; the code does not.
        assert_eq!(
            error(program).code,
            DiagnosticCode::OutcomeMismatch,
            "{program}"
        );
    }
}

/// The shapes that could not be combined are named, and so are the two values
/// that had them — which is what the error can point at now that no expression
/// is at fault for holding them together.
#[test]
fn a_shape_mismatch_names_both_shapes_and_values() {
    let diagnostic = error("output {[tuple 1 2], 7, [tuple 1 2 3]}");
    assert_eq!(
        diagnostic.summary,
        "A tuple of 2 fields and a single value cannot be combined"
    );
    let primary = diagnostic
        .labels
        .iter()
        .find(|label| label.style == LabelStyle::Primary)
        .expect("a primary label");
    let message = primary.message.as_deref().expect("a labelled primary");
    assert!(message.contains("displaying a distribution"), "{message}");
    assert!(
        message.contains("`(1, 2)` is a tuple of 2 fields"),
        "{message}"
    );
    assert!(message.contains("`7` is a single value"), "{message}");

    // Arity is named too, not just "tuple".
    assert_eq!(
        error("output {[tuple 1 2], [tuple 1 2 3]}").summary,
        "A tuple of 2 fields and a tuple of 3 fields cannot be combined"
    );
}

/// A function returning different shapes for different multisets builds a die
/// that holds both. Displaying it is what cannot be done, and the error names
/// the two outcomes rather than the call.
#[test]
fn mismatched_function_results_are_reported_when_displayed() {
    let diagnostic =
        error("function: f X:n { if X { result: 1 } result: [tuple 1 2] } output [f d{0, 1}]");
    assert_eq!(
        diagnostic.summary,
        "A single value and a tuple of 2 fields cannot be combined"
    );
    let message = diagnostic.labels[0]
        .message
        .as_deref()
        .expect("a labelled primary");
    assert!(message.contains("`1` is a single value"), "{message}");
    assert!(
        message.contains("`(1, 2)` is a tuple of 2 fields"),
        "{message}"
    );
}

/// A pool that cannot be summed names an outcome the user can recognize, and
/// for a tuple says which field is at fault.
#[test]
fn a_non_additive_sum_names_the_offending_value() {
    let scalar = error("enum { A, B } output 2d{A, B}");
    assert_eq!(
        scalar.summary,
        "displaying a pool requires summing a pool of 2 dice, but they cannot be added together"
    );
    assert_eq!(
        scalar.labels[0].message.as_deref(),
        Some("outcomes like `A` are not numbers")
    );
    assert!(
        scalar.help.as_deref().unwrap().contains("sum integers in"),
        "{:?}",
        scalar.help
    );

    let tuple = error("enum { A } output 2d{[tuple 1 2 A]}");
    assert_eq!(
        tuple.labels[0].message.as_deref(),
        Some("field 3 of outcomes like `(1, 2, A)` is not a number")
    );
    assert!(
        tuple.help.as_deref().unwrap().contains("field by field"),
        "{:?}",
        tuple.help
    );

    // A multiset returned from a pool evaluation is a sequence, not a pool.
    let sequence = error("enum { A, B } function: f X:s { result: X } output [f 2d{A, B}]");
    assert!(
        sequence.summary.contains("summing a sequence of 2 values"),
        "{}",
        sequence.summary
    );
    assert_eq!(
        sequence.labels[0].message.as_deref(),
        Some("values like `A` are not numbers")
    );
}

/// The empty sum has no shape of its own, so it shares a sequence with anything
/// at all. It settles into the `0` it stands for the moment something numeric
/// is alongside it, and is rendered as a categorical zero beside a symbol.
#[test]
fn the_empty_sum_takes_the_shape_of_whatever_it_meets() {
    assert_eq!(
        probabilities("X: {} + {} output {X, 1}"),
        vec![vec![(vec![0], 0.5f64), (vec![1], 0.5f64)]]
    );

    let (outputs, symbols) = run("enum { A } X: {} + {} output {X, A}").expect("mixed output runs");
    let distribution = Distribution::from_runtime(outputs[0].value.clone(), None, &symbols);
    assert_eq!(
        distribution.fields,
        vec![FieldSchema::Enum {
            labels: vec!["0".into(), "A".into()],
        }]
    );
    assert_eq!(
        distribution.probabilities,
        vec![(vec![0], 0.5), (vec![1], 0.5)]
    );
}

/// A field with numeric and symbolic outcomes is serialized categorically. Its
/// dense ordinals keep unlike values distinct even when their ordinary numeric
/// encodings would collide.
#[test]
fn mixed_fields_display_as_categories() {
    for program in ["enum { A } output {A, 1}", "enum { A } output d{A, 1}"] {
        let (outputs, symbols) = run(program).expect("mixed output runs");
        let distribution = Distribution::from_runtime(outputs[0].value.clone(), None, &symbols);
        assert_eq!(
            distribution.fields,
            vec![FieldSchema::Enum {
                labels: vec!["1".into(), "A".into()],
            }],
            "{program}"
        );
        assert_eq!(
            distribution.probabilities,
            vec![(vec![0], 0.5), (vec![1], 0.5)],
            "{program}"
        );
    }

    let (outputs, symbols) =
        run("enum { A } output [tuple 1, d{A, 2}]").expect("mixed tuple output runs");
    let distribution = Distribution::from_runtime(outputs[0].value.clone(), None, &symbols);
    assert_eq!(
        distribution.fields,
        vec![
            FieldSchema::Int,
            FieldSchema::Enum {
                labels: vec!["2".into(), "A".into()],
            },
        ]
    );
}

/// A distribution that is *all* symbols is an ordinary categorical output, even
/// though its outcome type is the same one a mixed pool has.
#[test]
fn all_symbol_distributions_still_display() {
    let (outputs, symbols) = run("enum { MISS, HIT } output d{MISS, HIT}").expect("runs");
    let distribution = Distribution::from_runtime(outputs[0].value.clone(), None, &symbols);
    let FieldSchema::Enum { labels } = &distribution.fields[0] else {
        panic!("expected a symbol field");
    };
    assert_eq!(labels, &["MISS", "HIT"]);
}

/// Displaying one mixed die does not make a multidimensional mixed pool
/// additive. Multiset iteration still works, while summing, arithmetic, and
/// ordering remain unavailable just as they are for an all-symbol pool.
#[test]
fn mixed_pools_gain_no_arithmetic_or_ordering() {
    for program in [
        "output 4dDIE",
        "output 2dDIE + 1",
        "output [highest 2 of 4dDIE]",
        "output [sort {1, TIMES_TWO}]",
        "output DIE < 1",
        "output -2dDIE",
        "output [maximum of 4dDIE]",
        "output [explode 4dDIE]",
    ] {
        let program = format!("{DIE} {program}");
        assert!(run(&program).is_err(), "{program}");
    }
}

/// Iteration order across kinds: integers first, in numeric order, then
/// symbols in declaration order. Selecting one position is the only way to
/// observe it, since mixed outcomes have no ordering comparisons.
#[test]
fn integers_sort_before_symbols_in_multisets() {
    let program = |setting: &str| {
        format!(
            r#"
            enum {{ A }}
            function: first S:s {{ result: [1@S is integer] }}
            {setting}
            output [first 2d{{1, A}}]
            "#
        )
    };
    // Highest first: position 1 is an integer only when both dice roll 1.
    assert_eq!(
        probabilities(&program("")),
        vec![vec![(vec![0], 0.75), (vec![1], 0.25)]]
    );
    // Lowest first: position 1 is a symbol only when both dice roll A.
    assert_eq!(
        probabilities(&program(r#"set "position order" to "lowest first""#)),
        vec![vec![(vec![0], 0.25), (vec![1], 0.75)]]
    );
}
