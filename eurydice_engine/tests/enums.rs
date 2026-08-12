//! Declared symbols (`enum`) and the outcome type they share.

mod common;

use common::{
    distributions, error, error_summary, only_probabilities, owned_entries, probabilities, run,
};
use eurydice_engine::FieldSchema;

/// Every way of binding a variable reports a declared name the same way, and
/// points at the name itself.
#[test]
fn declared_names_cannot_be_bound_as_variables() {
    for (program, expected) in [
        (
            "enum { MISS, HIT } MISS: 3",
            "`MISS` is a declared symbol, so it cannot be used as a variable name",
        ),
        (
            "enum { MISS, HIT } loop MISS over {1} { output 1 }",
            "`MISS` is a declared symbol, so it cannot be used as a variable name",
        ),
        (
            "enum { MISS, HIT } function: f MISS:n { result: 1 }",
            "`MISS` is a declared symbol, so it cannot be used as a variable name",
        ),
    ] {
        assert_eq!(error_summary(program), expected, "{program}");
    }

    // The span covers the name and nothing else, wherever it appears.
    for (program, name) in [
        ("enum { A } A : 1", "A"),
        ("enum { A } loop A over {1} { output 1 }", "A"),
    ] {
        let range = error(program).primary_label.range;
        assert_eq!(
            &program[range.range.start..range.range.end],
            name,
            "{program}"
        );
    }
}

#[test]
fn symbol_declarations_do_not_replace_or_reuse_other_bindings() {
    for program in [
        "A: 1 enum: A",
        "enum: A enum: A",
        "enum { A, A }",
        "enum: A function: f A:n { result: A }",
        "enum: A loop A over {1} { output A }",
    ] {
        assert!(run(program).is_err(), "{program}");
    }
}

#[test]
fn grouped_declarations_are_sugar_for_individual_symbols() {
    let render = |program: &str| distributions(program).remove(0);
    let grouped = render("enum { A, B, C } output d{A, B, C}");
    let individual = render("enum: A enum: B enum: C output d{A, B, C}");
    assert_eq!(grouped, individual);
}

#[test]
fn rejects_enum_arithmetic_ordering_and_multidimensional_output() {
    for (program, expected) in [
        ("enum { A, B } output A + B", "requires numbers"),
        ("enum { A, B } output A < B", "requires numbers"),
        ("enum { A, B } output 2d{A, B}", "cannot be added together"),
    ] {
        let error = error_summary(program);
        assert!(error.contains(expected), "{error}");
    }
}

/// A pool of enum outcomes may have any dimension. It cannot be summed, so it
/// is only usable where multisets, not sums, are required.
#[test]
fn multidimensional_enum_pools_are_usable_without_being_summed() {
    // Multidimensional enum pools are constructible.
    let outputs = probabilities(
        r#"
        enum { MISS, HIT }
        function: hits S:s { result: [count {HIT} in S] }
        output [hits 2d{MISS, HIT}]
        output [count {HIT} in 2d{MISS, HIT}]
        output [2d{MISS, HIT} contains HIT]
        X: 2d{MISS, HIT}
        output #X
        "#,
    );
    // Two coin flips: 0, 1 or 2 hits with probabilities 1/4, 1/2, 1/4.
    let expected = vec![(vec![0], 0.25f64), (vec![1], 0.5f64), (vec![2], 0.25f64)];
    assert_eq!(outputs[0], expected);
    // Icepool's `count` must agree with iterating the multisets by hand.
    assert_eq!(outputs[1], expected);
    assert_eq!(outputs[2], vec![(vec![0], 0.25f64), (vec![1], 0.75f64)]);
    assert_eq!(outputs[3], vec![(vec![2], 1.0f64)]);
}

/// Multisets are sorted by declaration order, and `position order` picks the
/// end they start from. Selecting a single position is the only way to observe
/// this, since enum members support no ordering comparisons of their own.
#[test]
fn enum_multisets_are_ordered_by_declaration_and_respect_position_order() {
    let program = |setting: &str| {
        format!(
            r#"
            enum {{ MISS, HIT }}
            function: first S:s {{ result: 1@S }}
            {setting}
            output [first 3d{{MISS, HIT}}]
            "#
        )
    };
    let first = |program: String| only_probabilities(&program);
    // Highest first (the default): position 1 is MISS only when all three miss.
    assert_eq!(
        first(program("")),
        vec![(vec![0], 0.125f64), (vec![1], 0.875f64)]
    );
    // Lowest first: position 1 is HIT only when all three hit.
    assert_eq!(
        first(program(r#"set "position order" to "lowest first""#)),
        vec![(vec![0], 0.875f64), (vec![1], 0.125f64)]
    );
}

/// Every operation that would sum a multidimensional non-additive pool must
/// report an error rather than silently ignoring the pool's dimension.
#[test]
fn operations_that_would_sum_a_multidimensional_enum_pool_are_rejected() {
    for program in [
        "enum { A, B } output 2d{A, B}",
        "enum { A, B } output {2d{A, B}}",
        "enum { A, B } output {2d{A, B}:2}",
        "enum { A, B } function: f X:n { result: X } output [f 2d{A, B}]",
        "enum { A, B } output 2d{A, B} = A",
        "enum { A, B } output A != 2d{A, B}",
        "enum { A, B } output -2d{A, B}",
        "enum { A, B } output (0-2)d{A, B}",
        "enum { A, B } output (0-1)d(d{A, B})",
        "enum { A, B } output d2 d {A, B}",
        "enum { A, B } output d2 d (2d{A, B})",
        "enum { A, B } function: f S:s { result: S } output [f 2d{A, B}]",
        "enum { A, B } function: f D:d { result: D } output [f 2d{A, B}]",
        "enum { A, B } output [choose 2d{A, B} if 1 else 2d{A, B}]",
        "enum { A, B } function: g X:n { result: 2d{A, B} } output [g d2]",
    ] {
        assert!(run(program).is_err(), "{program}");
    }
}

/// Rolling no dice adds nothing, so it never meets the faces at all: `0d{A, B}`
/// is the empty sum, and displays as the `0` that `0d6` does. Nothing about the
/// faces can make an empty sum fail, because nothing is ever added to it.
#[test]
fn rolling_no_enum_dice_yields_the_empty_sum() {
    // No dice, nothing to add.
    assert_eq!(
        only_probabilities("enum { A, B } output 0d{A, B}"),
        vec![(vec![0], 1.0f64)]
    );
}

/// Loosening pool construction must not loosen the operations that were already
/// unavailable for enum outcomes.
#[test]
fn multidimensional_enum_pools_do_not_gain_ordering_or_arithmetic() {
    for program in [
        "enum { A, B } output 1@2d{A, B}",
        "enum { A, B } output [sort {A, B}]",
        "enum { A, B } output [highest 1 of 2d{A, B}]",
        "enum { A, B } output [lowest 1 of 2d{A, B}]",
        "enum { A, B } output [maximum of 2d{A, B}]",
        "enum { A, B } output [explode 2d{A, B}]",
        "enum { A, B } output 2d{A, B} < A",
        "enum { A, B } output 2d{A, B} + A",
    ] {
        assert!(run(program).is_err(), "{program}");
    }
}

#[test]
fn rejects_nested_declarations() {
    let nested = error_summary("if 1 { enum { A } }");
    assert!(nested.contains("top level"), "{nested}");
}

/// Independently declared symbols share one outcome type and can share a
/// collection. The display includes the symbols that actually occur.
#[test]
fn independently_declared_symbols_share_one_outcome_type() {
    let distribution = distributions("enum: A enum: B output {A, B}").remove(0);
    let FieldSchema::Categorical { labels, .. } = &distribution.fields()[0].schema else {
        panic!("expected a symbol field");
    };
    assert_eq!(labels, &["A", "B"]);
    assert_eq!(
        owned_entries(&distribution),
        vec![(vec![0], 0.5), (vec![1], 0.5)]
    );
}

/// Equality is total: independently declared symbols are simply never equal.
#[test]
fn equality_aware_operations_are_total_across_symbols() {
    for program in [
        "enum { A } enum { B } output A = B",
        "enum { A } enum { B } output [{A} contains B]",
        "enum { A } enum { B } output [d{A} contains B]",
        "enum { A } enum { B } output [count {A} in {B}]",
        "enum { A } enum { B } output [count {A} in d{B}]",
    ] {
        assert_eq!(
            only_probabilities(program),
            vec![(vec![0], 1.0)],
            "{program}"
        );
    }

    // Inequality is the complement, not an error.
    assert_eq!(
        only_probabilities("enum { A } enum { B } output A != B"),
        vec![(vec![1], 1.0)]
    );
}

#[test]
fn distribution_keeps_numeric_probabilities_and_enum_labels() {
    let output = distributions("enum { MISS, HIT } output d{MISS, HIT}").remove(0);
    assert_eq!(output.entries().len(), 2);
    let FieldSchema::Categorical { labels, .. } = &output.fields()[0].schema else {
        panic!("expected enum field");
    };
    assert_eq!(labels, &["MISS", "HIT"]);
}

#[test]
fn symbol_fields_only_include_observed_values() {
    let output = distributions("enum { A, B, C } output d{A, B}").remove(0);
    let FieldSchema::Categorical { labels, .. } = &output.fields()[0].schema else {
        panic!("expected enum field");
    };
    assert_eq!(labels, &["A", "B"]);
}

#[test]
fn tuple_distribution_hoists_field_schema() {
    let dist =
        distributions("enum { MISS, HIT } A: d2 B: d{MISS, HIT} output [tuple A B]").remove(0);
    assert!(dist.fields().iter().all(|field| field.name.is_none()));

    // The per-field schema is stored once, not repeated on each outcome.
    assert!(matches!(dist.fields()[0].schema, FieldSchema::Int));
    let FieldSchema::Categorical { labels, .. } = &dist.fields()[1].schema else {
        panic!("expected enum field schema");
    };
    assert_eq!(labels, &["MISS", "HIT"]);

    // Every outcome is a raw i32 vector matching the field count, and the
    // probabilities form a valid distribution.
    assert!(dist.entries().all(|(values, _)| values.len() == 2));
    let total: f64 = dist.entries().map(|(_, probability)| probability).sum();
    assert!((total - 1.0).abs() < 1e-9, "probabilities should sum to 1");
}
