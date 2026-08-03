use eurydice_engine::{
    Engine,
    eval::{EvaluatedOutput, Evaluator},
    grammar,
    output::{Distribution, FieldSchema},
};

fn run(program: &str) -> Result<Vec<EvaluatedOutput>, String> {
    let statements = grammar::BodyParser::new()
        .parse(program)
        .map_err(|error| format!("{error:?}"))?;
    let mut evaluator = Evaluator::new();
    for statement in statements {
        evaluator
            .execute(&statement)
            .map_err(|error| format!("{error:?}"))?;
    }
    Ok(evaluator.take_outputs())
}

/// The message a failing program shows the user.
fn error_summary(program: &str) -> String {
    Engine::new()
        .run_with_diagnostics(program)
        .error()
        .unwrap_or_else(|| panic!("expected an error for {program}"))
        .summary
        .clone()
}

#[test]
fn rejects_mixed_enum_list() {
    let error = error_summary(
        r#"
        enum: RESULT { MISS, HIT }
        output {MISS, 1}
        "#,
    );
    assert!(error.contains("mixed outcome types"), "{error}");
}

#[test]
fn enum_names_and_members_are_immutable() {
    for program in [
        "enum: RESULT { MISS, HIT } MISS: 3",
        "enum: RESULT { MISS, HIT } RESULT: 3",
    ] {
        let error = error_summary(program);
        assert!(error.contains("immutable"), "{error}");
    }
}

#[test]
fn enum_names_do_not_replace_or_get_reused_by_other_bindings() {
    for program in [
        "RESULT: 1 enum: RESULT { MISS }",
        "enum: RESULT { RESULT }",
        "enum: RESULT { MISS } enum: OTHER { RESULT }",
        "enum: RESULT { MISS } function: f RESULT:n { result: RESULT }",
        "enum: RESULT { MISS } loop RESULT over {1} { output RESULT }",
    ] {
        assert!(run(program).is_err(), "{program}");
    }
}

#[test]
fn rejects_enum_arithmetic_ordering_and_multidimensional_output() {
    for (program, expected) in [
        ("enum: R { A, B } output A + B", "not defined"),
        ("enum: R { A, B } output A < B", "not defined"),
        (
            "enum: R { A, B } output 2d{A, B}",
            "cannot be added together",
        ),
    ] {
        let error = error_summary(program);
        assert!(error.contains(expected), "{error}");
    }
}

/// A pool of enum outcomes may have any dimension. It cannot be summed, so it
/// is only usable where multisets, not sums, are required.
#[test]
fn multidimensional_enum_pools_are_usable_without_being_summed() {
    let outputs = run(r#"
        enum: R { MISS, HIT }
        function: hits S:s { result: [count {HIT} in S] }
        output [hits 2d{MISS, HIT}]
        output [count {HIT} in 2d{MISS, HIT}]
        output [2d{MISS, HIT} contains HIT]
        X: 2d{MISS, HIT}
        output #X
        "#)
    .expect("multidimensional enum pools are constructible");
    let probabilities = |output: &EvaluatedOutput| {
        Distribution::from_runtime(output.value.clone(), None).probabilities
    };
    // Two coin flips: 0, 1 or 2 hits with probabilities 1/4, 1/2, 1/4.
    let expected = vec![(vec![0], 0.25f64), (vec![1], 0.5f64), (vec![2], 0.25f64)];
    assert_eq!(probabilities(&outputs[0]), expected);
    // Icepool's `count` must agree with iterating the multisets by hand.
    assert_eq!(probabilities(&outputs[1]), expected);
    assert_eq!(
        probabilities(&outputs[2]),
        vec![(vec![0], 0.25f64), (vec![1], 0.75f64)]
    );
    assert_eq!(probabilities(&outputs[3]), vec![(vec![2], 1.0f64)]);
}

/// Multisets are sorted by declaration order, and `position order` picks the end
/// they start from. Selecting a single position is the only way to observe this,
/// since enum members support no ordering comparisons of their own.
#[test]
fn enum_multisets_are_ordered_by_declaration_and_respect_position_order() {
    let program = |setting: &str| {
        format!(
            r#"
            enum: R {{ MISS, HIT }}
            function: first S:s {{ result: 1@S }}
            {setting}
            output [first 3d{{MISS, HIT}}]
            "#
        )
    };
    let probabilities = |program: String| {
        let outputs = run(&program).expect("program runs");
        Distribution::from_runtime(outputs[0].value.clone(), None).probabilities
    };
    // Highest first (the default): position 1 is MISS only when all three miss.
    assert_eq!(
        probabilities(program("")),
        vec![(vec![0], 0.125f64), (vec![1], 0.875f64)]
    );
    // Lowest first: position 1 is HIT only when all three hit.
    assert_eq!(
        probabilities(program(r#"set "position order" to "lowest first""#)),
        vec![(vec![0], 0.875f64), (vec![1], 0.125f64)]
    );
}

/// Every operation that would sum a multidimensional non-additive pool must
/// report an error rather than silently ignoring the pool's dimension.
#[test]
fn operations_that_would_sum_a_multidimensional_enum_pool_are_rejected() {
    for program in [
        "enum: R { A, B } output 2d{A, B}",
        "enum: R { A, B } output {2d{A, B}}",
        "enum: R { A, B } output {2d{A, B}:2}",
        "enum: R { A, B } function: f X:n { result: X } output [f 2d{A, B}]",
        "enum: R { A, B } output 2d{A, B} = A",
        "enum: R { A, B } output A != 2d{A, B}",
        "enum: R { A, B } output -2d{A, B}",
        "enum: R { A, B } output (0-2)d{A, B}",
        "enum: R { A, B } output (0-1)d(d{A, B})",
        "enum: R { A, B } output d2 d {A, B}",
        "enum: R { A, B } output d2 d (2d{A, B})",
        "enum: R { A, B } output 0d{A, B}",
        "enum: R { A, B } function: f S:s { result: S } output [f 2d{A, B}]",
        "enum: R { A, B } function: f D:d { result: D } output [f 2d{A, B}]",
        "enum: R { A, B } output [choose 2d{A, B} if 1 else 2d{A, B}]",
        "enum: R { A, B } function: g X:n { result: 2d{A, B} } output [g d2]",
    ] {
        assert!(run(program).is_err(), "{program}");
    }
}

/// Loosening pool construction must not loosen the operations that were already
/// unavailable for enum outcomes.
#[test]
fn multidimensional_enum_pools_do_not_gain_ordering_or_arithmetic() {
    for program in [
        "enum: R { A, B } output 1@2d{A, B}",
        "enum: R { A, B } output [sort {A, B}]",
        "enum: R { A, B } output [highest 1 of 2d{A, B}]",
        "enum: R { A, B } output [lowest 1 of 2d{A, B}]",
        "enum: R { A, B } output [maximum of 2d{A, B}]",
        "enum: R { A, B } output [explode 2d{A, B}]",
        "enum: R { A, B } output 2d{A, B} < A",
        "enum: R { A, B } output 2d{A, B} + A",
    ] {
        assert!(run(program).is_err(), "{program}");
    }
}

#[test]
fn rejects_different_enum_types_and_nested_declarations() {
    let mixed = error_summary("enum: A_TYPE { A } enum: B_TYPE { B } output {A, B}");
    assert!(mixed.contains("mixed outcome types"), "{mixed}");

    let nested = error_summary("if 1 { enum: RESULT { A } }");
    assert!(nested.contains("top level"), "{nested}");
}

#[test]
fn equality_aware_operations_require_the_same_enum_type() {
    for program in [
        "enum: A_TYPE { A } enum: B_TYPE { B } output A = B",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [{A} contains B]",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [d{A} contains B]",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [count {A} in {B}]",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [count {A} in d{B}]",
    ] {
        let error = error_summary(program);
        assert!(error.contains("same outcome type"), "{error}");
    }
}

#[test]
fn serialized_distribution_keeps_numeric_probabilities_and_enum_labels() {
    let mut outputs = run("enum: RESULT { MISS, HIT } output d{MISS, HIT}").unwrap();
    let output = Distribution::from(outputs.remove(0).value);
    assert_eq!(output.probabilities.len(), 2);
    let FieldSchema::Enum { enum_name, labels } = &output.fields[0] else {
        panic!("expected enum field");
    };
    assert_eq!(enum_name, "RESULT");
    assert_eq!(labels, &["MISS", "HIT"]);
}

#[test]
fn serialized_tuple_distribution_hoists_field_schema() {
    let mut outputs =
        run("enum: RESULT { MISS, HIT } A: d2 B: d{MISS, HIT} output [tuple A B]").unwrap();
    let dist = Distribution::from(outputs.remove(0).value);
    assert!(dist.field_names.is_none());

    // The per-field schema is stored once, not repeated on each outcome.
    assert!(matches!(dist.fields[0], FieldSchema::Int));
    let FieldSchema::Enum { enum_name, labels } = &dist.fields[1] else {
        panic!("expected enum field schema");
    };
    assert_eq!(enum_name, "RESULT");
    assert_eq!(labels, &["MISS", "HIT"]);

    // Every outcome is a raw i32 vector matching the field count, and the
    // probabilities form a valid distribution.
    assert!(
        dist.probabilities
            .iter()
            .all(|(values, _)| values.len() == 2)
    );
    let total: f64 = dist.probabilities.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-9, "probabilities should sum to 1");
}
