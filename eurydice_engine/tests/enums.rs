use eurydice_engine::{
    Engine,
    eval::{EvaluatedOutput, Evaluator, SymbolTable},
    grammar,
    output::{Distribution, FieldSchema},
};

/// Runs a program, returning its outputs together with the symbol table
/// needed to render them: a symbol value is an index, and the evaluator that
/// assigned it does not outlive this call.
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

/// The message a failing program shows the user.
fn error_summary(program: &str) -> String {
    Engine::new()
        .run_with_diagnostics(program)
        .error()
        .unwrap_or_else(|| panic!("expected an error for {program}"))
        .summary
        .clone()
}

/// Every way of binding a variable reports a declared name the same way, and
/// points at the name itself.
#[test]
fn declared_names_cannot_be_bound_as_variables() {
    for (program, expected) in [
        (
            "enum: RESULT { MISS, HIT } MISS: 3",
            "`MISS` is a declared symbol, so it cannot be used as a variable name",
        ),
        (
            "enum: RESULT { MISS, HIT } loop MISS over {1} { output 1 }",
            "`MISS` is a declared symbol, so it cannot be used as a variable name",
        ),
        (
            "enum: RESULT { MISS, HIT } function: f MISS:n { result: 1 }",
            "`MISS` is a declared symbol, so it cannot be used as a variable name",
        ),
        (
            "enum: RESULT { MISS, HIT } RESULT: 3",
            "`RESULT` is a declared symbol set, so it cannot be used as a variable name",
        ),
        (
            "enum: RESULT { MISS, HIT } loop RESULT over {1} { output 1 }",
            "`RESULT` is a declared symbol set, so it cannot be used as a variable name",
        ),
    ] {
        assert_eq!(error_summary(program), expected, "{program}");
    }

    // The span covers the name and nothing else, wherever it appears.
    for (program, name) in [
        ("enum: R { A } A : 1", "A"),
        ("enum: R { A } loop A over {1} { output 1 }", "A"),
    ] {
        let diagnostic = Engine::new()
            .run_with_diagnostics(program)
            .error()
            .unwrap_or_else(|| panic!("expected an error for {program}"))
            .clone();
        let range = diagnostic.primary_range().expect("a primary range");
        assert_eq!(
            &program[range.range.start..range.range.end],
            name,
            "{program}"
        );
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
    let (outputs, symbols) = run(r#"
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
        Distribution::from_runtime(output.value.clone(), None, &symbols).probabilities
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
        let (outputs, symbols) = run(&program).expect("program runs");
        Distribution::from_runtime(outputs[0].value.clone(), None, &symbols).probabilities
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
fn rejects_nested_declarations() {
    let nested = error_summary("if 1 { enum: RESULT { A } }");
    assert!(nested.contains("top level"), "{nested}");
}

/// Declared sets name a domain; they are not separate types. Symbols from two
/// sets share one outcome type, so they can share a collection — and the
/// display picks up both sets' labels.
#[test]
fn symbols_from_different_sets_share_one_outcome_type() {
    let (mut outputs, symbols) =
        run("enum: A_TYPE { A } enum: B_TYPE { B } output {A, B}").expect("program runs");
    let distribution = Distribution::from_runtime(outputs.remove(0).value, None, &symbols);
    let FieldSchema::Enum { enum_name, labels } = &distribution.fields[0] else {
        panic!("expected a symbol field");
    };
    assert_eq!(enum_name, "A_TYPE | B_TYPE");
    assert_eq!(labels, &["A", "B"]);
    assert_eq!(
        distribution.probabilities,
        vec![(vec![0], 0.5), (vec![1], 0.5)]
    );
}

/// Equality is total: symbols from different sets are simply never equal.
#[test]
fn equality_aware_operations_are_total_across_sets() {
    for program in [
        "enum: A_TYPE { A } enum: B_TYPE { B } output A = B",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [{A} contains B]",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [d{A} contains B]",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [count {A} in {B}]",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [count {A} in d{B}]",
    ] {
        let (outputs, symbols) = run(program).expect("program runs");
        let distribution = Distribution::from_runtime(outputs[0].value.clone(), None, &symbols);
        assert_eq!(
            distribution.probabilities,
            vec![(vec![0], 1.0)],
            "{program}"
        );
    }

    // Inequality is the complement, not an error.
    let (outputs, symbols) =
        run("enum: A_TYPE { A } enum: B_TYPE { B } output A != B").expect("program runs");
    assert_eq!(
        Distribution::from_runtime(outputs[0].value.clone(), None, &symbols).probabilities,
        vec![(vec![1], 1.0)]
    );
}

#[test]
fn serialized_distribution_keeps_numeric_probabilities_and_enum_labels() {
    let (mut outputs, symbols) = run("enum: RESULT { MISS, HIT } output d{MISS, HIT}").unwrap();
    let output = Distribution::from_runtime(outputs.remove(0).value, None, &symbols);
    assert_eq!(output.probabilities.len(), 2);
    let FieldSchema::Enum { enum_name, labels } = &output.fields[0] else {
        panic!("expected enum field");
    };
    assert_eq!(enum_name, "RESULT");
    assert_eq!(labels, &["MISS", "HIT"]);
}

#[test]
fn serialized_tuple_distribution_hoists_field_schema() {
    let (mut outputs, symbols) =
        run("enum: RESULT { MISS, HIT } A: d2 B: d{MISS, HIT} output [tuple A B]").unwrap();
    let dist = Distribution::from_runtime(outputs.remove(0).value, None, &symbols);
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
