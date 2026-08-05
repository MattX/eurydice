use eurydice_engine::{
    Engine, EngineDiagnostic,
    eval::{EvaluatedOutput, Evaluator, SymbolTable},
    grammar,
    output::Distribution,
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

/// The diagnostic a failing program produces, or `None` if it succeeded.
///
/// Assertions here use the diagnostic's stable code rather than its wording,
/// which is free to change without breaking anyone.
fn diagnostic(program: &str) -> Option<EngineDiagnostic> {
    Engine::new().run_with_diagnostics(program).error().cloned()
}

#[test]
fn tuple_output_labels_require_tuple_outcomes_and_matching_arity() {
    for (program, expected_code) in [
        ("output 1 labeled \"A\", \"B\"", "type.labels_require_tuple"),
        (
            "enum: RESULT { A } output A labeled \"Value\"",
            "type.labels_require_tuple",
        ),
        (
            "output [tuple 1 2] labeled \"Only one\"",
            "value.output_label_count",
        ),
        (
            "output [tuple 1 2] labeled \"A\", \"B\", \"C\"",
            "value.output_label_count",
        ),
        (
            "output [tuple 1 2] labeled \"[MISSING]\", \"B\"",
            "name.undefined_variable",
        ),
    ] {
        let error =
            diagnostic(program).unwrap_or_else(|| panic!("expected an error for {program}"));
        assert_eq!(error.code, expected_code, "{program}");
    }

    assert!(diagnostic("output [tuple 1 2] labeled \"A\", \"B\" labeled \"C\", \"D\"").is_some());
}

#[test]
fn rejects_invalid_tuple_operations() {
    for program in [
        "output [tuple 1 [tuple 2 3]]",
        "output [field 0 of [tuple 1 2]]",
        "output [field -2147483648 of [tuple 1 2]]",
        "output [field 3 of [tuple 1 2]]",
        "output [tuple 1 2] + 3",
        "output 3 / [tuple 1 2]",
        "output [tuple 1 2] + [tuple 1 2 3]",
        "output [tuple 1 2] / 0",
        "output [tuple 2147483647 0] + [tuple 1 0]",
        "enum: RESULT { A } output [tuple 1 A] * 2",
        "output 0d6 + [tuple 1 2]",
        "output 2d{1:0} + [tuple 1 2]",
        "enum: RESULT { A, B } output 2d{[tuple 1 A], [tuple 2 B]}",
        // A tuple with an enum field is not additive, so summing it is an error
        // however the sum is reached.
        "enum: RESULT { A, B } function: f X:n { result: X } output [f 2d{[tuple 1 A], [tuple 2 B]}]",
        "enum: RESULT { A, B } output {2d{[tuple 1 A], [tuple 2 B]}}",
        "enum: RESULT { A, B } output 2d{[tuple 1 A], [tuple 2 B]} = [tuple 1 A]",
    ] {
        assert!(
            diagnostic(program).is_some(),
            "expected error for {program}"
        );
    }
}

/// Equality is total, so tuples that could never match compare unequal rather
/// than failing — the same rule that lets a symbol be compared with a number.
#[test]
fn tuples_of_different_arity_compare_unequal() {
    let (outputs, symbols) =
        run("output [tuple 1 2] = [tuple 1 2 3] output [tuple 1 2] != [tuple 1 2 3]")
            .expect("comparing different tuple types is not an error");
    assert_eq!(
        Distribution::from_runtime(outputs[0].value.clone(), None, &symbols).probabilities,
        vec![(vec![0], 1.0)]
    );
    assert_eq!(
        Distribution::from_runtime(outputs[1].value.clone(), None, &symbols).probabilities,
        vec![(vec![1], 1.0)]
    );
}

/// Multisets work for every non-additive element type, not just bare enums.
#[test]
fn multidimensional_pools_of_tuples_with_enum_fields_iterate_as_multisets() {
    let (outputs, symbols) = run(r#"
        enum: RESULT { MISS, HIT }
        FACES: {[tuple 1 MISS], [tuple 2 HIT]}
        function: hits SEQ:s { result: [count {[tuple 2 HIT]} in SEQ] }
        output [hits 2dFACES]
        output [count {[tuple 2 HIT]} in 2dFACES]
        "#)
    .expect("pools of tuples with an enum field are constructible");
    let expected = vec![(vec![0], 0.25f64), (vec![1], 0.5f64), (vec![2], 0.25f64)];
    for output in &outputs {
        assert_eq!(
            Distribution::from_runtime(output.value.clone(), None, &symbols).probabilities,
            expected
        );
    }
}

/// Loosening pool construction must not stop all-`int` tuples from summing.
#[test]
fn multidimensional_pools_of_int_tuples_still_sum_componentwise() {
    let (outputs, symbols) = run("output 2d{[tuple 1 10], [tuple 2 20]}")
        .expect("pools of additive tuples are summable");
    assert_eq!(
        Distribution::from_runtime(outputs[0].value.clone(), None, &symbols).probabilities,
        vec![
            (vec![2, 20], 0.25f64),
            (vec![3, 30], 0.5f64),
            (vec![4, 40], 0.25f64),
        ]
    );
}
