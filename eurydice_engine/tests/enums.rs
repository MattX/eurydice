use eurydice_engine::{
    eval::{EvaluatedOutput, Evaluator},
    grammar,
    output::{Distribution, FieldSchema},
};

fn run(program: &str) -> Result<Vec<EvaluatedOutput>, String> {
    let statements = grammar::BodyParser::new()
        .parse(program)
        .map_err(|error| error.to_string())?;
    let mut evaluator = Evaluator::new();
    for statement in statements {
        evaluator
            .execute(&statement)
            .map_err(|error| error.to_string())?;
    }
    Ok(evaluator.take_outputs())
}

#[test]
fn rejects_mixed_enum_list() {
    let error = run(r#"
        enum: RESULT { MISS, HIT }
        output {MISS, 1}
        "#)
    .unwrap_err();
    assert!(error.contains("mixed outcome types"), "{error}");
}

#[test]
fn enum_names_and_members_are_immutable() {
    for program in [
        "enum: RESULT { MISS, HIT } MISS: 3",
        "enum: RESULT { MISS, HIT } RESULT: 3",
    ] {
        let error = run(program).unwrap_err();
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
        ("enum: R { A, B } output 2d{A, B}", "dimension one"),
    ] {
        let error = run(program).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn rejects_different_enum_types_and_nested_declarations() {
    let mixed = run("enum: A_TYPE { A } enum: B_TYPE { B } output {A, B}").unwrap_err();
    assert!(mixed.contains("mixed outcome types"), "{mixed}");

    let nested = run("if 1 { enum: RESULT { A } }").unwrap_err();
    assert!(nested.contains("top level"), "{nested}");
}

#[test]
fn equality_aware_operations_require_the_same_enum_type() {
    for program in [
        "enum: A_TYPE { A } enum: B_TYPE { B } output A = B",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [{A} contains B]",
        "enum: A_TYPE { A } enum: B_TYPE { B } output [count {A} in {B}]",
    ] {
        let error = run(program).unwrap_err();
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
    assert!(dist
        .probabilities
        .iter()
        .all(|(values, _)| values.len() == 2));
    let total: f64 = dist.probabilities.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-9, "probabilities should sum to 1");
}
