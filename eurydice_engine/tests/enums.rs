use eurydice_engine::{
    eval::{ElementValue, EvaluatedOutput, Evaluator, RuntimeValue},
    grammar,
    output::{Distribution, OutputValue, TupleFieldSchema},
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
fn attack_function_returns_enum_distribution() {
    let outputs = run(r#"
        enum: ATTACK_RESULT { MISS, HIT, CRITICALHIT }
        ATTACK: 5
        function: attack ROLL:n vs DEFENSE:n {
          if ROLL = 1 { result: MISS }
          if ROLL = 20 {
            if ROLL + ATTACK >= DEFENSE { result: CRITICALHIT }
            result: HIT
          }
          if ROLL + ATTACK >= DEFENSE { result: HIT }
          result: MISS
        }
        output [attack d20 vs 15]
        "#)
    .unwrap();

    let RuntimeValue::Pool(pool, _) = &outputs[0].value else {
        panic!("expected enum pool, got {:?}", outputs[0].value);
    };
    let ElementValue::Enum { ty: enum_type, .. } = &pool.ordered_outcomes()[0].0 else {
        panic!("expected enum outcome");
    };
    assert_eq!(enum_type.name, "ATTACK_RESULT");
    assert_eq!(enum_type.members, ["MISS", "HIT", "CRITICALHIT"]);
    assert_eq!(pool.ordered_outcomes().len(), 3);
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
fn untyped_empty_list_adopts_an_enum_outcome_type() {
    let outputs = run(r#"
        enum: RESULT { MISS, HIT }
        output {{}, MISS}
        "#)
    .unwrap();
    let RuntimeValue::List(values, _) = &outputs[0].value else {
        panic!("expected enum list");
    };
    assert!(matches!(
        values.as_slice(),
        [ElementValue::Enum { value: 0, .. }]
    ));
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
fn supports_enum_shape_constraints_and_safe_operations() {
    let outputs = run(r#"
        enum: RESULT { MISS, HIT }
        function: pool D:d { result: D }
        function: sequence S:s { result: [reverse S] }
        function: generic element X:n { result: X }
        output [pool d{MISS, HIT}]
        output [generic element d{MISS, HIT}]
        output [[sequence {MISS, HIT}] contains MISS]
        output [count {HIT} in [sequence {MISS, HIT, HIT}]]
        output 1@{MISS, HIT}
        output #{MISS, HIT}
        "#)
    .unwrap();
    assert!(matches!(outputs[0].value, RuntimeValue::Pool(_, _)));
    assert!(matches!(outputs[1].value, RuntimeValue::Pool(_, _)));
    assert_eq!(outputs[2].value, 1.into());
    assert_eq!(outputs[3].value, 2.into());
    assert!(matches!(
        outputs[4].value,
        RuntimeValue::Element(ElementValue::Enum { value: 0, .. })
    ));
    assert_eq!(outputs[5].value, 2.into());
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
fn shape_constraints_accept_typed_empty_enum_values() {
    let outputs =
        run("enum: RESULT { A } function: typed X:s { result: X } output [typed {A:0}]").unwrap();
    assert!(matches!(outputs[0].value, RuntimeValue::List(_, _)));
}

#[test]
fn serialized_distribution_keeps_numeric_probabilities_and_enum_labels() {
    let mut outputs = run("enum: RESULT { MISS, HIT } output d{MISS, HIT}").unwrap();
    let output = OutputValue::from(outputs.remove(0).value);
    let OutputValue::Distribution(Distribution {
        probabilities,
        enum_name,
        labels,
    }) = output
    else {
        panic!("expected distribution");
    };
    assert_eq!(probabilities.len(), 2);
    assert_eq!(enum_name.as_deref(), Some("RESULT"));
    assert_eq!(labels.unwrap(), ["MISS", "HIT"]);
}

#[test]
fn enum_elements_and_lists_are_converted_to_distributions_in_the_engine() {
    let mut outputs = run(
        "enum: RESULT { MISS, HIT } output HIT named \"element\" output {MISS, HIT, HIT} named \"list\"",
    )
    .unwrap();

    let OutputValue::Distribution(element) = OutputValue::from(outputs.remove(0).value) else {
        panic!("expected element distribution");
    };
    assert_eq!(element.probabilities, [(1, 1.0)]);
    assert_eq!(element.enum_name.as_deref(), Some("RESULT"));
    assert_eq!(element.labels.unwrap(), ["MISS", "HIT"]);

    let OutputValue::Distribution(list) = OutputValue::from(outputs.remove(0).value) else {
        panic!("expected list distribution");
    };
    assert_eq!(list.probabilities, [(0, 1.0 / 3.0), (1, 2.0 / 3.0)]);
    assert_eq!(list.enum_name.as_deref(), Some("RESULT"));
    assert_eq!(list.labels.unwrap(), ["MISS", "HIT"]);
}

#[test]
fn serialized_tuple_distribution_hoists_field_schema() {
    let mut outputs =
        run("enum: RESULT { MISS, HIT } A: d2 B: d{MISS, HIT} output [tuple A B]").unwrap();
    let output = OutputValue::from(outputs.remove(0).value);
    let OutputValue::TupleDistribution(dist) = output else {
        panic!("expected tuple distribution");
    };
    assert!(dist.field_names.is_none());

    // The per-field schema is stored once, not repeated on each outcome.
    assert!(matches!(dist.fields[0], TupleFieldSchema::Int));
    let TupleFieldSchema::Enum { enum_name, labels } = &dist.fields[1] else {
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
