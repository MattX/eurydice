use eurydice_engine::{
    eval::{Evaluator, RuntimeValue},
    grammar,
    output::{Distribution, OutputValue},
};

fn run(program: &str) -> Result<Vec<(RuntimeValue, String)>, String> {
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

    let RuntimeValue::Pool(pool, Some(enum_type)) = &outputs[0].0 else {
        panic!("expected enum pool, got {:?}", outputs[0].0);
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
fn checks_numeric_and_named_constraints() {
    let numeric_error = run(r#"
        enum: RESULT { MISS, HIT }
        function: numeric X:int { result: X }
        output [numeric HIT]
        "#)
    .unwrap_err();
    assert!(numeric_error.contains("expected int"), "{numeric_error}");

    let enum_error = run(r#"
        enum: RESULT { MISS, HIT }
        function: typed X:RESULT { result: X }
        output [typed 1]
        "#)
    .unwrap_err();
    assert!(enum_error.contains("expected RESULT"), "{enum_error}");
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
        function: pool D:d<RESULT> { result: D }
        function: sequence S:s<RESULT> { result: [reverse S] }
        function: generic scalar X:n { result: X }
        output [pool d{MISS, HIT}]
        output [generic scalar d{MISS, HIT}]
        output [[sequence {MISS, HIT}] contains MISS]
        output [count {HIT} in [sequence {MISS, HIT, HIT}]]
        output 1@{MISS, HIT}
        output #{MISS, HIT}
        "#)
    .unwrap();
    assert!(matches!(outputs[0].0, RuntimeValue::Pool(_, Some(_))));
    assert!(matches!(outputs[1].0, RuntimeValue::Pool(_, Some(_))));
    assert_eq!(outputs[2].0, RuntimeValue::Int(1, None));
    assert_eq!(outputs[3].0, RuntimeValue::Int(2, None));
    assert!(matches!(outputs[4].0, RuntimeValue::Int(0, Some(_))));
    assert_eq!(outputs[5].0, RuntimeValue::Int(2, None));
}

#[test]
fn rejects_enum_arithmetic_ordering_and_multidimensional_output() {
    for (program, expected) in [
        ("enum: R { A, B } output A + B", "not defined for enum"),
        ("enum: R { A, B } output A < B", "not defined for enum"),
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
fn named_types_must_be_declared_before_functions() {
    let error = run("function: typed X:RESULT { result: X } enum: RESULT { A }").unwrap_err();
    assert!(error.contains("RESULT is not defined"), "{error}");
}

#[test]
fn numeric_container_constraints_reject_enums() {
    for annotation in ["s<int>", "d<int>"] {
        let program = format!(
            "enum: RESULT {{ A, B }} function: typed X:{annotation} {{ result: X }} output [typed {{A, B}}]"
        );
        let error = run(&program).unwrap_err();
        assert!(error.contains("expected int"), "{error}");
    }
}

#[test]
fn untyped_empty_values_do_not_infer_an_enum_type() {
    let error =
        run("enum: RESULT { A } function: typed X:s<RESULT> { result: X } output [typed {}]")
            .unwrap_err();
    assert!(error.contains("expected RESULT, found int"), "{error}");

    let outputs =
        run("enum: RESULT { A } function: typed X:s<RESULT> { result: X } output [typed {A:0}]")
            .unwrap();
    assert!(matches!(outputs[0].0, RuntimeValue::List(_, Some(_))));
}

#[test]
fn serialized_distribution_keeps_numeric_probabilities_and_enum_labels() {
    let mut outputs = run("enum: RESULT { MISS, HIT } output d{MISS, HIT}").unwrap();
    let output = OutputValue::from(outputs.remove(0).0);
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
