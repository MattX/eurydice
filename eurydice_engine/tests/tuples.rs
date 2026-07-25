use eurydice_engine::{eval::Evaluator, grammar};

fn run(program: &str) -> Result<(), String> {
    let statements = grammar::BodyParser::new()
        .parse(program)
        .map_err(|error| error.to_string())?;
    let mut evaluator = Evaluator::new();
    for statement in statements {
        evaluator
            .execute(&statement)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[test]
fn tuple_output_labels_require_tuple_outcomes_and_matching_arity() {
    for (program, expected) in [
        ("output 1 labeled \"A\", \"B\"", "non-tuple"),
        ("enum: RESULT { A } output A labeled \"Value\"", "non-tuple"),
        (
            "output [tuple 1 2] labeled \"Only one\"",
            "expected 2, found 1",
        ),
        (
            "output [tuple 1 2] labeled \"A\", \"B\", \"C\"",
            "expected 2, found 3",
        ),
        (
            "output [tuple 1 2] labeled \"[MISSING]\", \"B\"",
            "undefined",
        ),
    ] {
        let error = run(program).unwrap_err();
        assert!(
            error.to_lowercase().contains(&expected.to_lowercase()),
            "{error}"
        );
    }

    assert!(run("output [tuple 1 2] labeled \"A\", \"B\" labeled \"C\", \"D\"").is_err());
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
        "output [tuple 1 2] = [tuple 1 2 3]",
        "output 0d6 + [tuple 1 2]",
        "output 2d{1:0} + [tuple 1 2]",
        "enum: RESULT { A, B } output 2d{[tuple 1 A], [tuple 2 B]}",
    ] {
        assert!(run(program).is_err(), "expected error for {program}");
    }
}
