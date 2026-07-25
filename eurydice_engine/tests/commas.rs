//! Optional comma separators between adjacent function arguments.
//!
//! Without a comma an argument following another must be a `SafeExpr`, which
//! cannot start with `-` or `d`. A comma removes that ambiguity, so any
//! expression may follow one. Commas are pure separators: they do not affect
//! which function an identifier resolves to.

use eurydice_engine::{eval::Evaluator, grammar};

fn run(program: &str) -> Result<Vec<eurydice_engine::eval::EvaluatedOutput>, String> {
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
fn commas_separate_arguments_that_would_otherwise_merge() {
    // Without the comma this parses as the single argument `d(6d8)`.
    let outputs = run("output #[tuple d6, d8]").unwrap();
    assert_eq!(outputs[0].value, 1.into(), "expected a dimension-one pool");

    // Comparing the two forms with `=` would compare two independent rolls, so
    // compare the distributions they produce instead.
    let outputs = run("output [tuple d6, d8]\noutput [tuple (d6) (d8)]").unwrap();
    assert_eq!(outputs[0].value, outputs[1].value);
}

#[test]
fn commas_allow_a_leading_minus() {
    let outputs = run("output [tuple 1, -2] = [tuple 1 (-2)]").unwrap();
    assert_eq!(outputs[0].value, 1.into());
}

#[test]
fn commas_do_not_affect_function_identity() {
    // All four spellings must resolve to the same two-argument function.
    for program in [
        "function: add A:n B:n { result: A + B }\noutput [add 1 2]",
        "function: add A:n B:n { result: A + B }\noutput [add 1, 2]",
        "function: add A:n, B:n { result: A + B }\noutput [add 1 2]",
        "function: add A:n, B:n { result: A + B }\noutput [add 1, 2]",
    ] {
        let outputs = run(program).unwrap();
        assert_eq!(outputs[0].value, 3.into(), "for {program}");
    }
}

#[test]
fn commas_interleave_with_words() {
    let outputs =
        run("function: mix A:n and B:n, C:n { result: A + B + C }\noutput [mix 1 and 2, 3]")
            .unwrap();
    assert_eq!(outputs[0].value, 6.into());
}

#[test]
fn commas_must_sit_between_two_arguments() {
    for program in [
        "output [tuple , 1 2]",
        "output [tuple 1, 2,]",
        "output [tuple 1,, 2]",
        "output [add 1, and 2]",
    ] {
        assert!(
            run(program).is_err(),
            "expected a parse error for {program}"
        );
    }
}

#[test]
fn unresolved_calls_suggest_a_comma_when_the_arity_is_wrong() {
    let error = run("output [tuple d6 d8]").unwrap_err();
    assert!(error.contains("tuple {}"), "got {error}");

    // The hint is only attached when a same-word function of another arity
    // exists; an entirely unknown function gets none.
    let error = run("output [nonexistent thing]").unwrap_err();
    assert!(!error.contains("separate them with a comma"), "got {error}");
}
