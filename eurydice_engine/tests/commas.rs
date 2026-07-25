//! Invalid uses of optional comma separators between function arguments.

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
