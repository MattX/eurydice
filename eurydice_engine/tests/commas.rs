//! Invalid uses of optional comma separators between function arguments.

use eurydice_engine::{Engine, EngineDiagnostic};

fn diagnostic(program: &str) -> Option<EngineDiagnostic> {
    Engine::new().run_with_diagnostics(program).error().cloned()
}

fn mentions_a_comma(diagnostic: &EngineDiagnostic) -> bool {
    diagnostic
        .help
        .as_deref()
        .is_some_and(|help| help.contains("separate them with a comma"))
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
            diagnostic(program).is_some(),
            "expected a parse error for {program}"
        );
    }
}

#[test]
fn unresolved_calls_suggest_a_comma_when_the_arity_is_wrong() {
    let joined = diagnostic("output [tuple d6 d8]").unwrap();
    assert_eq!(joined.code, "name.undefined_function");
    assert!(mentions_a_comma(&joined), "got {:?}", joined.help);
    assert_eq!(joined.fixes.len(), 1);

    // The hint is only attached when a same-word function of another arity
    // exists; an entirely unknown function gets none.
    let unknown = diagnostic("output [nonexistent thing]").unwrap();
    assert_eq!(unknown.code, "name.undefined_function");
    assert!(!mentions_a_comma(&unknown), "got {:?}", unknown.help);
    assert!(unknown.fixes.is_empty());
}
