use eurydice_engine::{Engine, EngineDiagnostic};

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
        "output [tuple 1 2] = [tuple 1 2 3]",
        "output 0d6 + [tuple 1 2]",
        "output 2d{1:0} + [tuple 1 2]",
        "enum: RESULT { A, B } output 2d{[tuple 1 A], [tuple 2 B]}",
    ] {
        assert!(
            diagnostic(program).is_some(),
            "expected error for {program}"
        );
    }
}
