//! Tuple outcomes: construction, arity, and the operations they support.
//!
//! Error assertions use each diagnostic's stable code rather than its wording,
//! which is free to change without breaking anyone.

mod common;

use common::{diagnostic, only_probabilities, probabilities};
use eurydice_engine::DiagnosticCode;

#[test]
fn tuple_output_labels_require_tuple_outcomes_and_matching_arity() {
    for (program, expected_code) in [
        (
            "output 1 labeled \"A\", \"B\"",
            DiagnosticCode::LabelsRequireTuple,
        ),
        (
            "enum { A } output A labeled \"Value\"",
            DiagnosticCode::LabelsRequireTuple,
        ),
        (
            "output [tuple 1 2] labeled \"Only one\"",
            DiagnosticCode::OutputLabelCount,
        ),
        (
            "output [tuple 1 2] labeled \"A\", \"B\", \"C\"",
            DiagnosticCode::OutputLabelCount,
        ),
        (
            "output [tuple 1 2] labeled \"[MISSING]\", \"B\"",
            DiagnosticCode::UndefinedVariable,
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
        "enum { A } output [tuple 1 A] * 2",
        // `0d6` has faces, so its sum is the `0` they are shaped like, and `0`
        // is not a tuple. A die with *no* faces is a different matter: see
        // `a_die_with_no_faces_is_the_identity_whatever_it_meets`.
        "output 0d6 + [tuple 1 2]",
        "enum { A, B } output 2d{[tuple 1 A], [tuple 2 B]}",
        // A tuple with an enum field is not additive, so summing it is an error
        // however the sum is reached.
        "enum { A, B } function: f X:n { result: X } output [f 2d{[tuple 1 A], [tuple 2 B]}]",
        "enum { A, B } output {2d{[tuple 1 A], [tuple 2 B]}}",
        "enum { A, B } output 2d{[tuple 1 A], [tuple 2 B]} = [tuple 1 A]",
    ] {
        assert!(
            diagnostic(program).is_some(),
            "expected error for {program}"
        );
    }
}

/// A die with no faces contributes nothing to a sum, so adding it to a tuple
/// gives the tuple back — whichever way the faceless die was written.
///
/// This is the one place a die's faces do settle a shape: `0d6` has faces, so
/// rolling none of them still yields the `0` they are shaped like, and `0` and
/// a tuple cannot be added. With no faces at all there is nothing to take a
/// shape from, and the empty sum takes the tuple's.
#[test]
fn a_die_with_no_faces_is_the_identity_whatever_it_meets() {
    // A die with no faces adds nothing, however it was written.
    for outcomes in probabilities("output 2d{} + [tuple 1 2] output 2d{1:0} + [tuple 1 2]") {
        assert_eq!(outcomes, vec![(vec![1, 2], 1.0)]);
    }
}

/// Equality is total, so tuples that could never match compare unequal rather
/// than failing — the same rule that lets a symbol be compared with a number.
#[test]
fn tuples_of_different_arity_compare_unequal() {
    // Comparing different tuple types is not an error.
    let outputs =
        probabilities("output [tuple 1 2] = [tuple 1 2 3] output [tuple 1 2] != [tuple 1 2 3]");
    assert_eq!(outputs[0], vec![(vec![0], 1.0)]);
    assert_eq!(outputs[1], vec![(vec![1], 1.0)]);
}

/// Multisets work for every non-additive element type, not just bare enums.
#[test]
fn multidimensional_pools_of_tuples_with_enum_fields_iterate_as_multisets() {
    // Pools of tuples with an enum field are constructible.
    let outputs = probabilities(
        r#"
        enum { MISS, HIT }
        FACES: {[tuple 1 MISS], [tuple 2 HIT]}
        function: hits SEQ:s { result: [count {[tuple 2 HIT]} in SEQ] }
        output [hits 2dFACES]
        output [count {[tuple 2 HIT]} in 2dFACES]
        "#,
    );
    let expected = vec![(vec![0], 0.25f64), (vec![1], 0.5f64), (vec![2], 0.25f64)];
    for outcomes in &outputs {
        assert_eq!(outcomes, &expected);
    }
}

/// Loosening pool construction must not stop all-`int` tuples from summing.
#[test]
fn multidimensional_pools_of_int_tuples_still_sum_componentwise() {
    // Pools of additive tuples are summable.
    assert_eq!(
        only_probabilities("output 2d{[tuple 1 10], [tuple 2 20]}"),
        vec![
            (vec![2, 20], 0.25f64),
            (vec![3, 30], 0.5f64),
            (vec![4, 40], 0.25f64),
        ]
    );
}
