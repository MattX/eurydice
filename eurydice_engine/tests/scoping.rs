use eurydice_engine::Engine;

/// An error raised inside a function unwinds the frame the call pushed. The
/// binding the caller sees afterwards is the top-level `X: 1`, not the `9` the
/// failing call bound to the parameter that shadows it.
///
/// A submission that fails is abandoned, so the surviving binding is observed
/// from the next one — which is also how an interactive session meets it.
#[test]
fn a_function_frame_is_popped_after_an_error() {
    let mut engine = Engine::new();
    let setup = engine.run_with_diagnostics("X: 1\nfunction: fail X:n { result: X / 0 }");
    assert_eq!(setup.error(), None);

    let failed = engine.run_with_diagnostics("output [fail 9]");
    assert!(failed.error().is_some());

    let report = engine.run_with_diagnostics("output X");
    assert_eq!(report.error(), None);
    assert_eq!(
        report.outputs[0].distribution.probabilities,
        vec![(vec![1], 1.0)]
    );
}
