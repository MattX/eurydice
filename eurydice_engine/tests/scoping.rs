use eurydice_engine::{
    eval::{Evaluator, RuntimeValue},
    grammar,
};

#[test]
fn a_function_frame_is_popped_after_an_error() {
    let statements = grammar::BodyParser::new()
        .parse(
            r#"
            X: 1
            function: fail X:n { result: X / 0 }
            output [fail 9]
            output X
            "#,
        )
        .unwrap();
    let mut evaluator = Evaluator::new();
    let mut statements = statements.into_iter();

    evaluator.execute(&statements.next().unwrap()).unwrap();
    evaluator.execute(&statements.next().unwrap()).unwrap();
    assert!(evaluator.execute(&statements.next().unwrap()).is_err());
    evaluator.execute(&statements.next().unwrap()).unwrap();

    let outputs = evaluator.take_outputs();
    let RuntimeValue::Element(value) = &outputs[0].value else {
        panic!("expected an element, got {:?}", outputs[0].value);
    };
    assert_eq!(value.as_int(), Some(1));
}
