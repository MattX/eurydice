mod utils;

use eurydice_engine::{ast::ParseActionError, eval::RuntimeValue, output::Distribution};
use js_sys::Function;
use lalrpop_util::ParseError;
use serde::Serialize;
use utils::set_panic_hook;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn run(input: &str, print_callback: Function) -> JsValue {
    set_panic_hook();
    let callback = Box::new(move |value: RuntimeValue, name: String| {
        print_callback
            .call2(&JsValue::NULL, &format!("{}", value).into(), &name.into())
            .unwrap();
    });
    serde_wasm_bindgen::to_value(&run_inner(input, callback)).unwrap()
}

#[derive(Debug, Clone, Serialize)]
pub struct Error {
    message: String,
    from: usize,
    to: usize,
}

fn run_inner(
    input: &str,
    print_callback: Box<dyn Fn(RuntimeValue, String)>,
) -> Result<Vec<(String, Distribution)>, Error> {
    let mut evaluator = eurydice_engine::eval::Evaluator::new();
    evaluator.set_print_callback(print_callback);
    let parser = eurydice_engine::grammar::BodyParser::new();
    let statements = match parser.parse(input) {
        Ok(expr) => expr,
        Err(e) => return Err(lalrpop_to_error(&e)),
    };

    for statement in statements {
        match evaluator.execute(&statement) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error {
                    message: e.to_string(),
                    from: e.range().start,
                    to: e.range().end,
                })
            }
        }
    }

    Ok(evaluator
        .take_outputs()
        .into_iter()
        .map(|output| {
            (
                output.name,
                Distribution::from_runtime(output.value, output.field_names),
            )
        })
        .collect())
}

fn lalrpop_to_error<T: std::fmt::Display>(e: &ParseError<usize, T, ParseActionError>) -> Error {
    let range = match e {
        ParseError::UnrecognizedToken { token, .. } => (token.0, token.2),
        ParseError::UnrecognizedEof { location, .. } => (*location, *location),
        ParseError::ExtraToken { token } => (token.0, token.2),
        ParseError::InvalidToken { location } => (*location, *location),
        ParseError::User { error } => match error {
            ParseActionError::InvalidIntegerLiteral { range, .. } => (range.start, range.end),
            ParseActionError::EmptyFunctionCall { range } => (range.start, range.end),
        },
    };
    Error {
        message: e.to_string(),
        from: range.0,
        to: range.1,
    }
}

#[cfg(test)]
mod tests {
    use super::run_inner;

    #[test]
    fn labeled_tuple_metadata_reaches_wasm_output() {
        let outputs = run_inner(
            "output [tuple 1 2] labeled \"Left\", \"Right\"",
            Box::new(|_, _| {}),
        )
        .unwrap();
        let distribution = &outputs[0].1;
        assert_eq!(
            distribution.field_names.as_ref().unwrap(),
            &["Left".to_string(), "Right".to_string()]
        );
    }
}
