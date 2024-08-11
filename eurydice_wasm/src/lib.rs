mod utils;

use std::collections::HashMap;

use eurydice_engine::output::OutputValue;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn return_three() -> i32 {
    3
}

#[wasm_bindgen]
pub fn run(input: &str) -> JsValue {
    serde_wasm_bindgen::to_value(&run_inner(input)).unwrap()
}

fn run_inner(input: &str) -> Result<HashMap<String, OutputValue>, String> {
    let mut evaluator = eurydice_engine::eval::Evaluator::new();
    let parser = eurydice_engine::grammar::BodyParser::new();
    let statements = match parser.parse(input) {
        Ok(expr) => expr,
        Err(err) => return Err(err.to_string()),
    };

    for statement in statements {
        match evaluator.execute(&statement) {
            Ok(()) => {}
            Err(e) => return Err(e.to_string()),
        }
    }

    Ok(evaluator
        .take_outputs()
        .into_iter()
        .map(|(val, name)| (name, OutputValue::from(val)))
        .collect::<HashMap<_, _>>())
}
