mod utils;

use eurydice_engine::{Engine, output::Distribution};
use js_sys::Function;
use serde::Serialize;
use utils::set_panic_hook;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn run(input: &str, print_callback: Function) -> JsValue {
    set_panic_hook();
    let callback = move |value: String, name: String| {
        print_callback
            .call2(&JsValue::NULL, &value.into(), &name.into())
            .unwrap();
    };
    serde_wasm_bindgen::to_value(&run_inner(input, callback)).unwrap()
}

#[derive(Debug, Clone, Serialize)]
pub struct Error {
    message: String,
    from: usize,
    to: usize,
}

fn run_inner<F>(input: &str, print_callback: F) -> Result<Vec<(String, Distribution)>, Error>
where
    F: Fn(String, String) + 'static,
{
    let mut engine = Engine::new();
    engine.set_print_callback(print_callback);
    engine
        .run(input)
        .map_err(|error| {
            let range = error.range();
            Error {
                message: error.to_string(),
                from: range.start,
                to: range.end,
            }
        })
        .map(|outputs| {
            outputs
                .into_iter()
                .map(|output| (output.name, output.distribution))
                .collect()
        })
}

#[cfg(test)]
mod tests {
    use super::run_inner;

    #[test]
    fn labeled_tuple_metadata_reaches_wasm_output() {
        let outputs =
            run_inner("output [tuple 1 2] labeled \"Left\", \"Right\"", |_, _| {}).unwrap();
        let distribution = &outputs[0].1;
        assert_eq!(
            distribution.field_names.as_ref().unwrap(),
            &["Left".to_string(), "Right".to_string()]
        );
    }
}
