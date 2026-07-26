mod utils;

use std::collections::HashMap;

use eurydice_engine::{
    Engine, RunReport, SourceRange, output::Distribution,
    primitive_metadata as engine_primitive_metadata,
};
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

/// Structured engine output for richer diagnostic frontends. The existing
/// `run` export remains unchanged for compatibility with the current website.
#[wasm_bindgen(js_name = runWithDiagnostics)]
pub fn run_with_diagnostics(input: &str, print_callback: Function) -> JsValue {
    set_panic_hook();
    let callback = move |value: String, name: String| {
        print_callback
            .call2(&JsValue::NULL, &value.into(), &name.into())
            .unwrap();
    };
    let mut engine = Engine::new();
    engine.set_print_callback(callback);
    let mut report = engine.run_with_diagnostics(input);
    convert_report_offsets(&mut report);
    serde_wasm_bindgen::to_value(&report).unwrap()
}

/// Metadata used by editor integrations to complete built-in function calls.
#[wasm_bindgen(js_name = primitiveMetadata)]
pub fn primitive_metadata() -> JsValue {
    serde_wasm_bindgen::to_value(engine_primitive_metadata()).unwrap()
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
                from: byte_to_utf16(input, range.start),
                to: byte_to_utf16(input, range.end),
            }
        })
        .map(|outputs| {
            outputs
                .into_iter()
                .map(|output| (output.name, output.distribution))
                .collect()
        })
}

fn byte_to_utf16(source: &str, byte_offset: usize) -> usize {
    let mut byte_offset = byte_offset.min(source.len());
    while !source.is_char_boundary(byte_offset) {
        byte_offset -= 1;
    }
    source[..byte_offset].encode_utf16().count()
}

fn convert_report_offsets(report: &mut RunReport) {
    let sources = report
        .sources
        .iter()
        .map(|source| (source.id, source.text.clone()))
        .collect::<HashMap<_, _>>();
    let convert = |range: &mut SourceRange| {
        let Some(source) = sources.get(&range.source) else {
            return;
        };
        range.range.start = byte_to_utf16(source, range.range.start);
        range.range.end = byte_to_utf16(source, range.range.end);
    };

    for diagnostic in &mut report.diagnostics {
        for label in &mut diagnostic.labels {
            convert(&mut label.range);
        }
        for fix in &mut diagnostic.fixes {
            for edit in &mut fix.edits {
                convert(&mut edit.range);
            }
        }
        for frame in &mut diagnostic.trace {
            convert(&mut frame.call);
            if let Some(definition) = &mut frame.definition {
                convert(definition);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{byte_to_utf16, convert_report_offsets, run_inner};
    use eurydice_engine::Engine;

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

    #[test]
    fn converts_engine_byte_offsets_to_javascript_offsets() {
        assert_eq!(byte_to_utf16("éMISSING", 2), 1);
        assert_eq!(byte_to_utf16("😀MISSING", 4), 2);
    }

    #[test]
    fn converts_every_structured_diagnostic_range() {
        let input = "print 1 named \"é\"\noutput MISSING";
        let mut report = Engine::new().run_with_diagnostics(input);
        let byte_start = report.error().unwrap().labels[0].range.range.start;
        convert_report_offsets(&mut report);

        assert_eq!(
            report.error().unwrap().labels[0].range.range.start,
            input[..byte_start].encode_utf16().count()
        );
    }
}
