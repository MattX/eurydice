mod utils;

use std::collections::HashMap;

use eurydice_engine::{
    Engine, RunReport, SourceRange, primitive_metadata as engine_primitive_metadata,
};
use js_sys::Function;
use utils::set_panic_hook;
use wasm_bindgen::prelude::*;

/// Runs a program, returning its outputs and structured diagnostics.
#[wasm_bindgen(js_name = runWithDiagnostics)]
pub fn run_source(input: &str, print_callback: Function) -> JsValue {
    set_panic_hook();
    let callback = move |value: String, name: String| {
        print_callback
            .call2(&JsValue::NULL, &value.into(), &name.into())
            .unwrap();
    };
    let mut engine = Engine::new();
    engine.set_print_callback(callback);
    let mut report = engine.run_source(input);
    convert_report_offsets(&mut report);
    serde_wasm_bindgen::to_value(&report).unwrap()
}

/// Metadata used by editor integrations to complete built-in function calls.
#[wasm_bindgen(js_name = primitiveMetadata)]
pub fn primitive_metadata() -> JsValue {
    serde_wasm_bindgen::to_value(engine_primitive_metadata()).unwrap()
}

fn byte_to_utf16(source: &str, byte_offset: usize) -> usize {
    let mut byte_offset = byte_offset.min(source.len());
    while !source.is_char_boundary(byte_offset) {
        byte_offset -= 1;
    }
    source[..byte_offset].encode_utf16().count()
}

fn convert_report_offsets(report: &mut RunReport) {
    let RunReport {
        diagnostics,
        sources,
        ..
    } = report;
    let texts = sources
        .iter()
        .map(|source| (source.id, source.text.as_str()))
        .collect::<HashMap<_, _>>();
    let convert = |range: &mut SourceRange| {
        let Some(text) = texts.get(&range.source) else {
            return;
        };
        range.range.start = byte_to_utf16(text, range.range.start);
        range.range.end = byte_to_utf16(text, range.range.end);
    };

    for diagnostic in diagnostics {
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
    use super::{byte_to_utf16, convert_report_offsets};
    use eurydice_engine::Engine;

    #[test]
    fn labeled_tuple_metadata_reaches_wasm_output() {
        let report = Engine::new().run_source("output [tuple 1 2] labeled \"Left\", \"Right\"");
        let distribution = &report.outputs[0].distribution;
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
        let mut report = Engine::new().run_source(input);
        let byte_start = report.error().unwrap().labels[0].range.range.start;
        convert_report_offsets(&mut report);

        assert_eq!(
            report.error().unwrap().labels[0].range.range.start,
            input[..byte_start].encode_utf16().count()
        );
    }
}
