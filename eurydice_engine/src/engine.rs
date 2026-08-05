//! High-level source-to-distribution engine API.

use std::collections::HashSet;

use serde::Serialize;

use crate::{
    diagnostic::{
        DiagnosticSeverity, DiagnosticSource, EngineDiagnostic, SourceId, UndefinedName,
        add_later_definition, parse_error, runtime_diagnostic, undefined_name,
    },
    eval::Evaluator,
    grammar,
    output::Distribution,
};

/// A named distribution produced by an `output` statement.
#[derive(Debug, Clone, Serialize)]
pub struct EngineOutput {
    pub name: String,
    pub distribution: Distribution,
}

/// Outputs and diagnostics produced by one source submission.
#[derive(Debug, Clone, Serialize)]
pub struct RunReport {
    pub outputs: Vec<EngineOutput>,
    pub diagnostics: Vec<EngineDiagnostic>,
    /// Every source the report's diagnostics point into, plus the submission
    /// that produced it. The submission is always last.
    pub sources: Vec<DiagnosticSource>,
}

impl RunReport {
    pub fn error(&self) -> Option<&EngineDiagnostic> {
        self.diagnostics
            .iter()
            .find(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    }
}

/// Stateful façade for parsing and evaluating Eurydice programs.
///
/// Definitions and settings persist between calls to
/// [`Engine::run_with_diagnostics`], making the same API suitable for both
/// one-shot execution and interactive sessions.
pub struct Engine {
    evaluator: Evaluator,
    next_source_id: u64,
    sources: Vec<DiagnosticSource>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Self {
            evaluator: Evaluator::new(),
            next_source_id: 0,
            sources: Vec::new(),
        }
    }

    /// Installs the callback used by `print` statements.
    pub fn set_print_callback<F>(&mut self, callback: F)
    where
        F: Fn(String, String) + 'static,
    {
        self.evaluator.set_print_callback(Box::new(callback));
    }

    /// Parses and executes a source submission, returning its outputs and any
    /// diagnostics it produced.
    ///
    /// A diagnostic may point into an earlier submission that defined a
    /// function, so the report carries the text of every source its own
    /// diagnostics reference.
    pub fn run_with_diagnostics(&mut self, source: &str) -> RunReport {
        let outcome = self.run_internal(source);
        // Warnings are collected during evaluation, so they precede the error
        // that stopped the submission.
        let mut diagnostics = self.evaluator.take_diagnostics();
        let outputs = match outcome {
            Ok(outputs) => outputs,
            // Outputs from a failed submission are discarded.
            Err(diagnostic) => {
                diagnostics.push(*diagnostic);
                Vec::new()
            }
        };
        let sources = self.referenced_sources(&diagnostics);
        RunReport {
            outputs,
            diagnostics,
            sources,
        }
    }

    /// The sources the report needs to render, in submission order.
    fn referenced_sources(&self, diagnostics: &[EngineDiagnostic]) -> Vec<DiagnosticSource> {
        let mut referenced = diagnostics
            .iter()
            .flat_map(|diagnostic| {
                let labels = diagnostic.labels.iter().map(|label| label.range.source);
                let edits = diagnostic
                    .fixes
                    .iter()
                    .flat_map(|fix| fix.edits.iter().map(|edit| edit.range.source));
                let frames = diagnostic.trace.iter().flat_map(|frame| {
                    std::iter::once(frame.call.source)
                        .chain(frame.definition.map(|range| range.source))
                });
                labels.chain(edits).chain(frames)
            })
            .collect::<HashSet<_>>();
        // The current submission is always included: a frontend renders it
        // whether or not anything went wrong in it.
        referenced.extend(self.sources.last().map(|source| source.id));
        self.sources
            .iter()
            .filter(|source| referenced.contains(&source.id))
            .cloned()
            .collect()
    }

    fn run_internal(&mut self, source: &str) -> Result<Vec<EngineOutput>, Box<EngineDiagnostic>> {
        // A source outlives its submission only while a persisted function can
        // still be traced back to it. Dropping the rest here keeps an
        // interactive session — which resubmits a growing program on every
        // keystroke — from grinding to a halt.
        let live = self.evaluator.live_source_ids();
        self.sources.retain(|source| live.contains(&source.id));

        let source_id = SourceId(self.next_source_id);
        self.next_source_id += 1;
        self.sources.push(DiagnosticSource {
            id: source_id,
            name: format!("submission {}", source_id.0 + 1),
            text: source.to_string(),
        });

        // Outputs belong to one submission and must never leak out of a failed
        // previous run.
        self.evaluator.take_outputs();
        self.evaluator.begin_submission(source_id);

        let statements = match grammar::BodyParser::new().parse(source) {
            Ok(statements) => statements,
            Err(error) => return Err(Box::new(parse_error(error, source_id, source))),
        };

        for (statement_index, statement) in statements.iter().enumerate() {
            if let Err(error) = self.evaluator.execute(statement) {
                self.evaluator.take_outputs();
                let suggestions = match undefined_name(&error) {
                    Some(UndefinedName::Variable(name)) => {
                        self.evaluator.variable_suggestions(name)
                    }
                    Some(UndefinedName::Function(name)) => {
                        self.evaluator.function_suggestions(name)
                    }
                    None => Vec::new(),
                };
                let mut diagnostic =
                    runtime_diagnostic(&error, source_id, &suggestions, self.evaluator.symbols());
                add_later_definition(
                    &mut diagnostic,
                    &error,
                    &statements[statement_index + 1..],
                    source_id,
                );
                return Err(Box::new(diagnostic));
            }
        }

        let outputs = self.evaluator.take_outputs();
        let symbols = self.evaluator.symbols();
        Ok(outputs
            .into_iter()
            .map(|output| EngineOutput {
                name: output.name,
                distribution: Distribution::from_runtime(output.value, output.field_names, symbols),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::FixApplicability;

    /// Runs a submission that is expected to succeed, returning its outputs.
    fn run(engine: &mut Engine, source: &str) -> Vec<EngineOutput> {
        let report = engine.run_with_diagnostics(source);
        assert_eq!(report.error(), None, "{source}");
        report.outputs
    }

    #[test]
    fn runs_source_to_named_distributions() {
        let outputs = run(&mut Engine::new(), "output 2d2 named \"roll\"");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "roll");
        assert_eq!(
            outputs[0].distribution.probabilities,
            vec![(vec![2], 0.25), (vec![3], 0.5), (vec![4], 0.25)]
        );
    }

    #[test]
    fn state_persists_between_runs() {
        let mut engine = Engine::new();
        run(&mut engine, "X: 4");

        let outputs = run(&mut engine, "output X");

        assert_eq!(outputs[0].distribution.probabilities, vec![(vec![4], 1.0)]);
    }

    #[test]
    fn reports_parse_and_runtime_ranges() {
        let parse_report = Engine::new().run_with_diagnostics("output (");
        let parse_error = parse_report.error().unwrap();
        assert!(parse_error.incomplete);
        assert_eq!(parse_error.primary_range().unwrap().range.start, 8);

        let runtime_report = Engine::new().run_with_diagnostics("output MISSING");
        let runtime_error = runtime_report.error().unwrap();
        assert!(!runtime_error.incomplete);
        assert_eq!(runtime_error.primary_range().unwrap().range.start, 7);
    }

    #[test]
    fn failed_runs_do_not_leak_outputs() {
        let mut engine = Engine::new();
        assert!(
            engine
                .run_with_diagnostics("output 1\noutput MISSING")
                .error()
                .is_some()
        );

        assert!(run(&mut engine, "X: 1").is_empty());
    }

    #[test]
    fn reports_common_syntax_mistakes_with_fixes() {
        let cases = [
            ("X = 1", "syntax.assignment_separator", ":"),
            ("output 1 == 1", "syntax.equality_operator", ""),
            ("output d6;", "syntax.semicolon", ""),
            ("output x", "syntax.variable_case", "X"),
            ("output total", "syntax.variable_case", "TOTAL"),
            ("output (1", "syntax.unclosed_delimiter", ")"),
        ];

        for (source, code, replacement) in cases {
            let report = Engine::new().run_with_diagnostics(source);
            let error = report.error().unwrap();
            assert_eq!(error.code, code);
            assert_eq!(error.fixes[0].edits[0].replacement, replacement);
        }
    }

    #[test]
    fn single_letter_function_names_are_accepted() {
        let mut engine = Engine::new();
        run(&mut engine, "function: f X:n { result: X }");

        let outputs = run(&mut engine, "output [f 3]");

        assert_eq!(outputs[0].distribution.probabilities, vec![(vec![3], 1.0)]);
    }

    #[test]
    fn misplaced_keywords_are_not_mistaken_for_lowercase_variables() {
        let report = Engine::new().run_with_diagnostics("output result");
        let error = report.error().unwrap();

        assert_eq!(error.code, "syntax.unexpected_token");
        assert!(error.fixes.is_empty());
    }

    #[test]
    fn unclosed_delimiters_ignore_strings_and_comments() {
        let cases = ["output (1 \\ ) \\", "output (1 \\\\\\ )"];

        for source in cases {
            let report = Engine::new().run_with_diagnostics(source);
            let error = report.error().unwrap();

            assert_eq!(error.code, "syntax.unclosed_delimiter", "{source}");
            assert_eq!(error.fixes[0].edits[0].replacement, ")", "{source}");
        }
    }

    #[test]
    fn explains_that_top_level_expressions_need_output() {
        for source in ["1d6", "D: d6\nD + D"] {
            let report = Engine::new().run_with_diagnostics(source);
            let error = report.error().unwrap();

            assert_eq!(error.code, "syntax.unexpected_token");
            assert_eq!(
                error.notes,
                ["At the top level, expressions you want to show must start with `output`."]
            );
            assert!(error.fixes.is_empty());
        }
    }

    #[test]
    fn suggests_names_and_marks_definitions_that_execute_later() {
        let mut engine = Engine::new();
        run(&mut engine, "FOOD: 1");
        let report = engine.run_with_diagnostics("output FOO\nFOO: 2");
        let error = report.error().unwrap();

        assert_eq!(error.labels[0].message, None);
        assert!(
            error
                .help
                .as_deref()
                .is_some_and(|help| help.contains("Did you mean `FOOD`?")),
            "{:?}",
            error.help
        );
        assert!(
            error
                .labels
                .iter()
                .any(|label| label.message.as_deref() == Some("defined later"))
        );
    }

    #[test]
    fn suggests_a_comma_for_ambiguously_joined_dice_arguments() {
        // The comma goes after the first complete roll, however many were run
        // together, and a die count is never mistaken for one.
        for (source, remainder) in [
            ("output [tuple d6 d8]", " d8]"),
            ("output [tuple d6 d8 d10]", " d8 d10]"),
            ("output [tuple 2d6 d8]", " d8]"),
        ] {
            let report = Engine::new().run_with_diagnostics(source);
            let error = report.error().unwrap();

            assert_eq!(error.code, "name.undefined_function", "{source}");
            assert_eq!(error.fixes.len(), 1, "{source}");
            assert_eq!(
                error.fixes[0].applicability,
                FixApplicability::Suggested,
                "{source}"
            );
            let edit = &error.fixes[0].edits[0];
            assert_eq!(
                &report.sources[0].text[edit.range.range.start..],
                remainder,
                "{source}"
            );
            assert_eq!(edit.replacement, ",", "{source}");
        }
    }

    /// A call can also pass too many arguments, which no comma would fix.
    #[test]
    fn does_not_suggest_a_comma_when_no_arity_takes_more_arguments() {
        let report = Engine::new().run_with_diagnostics("output [absolute 1, 2]");
        let error = report.error().unwrap();

        assert_eq!(error.code, "name.undefined_function");
        assert!(error.fixes.is_empty());
    }

    #[test]
    fn traces_pool_expansion_with_the_failing_argument_value() {
        let source = "\
function: pick I:n {
  result: [field I of [tuple 10 20]]
}
output [pick d3]";
        let report = Engine::new().run_with_diagnostics(source);
        let error = report.error().unwrap();

        assert_eq!(error.code, "value.out_of_range");
        assert_eq!(error.trace.len(), 1);
        assert_eq!(error.trace[0].function, "pick {}");
        assert_eq!(error.trace[0].bindings[0].name, "I");
        assert_eq!(error.trace[0].bindings[0].value, "3");
    }

    #[test]
    fn primitive_diagnostics_identify_invalid_field_arguments() {
        let source = "output [field d2 of 1]";
        let report = Engine::new().run_with_diagnostics(source);
        let error = report.error().unwrap();

        assert_eq!(error.code, "type.function_argument");
        assert_eq!(
            error.summary,
            "`[field INDEX:n of TUPLE:n]` requires `INDEX` to be an integer and `TUPLE` to be a tuple"
        );
        assert_eq!(error.labels.len(), 1);
        let label = &error.labels[0];
        assert_eq!(&source[label.range.range.start..label.range.range.end], "1");
        let message = label.message.as_deref().unwrap();
        assert!(message.contains("`TUPLE` is an integer: `1`"));
        assert!(message.contains("expected a tuple"));
        assert_eq!(
            error.help.as_deref(),
            Some("Build a tuple with `[tuple A B]`. Tuple field positions start at 1.")
        );
    }

    #[test]
    fn primitive_diagnostics_report_every_bad_tuple_or_field_argument() {
        let nested =
            Engine::new().run_with_diagnostics("output [tuple [tuple 1, 2], [tuple 3, 4]]");
        let nested_error = nested.error().unwrap();
        assert_eq!(nested_error.code, "type.function_argument");
        assert_eq!(
            nested_error.summary,
            "`[tuple A:n B:n]` cannot contain another tuple"
        );
        assert_eq!(nested_error.labels.len(), 2);
        assert!(
            nested_error.labels[0]
                .message
                .as_deref()
                .unwrap()
                .contains("`A` is a tuple")
        );
        assert!(
            nested_error.labels[1]
                .message
                .as_deref()
                .unwrap()
                .contains("`B` is a tuple")
        );

        let field = Engine::new().run_with_diagnostics("output [field [tuple 1, 2] of 1]");
        let field_error = field.error().unwrap();
        assert_eq!(field_error.code, "type.function_argument");
        assert_eq!(field_error.labels.len(), 2);
        assert!(
            field_error.labels[0]
                .message
                .as_deref()
                .unwrap()
                .contains("`INDEX` is a tuple")
        );
        assert!(
            field_error.labels[1]
                .message
                .as_deref()
                .unwrap()
                .contains("`TUPLE` is an integer")
        );
    }

    /// The equality-based primitives are as total as `=` itself: a needle that
    /// could never match counts zero matches rather than failing.
    #[test]
    fn equality_primitives_count_no_matches_across_outcome_types() {
        for source in [
            "enum: R { A, B } output [{A} contains 1]",
            "enum: R { A, B } output [count {A} in {1}]",
            "enum: R { A, B } output [count {1} in 2d{A, B}]",
            "output [{[tuple 1, 2]} contains 1]",
        ] {
            let mut engine = Engine::new();
            let outputs = run(&mut engine, source);
            assert_eq!(
                outputs[0].distribution.probabilities,
                vec![(vec![0], 1.0)],
                "{source}"
            );
        }
    }

    #[test]
    fn primitive_diagnostics_explain_outcome_and_value_constraints() {
        let bounds = Engine::new().run_with_diagnostics("output [field 3 of [tuple 1, 2]]");
        let bounds_error = bounds.error().unwrap();
        assert_eq!(bounds_error.code, "value.out_of_range");
        assert_eq!(
            bounds_error.summary,
            "`[field INDEX:n of TUPLE:n]` cannot select field `3`"
        );
        assert!(
            bounds_error.labels[0]
                .message
                .as_deref()
                .unwrap()
                .contains("expected an index from 1 through 2")
        );

        let negative = Engine::new().run_with_diagnostics("output [highest -1 of 2d6]");
        assert_eq!(
            negative.error().unwrap().summary,
            "`[highest COUNT:n of POOL:d]` needs a nonnegative count"
        );
    }

    #[test]
    fn non_primitive_type_errors_include_the_actual_value_and_type() {
        let cases = [
            (
                "loop X over d6 { print X }",
                "type.expected_sequence",
                "expected a sequence",
            ),
            (
                "if d6 { print 1 }",
                "type.expected_number",
                "expected an integer",
            ),
            ("output {1:d6}", "type.repeat_count", "expected an integer"),
            (
                "output {1..d6}",
                "type.range_endpoint",
                "expected an integer",
            ),
        ];

        for (source, code, expected_message) in cases {
            let report = Engine::new().run_with_diagnostics(source);
            let error = report.error().unwrap();
            assert_eq!(error.code, code, "{source}");
            let message = error.labels[0].message.as_deref().unwrap_or_default();
            // The label names the type that was found as well as the one the
            // position required.
            assert!(message.contains(expected_message), "{source}: {message}");
            assert!(message.contains("is a dice pool"), "{source}: {message}");
        }
    }

    #[test]
    fn output_label_errors_explain_the_output_shape_and_arity() {
        let non_tuple = Engine::new().run_with_diagnostics("output 1 labeled \"Value\"");
        let error = non_tuple.error().unwrap();
        assert_eq!(error.code, "type.labels_require_tuple");
        assert_eq!(error.labels.len(), 2);
        assert!(
            error.labels[1]
                .message
                .as_deref()
                .is_some_and(|message| message.contains("`1` is an integer; expected a tuple"))
        );

        let wrong_count =
            Engine::new().run_with_diagnostics("output [tuple 1 2] labeled \"Only one\"");
        let error = wrong_count.error().unwrap();
        assert_eq!(
            error.summary,
            "This tuple has 2 fields, but 1 output label was provided"
        );
        assert_eq!(error.labels[0].message, None);
    }

    #[test]
    fn non_primitive_diagnostics_do_not_repeat_the_summary_as_a_label() {
        for source in ["output [missing]", "enum: R { A, B } output A + B"] {
            let report = Engine::new().run_with_diagnostics(source);
            let error = report.error().unwrap();
            assert_eq!(error.labels[0].message, None, "{source}");
        }

        let operator = Engine::new().run_with_diagnostics("output d6 @ d6");
        let error = operator.error().unwrap();
        assert_eq!(
            error.summary,
            "Operator `@` expects an integer or a sequence"
        );
        assert_eq!(error.labels[0].message, None);
        assert!(
            error.labels[1]
                .message
                .as_deref()
                .is_some_and(|message| message.contains("is a dice pool"))
        );
    }

    #[test]
    fn reports_carry_only_the_sources_they_point_into() {
        let mut engine = Engine::new();

        // The failed submission is still rendered against its own source.
        let failed = engine.run_with_diagnostics("output (");
        assert_eq!(failed.sources.len(), 1);
        assert_eq!(
            failed.error().unwrap().primary_range().unwrap().source,
            SourceId(0)
        );

        let succeeded = engine.run_with_diagnostics("output 1");
        assert_eq!(succeeded.sources.len(), 1);
        assert_eq!(succeeded.sources[0].id, SourceId(1));
    }

    /// An interactive frontend resubmits a growing program on every keystroke,
    /// so anything the engine keeps per submission has to be released again.
    #[test]
    fn only_sources_behind_a_live_function_are_kept() {
        let mut engine = Engine::new();
        for _ in 0..10 {
            run(&mut engine, "X: 1\noutput X");
        }
        assert_eq!(engine.sources.len(), 1);

        run(&mut engine, "function: keep me { result: 1 }");
        run(&mut engine, "output 1");
        // The submission that defined `keep me` is still needed to render a
        // diagnostic raised inside its body.
        assert_eq!(engine.sources.len(), 2);

        // Redefining the function releases the source that first defined it.
        run(&mut engine, "function: keep me { result: 2 }");
        run(&mut engine, "output 1");
        assert_eq!(engine.sources.len(), 2);
    }

    #[test]
    fn diagnostics_retain_sources_for_persisted_functions() {
        let mut engine = Engine::new();
        run(&mut engine, "function: broken { result: MISSING }");
        let report = engine.run_with_diagnostics("output [broken]");
        let error = report.error().unwrap();

        assert_eq!(error.labels[0].range.source, SourceId(0));
        assert_eq!(error.trace[0].call.source, SourceId(1));
        assert_eq!(report.sources.len(), 2);
    }

    #[test]
    fn warns_when_the_same_pool_is_sampled_independently() {
        let report = Engine::new().run_with_diagnostics("D: d6\noutput D + D");
        assert!(report.error().is_none());
        assert_eq!(
            report.diagnostics[0].code,
            "evaluation.independent_pool_reuse"
        );

        let no_warning = Engine::new().run_with_diagnostics("D: d6\noutput #D + #D");
        assert!(no_warning.diagnostics.is_empty());
    }

    #[test]
    fn warns_when_a_function_can_finish_without_a_result() {
        let cases = [
            "function: empty {}",
            "function: partial X:n { if X { result: 1 } }",
        ];

        for source in cases {
            let report = Engine::new().run_with_diagnostics(source);
            let warning = &report.diagnostics[0];
            assert_eq!(warning.code, "control_flow.missing_result", "{source}");
            assert_eq!(warning.severity, DiagnosticSeverity::Warning);
            assert!(warning.fixes.is_empty());
        }
    }

    #[test]
    fn accepts_functions_that_return_on_every_path() {
        let cases = [
            "function: direct { result: 1 }",
            "function: branched X:n { if X { result: 1 } else { result: 2 } }",
            "function: fallback X:n { if X { result: 1 } result: 2 }",
            "function: chained X:n { if X { result: 1 } else if X = 2 { result: 2 } else { result: 3 } }",
            // Returning from a loop is only certain for a non-empty sequence,
            // but warning about the idiom would cost more than it catches.
            "function: first S:s { loop X over S { result: X } }",
        ];

        for source in cases {
            let report = Engine::new().run_with_diagnostics(source);
            assert!(report.diagnostics.is_empty(), "{source}");
        }
    }

    #[test]
    fn warns_when_depth_settings_bound_results() {
        let explode = Engine::new().run_with_diagnostics("output [explode d6]");
        assert_eq!(explode.diagnostics[0].code, "evaluation.explode_depth");

        let recursion = Engine::new().run_with_diagnostics(
            "function: recurse N:n { result: [recurse N] }\noutput [recurse 1]",
        );
        assert_eq!(
            recursion.diagnostics[0].code,
            "evaluation.maximum_function_depth"
        );
    }
}
