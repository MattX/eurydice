//! High-level source-to-distribution engine API.

use miette::{Diagnostic, SourceSpan};
use serde::Serialize;
use thiserror::Error;

use crate::{
    ast::Range,
    diagnostic::{
        DiagnosticSeverity, DiagnosticSource, EngineDiagnostic, SourceId, add_adjacent_dice_fix,
        add_later_definition, parse_error, runtime_diagnostic, undefined_name,
    },
    eval::{Evaluator, RuntimeError},
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
    pub sources: Vec<DiagnosticSource>,
}

impl RunReport {
    pub fn error(&self) -> Option<&EngineDiagnostic> {
        self.diagnostics
            .iter()
            .find(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    }
}

/// An error encountered while parsing or evaluating source text.
#[derive(Debug, Error, Diagnostic)]
pub enum EngineError {
    #[error("Parse error: {message}")]
    Parse {
        message: String,
        #[label = "here"]
        range: SourceSpan,
        incomplete: bool,
    },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Runtime(Box<RuntimeError>),
}

impl From<RuntimeError> for EngineError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(Box::new(error))
    }
}

impl EngineError {
    /// The source range associated with this error.
    pub fn range(&self) -> Range {
        match self {
            Self::Parse { range, .. } => range.into(),
            Self::Runtime(error) => error.range(),
        }
    }

    /// Whether parsing failed only because more input was expected.
    ///
    /// Interactive frontends can use this to prompt for another line.
    pub fn is_incomplete(&self) -> bool {
        matches!(
            self,
            Self::Parse {
                incomplete: true,
                ..
            }
        )
    }
}

/// Stateful façade for parsing and evaluating Eurydice programs.
///
/// Definitions and settings persist between calls to [`Engine::run`], making
/// the same API suitable for both one-shot execution and interactive sessions.
pub struct Engine {
    evaluator: Evaluator,
    next_source_id: u64,
    sources: Vec<DiagnosticSource>,
}

struct RunFailure {
    legacy: EngineError,
    diagnostic: EngineDiagnostic,
    warnings: Vec<EngineDiagnostic>,
}

type InternalRunResult = Result<(Vec<EngineOutput>, Vec<EngineDiagnostic>), Box<RunFailure>>;

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
        self.evaluator
            .set_print_callback(Box::new(move |value, name| {
                callback(value.to_string(), name);
            }));
    }

    /// Parses and executes a source submission, returning its output
    /// distributions.
    pub fn run(&mut self, source: &str) -> Result<Vec<EngineOutput>, EngineError> {
        self.run_internal(source)
            .map(|(outputs, _)| outputs)
            .map_err(|failure| failure.legacy)
    }

    /// Parses and executes a source submission, returning frontend-neutral
    /// structured diagnostics in addition to outputs.
    pub fn run_with_diagnostics(&mut self, source: &str) -> RunReport {
        match self.run_internal(source) {
            Ok((outputs, warnings)) => RunReport {
                outputs,
                diagnostics: warnings,
                sources: self.sources.clone(),
            },
            Err(failure) => {
                let mut diagnostics = failure.warnings;
                diagnostics.push(failure.diagnostic);
                // Match the legacy API: outputs from a failed submission are
                // discarded.
                RunReport {
                    outputs: Vec::new(),
                    diagnostics,
                    sources: self.sources.clone(),
                }
            }
        }
    }

    fn run_internal(&mut self, source: &str) -> InternalRunResult {
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
            Err(error) => {
                let (legacy, diagnostic) = parse_error(error, source_id, source);
                return Err(Box::new(RunFailure {
                    legacy,
                    diagnostic,
                    warnings: self.evaluator.take_diagnostics(),
                }));
            }
        };

        for (statement_index, statement) in statements.iter().enumerate() {
            if let Err(error) = self.evaluator.execute(statement) {
                self.evaluator.take_outputs();
                let mut diagnostic = runtime_diagnostic(
                    &error,
                    source_id,
                    &self
                        .evaluator
                        .variable_suggestions(undefined_name(&error).unwrap_or_default()),
                    &self
                        .evaluator
                        .function_suggestions(undefined_name(&error).unwrap_or_default()),
                );
                add_adjacent_dice_fix(&mut diagnostic, &error, &self.sources);
                add_later_definition(
                    &mut diagnostic,
                    &error,
                    &statements[statement_index + 1..],
                    source_id,
                );
                return Err(Box::new(RunFailure {
                    legacy: error.into(),
                    diagnostic,
                    warnings: self.evaluator.take_diagnostics(),
                }));
            }
        }

        let outputs = self
            .evaluator
            .take_outputs()
            .into_iter()
            .map(|output| EngineOutput {
                name: output.name,
                distribution: Distribution::from_runtime(output.value, output.field_names),
            })
            .collect();
        Ok((outputs, self.evaluator.take_diagnostics()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::{DiagnosticDetails, FixApplicability};

    #[test]
    fn runs_source_to_named_distributions() {
        let outputs = Engine::new().run("output 2d2 named \"roll\"").unwrap();

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
        engine.run("X: 4").unwrap();

        let outputs = engine.run("output X").unwrap();

        assert_eq!(outputs[0].distribution.probabilities, vec![(vec![4], 1.0)]);
    }

    #[test]
    fn reports_parse_and_runtime_ranges() {
        let parse_error = Engine::new().run("output (").unwrap_err();
        assert!(parse_error.is_incomplete());
        assert_eq!(parse_error.range().start, 8);

        let runtime_error = Engine::new().run("output MISSING").unwrap_err();
        assert!(!runtime_error.is_incomplete());
        assert_eq!(runtime_error.range().start, 7);
    }

    #[test]
    fn failed_runs_do_not_leak_outputs() {
        let mut engine = Engine::new();
        assert!(engine.run("output 1\noutput MISSING").is_err());

        assert!(engine.run("X: 1").unwrap().is_empty());
    }

    #[test]
    fn reports_common_syntax_mistakes_with_fixes() {
        let cases = [
            ("X = 1", "syntax.assignment_separator", ":"),
            ("output 1 == 1", "syntax.equality_operator", ""),
            ("output d6;", "syntax.semicolon", ""),
            ("output x", "syntax.variable_case", "X"),
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
        engine.run("FOOD: 1").unwrap();
        let report = engine.run_with_diagnostics("output FOO\nFOO: 2");
        let error = report.error().unwrap();

        let DiagnosticDetails::UndefinedName { suggestions, .. } = &error.details else {
            panic!("expected undefined-name details");
        };
        assert_eq!(error.labels[0].message, None);
        assert_eq!(suggestions, &["FOOD"]);
        assert!(
            error
                .labels
                .iter()
                .any(|label| label.message.as_deref() == Some("defined later"))
        );
    }

    #[test]
    fn suggests_a_comma_for_ambiguously_joined_dice_arguments() {
        let report = Engine::new().run_with_diagnostics("output [tuple d6 d8]");
        let error = report.error().unwrap();

        assert_eq!(error.code, "name.undefined_function");
        assert_eq!(error.fixes.len(), 1);
        assert_eq!(error.fixes[0].applicability, FixApplicability::Suggested);
        let edit = &error.fixes[0].edits[0];
        assert_eq!(&report.sources[0].text[edit.range.range.start..], " d8]");
        assert_eq!(edit.replacement, ",");
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
        assert_eq!(error.trace[0].bindings[0].value.preview, "3");
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
        let DiagnosticDetails::TypeMismatch {
            actual: Some(actual),
            ..
        } = &error.details
        else {
            panic!("expected a type mismatch with the actual value");
        };
        assert_eq!(actual.preview, "1");
        assert_eq!(actual.outcome_type, "int");
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

    #[test]
    fn primitive_diagnostics_explain_outcome_and_value_constraints() {
        for (source, signature) in [
            (
                "enum: R { A, B } output [{A} contains 1]",
                "[SEQ:s contains N:n]",
            ),
            (
                "enum: R { A, B } output [count {A} in {1}]",
                "[count NEEDLES:s in HAYSTACK:s]",
            ),
        ] {
            let report = Engine::new().run_with_diagnostics(source);
            let error = report.error().unwrap();
            assert_eq!(error.code, "type.outcome_mismatch");
            assert!(error.summary.contains(signature));
            assert_eq!(error.labels.len(), 2);
            assert!(
                error.labels[0]
                    .message
                    .as_deref()
                    .unwrap()
                    .contains("sequence of `R` values")
            );
            assert!(
                error.labels[1]
                    .message
                    .as_deref()
                    .unwrap()
                    .contains("sequence of `int` values")
                    || error.labels[1]
                        .message
                        .as_deref()
                        .unwrap()
                        .contains("an integer")
            );
        }

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
            assert!(
                error.labels[0]
                    .message
                    .as_deref()
                    .is_some_and(|message| message.contains(expected_message)),
                "{source}: {:?}",
                error.labels[0].message
            );
            let DiagnosticDetails::TypeMismatch {
                actual: Some(actual),
                ..
            } = &error.details
            else {
                panic!("expected the actual value for {source}");
            };
            assert_eq!(actual.shape, "dice_pool", "{source}");
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
        let DiagnosticDetails::TypeMismatch {
            actual: Some(actual),
            ..
        } = &error.details
        else {
            panic!("expected the actual output value");
        };
        assert_eq!(actual.preview, "1");

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
    fn diagnostics_retain_sources_for_persisted_functions() {
        let mut engine = Engine::new();
        engine.run("function: broken { result: MISSING }").unwrap();
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
            "function: loop only S:s { loop X over S { result: X } }",
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
