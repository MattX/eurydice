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
    fn suggests_names_and_marks_definitions_that_execute_later() {
        let mut engine = Engine::new();
        engine.run("FOOD: 1").unwrap();
        let report = engine.run_with_diagnostics("output FOO\nFOO: 2");
        let error = report.error().unwrap();

        let DiagnosticDetails::UndefinedName { suggestions, .. } = &error.details else {
            panic!("expected undefined-name details");
        };
        assert_eq!(suggestions, &["FOOD"]);
        assert!(
            error
                .labels
                .iter()
                .any(|label| label.message == "defined later")
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
