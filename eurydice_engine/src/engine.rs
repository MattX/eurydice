//! High-level source-to-distribution engine API.

use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{
    ast::{Statement, WithRange},
    diagnostic::{
        Diagnostic, DiagnosticSeverity, DiagnosticSource, SourceId, UndefinedName,
        add_later_definition, parse_error, runtime_diagnostic, undefined_name,
    },
    eval::Evaluator,
    grammar,
    output::Distribution,
};

/// A named distribution produced by an `output` statement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct EngineOutput {
    pub name: String,
    pub distribution: Distribution,
}

/// Outputs and diagnostics produced by one source submission.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[must_use]
#[non_exhaustive]
pub struct RunReport {
    pub outputs: Vec<EngineOutput>,
    pub diagnostics: Vec<Diagnostic>,
    /// Every source the report's diagnostics point into, plus the submission
    /// that produced it. The submission is always last.
    pub sources: Vec<DiagnosticSource>,
}

impl RunReport {
    pub fn has_errors(&self) -> bool {
        self.error().is_some()
    }

    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code.severity() == DiagnosticSeverity::Error)
    }

    pub fn error(&self) -> Option<&Diagnostic> {
        self.errors().next()
    }
}

/// An error returned by the compiler.
#[derive(Debug, Clone)]
pub struct CompileError {
    report: RunReport,
}

impl CompileError {
    /// Returns the diagnostic that caused the compilation to fail.
    pub fn diagnostic(&self) -> &Diagnostic {
        self.report
            .error()
            .expect("a compilation error report contains an error diagnostic")
    }

    /// Returns the report that caused the compilation to fail.
    pub fn report(&self) -> &RunReport {
        &self.report
    }

    /// Consumes the error and returns the report that caused the compilation to fail.
    pub fn into_report(self) -> RunReport {
        self.report
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.report.error() {
            Some(error) => write!(f, "{}: {}", error.code, error.summary),
            None => f.write_str("program compilation failed"),
        }
    }
}

impl std::error::Error for CompileError {}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PrintEvent {
    pub name: Option<String>,
    pub value: String,
}

/// A compiled program, ready to run.
///
/// Parsing is a substantial share of a short program's cost — a quarter to a
/// third of the total for a small pool program — so a caller that runs the same
/// source repeatedly should compile it once with [`Program::compile`] and then
/// run it with [`Engine::run_program`].
///
/// A program carries its own source text and is independent of the engine that
/// compiled it: it may be run on any engine, any number of times. Each run is a
/// separate submission, exactly as if the text had been submitted again.
#[derive(Debug, Clone)]
pub struct Program {
    name: Option<String>,
    text: String,
    statements: Vec<WithRange<Statement>>,
}

impl Program {
    /// Compiles source without involving an engine session.
    ///
    /// A parse failure is returned as a self-contained report whose source has
    /// ID zero. Running a source through [`Engine::run_source`] instead assigns
    /// the failure an ID from that engine's submission history.
    pub fn compile(source: &str) -> Result<Self, CompileError> {
        Self::compile_with_name(None, source)
    }

    pub fn compile_named(name: impl Into<String>, source: &str) -> Result<Self, CompileError> {
        Self::compile_with_name(Some(name.into()), source)
    }

    fn compile_with_name(name: Option<String>, source: &str) -> Result<Self, CompileError> {
        let source_id = SourceId(0);
        compile_program(source, source_id, name.clone()).map_err(|diagnostic| CompileError {
            report: RunReport {
                outputs: Vec::new(),
                diagnostics: vec![*diagnostic],
                sources: vec![DiagnosticSource {
                    id: source_id,
                    name: name.unwrap_or_else(|| "source".to_string()),
                    text: source.to_string(),
                }],
            },
        })
    }

    /// The source text this program was parsed from.
    pub fn source(&self) -> &str {
        &self.text
    }

    pub fn source_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Stateful façade for parsing and evaluating Eurydice programs.
///
/// Definitions and settings persist between calls to [`Engine::run_program`]
/// and [`Engine::run_source`], making the same API suitable for both
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
        F: FnMut(PrintEvent) + 'static,
    {
        self.evaluator.set_print_callback(Box::new(callback));
    }

    pub fn clear_print_callback(&mut self) {
        self.evaluator.clear_print_callback();
    }

    /// Executes a compiled program, returning its outputs and any diagnostics
    /// it produced.
    ///
    /// Each call is a fresh submission: the program's text is recorded again
    /// under a new source, so a program may be run repeatedly and diagnostics
    /// from each run point into that run's own submission.
    ///
    /// A diagnostic may point into an earlier submission that defined a
    /// function, so the report carries the text of every source its own
    /// diagnostics reference.
    pub fn run_program(&mut self, program: &Program) -> RunReport {
        let source_id = self.begin_submission(program.name.as_deref(), &program.text);
        self.execute_report(program, source_id)
    }

    /// Parses and executes a source submission, returning its outputs and any
    /// diagnostics it produced.
    ///
    /// To run one program repeatedly, compile it once with [`Program::compile`]
    /// and call [`Engine::run_program`] directly.
    pub fn run_source(&mut self, source: &str) -> RunReport {
        self.run_source_with_name(None, source)
    }

    pub fn run_named_source(&mut self, name: impl Into<String>, source: &str) -> RunReport {
        self.run_source_with_name(Some(name.into()), source)
    }

    fn run_source_with_name(&mut self, name: Option<String>, source: &str) -> RunReport {
        let source_id = self.begin_submission(name.as_deref(), source);
        match compile_program(source, source_id, name) {
            Ok(program) => self.execute_report(&program, source_id),
            Err(diagnostic) => self.report(Vec::new(), vec![*diagnostic]),
        }
    }

    /// Starts a fresh submission and returns the ID assigned to its source.
    fn begin_submission(&mut self, name: Option<&str>, source: &str) -> SourceId {
        self.collect_garbage();
        let source_id = self.register_source(name, source);
        // Outputs belong to one submission and must never leak out of a failed
        // previous run.
        self.evaluator.take_outputs();
        self.evaluator.begin_submission(source_id);
        source_id
    }

    fn execute_report(&mut self, program: &Program, source_id: SourceId) -> RunReport {
        let outcome = self.execute(program, source_id);
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
        self.report(outputs, diagnostics)
    }

    /// A source outlives its submission only while a persisted function can
    /// still be traced back to it. Dropping the rest keeps an interactive
    /// session — which resubmits a growing program on every keystroke — from
    /// grinding to a halt.
    fn collect_garbage(&mut self) {
        let live = self.evaluator.live_source_ids();
        self.sources.retain(|source| live.contains(&source.id));
    }

    /// Records a submission's text so diagnostics can point into it.
    fn register_source(&mut self, name: Option<&str>, text: &str) -> SourceId {
        let source_id = SourceId(self.next_source_id);
        self.next_source_id += 1;
        self.sources.push(DiagnosticSource {
            id: source_id,
            name: name
                .map(str::to_owned)
                .unwrap_or_else(|| format!("submission {}", source_id.0 + 1)),
            text: text.to_string(),
        });
        source_id
    }

    fn report(&self, outputs: Vec<EngineOutput>, diagnostics: Vec<Diagnostic>) -> RunReport {
        let sources = self.referenced_sources(&diagnostics);
        RunReport {
            outputs,
            diagnostics,
            sources,
        }
    }

    /// The sources the report needs to render, in submission order.
    fn referenced_sources(&self, diagnostics: &[Diagnostic]) -> Vec<DiagnosticSource> {
        let mut referenced = diagnostics
            .iter()
            .flat_map(|diagnostic| {
                let labels = diagnostic.labels().map(|label| label.range.source);
                let edits = diagnostic
                    .fix
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

    fn execute(
        &mut self,
        program: &Program,
        source_id: SourceId,
    ) -> Result<Vec<EngineOutput>, Box<Diagnostic>> {
        let statements = &program.statements;
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

fn compile_program(
    source: &str,
    source_id: SourceId,
    name: Option<String>,
) -> Result<Program, Box<Diagnostic>> {
    grammar::BodyParser::new()
        .parse(source)
        .map(|statements| Program {
            name,
            text: source.to_string(),
            statements,
        })
        .map_err(|error| Box::new(parse_error(error, source_id, source)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::{DiagnosticCode, SuggestedFix};

    /// The fix a diagnostic is expected to carry.
    fn fix(diagnostic: &Diagnostic) -> &SuggestedFix {
        diagnostic.fix.as_ref().expect("diagnostic carries a fix")
    }

    /// Runs a submission that is expected to succeed, returning its outputs.
    fn run(engine: &mut Engine, source: &str) -> Vec<EngineOutput> {
        let report = engine.run_source(source);
        assert_eq!(report.error(), None, "{source}");
        report.outputs
    }

    /// The point of compiling separately: one parse, many runs. Each run is its
    /// own submission, so a program that binds a name binds it again.
    #[test]
    fn a_compiled_program_can_be_run_repeatedly() {
        let mut engine = Engine::new();
        let program = Program::compile("X: 1\noutput X named \"x\"").expect("parses");

        for _ in 0..3 {
            let report = engine.run_program(&program);
            assert_eq!(report.error(), None);
            assert_eq!(report.outputs[0].distribution.entries, vec![(vec![1], 1.0)]);
        }
    }

    /// A program carries its own text and holds no reference to the engine that
    /// parsed it, so it runs anywhere — including on an engine that has never
    /// seen the source.
    #[test]
    fn a_program_is_independent_of_the_engine_that_compiled_it() {
        let program = Program::compile("output 2d2").expect("parses");

        let report = Engine::new().run_program(&program);

        assert_eq!(report.error(), None);
        assert_eq!(
            report.outputs[0].distribution.entries,
            vec![(vec![2], 0.25), (vec![3], 0.5), (vec![4], 0.25)]
        );
        // The run registered the program's text as its own submission, so the
        // report can be rendered without the caller supplying the source.
        assert_eq!(report.sources.len(), 1);
        assert_eq!(report.sources[0].text, "output 2d2");
    }

    /// A failed parse comes back as a report rather than a bare diagnostic,
    /// because rendering it needs the source the diagnostic points into.
    #[test]
    fn a_failed_compile_reports_against_its_own_source() {
        let error = Program::compile("output (1").expect_err("does not parse");
        let report = error.report();

        let error = report.error().expect("a parse error");
        assert_eq!(error.code, DiagnosticCode::UnclosedDelimiter);
        let range = error.primary_label.range;
        assert_eq!(report.sources.len(), 1);
        assert_eq!(report.sources[0].id, range.source);
    }

    #[test]
    fn compile_errors_integrate_with_standard_error_handling() {
        fn compile() -> Result<(), Box<dyn std::error::Error>> {
            Program::compile("output (1")?;
            Ok(())
        }

        let error = compile().expect_err("does not parse");
        assert!(error.to_string().contains("syntax.unclosed_delimiter"));
    }

    #[test]
    fn named_programs_keep_their_name_for_parse_and_runtime_diagnostics() {
        let compile_error =
            Program::compile_named("broken.eurydice", "output (").expect_err("does not parse");
        assert_eq!(compile_error.report().sources[0].name, "broken.eurydice");

        let program = Program::compile_named("rules.eurydice", "output MISSING")
            .expect("runtime errors still compile");
        assert_eq!(program.source_name(), Some("rules.eurydice"));
        let report = Engine::new().run_program(&program);
        assert_eq!(report.sources.last().unwrap().name, "rules.eurydice");
    }

    #[test]
    fn direct_source_runs_can_be_named() {
        let report = Engine::new().run_named_source("request.eurydice", "output MISSING");

        assert_eq!(report.sources.last().unwrap().name, "request.eurydice");
    }

    #[test]
    fn print_callbacks_are_mutable_and_removable() {
        use std::{cell::RefCell, rc::Rc};

        let events = Rc::new(RefCell::new(Vec::new()));
        let callback_events = Rc::clone(&events);
        let mut engine = Engine::new();
        engine.set_print_callback(move |event| callback_events.borrow_mut().push(event));

        let _ = engine.run_source("print 1\nprint 2 named \"two\"");
        engine.clear_print_callback();
        let _ = engine.run_source("print 3");

        let events = events.borrow();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].name, None);
        assert_eq!(events[0].value, "1");
        assert_eq!(events[1].name.as_deref(), Some("two"));
        assert_eq!(events[1].value, "2");
    }

    #[test]
    fn runs_source_to_named_distributions() {
        let outputs = run(&mut Engine::new(), "output 2d2 named \"roll\"");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "roll");
        assert_eq!(
            outputs[0].distribution.entries,
            vec![(vec![2], 0.25), (vec![3], 0.5), (vec![4], 0.25)]
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn run_reports_serialize_the_public_wire_values() {
        let report = Engine::new().run_source("D: d6\noutput D + D named \"sum\"");
        let serialized = serde_lexpr::to_string(&report).expect("report serializes");

        assert!(serialized.contains("evaluation.independent_pool_reuse"));
        assert!(serialized.contains("primary_label"));
        assert!(serialized.contains("secondary_labels"));
        assert!(serialized.contains("sum"));
        assert!(serialized.contains("D: d6\\noutput D + D"));
    }

    #[test]
    fn state_persists_between_runs() {
        let mut engine = Engine::new();
        run(&mut engine, "X: 4");

        let outputs = run(&mut engine, "output X");

        assert_eq!(outputs[0].distribution.entries, vec![(vec![4], 1.0)]);
    }

    #[test]
    fn reports_parse_and_runtime_ranges() {
        let parse_report = Engine::new().run_source("output (");
        let parse_error = parse_report.error().unwrap();
        assert!(parse_error.code.is_incomplete());
        assert_eq!(parse_error.primary_label.range.range.start, 8);

        let runtime_report = Engine::new().run_source("output MISSING");
        let runtime_error = runtime_report.error().unwrap();
        assert!(!runtime_error.code.is_incomplete());
        assert_eq!(runtime_error.primary_label.range.range.start, 7);
    }

    #[test]
    fn failed_runs_do_not_leak_outputs() {
        let mut engine = Engine::new();
        assert!(
            engine
                .run_source("output 1\noutput MISSING")
                .error()
                .is_some()
        );

        assert!(run(&mut engine, "X: 1").is_empty());
    }

    #[test]
    fn reports_common_syntax_mistakes_with_fixes() {
        let cases = [
            ("X = 1", DiagnosticCode::AssignmentSeparator, ":"),
            ("output 1 == 1", DiagnosticCode::EqualityOperator, ""),
            ("output d6;", DiagnosticCode::UnexpectedSemicolon, ""),
            ("output x", DiagnosticCode::VariableCase, "X"),
            ("output total", DiagnosticCode::VariableCase, "TOTAL"),
            ("output (1", DiagnosticCode::UnclosedDelimiter, ")"),
        ];

        for (source, code, replacement) in cases {
            let report = Engine::new().run_source(source);
            let error = report.error().unwrap();
            assert_eq!(error.code, code);
            assert_eq!(fix(error).edits[0].replacement, replacement);
        }
    }

    #[test]
    fn single_letter_function_names_are_accepted() {
        let mut engine = Engine::new();
        run(&mut engine, "function: f X:n { result: X }");

        let outputs = run(&mut engine, "output [f 3]");

        assert_eq!(outputs[0].distribution.entries, vec![(vec![3], 1.0)]);
    }

    #[test]
    fn misplaced_keywords_are_not_mistaken_for_lowercase_variables() {
        let report = Engine::new().run_source("output result");
        let error = report.error().unwrap();

        assert_eq!(error.code, DiagnosticCode::UnexpectedToken);
        assert!(error.fix.is_none());
    }

    #[test]
    fn rejects_result_outside_a_function() {
        let report = Engine::new().run_source("result: 1");

        assert_eq!(
            report.error().map(|error| error.code),
            Some(DiagnosticCode::ResultOutsideFunction)
        );
    }

    #[test]
    fn unclosed_delimiters_ignore_strings_and_comments() {
        let cases = ["output (1 \\ ) \\", "output (1 \\\\\\ )"];

        for source in cases {
            let report = Engine::new().run_source(source);
            let error = report.error().unwrap();

            assert_eq!(error.code, DiagnosticCode::UnclosedDelimiter, "{source}");
            assert_eq!(fix(error).edits[0].replacement, ")", "{source}");
        }
    }

    #[test]
    fn explains_that_top_level_expressions_need_output() {
        for source in ["1d6", "D: d6\nD + D"] {
            let report = Engine::new().run_source(source);
            let error = report.error().unwrap();

            assert_eq!(error.code, DiagnosticCode::UnexpectedToken);
            assert_eq!(
                error.help.as_deref(),
                Some("At the top level, expressions you want to show must start with `output`.")
            );
            assert!(error.fix.is_none());
        }
    }

    #[test]
    fn suggests_names_and_marks_definitions_that_execute_later() {
        let mut engine = Engine::new();
        run(&mut engine, "FOOD: 1");
        let report = engine.run_source("output FOO\nFOO: 2");
        let error = report.error().unwrap();

        assert_eq!(error.primary_label.message, None);
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
                .help
                .as_deref()
                .is_some_and(|help| help.contains("this definition is not available yet")),
            "{:?}",
            error.help
        );
        assert!(
            error
                .secondary_labels
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
            let report = Engine::new().run_source(source);
            let error = report.error().unwrap();

            assert_eq!(error.code, DiagnosticCode::UndefinedFunction, "{source}");
            let edit = &fix(error).edits[0];
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
        let report = Engine::new().run_source("output [absolute 1, 2]");
        let error = report.error().unwrap();

        assert_eq!(error.code, DiagnosticCode::UndefinedFunction);
        assert!(error.fix.is_none());
    }

    #[test]
    fn traces_pool_expansion_with_the_failing_argument_value() {
        let source = "\
function: pick I:n {
  result: [field I of [tuple 10 20]]
}
output [pick d3]";
        let report = Engine::new().run_source(source);
        let error = report.error().unwrap();

        assert_eq!(error.code, DiagnosticCode::OutOfRange);
        assert_eq!(error.trace.len(), 1);
        assert_eq!(error.trace[0].function, "pick {}");
        assert_eq!(error.trace[0].bindings[0].name, "I");
        assert_eq!(error.trace[0].bindings[0].value, "3");
    }

    #[test]
    fn primitive_diagnostics_identify_invalid_field_arguments() {
        let source = "output [field d2 of 1]";
        let report = Engine::new().run_source(source);
        let error = report.error().unwrap();

        assert_eq!(error.code, DiagnosticCode::FunctionArgument);
        assert_eq!(
            error.summary,
            "`[field INDEX:n of TUPLE:n]` requires `INDEX` to be an integer and `TUPLE` to be a tuple"
        );
        assert!(error.secondary_labels.is_empty());
        let label = &error.primary_label;
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
        let nested = Engine::new().run_source("output [tuple [tuple 1, 2], [tuple 3, 4]]");
        let nested_error = nested.error().unwrap();
        assert_eq!(nested_error.code, DiagnosticCode::FunctionArgument);
        assert_eq!(
            nested_error.summary,
            "`[tuple A:n B:n]` cannot contain another tuple"
        );
        assert_eq!(nested_error.secondary_labels.len(), 1);
        assert!(
            nested_error
                .primary_label
                .message
                .as_deref()
                .unwrap()
                .contains("`A` is a tuple")
        );
        assert!(
            nested_error.secondary_labels[0]
                .message
                .as_deref()
                .unwrap()
                .contains("`B` is a tuple")
        );

        let field = Engine::new().run_source("output [field [tuple 1, 2] of 1]");
        let field_error = field.error().unwrap();
        assert_eq!(field_error.code, DiagnosticCode::FunctionArgument);
        assert_eq!(field_error.secondary_labels.len(), 1);
        assert!(
            field_error
                .primary_label
                .message
                .as_deref()
                .unwrap()
                .contains("`INDEX` is a tuple")
        );
        assert!(
            field_error.secondary_labels[0]
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
            "enum { A, B } output [{A} contains 1]",
            "enum { A, B } output [count {A} in {1}]",
            "enum { A, B } output [count {1} in 2d{A, B}]",
            "output [{[tuple 1, 2]} contains 1]",
        ] {
            let mut engine = Engine::new();
            let outputs = run(&mut engine, source);
            assert_eq!(
                outputs[0].distribution.entries,
                vec![(vec![0], 1.0)],
                "{source}"
            );
        }
    }

    #[test]
    fn primitive_diagnostics_explain_outcome_and_value_constraints() {
        let bounds = Engine::new().run_source("output [field 3 of [tuple 1, 2]]");
        let bounds_error = bounds.error().unwrap();
        assert_eq!(bounds_error.code, DiagnosticCode::OutOfRange);
        assert_eq!(
            bounds_error.summary,
            "`[field INDEX:n of TUPLE:n]` cannot select field `3`"
        );
        assert!(
            bounds_error
                .primary_label
                .message
                .as_deref()
                .unwrap()
                .contains("expected an index from 1 through 2")
        );

        let negative = Engine::new().run_source("output [highest -1 of 2d6]");
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
                DiagnosticCode::ExpectedSequence,
                "expected a sequence",
            ),
            (
                "if d6 { print 1 }",
                DiagnosticCode::ExpectedNumber,
                "expected an integer",
            ),
            (
                "output {1:d6}",
                DiagnosticCode::RepeatCount,
                "expected an integer",
            ),
            (
                "output {1..d6}",
                DiagnosticCode::RangeEndpoint,
                "expected an integer",
            ),
        ];

        for (source, code, expected_message) in cases {
            let report = Engine::new().run_source(source);
            let error = report.error().unwrap();
            assert_eq!(error.code, code, "{source}");
            let message = error.primary_label.message.as_deref().unwrap_or_default();
            // The label names the type that was found as well as the one the
            // position required.
            assert!(message.contains(expected_message), "{source}: {message}");
            assert!(message.contains("is a dice pool"), "{source}: {message}");
        }
    }

    #[test]
    fn output_label_errors_explain_the_output_shape_and_arity() {
        let non_tuple = Engine::new().run_source("output 1 labeled \"Value\"");
        let error = non_tuple.error().unwrap();
        assert_eq!(error.code, DiagnosticCode::LabelsRequireTuple);
        assert_eq!(error.secondary_labels.len(), 1);
        assert!(
            error
                .primary_label
                .message
                .as_deref()
                .is_some_and(|message| message.contains("`1` is an integer; expected a tuple"))
        );

        let wrong_count = Engine::new().run_source("output [tuple 1 2] labeled \"Only one\"");
        let error = wrong_count.error().unwrap();
        assert_eq!(
            error.summary,
            "This tuple has 2 fields, but 1 output label was provided"
        );
        assert_eq!(error.primary_label.message, None);
    }

    #[test]
    fn non_primitive_diagnostics_do_not_repeat_the_summary_as_a_label() {
        let report = Engine::new().run_source("output [missing]");
        assert_eq!(report.error().unwrap().primary_label.message, None);

        // A label may still carry something the summary does not. Adding two
        // symbols is reported at the value that cannot be added, which the
        // summary has no room to name.
        let report = Engine::new().run_source("enum { A, B } output A + B");
        let error = report.error().unwrap();
        assert_eq!(error.summary, "this operator requires numbers");
        assert_eq!(
            error.primary_label.message.as_deref(),
            Some("values like `B` are not numbers")
        );

        let operator = Engine::new().run_source("output d6 @ d6");
        let error = operator.error().unwrap();
        assert_eq!(
            error.summary,
            "Operator `@` expects an integer or a sequence"
        );
        assert_eq!(error.secondary_labels[0].message, None);
        assert!(
            error
                .primary_label
                .message
                .as_deref()
                .is_some_and(|message| message.contains("is a dice pool"))
        );
    }

    #[test]
    fn reports_carry_only_the_sources_they_point_into() {
        let mut engine = Engine::new();

        // The failed submission is still rendered against its own source.
        let failed = engine.run_source("output (");
        assert_eq!(failed.sources.len(), 1);
        assert_eq!(
            failed.error().unwrap().primary_label.range.source,
            SourceId(0)
        );

        let succeeded = engine.run_source("output 1");
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
        let report = engine.run_source("output [broken]");
        let error = report.error().unwrap();

        assert_eq!(error.primary_label.range.source, SourceId(0));
        assert_eq!(error.trace[0].call.source, SourceId(1));
        assert_eq!(report.sources.len(), 2);
    }

    #[test]
    fn warns_when_the_same_pool_is_sampled_independently() {
        let report = Engine::new().run_source("D: d6\noutput D + D");
        assert!(report.error().is_none());
        assert_eq!(
            report.diagnostics[0].code,
            DiagnosticCode::IndependentPoolReuse
        );
        let help = report.diagnostics[0].help.as_deref().unwrap();
        assert!(help.contains("stores its distribution"));
        assert!(help.contains("To reuse one roll"));

        let no_warning = Engine::new().run_source("D: d6\noutput #D + #D");
        assert!(no_warning.diagnostics.is_empty());
    }

    #[test]
    fn warns_when_a_function_can_finish_without_a_result() {
        let cases = [
            "function: empty {}",
            "function: partial X:n { if X { result: 1 } }",
        ];

        for source in cases {
            let report = Engine::new().run_source(source);
            let warning = &report.diagnostics[0];
            assert_eq!(warning.code, DiagnosticCode::MissingResult, "{source}");
            assert_eq!(warning.code.severity(), DiagnosticSeverity::Warning);
            let help = warning.help.as_deref().unwrap();
            assert!(help.contains("produces an empty die"), "{source}");
            assert!(help.contains("Add `result:`"), "{source}");
            assert!(warning.fix.is_none());
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
            let report = Engine::new().run_source(source);
            assert!(report.diagnostics.is_empty(), "{source}");
        }
    }

    /// An empty die displays as no outcomes at all — not as a single outcome at
    /// zero — and stays that way after passing through a function that demands
    /// `int` arguments.
    ///
    /// Summing an empty die turns its additive identity into a real outcome, so
    /// the display rule leaves it alone while its outcome type still says
    /// nothing about what it would have produced. Coercing it to an `int`
    /// parameter used to overwrite that type, which made `[explode d{}]`
    /// disagree with the `d{}` it was given.
    #[test]
    fn an_empty_die_outputs_no_outcomes_even_after_coercion() {
        // `explode` and `reroll` hand an empty die straight back, so what they
        // display is decided entirely by the coercion and the display rule.
        // `highest`/`lowest`/`middle` are not listed: they sum the dice they
        // keep, and the sum of no dice is `0`, which is a claim about those
        // primitives rather than about empty dice.
        for source in [
            "function: nothing {}\noutput [nothing]",
            "output d{}",
            "output [explode d{}]",
            "output [reroll d{}]",
        ] {
            let report = Engine::new().run_source(source);
            assert_eq!(report.error(), None, "{source}");
            assert_eq!(report.outputs.len(), 1, "{source}");
            assert_eq!(report.outputs[0].distribution.entries, vec![], "{source}");
        }
    }

    /// Knowing what an empty die's faces *would* have been does not make it
    /// rollable, so a die with no faces displays as no outcomes whatever its
    /// outcome type. `d{}` and `d{5:0}` differ only in whether a face was
    /// written down, and neither has an outcome to show.
    #[test]
    fn an_empty_die_outputs_no_outcomes_whatever_its_type() {
        for source in ["output d{}", "output d{5:0}", "output d{[tuple 5 6]:0}"] {
            let report = Engine::new().run_source(source);
            assert_eq!(report.error(), None, "{source}");
            assert_eq!(report.outputs[0].distribution.entries, vec![], "{source}");
        }
    }

    /// The contrast that keeps the rule honest: it asks about faces, not dice.
    /// `0d6` has faces and no dice, so it still sums — to the `0` that rolling
    /// no dice yields.
    #[test]
    fn rolling_no_dice_still_sums_to_zero() {
        for source in ["output 0d6", "output d0"] {
            let outputs = run(&mut Engine::new(), source);

            assert_eq!(
                outputs[0].distribution.entries,
                vec![(vec![0], 1.0)],
                "{source}"
            );
        }
    }

    #[test]
    fn warns_when_depth_settings_bound_results() {
        let explode = Engine::new().run_source("output [explode d6]");
        assert_eq!(explode.diagnostics[0].code, DiagnosticCode::ExplodeDepth);
        let help = explode.diagnostics[0].help.as_deref().unwrap();
        assert!(help.contains("bounded by this setting"));
        assert!(help.contains("Change it with"));

        let recursion = Engine::new()
            .run_source("function: recurse N:n { result: [recurse N] }\noutput [recurse 1]");
        assert_eq!(
            recursion.diagnostics[0].code,
            DiagnosticCode::MaximumFunctionDepth
        );
    }
}
