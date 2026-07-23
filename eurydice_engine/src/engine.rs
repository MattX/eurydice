//! High-level source-to-distribution engine API.

use lalrpop_util::ParseError;
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{ParseActionError, Range},
    eval::{Evaluator, RuntimeError},
    grammar,
    output::Distribution,
};

/// A named distribution produced by an `output` statement.
#[derive(Debug, Clone)]
pub struct EngineOutput {
    pub name: String,
    pub distribution: Distribution,
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
    Runtime(#[from] RuntimeError),
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
        // Outputs belong to one submission and must never leak out of a failed
        // previous run.
        self.evaluator.take_outputs();

        let statements = grammar::BodyParser::new()
            .parse(source)
            .map_err(parse_error)?;

        for statement in statements {
            if let Err(error) = self.evaluator.execute(&statement) {
                self.evaluator.take_outputs();
                return Err(error.into());
            }
        }

        Ok(self
            .evaluator
            .take_outputs()
            .into_iter()
            .map(|output| EngineOutput {
                name: output.name,
                distribution: Distribution::from_runtime(output.value, output.field_names),
            })
            .collect())
    }
}

fn parse_error<T: std::fmt::Display>(error: ParseError<usize, T, ParseActionError>) -> EngineError {
    let message = error.to_string();
    let (range, incomplete) = match error {
        ParseError::UnrecognizedToken { token, .. } => ((token.0, token.2), false),
        ParseError::UnrecognizedEof { location, .. } => ((location, location), true),
        ParseError::ExtraToken { token } => ((token.0, token.2), false),
        ParseError::InvalidToken { location } => ((location, location), false),
        ParseError::User { error } => {
            let range = match error {
                ParseActionError::InvalidIntegerLiteral { range, .. }
                | ParseActionError::EmptyFunctionCall { range } => (range.start, range.end),
            };
            (range, false)
        }
    };
    EngineError::Parse {
        message,
        range: Range::from(range).into(),
        incomplete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
