//! Runtime diagnostics.

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{self, BinaryOp, StaticType},
    diagnostic::{EvaluationFrame, SourceId},
    value::RuntimeValue,
};

#[derive(Debug)]
pub struct PrimitiveArgumentError {
    pub name: &'static str,
    pub range: SourceSpan,
    pub expected: String,
    pub value: RuntimeValue,
}

#[derive(Debug, Clone, Copy)]
pub enum PrimitiveArgumentErrorKind {
    Type,
    OutcomeType,
}

#[derive(Debug, Error, Diagnostic)]
#[error("Invalid arguments to primitive function [{function}]: {requirement}")]
pub struct PrimitiveArgumentsError {
    #[label = "{requirement}"]
    pub range: SourceSpan,
    pub function: &'static str,
    pub requirement: String,
    #[help]
    pub help: Option<String>,
    pub kind: PrimitiveArgumentErrorKind,
    pub arguments: Vec<PrimitiveArgumentError>,
}

#[derive(Debug, Error, Diagnostic)]
#[error("Invalid value passed to primitive function [{function}]: {requirement}")]
pub struct PrimitiveValueError {
    #[label = "{requirement}"]
    pub range: SourceSpan,
    pub function: &'static str,
    pub requirement: String,
    pub argument: &'static str,
    pub found_range: SourceSpan,
    pub value: RuntimeValue,
    pub constraint: String,
    #[help]
    pub help: Option<String>,
}

impl From<ast::Range> for SourceSpan {
    fn from(range: ast::Range) -> Self {
        SourceSpan::new(range.start.into(), range.end - range.start)
    }
}

#[derive(Debug, Error, Diagnostic)]
#[error("Runtime error")]
pub enum RuntimeError {
    #[error("{source}")]
    InFunction {
        #[label = "error occurred here"]
        range: SourceSpan,
        source: Box<RuntimeError>,
        body_source: SourceId,
        frame: Box<EvaluationFrame>,
    },

    #[error("Enum type error: {message}")]
    EnumTypeError {
        #[label = "{message}"]
        range: SourceSpan,
        message: String,
    },

    #[error("Tuple labels on a non-tuple output")]
    LabelsOnNonTupleOutput {
        #[label = "Labels can only be specified for tuple-valued outputs"]
        range: SourceSpan,
        value_range: SourceSpan,
        value: RuntimeValue,
    },

    #[error("Wrong number of tuple output labels: expected {expected}, found {found}")]
    OutputLabelCountMismatch {
        #[label = "Expected {expected} labels, found {found}"]
        range: SourceSpan,
        expected: usize,
        found: usize,
    },

    #[error("Output statement inside a function")]
    #[diagnostic(help("Output statements can only appear outside functions."))]
    OutputNotAtTopLevel {
        #[label = "Output statement inside a function"]
        range: SourceSpan,
    },

    #[error("Set statement inside a function")]
    #[diagnostic(help("Set statements can only appear outside functions."))]
    SetNotAtTopLevel {
        #[label = "Set statement inside a function"]
        range: SourceSpan,
    },

    #[error("Return statement outside a function")]
    ReturnOutsideFunction {
        #[label = "Return statement outside a function"]
        range: SourceSpan,
    },

    #[error("Loops must iterate over sequences")]
    LoopOverNonSequence {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("Reference to undefined variable [{name}]")]
    UndefinedReference {
        #[label = "Variable not defined"]
        range: SourceSpan,
        name: String,
    },

    #[error("Reference to undefined function [{name}]")]
    UndefinedFunction {
        #[label = "No function named [{name}]"]
        range: SourceSpan,
        name: String,
        /// Set when a function with the same words but a different number of
        /// arguments exists, which usually means adjacent arguments were
        /// accidentally parsed as a single expression.
        #[help]
        help: Option<String>,
    },

    #[error("Conditions to `if` statements must be numbers.")]
    InvalidCondition {
        #[label = "This is a {found} with value {value}."]
        range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("Both sides of a range constructor must evaluate to numbers.")]
    RangeHasNonSequenceEndpoints {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("Invalid argument to operator")]
    InvalidArgumentToOperator {
        #[label = "Operator {op} expects {expected}."]
        operator_range: SourceSpan,
        op: BinaryOp,
        expected: &'static str,

        #[label = "This is a {found} with value {value}."]
        found_range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("Invalid argument to function")]
    NegativeArgumentToFunction {
        #[label = "Function [{name}] expects a nonnegative integer."]
        range: SourceSpan,
        name: String,

        #[label = "This evaluates to {value}."]
        found_range: SourceSpan,
        value: i32,
    },

    #[error(transparent)]
    #[diagnostic(transparent)]
    InvalidPrimitiveArguments(Box<PrimitiveArgumentsError>),

    #[error(transparent)]
    #[diagnostic(transparent)]
    InvalidPrimitiveValue(Box<PrimitiveValueError>),

    #[error("Invalid repeat expression")]
    #[diagnostic(help("The expression inside the repeat operator must evaluate to an int."))]
    InvalidRepeatExpression {
        #[label = "This is a {found}: {value}."]
        range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("A mathematical error occurred.")]
    MathError {
        #[label = "{message}"]
        range: SourceSpan,
        message: String,
    },
}

impl RuntimeError {
    pub fn range(&self) -> ast::Range {
        match self {
            RuntimeError::InFunction { range, .. } => range.into(),
            RuntimeError::EnumTypeError { range, .. } => range.into(),
            RuntimeError::LabelsOnNonTupleOutput { range, .. } => range.into(),
            RuntimeError::OutputLabelCountMismatch { range, .. } => range.into(),
            RuntimeError::OutputNotAtTopLevel { range } => range.into(),
            RuntimeError::SetNotAtTopLevel { range } => range.into(),
            RuntimeError::ReturnOutsideFunction { range } => range.into(),
            RuntimeError::LoopOverNonSequence { range, .. } => range.into(),
            RuntimeError::UndefinedReference { range, .. } => range.into(),
            RuntimeError::UndefinedFunction { range, .. } => range.into(),
            RuntimeError::InvalidCondition { range, .. } => range.into(),
            RuntimeError::RangeHasNonSequenceEndpoints { range, .. } => range.into(),
            RuntimeError::InvalidArgumentToOperator { operator_range, .. } => operator_range.into(),
            RuntimeError::NegativeArgumentToFunction { range, .. } => range.into(),
            RuntimeError::InvalidPrimitiveArguments(error) => (&error.range).into(),
            RuntimeError::InvalidPrimitiveValue(error) => (&error.range).into(),
            RuntimeError::InvalidRepeatExpression { range, .. } => range.into(),
            RuntimeError::MathError { range, .. } => range.into(),
        }
    }
}
