//! Errors raised during evaluation.
//!
//! These carry the facts a diagnostic needs — spans, offending values, and the
//! reason a rule was broken — but no user-facing prose. Wording lives in
//! [`crate::diagnostic`], so there is exactly one place where an error's
//! presentation can change.

use crate::{
    ast::{self, BinaryOp},
    diagnostic::{DiagnosticCode, EvaluationFrame, SourceId},
    value::{ElementValue, RuntimeValue},
};

#[derive(Debug)]
pub struct PrimitiveArgumentError {
    pub name: &'static str,
    pub range: ast::ByteRange,
    pub expected: String,
    pub value: RuntimeValue,
}

#[derive(Debug)]
pub struct PrimitiveArgumentsError {
    pub range: ast::ByteRange,
    pub function: &'static str,
    pub requirement: String,
    pub help: Option<String>,
    pub arguments: Vec<PrimitiveArgumentError>,
}

#[derive(Debug)]
pub struct PrimitiveValueError {
    pub range: ast::ByteRange,
    pub function: &'static str,
    pub requirement: String,
    pub argument: &'static str,
    pub found_range: ast::ByteRange,
    pub value: RuntimeValue,
    pub constraint: String,
    pub help: Option<String>,
}

/// Two values whose shapes do not line up, found by an operation that needed
/// them to.
///
/// Values may differ in kind — a number here, a symbol there — and still share
/// a sequence or a pool. What no operation can reconcile is a difference in
/// *shape*: a scalar meeting a tuple, or two tuples of different sizes. Nothing
/// checks for this when the collection is built, so the values themselves are
/// what the diagnostic names.
#[derive(Debug)]
pub struct ShapeMismatchError {
    pub range: ast::ByteRange,
    /// What needed the two to line up; completes "<action> requires summing
    /// ...".
    pub action: &'static str,
    pub first: ElementValue,
    pub second: ElementValue,
}

/// What had to be added, and how much of it.
#[derive(Debug, Clone, Copy)]
pub enum NonAdditiveSubject {
    /// A pool of this many dice.
    Pool(u32),
    /// A sequence of this many values, such as one multiset of a pool.
    Sequence(usize),
    /// A single value an operator was applied to, where no collection is
    /// involved at all and there is nothing to count.
    Operand,
}

/// A value that had to be summed, but whose parts cannot be added together.
#[derive(Debug)]
pub struct NonAdditiveSumError {
    pub range: ast::ByteRange,
    /// What forced the sum; completes "<action> requires summing ...".
    pub action: &'static str,
    pub subject: NonAdditiveSubject,
    /// An outcome that cannot be added, so the diagnostic can name a concrete
    /// value rather than only its type. `None` for a pool with no outcomes.
    pub witness: Option<ElementValue>,
    /// For a tuple witness, the one-based field that is not a number.
    pub field: Option<usize>,
}

/// A call that named no function, but whose words match a function taking a
/// different number of arguments.
///
/// This usually means adjacent arguments were parsed as a single expression,
/// so the diagnostic can both explain the arity and offer a comma.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArityMismatch {
    /// The function's words, without argument slots.
    pub words: String,
    /// The argument counts the words are defined for, ascending.
    pub available: Vec<usize>,
    /// The number of arguments the call passed.
    pub found: usize,
    /// Where a comma would separate two arguments that were parsed as one,
    /// when the call's expression shows that is what happened.
    pub comma_insertion: Option<usize>,
}

#[derive(Debug)]
pub enum RuntimeError {
    /// An error raised inside a function body, wrapped with the call that
    /// reached it. Nested calls nest these.
    InFunction {
        range: ast::ByteRange,
        source: Box<RuntimeError>,
        body_source: SourceId,
        frame: Box<EvaluationFrame>,
    },

    /// An error whose message is built at the raise site, classified by the
    /// code it reports as. Everything a diagnostic would add beyond the
    /// message — a second span, a note, a fix — needs a variant of its own.
    Semantic {
        code: DiagnosticCode,
        range: ast::ByteRange,
        message: String,
    },

    LabelsOnNonTupleOutput {
        range: ast::ByteRange,
        value_range: ast::ByteRange,
        value: RuntimeValue,
    },

    OutputLabelCountMismatch {
        range: ast::ByteRange,
        expected: usize,
        found: usize,
    },

    OutputNotAtTopLevel {
        range: ast::ByteRange,
    },

    SetNotAtTopLevel {
        range: ast::ByteRange,
    },

    ReturnOutsideFunction {
        range: ast::ByteRange,
    },

    LoopOverNonSequence {
        range: ast::ByteRange,
        value: RuntimeValue,
    },

    UndefinedReference {
        range: ast::ByteRange,
        name: String,
    },

    UndefinedFunction {
        range: ast::ByteRange,
        name: String,
        arity_mismatch: Option<ArityMismatch>,
    },

    InvalidCondition {
        range: ast::ByteRange,
        value: RuntimeValue,
    },

    RangeHasNonSequenceEndpoints {
        range: ast::ByteRange,
        value: RuntimeValue,
    },

    InvalidArgumentToOperator {
        operator_range: ast::ByteRange,
        op: BinaryOp,
        expected: &'static str,
        found_range: ast::ByteRange,
        value: RuntimeValue,
    },

    NegativeArgumentToFunction {
        range: ast::ByteRange,
        name: String,
        found_range: ast::ByteRange,
        value: i32,
    },

    InvalidPrimitiveArguments(Box<PrimitiveArgumentsError>),

    InvalidPrimitiveValue(Box<PrimitiveValueError>),

    ShapeMismatch(Box<ShapeMismatchError>),

    NonAdditiveSum(Box<NonAdditiveSumError>),

    InvalidRepeatExpression {
        range: ast::ByteRange,
        value: RuntimeValue,
    },

    MathError {
        range: ast::ByteRange,
        message: String,
    },
}

impl RuntimeError {
    pub fn range(&self) -> ast::ByteRange {
        match self {
            RuntimeError::InFunction { range, .. } => *range,
            RuntimeError::Semantic { range, .. } => *range,
            RuntimeError::LabelsOnNonTupleOutput { range, .. } => *range,
            RuntimeError::OutputLabelCountMismatch { range, .. } => *range,
            RuntimeError::OutputNotAtTopLevel { range } => *range,
            RuntimeError::SetNotAtTopLevel { range } => *range,
            RuntimeError::ReturnOutsideFunction { range } => *range,
            RuntimeError::LoopOverNonSequence { range, .. } => *range,
            RuntimeError::UndefinedReference { range, .. } => *range,
            RuntimeError::UndefinedFunction { range, .. } => *range,
            RuntimeError::InvalidCondition { range, .. } => *range,
            RuntimeError::RangeHasNonSequenceEndpoints { range, .. } => *range,
            RuntimeError::InvalidArgumentToOperator { operator_range, .. } => *operator_range,
            RuntimeError::NegativeArgumentToFunction { range, .. } => *range,
            RuntimeError::InvalidPrimitiveArguments(error) => error.range,
            RuntimeError::InvalidPrimitiveValue(error) => error.range,
            RuntimeError::ShapeMismatch(error) => error.range,
            RuntimeError::NonAdditiveSum(error) => error.range,
            RuntimeError::InvalidRepeatExpression { range, .. } => *range,
            RuntimeError::MathError { range, .. } => *range,
        }
    }
}
