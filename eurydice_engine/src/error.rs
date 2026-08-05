//! Errors raised during evaluation.
//!
//! These carry the facts a diagnostic needs — spans, offending values, and the
//! reason a rule was broken — but no user-facing prose. Wording lives in
//! [`crate::diagnostic`], so there is exactly one place where an error's
//! presentation can change.

use miette::SourceSpan;

use crate::{
    ast::{self, BinaryOp},
    diagnostic::{EvaluationFrame, SourceId},
    value::{ElementType, ElementValue, RuntimeValue},
};

#[derive(Debug)]
pub struct PrimitiveArgumentError {
    pub name: &'static str,
    pub range: SourceSpan,
    pub expected: String,
    pub value: RuntimeValue,
}

/// Why a [`RuntimeError::Semantic`] was raised.
///
/// This determines the diagnostic code, so each raise site states it outright
/// rather than leaving it to be inferred from the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticErrorKind {
    /// An operator does not accept the types it was given.
    OperatorOperands,
    /// Values whose outcome types cannot be combined were used together.
    OutcomeMismatch,
    /// A value that cannot be summed was used where a sum is required.
    NonAdditiveValue,
    /// A value falls outside the range the operation allows.
    OutOfRange,
    /// A name is already taken, or cannot be bound in this position.
    BindingConflict,
    /// A statement appeared somewhere it is not allowed.
    TopLevelOnly,
}

#[derive(Debug)]
pub struct PrimitiveArgumentsError {
    pub range: SourceSpan,
    pub function: &'static str,
    pub requirement: String,
    pub help: Option<String>,
    pub arguments: Vec<PrimitiveArgumentError>,
}

#[derive(Debug)]
pub struct PrimitiveValueError {
    pub range: SourceSpan,
    pub function: &'static str,
    pub requirement: String,
    pub argument: &'static str,
    pub found_range: SourceSpan,
    pub value: RuntimeValue,
    pub constraint: String,
    pub help: Option<String>,
}

/// Two outcome types that cannot share a collection.
///
/// Outcomes may differ in kind — a number here, a symbol there — but not in
/// shape, so this is always a scalar meeting a tuple, two tuples of different
/// sizes, or the empty sum meeting something it cannot be added to.
#[derive(Debug)]
pub struct OutcomeMismatchError {
    pub range: SourceSpan,
    pub context: OutcomeMismatchContext,
    /// The outcome type the collection had already settled on.
    pub first: OutcomeConflict,
    /// The one that could not join it.
    pub second: OutcomeConflict,
}

/// One side of an [`OutcomeMismatchError`].
#[derive(Debug)]
pub struct OutcomeConflict {
    pub outcome_type: ElementType,
    /// The expression this outcome type came from, where the raise site can
    /// name one. Results collected from a pool evaluation cannot.
    pub range: Option<SourceSpan>,
}

#[derive(Debug, Clone, Copy)]
pub enum OutcomeMismatchContext {
    /// A `{...}` literal.
    SequenceLiteral,
    /// The results of one function evaluated once per multiset.
    FunctionResults,
}

/// What had to be summed, and how much of it.
#[derive(Debug, Clone, Copy)]
pub enum NonAdditiveSubject {
    /// A pool of this many dice.
    Pool(u32),
    /// A sequence of this many values, such as one multiset of a pool.
    Sequence(usize),
}

/// A value that had to be summed, but whose parts cannot be added together.
#[derive(Debug)]
pub struct NonAdditiveSumError {
    pub range: SourceSpan,
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

impl From<ast::Range> for SourceSpan {
    fn from(range: ast::Range) -> Self {
        SourceSpan::new(range.start.into(), range.end - range.start)
    }
}

#[derive(Debug)]
pub enum RuntimeError {
    /// An error raised inside a function body, wrapped with the call that
    /// reached it. Nested calls nest these.
    InFunction {
        range: SourceSpan,
        source: Box<RuntimeError>,
        body_source: SourceId,
        frame: Box<EvaluationFrame>,
    },

    Semantic {
        kind: SemanticErrorKind,
        range: SourceSpan,
        message: String,
    },

    LabelsOnNonTupleOutput {
        range: SourceSpan,
        value_range: SourceSpan,
        value: RuntimeValue,
    },

    OutputLabelCountMismatch {
        range: SourceSpan,
        expected: usize,
        found: usize,
    },

    OutputNotAtTopLevel {
        range: SourceSpan,
    },

    SetNotAtTopLevel {
        range: SourceSpan,
    },

    ReturnOutsideFunction {
        range: SourceSpan,
    },

    LoopOverNonSequence {
        range: SourceSpan,
        value: RuntimeValue,
    },

    UndefinedReference {
        range: SourceSpan,
        name: String,
    },

    UndefinedFunction {
        range: SourceSpan,
        name: String,
        arity_mismatch: Option<ArityMismatch>,
    },

    InvalidCondition {
        range: SourceSpan,
        value: RuntimeValue,
    },

    RangeHasNonSequenceEndpoints {
        range: SourceSpan,
        value: RuntimeValue,
    },

    InvalidArgumentToOperator {
        operator_range: SourceSpan,
        op: BinaryOp,
        expected: &'static str,
        found_range: SourceSpan,
        value: RuntimeValue,
    },

    NegativeArgumentToFunction {
        range: SourceSpan,
        name: String,
        found_range: SourceSpan,
        value: i32,
    },

    InvalidPrimitiveArguments(Box<PrimitiveArgumentsError>),

    InvalidPrimitiveValue(Box<PrimitiveValueError>),

    OutcomeMismatch(Box<OutcomeMismatchError>),

    NonAdditiveSum(Box<NonAdditiveSumError>),

    InvalidRepeatExpression {
        range: SourceSpan,
        value: RuntimeValue,
    },

    MathError {
        range: SourceSpan,
        message: String,
    },
}

impl RuntimeError {
    pub fn range(&self) -> ast::Range {
        match self {
            RuntimeError::InFunction { range, .. } => range.into(),
            RuntimeError::Semantic { range, .. } => range.into(),
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
            RuntimeError::OutcomeMismatch(error) => (&error.range).into(),
            RuntimeError::NonAdditiveSum(error) => (&error.range).into(),
            RuntimeError::InvalidRepeatExpression { range, .. } => range.into(),
            RuntimeError::MathError { range, .. } => range.into(),
        }
    }
}
