use std::rc::Rc;

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{self, BinaryOp, Expression, UnaryOp, WithRange},
    probability::{self, Distribution},
};

#[derive(Debug, Clone)]
pub enum RuntimeValue {
    Int(i32),
    List(Rc<Vec<i32>>),
    Distribution(Rc<Vec<Distribution>>),
    Primitive(fn(&[RuntimeValue]) -> Result<RuntimeValue, PrimitiveError>),
    // TODO ideally this would instead be a reference into the AST, but what can we do
    Function(Rc<WithRange<Expression>>),
}

impl From<i32> for RuntimeValue {
    fn from(value: i32) -> Self {
        RuntimeValue::Int(value)
    }
}

impl From<Rc<Vec<i32>>> for RuntimeValue {
    fn from(value: Rc<Vec<i32>>) -> Self {
        RuntimeValue::List(value)
    }
}

impl From<Vec<i32>> for RuntimeValue {
    fn from(value: Vec<i32>) -> Self {
        RuntimeValue::List(Rc::new(value))
    }
}

impl From<Rc<Vec<Distribution>>> for RuntimeValue {
    fn from(value: Rc<Vec<Distribution>>) -> Self {
        RuntimeValue::Distribution(value)
    }
}

impl From<Vec<Distribution>> for RuntimeValue {
    fn from(value: Vec<Distribution>) -> Self {
        RuntimeValue::Distribution(Rc::new(value))
    }
}

type ValEnv<'a> = crate::util::Environment<'a, RuntimeValue>;

pub struct Evaluator {
    global_env: ValEnv<'static>,
}

impl Evaluator {
    pub fn new() -> Self {
        Self {
            global_env: ValEnv::new(),
        }
    }

    pub fn evaluate(
        &mut self,
        expression: &WithRange<Expression>,
    ) -> Result<RuntimeValue, RuntimeError> {
        match &expression.value {
            Expression::Distribution(d) => Ok(spec_to_distribution(d)),
            Expression::UnaryOp { op, operand } => {
                let distributions = self.evaluate(operand)?;
                self.apply_unary_op(op, &distributions, expression.range)
            }
            Expression::BinaryOp { op, left, right } => {
                let left_dists = self.evaluate(left)?;
                let right_dists = self.evaluate(right)?;
                self.apply_binary_op(op, &left_dists, &right_dists, expression.range)
            }
            Expression::FunctionCall { .. } => todo!(),
            Expression::List(_) => todo!(),
            Expression::Reference(_) => todo!(),
            Expression::If { .. } => todo!(),
        }
    }

    fn apply_unary_op(
        &self,
        op: &WithRange<UnaryOp>,
        operand: &RuntimeValue,
        range: ast::Range,
    ) -> Result<RuntimeValue, RuntimeError> {
        todo!()
    }

    fn apply_binary_op(
        &self,
        op: &WithRange<BinaryOp>,
        left: &RuntimeValue,
        right: &RuntimeValue,
        range: ast::Range,
    ) -> Result<RuntimeValue, RuntimeError> {
        todo!()
    }
}

fn spec_to_distribution(spec: &DistributionSpec) -> RuntimeValue {
    match spec {
        DistributionSpec::Constant(c) => RuntimeValue::Int(*c),
        DistributionSpec::Dice { repeat, sides } => {
            vec![Distribution::uniform(1, *sides); *repeat as usize].into()
        }
    }
}

impl From<ast::Range> for SourceSpan {
    fn from(range: ast::Range) -> Self {
        SourceSpan::new(range.start.into(), range.end - range.start)
    }
}

#[derive(Debug, Error, Diagnostic)]
#[error("Runtime error")]
pub enum RuntimeError {
    #[error("Empty input")]
    EmptyInput {
        #[label = "Empty input"]
        range: SourceSpan,
    },

    #[error("Mismatched binary op arguments")]
    MismatchedBinaryOpArgs {
        op: BinaryOp,
        left_size: usize,
        right_size: usize,

        #[label = "This argument has size {left_size}..."]
        left_range: SourceSpan,

        #[label = "...but this argument has size {right_size}"]
        right_range: SourceSpan,
    },

    #[error("Probability error")]
    ProbabilityError {
        #[label = "Probability error: {underlying}"]
        range: SourceSpan,

        underlying: probability::ProbabilityError,
    },
}

#[derive(Debug, Error, Diagnostic)]
pub enum PrimitiveError {
}
