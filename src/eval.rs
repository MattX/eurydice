
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{self, BinaryOp, DistributionSpec, Expression, UnaryOp, WithRange},
    probability::{self, Distribution},
};

pub struct Evaluator {
}

impl Evaluator {
    pub fn evaluate(
        &mut self,
        expression: &WithRange<Expression>,
    ) -> Result<Vec<Distribution>, RuntimeError> {
        match &expression.value {
            Expression::Distribution(d) => Ok(spec_to_distribution(d)),
            Expression::Tuple(t) => {
                let mut distributions = Vec::new();
                for expr in t {
                    let dists = self.evaluate(expr)?;
                    distributions.extend(dists);
                }
                Ok(distributions)
            }
            Expression::UnaryOp { op, operand } => {
                let distributions = self.evaluate(operand)?;
                if distributions.is_empty() {
                    return Err(empty_input(expression.range));
                }
                match op {
                    UnaryOp::Negate => {
                        let mut result = distributions[0].clone();
                        result.negate();
                        Ok(vec![result])
                    }
                    UnaryOp::Invert => {
                        let mut result = distributions[0].clone();
                        result.invert();
                        Ok(vec![result])
                    }
                }
            }
            Expression::BinaryOp { op, left, right } => {
                let left_dists = self.evaluate(left)?;
                let right_dists = self.evaluate(right)?;
                if left_dists.is_empty() {
                    return Err(empty_input(expression.range));
                }
                if left_dists.len() != right_dists.len() {
                    return Err(mismatched_binary_op_args(
                        *op,
                        left_dists.len(),
                        right_dists.len(),
                        left.range,
                        right.range,
                    ));
                }
                left_dists
                    .into_iter()
                    .zip(right_dists.into_iter())
                    .map(|(l, r)| self.apply_binary_op(*op, &l, &r, expression.range))
                    .collect()
            }
            Expression::FunctionCall { name, args } => todo!(),
            Expression::List(_) => todo!(),
        }
    }

    fn apply_binary_op(
        &self,
        op: BinaryOp,
        left: &Distribution,
        right: &Distribution,
        range: ast::Range,
    ) -> Result<Distribution, RuntimeError> {
        let two_vec = vec![left.clone(), right.clone()];
        match op {
            BinaryOp::Add => Distribution::sum(two_vec),
            BinaryOp::Sub => Distribution::reduce_pairwise(two_vec, |a, b| a - b),
            BinaryOp::Mul => Distribution::product(two_vec),
            // TODO handle errors here gracefully
            BinaryOp::Div => Distribution::reduce_pairwise(two_vec, |a, b| a / b),
            BinaryOp::Eq => Distribution::reduce_pairwise(two_vec, |a, b| (a == b) as i32),
            BinaryOp::Ne => Distribution::reduce_pairwise(two_vec, |a, b| (a != b) as i32),
            BinaryOp::Lt => Distribution::reduce_pairwise(two_vec, |a, b| (a < b) as i32),
            BinaryOp::Le => Distribution::reduce_pairwise(two_vec, |a, b| (a <= b) as i32),
            BinaryOp::Gt => Distribution::reduce_pairwise(two_vec, |a, b| (a > b) as i32),
            BinaryOp::Ge => Distribution::reduce_pairwise(two_vec, |a, b| (a >= b) as i32),
        }
        .map_err(|e| probability_error(range, e))
    }
}

fn spec_to_distribution(spec: &DistributionSpec) -> Vec<Distribution> {
    match spec {
        DistributionSpec::Constant(c) => vec![Distribution::uniform(*c, *c)],
        DistributionSpec::Dice { repeat, sides } => {
            vec![Distribution::uniform(1, *sides); *repeat as usize]
        }
    }
}

impl Into<SourceSpan> for ast::Range {
    fn into(self) -> SourceSpan {
        SourceSpan::new(self.start.into(), self.end - self.start)
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

fn empty_input(range: crate::ast::Range) -> RuntimeError {
    RuntimeError::EmptyInput {
        range: range.into(),
    }
}

fn mismatched_binary_op_args(
    op: BinaryOp,
    left_size: usize,
    right_size: usize,
    left_range: ast::Range,
    right_range: ast::Range,
) -> RuntimeError {
    RuntimeError::MismatchedBinaryOpArgs {
        op,
        left_size,
        right_size,
        left_range: left_range.into(),
        right_range: right_range.into(),
    }
}

fn probability_error(
    range: ast::Range,
    underlying: probability::ProbabilityError,
) -> RuntimeError {
    RuntimeError::ProbabilityError {
        range: range.into(),
        underlying,
    }
}
