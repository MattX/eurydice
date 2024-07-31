
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{self, BinaryOp, DistributionSpec, Expression, UnaryOp, WithRange},
    probability::{self, Distribution},
};

pub struct Evaluator<'source> {
    source: &'source str,
}

impl<'source> Evaluator<'source> {
    pub fn new(source: &'source str) -> Self {
        Self { source }
    }

    pub fn evaluate(
        &self,
        expression: &WithRange<Expression>,
    ) -> Result<Vec<Distribution>, RuntimeError> {
        match &expression.value {
            Expression::Distribution(d) => Ok(self.spec_to_distribution(d)),
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
                    return Err(self.empty_input(expression.range));
                }
                if op == &UnaryOp::Negate {
                    return Ok(distributions.into_iter().map(|d| d.negate()).collect());
                }
                let result = match op {
                    UnaryOp::Sum => Distribution::sum(distributions),
                    UnaryOp::Product => Distribution::product(distributions),
                    UnaryOp::Max => Distribution::max(distributions),
                    UnaryOp::Min => Distribution::min(distributions),
                    _ => unreachable!(),
                };
                match result {
                    Ok(d) => Ok(vec![d]),
                    Err(e) => Err(self.probability_error(expression.range, e)),
                }
            }
            Expression::BinaryOp { op, left, right } => {
                let left_dists = self.evaluate(left)?;
                let right_dists = self.evaluate(right)?;
                if left_dists.is_empty() {
                    return Err(self.empty_input(expression.range));
                }
                if left_dists.len() != right_dists.len() {
                    return Err(self.mismatched_binary_op_args(
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
            Expression::IntBinaryOp { op, left, right } => todo!(),
        }
    }

    fn spec_to_distribution(&self, spec: &DistributionSpec) -> Vec<Distribution> {
        match spec {
            DistributionSpec::Constant(c) => vec![Distribution::uniform(*c, *c)],
            DistributionSpec::Dice { repeat, sides } => {
                vec![Distribution::uniform(1, *sides); *repeat as usize]
            }
        }
    }

    fn apply_binary_op(
        &self,
        op: BinaryOp,
        left: &Distribution,
        right: &Distribution,
        range: ast::Range,
    ) -> Result<Distribution, RuntimeError<'source>> {
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
        .map_err(|e| self.probability_error(range, e))
    }

    fn empty_input(&self, range: crate::ast::Range) -> RuntimeError<'source> {
        RuntimeError::EmptyInput {
            source_code: &self.source,
            range: range.into(),
        }
    }

    fn mismatched_binary_op_args(
        &self,
        op: BinaryOp,
        left_size: usize,
        right_size: usize,
        left_range: ast::Range,
        right_range: ast::Range,
    ) -> RuntimeError<'source> {
        RuntimeError::MismatchedBinaryOpArgs {
            op,
            left_size,
            right_size,
            left_range: left_range.into(),
            right_range: right_range.into(),
            source_code: &self.source,
        }
    }

    fn probability_error(
        &self,
        range: ast::Range,
        underlying: probability::ProbabilityError,
    ) -> RuntimeError<'source> {
        RuntimeError::ProbabilityError {
            source_code: &self.source,
            range: range.into(),
            underlying,
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
pub enum RuntimeError<'source> {
    #[error("Empty input")]
    EmptyInput {
        #[source_code]
        source_code: &'source str,

        #[label = "Empty input"]
        range: SourceSpan,
    },

    #[error("Mismatched binary op arguments")]
    MismatchedBinaryOpArgs {
        op: BinaryOp,
        left_size: usize,
        right_size: usize,

        #[source_code]
        source_code: &'source str,

        #[label = "This argument has size {left_size}..."]
        left_range: SourceSpan,

        #[label = "...but this argument has size {right_size}"]
        right_range: SourceSpan,
    },

    #[error("Probability error")]
    ProbabilityError {
        #[source_code]
        source_code: &'source str,

        #[label = "Probability error: {underlying}"]
        range: SourceSpan,

        underlying: probability::ProbabilityError,
    },
}
