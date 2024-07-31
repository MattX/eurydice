use std::vec;

use miette::Diagnostic;
use thiserror::Error;

use crate::{
    parse::{BinaryOp, DistributionSpec, Expression, UnaryOp},
    probability::Distribution,
};

pub fn evaluate(expression: &Expression) -> Result<Vec<Distribution>, RuntimeError> {
    match expression {
        Expression::Distribution(d) => Ok(spec_to_distribution(d)),
        Expression::Tuple(t) => {
            let mut distributions = Vec::new();
            for expr in t {
                let dists = evaluate(expr)?;
                distributions.extend(dists);
            }
            Ok(distributions)
        }
        Expression::UnaryOp { op, operand } => {
            let distributions = evaluate(operand)?;
            if distributions.is_empty() {
                return Err(RuntimeError::EmptyInput);
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
                Err(e) => Err(e.into()),
            }
        }
        Expression::BinaryOp { op, left, right } => {
            let left_dists = evaluate(left)?;
            let right_dists = evaluate(right)?;
            if left_dists.is_empty() {
                return Err(RuntimeError::EmptyInput);
            }
            if left_dists.len() != right_dists.len() {
                return Err(RuntimeError::MismatchedBinaryOpArgs {
                    op: *op,
                    left_size: left_dists.len(),
                    right_size: right_dists.len(),
                });
            }
            left_dists
                .into_iter()
                .zip(right_dists.into_iter())
                .map(|(l, r)| apply_binary_op(*op, &l, &r))
                .collect()
        }
        Expression::IntBinaryOp { op, left, right } => todo!(),
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

fn apply_binary_op(
    op: BinaryOp,
    left: &Distribution,
    right: &Distribution,
) -> Result<Distribution, RuntimeError> {
    let two_vec = vec![left.clone(), right.clone()];
    match op {
        BinaryOp::Add => Distribution::sum(two_vec),
        BinaryOp::Subtract => Distribution::reduce_pairwise(two_vec, |a, b| a - b),
        BinaryOp::Multiply => Distribution::product(two_vec),
        // TODO handle errors here gracefully
        BinaryOp::Divide => Distribution::reduce_pairwise(two_vec, |a, b| a / b),
        BinaryOp::Eq => Distribution::reduce_pairwise(two_vec, |a, b| (a == b) as i32),
        BinaryOp::Ne => Distribution::reduce_pairwise(two_vec, |a, b| (a != b) as i32),
        BinaryOp::Lt => Distribution::reduce_pairwise(two_vec, |a, b| (a < b) as i32),
        BinaryOp::Le => Distribution::reduce_pairwise(two_vec, |a, b| (a <= b) as i32),
        BinaryOp::Gt => Distribution::reduce_pairwise(two_vec, |a, b| (a > b) as i32),
        BinaryOp::Ge => Distribution::reduce_pairwise(two_vec, |a, b| (a >= b) as i32),
    }.map_err(|e| e.into())
}

#[derive(Debug, Error, Diagnostic)]
pub enum RuntimeError {
    #[error("Empty input")]
    EmptyInput,

    #[error("Mismatched binary op arguments")]
    MismatchedBinaryOpArgs {
        op: BinaryOp,
        left_size: usize,
        right_size: usize,
    },

    #[error("Probability error")]
    ProbabilityError(#[from] crate::probability::ProbabilityError),
}
