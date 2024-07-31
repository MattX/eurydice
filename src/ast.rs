use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    pub start: i32,
    pub end: i32,
}

pub struct WithRange<T> {
    pub value: T,
    pub range: Range,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    Distribution(DistributionSpec),
    Tuple(Vec<Expression>),
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expression>,
    },
    BinaryOp {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    IntBinaryOp {
        op: IntBinaryOp,
        left: Box<Expression>,
        right: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionSpec {
    Constant(i32),
    Dice { repeat: i32, sides: i32 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Sum,
    Product,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IntBinaryOp {
    Highest,
    Lowest,
}
