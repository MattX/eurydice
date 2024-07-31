#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct WithRange<T> {
    pub value: T,
    pub range: Range,
}

impl<T> WithRange<T> {
    pub fn new(start: usize, end: usize, value: T) -> Self {
        Self {
            value,
            range: Range { start, end },
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expression {
    Distribution(DistributionSpec),
    Tuple(Vec<WithRange<Expression>>),
    UnaryOp {
        op: UnaryOp,
        operand: Box<WithRange<Expression>>,
    },
    BinaryOp {
        op: BinaryOp,
        left: Box<WithRange<Expression>>,
        right: Box<WithRange<Expression>>,
    },
    IntBinaryOp {
        op: IntBinaryOp,
        left: Box<WithRange<Expression>>,
        right: usize,
    },
}

#[derive(Debug, Clone)]
pub enum DistributionSpec {
    Constant(i32),
    Dice { repeat: i32, sides: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Sum,
    Product,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy)]
pub enum IntBinaryOp {
    Highest,
    Lowest,
}
