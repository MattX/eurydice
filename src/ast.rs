use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct WithRange<T> {
    pub value: T,
    #[serde(skip)] 
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

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
pub enum DistributionSpec {
    Constant(i32),
    Dice { repeat: i32, sides: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UnaryOp {
    Negate,
    Sum,
    Product,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, Serialize)]
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

#[derive(Debug, Clone, Copy, Serialize)]
pub enum IntBinaryOp {
    Highest,
    Lowest,
}

// Prints the code in sexpr
pub fn print_expression<T>(expression: &T) -> String
where
    T: serde::ser::Serialize,
{
    let val = serde_lexpr::to_value(expression).unwrap();
    let doc = to_doc(&val);
    doc.pretty(80).to_string()
}

fn to_doc(value: &lexpr::Value) -> pretty::RcDoc {
    match value {
        lexpr::Value::Cons(cons) => {
            let mut doc = pretty::RcDoc::text("(");
            let mut inner_doc = pretty::RcDoc::<()>::nil();
            for (i, value) in cons.iter().enumerate() {
                if i > 0 {
                    inner_doc = inner_doc.append(pretty::RcDoc::line());
                }
                inner_doc = inner_doc.append(to_doc(value.car()));
                match value.cdr() {
                    lexpr::Value::Cons(_) | lexpr::Value::Nil | lexpr::Value::Null => (),
                    _ => {
                        // This can only happen during the loop's last iteration
                        inner_doc = inner_doc
                            .append(pretty::RcDoc::line())
                            .append(pretty::RcDoc::text("."))
                            .append(pretty::RcDoc::line())
                            .append(to_doc(value.cdr()));
                    }
                }
            }
            inner_doc = inner_doc.nest(1).group();
            doc = doc.append(inner_doc).append(pretty::RcDoc::text(")"));
            doc
        }
        lexpr::Value::Vector(_) => panic!("should not have vectors in code"),
        _ => pretty::RcDoc::text(lexpr::to_string(value).unwrap()),
    }
}
