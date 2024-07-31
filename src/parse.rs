use serde::{Deserialize, Serialize};

pub fn parse(input: &str) -> Result<Expression, serde_lexpr::Error> {
    serde_lexpr::from_str(input)
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
