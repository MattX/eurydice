use miette::Diagnostic;
use serde::Serialize;
use thiserror::Error;

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
pub enum Statement {
    Assignment {
        name: WithRange<String>,
        value: WithRange<Expression>,
    },
    FunctionDefinition(FunctionDefinition),
    Output {
        value: WithRange<Expression>,
        named: Option<WithRange<String>>,
    },
    Return {
        value: WithRange<Expression>,
    },
    If {
        condition: Box<WithRange<Expression>>,
        then_block: Vec<WithRange<Statement>>,
        else_block: Option<Vec<WithRange<Statement>>>,
    },
    Loop {
        variable: WithRange<String>,
        range_expression: Box<WithRange<Expression>>,
        body: Vec<WithRange<Statement>>,
    },
    Set(SetParam),
    Print {
        expr: WithRange<Expression>,
        named: Option<WithRange<String>>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDefinition {
    pub name: WithRange<String>,
    pub args: Vec<WithRange<ArgWithType>>,
    pub body: Vec<WithRange<Statement>>,
}

#[derive(Debug, Clone, Serialize)]
pub enum Expression {
    Int(i32),
    List(ListLiteral),
    UnaryOp {
        op: WithRange<UnaryOp>,
        operand: Box<WithRange<Expression>>,
    },
    BinaryOp {
        op: WithRange<BinaryOp>,
        left: Box<WithRange<Expression>>,
        right: Box<WithRange<Expression>>,
    },
    FunctionCall {
        name: WithRange<String>,
        args: Vec<WithRange<Expression>>,
    },
    Reference(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct ListLiteral {
    /// (Item, number of repetitions)
    pub items: Vec<ListItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListItem {
    pub item: BareListItem,
    pub repeat: Option<WithRange<Expression>>,
}

#[derive(Debug, Clone, Serialize)]
pub enum BareListItem {
    Expr(WithRange<Expression>),
    Range(WithRange<Expression>, WithRange<Expression>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum UnaryOp {
    Negate,
    Invert,
    Length,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum BinaryOp {
    Pow,
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
    D,
    At,
    Or,
    And,
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Negate => write!(f, "-"),
            UnaryOp::Invert => write!(f, "!"),
            UnaryOp::Length => write!(f, "#"),
            UnaryOp::D => write!(f, "d"),
        }
    }
}

impl std::fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::Ne => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Le => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Ge => write!(f, ">="),
            BinaryOp::D => write!(f, "d"),
            BinaryOp::At => write!(f, "@"),
            BinaryOp::Or => write!(f, "|"),
            BinaryOp::And => write!(f, "&"),
            BinaryOp::Pow => write!(f, "^"),
        }
    }
}

pub fn apply_string_escapes(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some(c) => {
                    result.push('\\');
                    result.push(c);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum StaticType {
    Int,
    List,
    Pool,
}

impl std::fmt::Display for StaticType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaticType::Int => write!(f, "int"),
            StaticType::List => write!(f, "sequence"),
            StaticType::Pool => write!(f, "die"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum SetParam {
    PositionOrder(PositionOrder),
    MaximumFunctionDepth(usize),
    ExplodeDepth(usize),
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum PositionOrder {
    Ascending,
    Descending,
}

#[derive(Debug, Clone)]
pub enum FunctionDefinitionItem {
    Word(String),
    ArgWithType(ArgWithType),
}

#[derive(Debug, Clone, Serialize)]
pub struct ArgWithType {
    pub name: String,
    pub ty: Option<StaticType>,
}

pub fn make_function_definition(
    items: Vec<WithRange<FunctionDefinitionItem>>,
    body: Vec<WithRange<Statement>>,
) -> FunctionDefinition {
    let range_start = items
        .first()
        .map(|i| i.range.start)
        .expect("empty function definition");
    let range_end = items
        .last()
        .map(|i| i.range.end)
        .expect("empty function definition");
    let mut name = Vec::new();
    let mut args = Vec::new();
    for item in items.into_iter() {
        match item.value {
            FunctionDefinitionItem::Word(word) => {
                name.push(word);
            }
            FunctionDefinitionItem::ArgWithType(a) => {
                name.push("{}".to_string());
                args.push(WithRange::new(item.range.start, item.range.end, a));
            }
        }
    }
    FunctionDefinition {
        name: WithRange::new(range_start, range_end, name.join(" ")),
        args,
        body,
    }
}

#[derive(Debug, Clone)]
pub enum FunctionCallItem {
    Word(String),
    Expr(Expression),
}

pub fn make_function_call(items: Vec<WithRange<FunctionCallItem>>) -> Expression {
    let range_start = items
        .first()
        .map(|i| i.range.start)
        .expect("empty function call");
    let range_end = items
        .last()
        .map(|i| i.range.end)
        .expect("empty function call");
    let mut name = Vec::new();
    let mut args = Vec::new();
    for item in items.into_iter() {
        match item.value {
            FunctionCallItem::Word(word) => {
                name.push(word);
            }
            FunctionCallItem::Expr(expr) => {
                name.push("{}".to_string());
                args.push(WithRange::new(item.range.start, item.range.end, expr));
            }
        }
    }
    Expression::FunctionCall {
        name: WithRange::new(range_start, range_end, name.join(" ")),
        args,
    }
}

/// This corresponds to inner errors in the grammar - when the LR parser succeeded,
/// but custom action code failed.
#[derive(Debug, Diagnostic, Error)]
pub enum ParseActionError {
    #[error("Invalid integer literal")]
    #[diagnostic(help("Integer literals must be in the range -2^31 to 2^31-1"))]
    InvalidIntegerLiteral,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar;

    #[test]
    fn test_apply_string_escapes() {
        assert_eq!(apply_string_escapes("hello"), "hello");
        assert_eq!(apply_string_escapes("hello\\nworld"), "hello\nworld");
        assert_eq!(apply_string_escapes("hello\\tworld"), "hello\tworld");
        assert_eq!(apply_string_escapes("hello\\rworld"), "hello\rworld");
        assert_eq!(apply_string_escapes("hello\\\\world"), "hello\\world");
        assert_eq!(apply_string_escapes("hello\\\"world"), "hello\"world");
        assert_eq!(apply_string_escapes("hello\\xworld"), "hello\\xworld");
    }

    #[test]
    fn test_parse_function_call() {
        let text = "[test 1 2]";
        let ast = grammar::ExprParser::new().parse(text).unwrap();
        assert_eq!(serde_lexpr::to_string(&ast).unwrap(), "((value FunctionCall (name (value . \"test {} {}\")) (args ((value Int . 1)) ((value Int . 2)))))");
    }

    #[test]
    fn test_parse_unop_function_call() {
        let text = "[test 1 - 2]";
        let ast = grammar::ExprParser::new().parse(text).unwrap();
        assert_eq!(serde_lexpr::to_string(&ast).unwrap(), "((value FunctionCall (name (value . \"test {}\")) (args ((value BinaryOp (op (value . Sub)) (left (value Int . 1)) (right (value Int . 2)))))))");
    }

    #[test]
    fn test_parse_function_definition() {
        let text = "function: explode DIE:d { result: DIE }";
        let ast = grammar::FunctionDefinitionParser::new()
            .parse(text)
            .unwrap();
        assert_eq!(serde_lexpr::to_string(&ast).unwrap(), "((name (value . \"explode {}\")) (args ((value (name . \"DIE\") (ty Pool)))) (body ((value Return (value (value Reference . \"DIE\"))))))");
    }

    #[test]
    fn test_parse_function_definition_no_type() {
        let text = "function: explode DIE { result: DIE }";
        let ast = grammar::FunctionDefinitionParser::new()
            .parse(text)
            .unwrap();
        assert_eq!(serde_lexpr::to_string(&ast).unwrap(), "((name (value . \"explode {}\")) (args ((value (name . \"DIE\") (ty)))) (body ((value Return (value (value Reference . \"DIE\"))))))");
    }

    #[test]
    fn test_parse_binary_op_precedence() {
        let text = "1 + 2 * 3";
        let ast = grammar::ExprParser::new().parse(text).unwrap();
        assert_eq!(serde_lexpr::to_string(&ast).unwrap(), "((value BinaryOp (op (value . Add)) (left (value Int . 1)) (right (value BinaryOp (op (value . Mul)) (left (value Int . 2)) (right (value Int . 3))))))");
    }
}
