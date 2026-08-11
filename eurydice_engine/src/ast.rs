use lalrpop_util::ParseError;
#[cfg(feature = "serde")]
use serde::Serialize;

/// A range of bytes in the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct ByteRange {
    /// The starting byte index (inclusive) of the range.
    pub start: usize,
    /// The ending byte index (exclusive) of the range.
    pub end: usize,
}

impl From<(usize, usize)> for ByteRange {
    fn from(range: (usize, usize)) -> Self {
        Self {
            start: range.0,
            end: range.1,
        }
    }
}

#[derive(Clone)]
pub struct WithRange<T> {
    pub value: T,
    pub range: ByteRange,
}

impl<T> WithRange<T> {
    pub fn new(start: usize, end: usize, value: T) -> Self {
        Self {
            value,
            range: ByteRange { start, end },
        }
    }
}

/// Ranges are noise when reading a tree: they are derivable from the source and
/// they bury the structure that is actually being inspected. Debugging a
/// `WithRange` shows the value alone, which is also what makes the AST tests
/// below readable.
impl<T: std::fmt::Debug> std::fmt::Debug for WithRange<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub enum Statement {
    Assignment {
        name: WithRange<String>,
        value: WithRange<Expression>,
    },
    FunctionDefinition(FunctionDefinition),
    SymbolDefinitions(Vec<WithRange<String>>),
    Output {
        expr: WithRange<Expression>,
        named: Option<WithRange<String>>,
        labeled: Option<WithRange<Vec<WithRange<String>>>>,
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

/// Optional presentation metadata parsed after an `output` expression.
///
/// This is kept separate from `Statement::Output` because the grammar accepts
/// `named` and `labeled` in either order before assembling the final statement.
#[derive(Debug, Clone, Default)]
pub struct OutputOptions {
    pub named: Option<WithRange<String>>,
    pub labeled: Option<WithRange<Vec<WithRange<String>>>>,
}

#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    pub name: WithRange<String>,
    pub args: Vec<WithRange<ArgWithType>>,
    pub body: Vec<WithRange<Statement>>,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct ListLiteral {
    /// (Item, number of repetitions)
    pub items: Vec<ListItem>,
}

#[derive(Debug, Clone)]
pub struct ListItem {
    pub item: BareListItem,
    pub repeat: Option<WithRange<Expression>>,
}

#[derive(Debug, Clone)]
pub enum BareListItem {
    Expr(WithRange<Expression>),
    Range(WithRange<Expression>, WithRange<Expression>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Negate,
    Invert,
    Length,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy)]
pub enum SetParam {
    PositionOrder(PositionOrder),
    MaximumFunctionDepth(usize),
    ExplodeDepth(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionOrder {
    Ascending,
    Descending,
}

#[derive(Debug, Clone)]
pub enum FunctionDefinitionItem {
    Word(String),
    ArgWithType(ArgWithType),
}

#[derive(Debug, Clone)]
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
        .expect("grammar does not parse empty function definitions");
    let range_end = items
        .last()
        .map(|i| i.range.end)
        .expect("grammar does not parse empty function definitions");
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

pub fn make_function_call<L: std::fmt::Debug, T: std::fmt::Debug>(
    range: ByteRange,
    items: Vec<WithRange<FunctionCallItem>>,
) -> Result<Expression, ParseError<L, T, ParseActionError>> {
    let Some((first, remaining)) = items.split_first() else {
        return Err(ParseError::<L, T, ParseActionError>::User {
            error: ParseActionError::EmptyFunctionCall { range },
        });
    };
    let range_start = first.range.start;
    let range_end = remaining.last().unwrap_or(first).range.end;
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
    Ok(Expression::FunctionCall {
        name: WithRange::new(range_start, range_end, name.join(" ")),
        args,
    })
}

/// This corresponds to inner errors in the grammar - when the LR parser
/// succeeded, but custom action code failed.
#[derive(Debug)]
pub enum ParseActionError {
    InvalidIntegerLiteral { range: ByteRange, error: String },

    EmptyFunctionCall { range: ByteRange },
}

/// Only so that a `ParseError` wrapping one of these can be printed.
/// User-facing wording for these lives in [`crate::diagnostic`], like every
/// other error's.
impl std::fmt::Display for ParseActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseActionError::InvalidIntegerLiteral { error, .. } => {
                write!(f, "invalid integer literal: {error}")
            }
            ParseActionError::EmptyFunctionCall { .. } => write!(f, "empty function call"),
        }
    }
}

impl std::error::Error for ParseActionError {}

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

    /// The parsed shape of an expression, with byte ranges elided by
    /// `WithRange`'s `Debug`.
    fn expr(text: &str) -> String {
        format!("{:?}", grammar::ExprParser::new().parse(text).unwrap())
    }

    /// As [`expr`], for a function definition.
    fn definition(text: &str) -> String {
        format!(
            "{:?}",
            grammar::FunctionDefinitionParser::new()
                .parse(text)
                .unwrap()
        )
    }

    #[test]
    fn test_parse_function_call() {
        assert_eq!(
            expr("[test 1 2]"),
            r#"FunctionCall { name: "test {} {}", args: [Int(1), Int(2)] }"#
        );
    }

    #[test]
    fn test_parse_unop_function_call() {
        assert_eq!(
            expr("[test 1 - 2]"),
            r#"FunctionCall { name: "test {}", args: [BinaryOp { op: Sub, left: Int(1), right: Int(2) }] }"#
        );
    }

    #[test]
    fn test_parse_function_definition() {
        assert_eq!(
            definition("function: explode DIE:d { result: DIE }"),
            r#"FunctionDefinition { name: "explode {}", args: [ArgWithType { name: "DIE", ty: Some(Pool) }], body: [Return { value: Reference("DIE") }] }"#
        );
    }

    #[test]
    fn test_parse_function_definition_no_type() {
        assert_eq!(
            definition("function: explode DIE { result: DIE }"),
            r#"FunctionDefinition { name: "explode {}", args: [ArgWithType { name: "DIE", ty: None }], body: [Return { value: Reference("DIE") }] }"#
        );
    }

    #[test]
    fn test_parse_single_letter_function_name() {
        assert_eq!(
            definition("function: f { result: 1 }"),
            r#"FunctionDefinition { name: "f", args: [], body: [Return { value: Int(1) }] }"#
        );
        assert_eq!(expr("[f]"), r#"FunctionCall { name: "f", args: [] }"#);
    }

    #[test]
    fn test_parse_binary_op_precedence() {
        assert_eq!(
            expr("1 + 2 * 3"),
            r#"BinaryOp { op: Add, left: Int(1), right: BinaryOp { op: Mul, left: Int(2), right: Int(3) } }"#
        );
    }

    #[test]
    fn test_parse_empty_function_call() {
        let text = "[]";
        let ast = grammar::ExprParser::new().parse(text);
        assert!(ast.is_err());
    }
}
