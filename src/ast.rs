use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub start: usize,
    pub end: usize,
}

pub fn merge_ranges(range: &[Range]) -> Range {
    let start = range.iter().map(|r| r.start).min().unwrap();
    let end = range.iter().map(|r| r.end).max().unwrap();
    Range { start, end }
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
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDefinition {
    pub name: WithRange<String>,
    pub args: Vec<WithRange<(String, StaticType)>>,
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
    pub items: Vec<(ListLiteralItem, i32)>,
}

#[derive(Debug, Clone, Serialize)]
pub enum ListLiteralItem {
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

#[derive(Debug, Clone, Serialize)]
pub enum StaticType {
    Int,
    Sequence,
    Distribution,
}

impl std::fmt::Display for StaticType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaticType::Int => write!(f, "int"),
            StaticType::Sequence => write!(f, "sequence"),
            StaticType::Distribution => write!(f, "distribution"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FunctionDefinitionItem {
    Word(String),
    ArgWithType(String, StaticType),
}

pub fn make_function_definition(
    items: Vec<WithRange<FunctionDefinitionItem>>,
    body: Vec<WithRange<Statement>>,
) -> FunctionDefinition {
    let range_start = items.first().map(|i| i.range.start).expect("empty function definition");
    let range_end = items.last().map(|i| i.range.end).expect("empty function definition");
    let mut name = Vec::new();
    let mut args = Vec::new();
    for item in items.into_iter() {
        match item.value {
            FunctionDefinitionItem::Word(word) => {
                name.push(word);
            }
            FunctionDefinitionItem::ArgWithType(arg, ty) => {
                name.push("{}".to_string());
                args.push(WithRange::new(item.range.start, item.range.end, (arg, ty)));
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

pub fn make_function_call(
    items: Vec<WithRange<FunctionCallItem>>,
) -> Expression {
    let range_start = items.first().map(|i| i.range.start).expect("empty function call");
    let range_end = items.last().map(|i| i.range.end).expect("empty function call");
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

/// Prints the code in sexpr
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
