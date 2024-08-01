use std::collections::HashMap;

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::ast::{self, DistributionSpec, Expression, WithRange};

pub fn infer(env: &TypeEnv, expr: &WithRange<Expression>) -> Result<StaticType, TypeError> {
    match &expr.value {
        Expression::Distribution(ds) => Ok(distribution_spec_type(ds)),
        Expression::Tuple(exprs) => {
            let types = exprs
                .iter()
                .map(|e| infer(env, &e))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(StaticType::Tuple(types))
        }
        Expression::List(exprs) => infer_list(env, exprs),
        Expression::UnaryOp { op, operand } => infer_function(env, &op.into(), &[*operand.clone()]),
        Expression::BinaryOp { op, left, right } => {
            infer_function(env, &op.into(), &[*left.clone(), *right.clone()])
        }
        Expression::FunctionCall { name, args } => {
            infer_function(env, &Identifier::Function(name.clone()), args)
        }
    }
}

fn infer_list(env: &TypeEnv, exprs: &[WithRange<Expression>]) -> Result<StaticType, TypeError> {
    let types = exprs
        .iter()
        .map(|e| infer(env, e))
        .collect::<Result<Vec<_>, _>>()?;
    let first = match types.first() {
        Some(t) => t,
        None => return Ok(StaticType::IntList),
    };
    let list_type = match &first {
        StaticType::Int | StaticType::IntList => StaticType::IntList,
        StaticType::Dist => StaticType::Dist,
        _ => {
            return Err(TypeError::InvalidTypeInList {
                range: exprs[0].range.into(),
                found: first.clone(),
            })
        }
    };
    for element_type in types.iter() {
        match (&list_type, element_type) {
            (StaticType::IntList, StaticType::Int) | (StaticType::IntList, StaticType::IntList) => {
            }
            (StaticType::Dist, StaticType::Dist) => {}
            _ => {
                return Err(TypeError::NonHomogeneousList {
                    first_range: exprs[0].range.into(),
                    first_type: first.clone(),
                    found_range: exprs[0].range.into(),
                    found: element_type.clone(),
                })
            }
        }
    }
    Ok(list_type)
}

fn infer_function(
    env: &TypeEnv,
    ident: &Identifier,
    args: &[WithRange<Expression>],
) -> Result<StaticType, TypeError> {
    let arg_types = args
        .iter()
        .map(|e| infer(env, e))
        .collect::<Result<Vec<_>, _>>()?;
    let func_type = match env.vars.get(ident) {
        Some(StaticType::Function(t)) => t,
        Some(_) => {
            return Err(TypeError::ApplyingNonFunction {
                range: args[0].range.into(),
                found: env.vars[ident].clone(),
            })
        }
        None => {
            return Err(TypeError::Undefined {
                range: args[0].range.into(),
                name: ident
                    .function_name()
                    .unwrap_or_else(|| format!("missing type for operator {:?}", ident)),
            })
        }
    };
    if func_type.args.len() != arg_types.len() {
        return Err(TypeError::WrongArgumentCount {
            range: args[0].range.into(),
            expected: func_type.args.len(),
            found: arg_types.len(),
        });
    }
    let non_matching = func_type
        .args
        .iter()
        .zip(arg_types.iter())
        .filter(|(a, b)| a != b)
        .next();
    let (expected, found) = match non_matching {
        Some((expected, found)) => (expected, found),
        None => return Ok(*func_type.ret.clone()),
    };

    // Special rules (ugh):
    // * Functions that are IntList -> Int or IntList -> IntList can also be Dist -> Dist (autocross)
    // * Functions that are (Int, Int, ...) -> Int or (Int, Int, ...) -> IntList can also be
    // (Dist, Dist, ...) -> Dist (autovector)
    if func_type.is_autocross() && matches!(arg_types[0], StaticType::Dist) {
        return Ok(StaticType::Dist);
    }
    if func_type.is_autovector() && arg_types.iter().all(|t| matches!(t, StaticType::Dist)) {
        return Ok(StaticType::Dist);
    }
    Err(TypeError::MismatchedTypes {
        range: args[0].range.into(),
        expected: expected.clone(),
        found: found.clone(),
        func_type: func_type.clone(),
    })
}

fn distribution_spec_type(ds: &DistributionSpec) -> StaticType {
    match ds {
        DistributionSpec::Constant(_) => StaticType::Int,
        DistributionSpec::Dice { .. } => StaticType::Dist,
    }
}

fn function(args: Vec<StaticType>, ret: StaticType) -> StaticType {
    StaticType::Function(Function {
        args,
        ret: Box::new(ret),
    })
}

fn int_list_to_int() -> StaticType {
    function(vec![StaticType::IntList], StaticType::Int)
}

fn int_pair_to_int() -> StaticType {
    function(vec![StaticType::Int, StaticType::Int], StaticType::Int)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum StaticType {
    // An integer
    Int,
    // A list of integers
    IntList,
    // A list of distributions
    Dist,
    // A tuple of arbitrary types
    Tuple(Vec<StaticType>),
    // A function
    Function(Function),
}

impl std::fmt::Display for StaticType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaticType::Int => write!(f, "int"),
            StaticType::IntList => write!(f, "list"),
            StaticType::Dist => write!(f, "dist"),
            StaticType::Tuple(types) => write!(
                f,
                "[{}]",
                types
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            StaticType::Function(func) => write!(f, "{}", func),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Function {
    pub args: Vec<StaticType>,
    pub ret: Box<StaticType>,
}

impl Function {
    fn is_autocross(&self) -> bool {
        match (&self.args[..], &*self.ret) {
            ([StaticType::IntList], StaticType::Int) => true,
            ([StaticType::IntList], StaticType::IntList) => true,
            _ => false,
        }
    }

    fn is_autovector(&self) -> bool {
        if !matches!(&*self.ret, StaticType::Int) && !matches!(&*self.ret, StaticType::IntList) {
            return false;
        }
        for arg in &self.args {
            if !matches!(arg, StaticType::Int) {
                return false;
            }
        }
        true
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "function[{}][{}]",
            self.args
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            self.ret
        )
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Identifier {
    BinaryOp(ast::BinaryOp),
    UnaryOp(ast::UnaryOp),
    Function(String),
}

impl Identifier {
    fn function_name(&self) -> Option<String> {
        match self {
            Identifier::Function(name) => Some(name.clone()),
            _ => None,
        }
    }
}

impl From<&ast::BinaryOp> for Identifier {
    fn from(op: &ast::BinaryOp) -> Self {
        Identifier::BinaryOp(*op)
    }
}

impl From<ast::BinaryOp> for Identifier {
    fn from(op: ast::BinaryOp) -> Self {
        Identifier::BinaryOp(op)
    }
}

impl From<&ast::UnaryOp> for Identifier {
    fn from(op: &ast::UnaryOp) -> Self {
        Identifier::UnaryOp(*op)
    }
}

impl From<ast::UnaryOp> for Identifier {
    fn from(op: ast::UnaryOp) -> Self {
        Identifier::UnaryOp(op)
    }
}

impl From<String> for Identifier {
    fn from(name: String) -> Self {
        Identifier::Function(name)
    }
}

// TODO use a representation where we can point at parents instead of copying
#[derive(Debug, Clone)]
pub struct TypeEnv {
    pub vars: HashMap<Identifier, StaticType>,
}

pub fn global_env() -> TypeEnv {
    let mut vars = HashMap::new();

    // Unops
    for op in &[ast::UnaryOp::Negate, ast::UnaryOp::Invert] {
        vars.insert(
            op.clone().into(),
            function(vec![StaticType::Dist], StaticType::Dist),
        );
    }

    // Binops
    for op in &[
        ast::BinaryOp::Add,
        ast::BinaryOp::Sub,
        ast::BinaryOp::Mul,
        ast::BinaryOp::Div,
        ast::BinaryOp::Eq,
        ast::BinaryOp::Ge,
        ast::BinaryOp::Gt,
        ast::BinaryOp::Le,
        ast::BinaryOp::Lt,
        ast::BinaryOp::Ne,
    ] {
        vars.insert(op.clone().into(), int_pair_to_int());
    }

    // Built-in functions
    for name in &["sum", "product", "min", "max"] {
        vars.insert(name.to_string().into(), int_list_to_int());
    }

    TypeEnv { vars }
}

#[derive(Debug, Error, Diagnostic)]
#[error("Type checking error")]
#[diagnostic()]
pub enum TypeError {
    #[error("Reference to undefined name: `{name}`")]
    Undefined {
        #[label("Name `{name}` is not defined")]
        range: SourceSpan,
        name: String,
    },

    #[error("Applying non-function")]
    ApplyingNonFunction {
        #[label("This identifier refers to a value of type `{found}`")]
        range: SourceSpan,
        found: StaticType,
    },

    #[error("Mismatched types")]
    #[diagnostic(help("Function has type {func_type}"))]
    MismatchedTypes {
        #[label("Expected type `{expected}`, found type `{found}`")]
        range: SourceSpan,
        expected: StaticType,
        found: StaticType,
        func_type: Function,
    },

    #[error("Non-homogeneous list")]
    NonHomogeneousList {
        #[label("The first element has type `{first_type}`...")]
        first_range: SourceSpan,
        first_type: StaticType,

        #[label("...but this element has type `{found}`")]
        found_range: SourceSpan,
        found: StaticType,
    },

    #[error("Invalid type in list")]
    #[diagnostic(help("List elements must be `int`s, `list`s, or `dist`s."))]
    InvalidTypeInList {
        #[label("This element has type `{found}`")]
        range: SourceSpan,
        found: StaticType,
    },

    #[error("Wrong argument count")]
    WrongArgumentCount {
        #[label("Expected {expected} arguments, found {found}")]
        range: SourceSpan,
        expected: usize,
        found: usize,
    },
}
