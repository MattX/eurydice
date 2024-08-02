use std::collections::HashMap;

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::ast::{self, DistributionSpec, Expression, WithRange};

pub fn infer(env: &TypeEnv, expr: &WithRange<Expression>) -> Result<StaticType, TypeError> {
    match &expr.value {
        Expression::Distribution(ds) => Ok(distribution_spec_type(ds)),
        Expression::Tuple(_) => todo!(),
        Expression::List(exprs) => infer_list(env, exprs),
        Expression::UnaryOp { op, operand } => infer_application(env, &op.into(), &[*operand.clone()]),
        Expression::BinaryOp { op, left, right } => {
            infer_application(env, &op.into(), &[*left.clone(), *right.clone()])
        }
        Expression::FunctionCall { name, args } => {
            infer_application(env, &Identifier::Function(name.clone()), args)
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
        None => return Ok(StaticType::List),
    };
    let list_type = match &first {
        StaticType::Int | StaticType::List => StaticType::List,
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
            (StaticType::List, StaticType::Int) | (StaticType::List, StaticType::List) => {
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

fn infer_application(
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

    match func_type.find_overload(&arg_types) {
        Ok(t) => Ok(t),
        Err(e) => Err(TypeError::ApplicationMismatch {
            range: args[0].range.into(),
            reasons: vec![e],
        }),
    }
}

fn distribution_spec_type(ds: &DistributionSpec) -> StaticType {
    match ds {
        DistributionSpec::Constant(_) => StaticType::Int,
        DistributionSpec::Dice { .. } => StaticType::Dist,
    }
}

fn function(args: Vec<StaticType>, ret: StaticType) -> StaticType {
    StaticType::Function(Function {
        overloads: vec![FunctionSignature {
            args,
            ret: Box::new(ret),
        }],
        has_reducer: false,
    })
}

fn int_list_to_int() -> StaticType {
    function(vec![StaticType::List], StaticType::Int)
}

fn int_pair_to_int() -> StaticType {
    function(vec![StaticType::Int, StaticType::Int], StaticType::Int)
}

fn numeric_pair() -> StaticType {
    StaticType::Function(Function {
        overloads: vec![
            FunctionSignature {
                args: vec![StaticType::Int, StaticType::Int],
                ret: Box::new(StaticType::Int),
            },
            FunctionSignature {
                args: vec![StaticType::List, StaticType::List],
                ret: Box::new(StaticType::List),
            },
            FunctionSignature {
                args: vec![StaticType::Dist, StaticType::Dist],
                ret: Box::new(StaticType::Dist),
            },
        ],
        has_reducer: false,
    })
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum StaticType {
    // An integer
    Int,
    // A list of integers
    List,
    // A list of distributions
    Dist,
    // A function
    Function(Function),
}

impl StaticType {
    fn is_subtype(&self, other: &StaticType) -> Option<NotSubtypeReason> {
        match (self, other) {
            (StaticType::Int, StaticType::Int) => None,
            (StaticType::Int, StaticType::List) => None,
            (StaticType::Int, StaticType::Dist) => None,
            (StaticType::List, StaticType::List) => None,
            (StaticType::List, StaticType::Dist) => None,
            (StaticType::Dist, StaticType::Dist) => None,
            (StaticType::Function(f1), StaticType::Function(f2)) => {
                if f2.has_reducer && !f1.has_reducer {
                    return Some(NotSubtypeReason::FunctionDoesNotHaveReducer {
                        subtype: f1.clone(),
                        supertype: f2.clone(),
                    });
                }
                // For each overload of f2, at least one overload of f1 must match.
                for (i, f2_overload) in f2.overloads.iter().enumerate() {
                    let mut reasons = Vec::new();
                    let mut matched = false;
                    for f1_overload in f1.overloads.iter() {
                        match f1_overload.is_subtype(f2_overload) {
                            None => {
                                matched = true;
                                break;
                            }
                            Some(reason) => reasons.push(reason),
                        }
                    }
                    if !matched {
                        return Some(NotSubtypeReason::FunctionOverloadNotMatched {
                            subtype: f1.clone(),
                            supertype: f2.clone(),
                            overload_index: i,
                            reasons,
                        });
                    }
                }
                None
            }
            _ => Some(NotSubtypeReason::Incompatible {
                subtype: self.clone(),
                supertype: other.clone(),
            }),
        }
    }
}

#[derive(Debug, Error, Diagnostic)]
#[error("Not a subtype")]
pub enum NotSubtypeReason {
    #[error("Types {subtype} and {supertype} are incompatible.")]
    Incompatible {
        subtype: StaticType,
        supertype: StaticType,
    },
    #[error("Function type {subtype} does not have a reducer, but {supertype} does.")]
    FunctionDoesNotHaveReducer {
        subtype: Function,
        supertype: Function,
    },
    #[error("The {overload_index}th overload of {supertype} is not matched by any overloads of {subtype}.")]
    FunctionOverloadNotMatched {
        subtype: Function,
        supertype: Function,
        overload_index: usize,
        #[related("Reasons:")]
        reasons: Vec<SignatureMismatchReason>,
    }
}

impl std::fmt::Display for StaticType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaticType::Int => write!(f, "int"),
            StaticType::List => write!(f, "list"),
            StaticType::Dist => write!(f, "dist"),
            StaticType::Function(func) => write!(f, "{}", func),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FunctionSignature {
    pub args: Vec<StaticType>,
    pub ret: Box<StaticType>,
}

impl std::fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {}",
            self.args
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            self.ret
        )
    }
}

impl FunctionSignature {
    fn is_subtype(&self, other: &FunctionSignature) -> Option<SignatureMismatchReason> {
        if self.args.len() != other.args.len() {
            return Some(SignatureMismatchReason::ArgumentCount);
        }
        for (i, (self_arg_type, super_arg_type)) in self.args.iter().zip(other.args.iter()).enumerate() {
            if let Some(reason) = super_arg_type.is_subtype(self_arg_type) {
                return Some(SignatureMismatchReason::ArgumentNotSupertype {
                    index: i,
                    expected: self_arg_type.clone(),
                    found: super_arg_type.clone(),
                    reason: Box::new(reason),
                });
            }
        }
        if let Some(reason) = self.ret.is_subtype(&*other.ret) {
            return Some(SignatureMismatchReason::ReturnNotSubType {
                expected: *self.ret.clone(),
                found: *other.ret.clone(),
                reason: Box::new(reason),
            });
        }
        None
    }
}

#[derive(Debug, Error, Diagnostic)]
#[error("Signatures do not match")]
pub enum SignatureMismatchReason {
    ArgumentCount,
    ArgumentNotSupertype {
        index: usize,
        expected: StaticType,
        found: StaticType,
        reason: Box<NotSubtypeReason>,
    },
    ReturnNotSubType {
        expected: StaticType,
        found: StaticType,
        reason: Box<NotSubtypeReason>,
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Function {
    pub overloads: Vec<FunctionSignature>,
    pub has_reducer: bool,
}

impl Function {
    fn find_overload(&self, args: &[StaticType]) -> Result<StaticType, ApplicationMismatchReason> {
        // Stores the indices of matching overloads as well as the number of non-exact matches.
        let mut matching_overloads = Vec::<(usize, usize)>::new();
        let mut mismatch_reasons = Vec::new();
        for (i, sig) in self.overloads.iter().enumerate() {
            match Function::non_exact_matches(sig, args) {
                Ok(upcast_count) => matching_overloads.push((i, upcast_count)),
                Err(reason) => mismatch_reasons.push(reason),
            }
        }
        if matching_overloads.is_empty() {
            return Err(ApplicationMismatchReason::NoOverloadMatched {
                func: self.clone(),
                args: args.to_vec(),
                reasons: mismatch_reasons,
            });
        }
        let min_count = matching_overloads.iter().map(|(_, upcast_count)| *upcast_count).min().unwrap();
        let matching_overloads = matching_overloads
            .into_iter()
            .filter(|(_, upcast_count)| *upcast_count == min_count)
            .collect::<Vec<_>>();
        if matching_overloads.len() > 1 {
            return Err(ApplicationMismatchReason::SeveralOverloadsMatched {
                func: self.clone(),
                args: args.to_vec(),
                overloads: matching_overloads.iter().map(|(i, _)| *i).collect(),
                upcasted_args: min_count,
            });
        }
        Ok(*self.overloads[matching_overloads[0].0].ret.clone())
    }

    /// Otherwise, returns the number of arguments that need to be upcasted, or an explanation if the signature
    /// does not match.
    fn non_exact_matches(sig: &FunctionSignature, args: &[StaticType]) -> Result<usize, SignatureMismatchReason> {
        if sig.args.len() != args.len() {
            return Err(SignatureMismatchReason::ArgumentCount);
        }
        let mut upcast_count = 0;
        for (sig_arg, arg) in sig.args.iter().zip(args.iter()) {
            match arg.is_subtype(sig_arg) {
                None => {
                    if arg != sig_arg {
                        upcast_count += 1;
                    }
                }
                Some(reason) => {
                    return Err(SignatureMismatchReason::ArgumentNotSupertype {
                        index: upcast_count,
                        expected: sig_arg.clone(),
                        found: arg.clone(),
                        reason: Box::new(reason),
                    });
                }
            }
        }
        Ok(upcast_count)
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}function[{}]",
            if self.has_reducer { "@reducer " } else { "" },
            self.overloads
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        )
    }
}

#[derive(Debug, Error, Diagnostic)]
#[error("No candidate found")]
pub enum ApplicationMismatchReason {
    // TODO: provide fmt for the args
    #[error("No overload of {func} matched {args:?}")]
    NoOverloadMatched {
        func: Function,
        args: Vec<StaticType>,
        reasons: Vec<SignatureMismatchReason>,
    },
    #[error("Several overloads of {func} had the same number of upcasts ({upcasted_args}) for args {args:?}")]
    SeveralOverloadsMatched {
        func: Function,
        args: Vec<StaticType>,
        overloads: Vec<usize>,
        upcasted_args: usize,
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
        ast::BinaryOp::Eq,
        ast::BinaryOp::Ge,
        ast::BinaryOp::Gt,
        ast::BinaryOp::Le,
        ast::BinaryOp::Lt,
        ast::BinaryOp::Ne,
    ] {
        vars.insert(op.clone().into(), int_pair_to_int());
    }

    for op in &[ast::BinaryOp::Add, ast::BinaryOp::Sub, ast::BinaryOp::Mul, ast::BinaryOp::Div] {
        vars.insert(op.clone().into(), numeric_pair());
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

    #[error("No suitable overload for function")]
    ApplicationMismatch {
        #[label("No suitable overload for function")]
        range: SourceSpan,

        #[related]
        reasons: Vec<ApplicationMismatchReason>,
    },
}
