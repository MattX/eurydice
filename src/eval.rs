//! A simple tree walking interpreter.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{self, BinaryOp, Expression, FunctionDefinition, ListLiteralItem, UnaryOp, WithRange},
    dice::Pool,
};

#[derive(Debug, Clone)]
pub enum RuntimeValue {
    Int(i32),
    List(Rc<Vec<i32>>),
    Pool(Rc<Pool>),
    Primitive(fn(&[RuntimeValue]) -> Result<RuntimeValue, PrimitiveError>),
    // TODO ideally this would instead be a reference into the AST, but what can we do
    Function(Box<Function>),
}

#[derive(Debug, Clone)]
struct Function {
    definition: Rc<FunctionDefinition>,
    env: RcValEnv,
}

impl RuntimeValue {
    fn runtime_type(&self) -> RuntimeType {
        match self {
            RuntimeValue::Int(_) => RuntimeType::Int,
            RuntimeValue::List(_) => RuntimeType::Sequence,
            RuntimeValue::Pool(_) => RuntimeType::Distribution,
            RuntimeValue::Primitive(_) => RuntimeType::Function,
            RuntimeValue::Function { .. } => RuntimeType::Function,
        }
    }

    fn to_list(&self, repeat: usize) -> Vec<i32> {
        match self {
            RuntimeValue::Int(i) => vec![*i; repeat],
            RuntimeValue::List(list) => Rc::clone(list).repeat(repeat),
            RuntimeValue::Pool(d) => (**d)
                .sum()
                .into_iter()
                .map(|(k, _)| k)
                .collect::<Vec<_>>()
                .repeat(repeat),
            _ => panic!("to_list called on {}", self.runtime_type()),
        }
    }
}

impl From<i32> for RuntimeValue {
    fn from(value: i32) -> Self {
        RuntimeValue::Int(value)
    }
}

impl From<Rc<Vec<i32>>> for RuntimeValue {
    fn from(value: Rc<Vec<i32>>) -> Self {
        RuntimeValue::List(value)
    }
}

impl From<Vec<i32>> for RuntimeValue {
    fn from(value: Vec<i32>) -> Self {
        RuntimeValue::List(Rc::new(value))
    }
}

impl From<Pool> for RuntimeValue {
    fn from(value: Pool) -> Self {
        RuntimeValue::Pool(Rc::new(value))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RuntimeType {
    Int,
    Sequence,
    Distribution,
    Function,
}

impl std::fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeType::Int => write!(f, "int"),
            RuntimeType::Sequence => write!(f, "sequence"),
            RuntimeType::Distribution => write!(f, "distribution"),
            RuntimeType::Function => write!(f, "function"),
        }
    }
}

/// Represents the environment at some point during program execution.
///
/// Lambdas capture their arguments, so we need RCs to keep the environments alive;
/// we also need RefCells because the bindings are mutable.
#[derive(Debug, Clone)]
struct ValEnv {
    parent: Option<RcValEnv>,
    env: HashMap<String, RuntimeValue>,
}
type RcValEnv = Rc<RefCell<ValEnv>>;

impl ValEnv {
    fn new() -> Self {
        Self {
            parent: None,
            env: HashMap::new(),
        }
    }

    fn with_parent(parent: RcValEnv) -> Self {
        Self {
            parent: Some(parent),
            env: HashMap::new(),
        }
    }

    fn get(&self, key: &str) -> Option<RuntimeValue> {
        if let Some(value) = self.env.get(key) {
            Some(value.clone())
        } else if let Some(parent) = &self.parent {
            parent.borrow().get(key)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, value: RuntimeValue) {
        self.env.insert(key, value);
    }
}

#[derive(Clone)]
struct EvalContext {
    env: RcValEnv,
    recursion_depth: usize,
    at_top_level: bool,
}

impl EvalContext {
    fn new(env: RcValEnv) -> Self {
        Self {
            env,
            recursion_depth: 0,
            at_top_level: true,
        }
    }
}

pub struct Evaluator {
    global_env: RcValEnv,
    outputs: Vec<RuntimeValue>,
}

impl Evaluator {
    pub fn new() -> Self {
        Self {
            global_env: Rc::new(RefCell::new(ValEnv::new())),
            outputs: Vec::new(),
        }
    }

    pub fn execute(&mut self, statement: &WithRange<ast::Statement>) -> Result<(), RuntimeError> {
        let eval_context = EvalContext::new(Rc::clone(&self.global_env));
        let result = self.execute_statement(&eval_context, statement)?;
        debug_assert!(
            result.is_none(),
            "value returned from a top-level statement"
        );
        Ok(())
    }

    pub fn get_outputs(&mut self) -> Vec<RuntimeValue> {
        std::mem::take(&mut self.outputs)
    }

    /// Executes a statement; returns a value if a `return` statement is encountered.
    fn execute_statement(
        &mut self,
        eval_context: &EvalContext,
        statement: &WithRange<ast::Statement>,
    ) -> Result<Option<RuntimeValue>, RuntimeError> {
        match &statement.value {
            ast::Statement::Assignment { name, value } => {
                let value = self.evaluate(eval_context, value)?;
                eval_context
                    .env
                    .borrow_mut()
                    .insert(name.value.clone(), value);
            }
            ast::Statement::FunctionDefinition(fd) => {
                let env = Rc::new(RefCell::new(ValEnv::with_parent(Rc::clone(
                    &eval_context.env,
                ))));
                let value = RuntimeValue::Function(Box::new(Function {
                    // TODO make the parser parse the FD directly into an Rc
                    definition: Rc::new(fd.clone()),
                    env,
                }));
                eval_context
                    .env
                    .borrow_mut()
                    .insert(fd.name.value.clone(), value);
            }
            ast::Statement::Output { value, named } => {
                if !eval_context.at_top_level {
                    return Err(RuntimeError::OutputNotAtTopLevel {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, value)?;
                self.outputs.push(value);
            }
            ast::Statement::Return { value } => {
                if eval_context.at_top_level {
                    return Err(RuntimeError::ReturnOutsideFunction {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, value)?;
                return Ok(Some(value));
            }
            ast::Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition = self.evaluate(eval_context, condition)?;
                let cond_value = match condition {
                    RuntimeValue::Int(i) => i,
                    _ => {
                        return Err(RuntimeError::InvalidCondition {
                            range: statement.range.into(),
                            found: condition.runtime_type(),
                        })
                    }
                };
                let block = if cond_value != 0 {
                    then_block
                } else if let Some(block) = else_block {
                    block
                } else {
                    return Ok(None);
                };
                for statement in block {
                    let res = self.execute_statement(eval_context, statement)?;
                    if res.is_some() {
                        return Ok(res);
                    }
                }
            }
            ast::Statement::Loop {
                variable,
                range_expression,
                body,
            } => {
                let range = self.evaluate(eval_context, range_expression)?;
                let range = match range {
                    RuntimeValue::List(range) => range,
                    _ => {
                        return Err(RuntimeError::LoopOverNonSequence {
                            range: range_expression.range.into(),
                            found: range.runtime_type(),
                        })
                    }
                };
                for value in range.iter() {
                    eval_context
                        .env
                        .borrow_mut()
                        .insert(variable.value.clone(), RuntimeValue::Int(*value));
                    for statement in body {
                        let res = self.execute_statement(eval_context, statement)?;
                        if res.is_some() {
                            return Ok(res);
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    fn evaluate(
        &mut self,
        eval_context: &EvalContext,
        expression: &WithRange<Expression>,
    ) -> Result<RuntimeValue, RuntimeError> {
        match &expression.value {
            Expression::UnaryOp { op, operand } => {
                let value = self.evaluate(eval_context, operand)?;
                apply_unary_op(op, &value, expression.range)
            }
            Expression::BinaryOp { op, left, right } => {
                let left_value = self.evaluate(eval_context, left)?;
                let right_value = self.evaluate(eval_context, right)?;
                apply_binary_op(op, &left_value, &right_value, expression.range)
            }
            Expression::List(list) => {
                let elems = list
                    .items
                    .iter()
                    .map(|item| self.evaluate_list_literal_item(eval_context, item))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<i32>>();
                Ok(elems.into())
            }
            Expression::FunctionCall { name, args } => todo!(),
            Expression::Reference(name) => eval_context.env.borrow().get(name).ok_or_else(|| {
                RuntimeError::UndefinedReference {
                    range: expression.range.into(),
                    name: name.clone(),
                }
            }),
            Expression::Int(i) => Ok(RuntimeValue::Int(*i)),
        }
    }

    /// Evaluates a list literal item.
    ///
    /// This doesn't take precomputed values, because literals can contain special non-expression syntax.
    fn evaluate_list_literal_item(
        &mut self,
        eval_context: &EvalContext,
        item: &(ListLiteralItem, usize),
    ) -> Result<Vec<i32>, RuntimeError> {
        let (item, repeats) = item;
        let base = match item {
            ListLiteralItem::Expr(expression) => self.evaluate(eval_context, expression),
            ListLiteralItem::Range(start_expr, end_expr) => {
                let start = self.evaluate(eval_context, start_expr)?;
                let start = match start {
                    RuntimeValue::Int(i) => i,
                    _ => {
                        return Err(RuntimeError::RangeHasNonSequenceEndpoints {
                            range: start_expr.range.into(),
                            found: start.runtime_type(),
                        })
                    }
                };
                let end = self.evaluate(eval_context, end_expr)?;
                let end = match end {
                    RuntimeValue::Int(i) => i,
                    _ => {
                        return Err(RuntimeError::RangeHasNonSequenceEndpoints {
                            range: end_expr.range.into(),
                            found: end.runtime_type(),
                        })
                    }
                };
                Ok(RuntimeValue::List(Rc::new((start..=end).collect())))
            }
        }?;
        Ok(base.to_list(*repeats))
    }
}

fn apply_unary_op(
    op: &WithRange<UnaryOp>,
    operand: &RuntimeValue,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    match &op.value {
        UnaryOp::D => make_d(None, operand, range),
        _ => todo!(),
    }
}

fn apply_binary_op(
    op: &WithRange<BinaryOp>,
    left: &RuntimeValue,
    right: &RuntimeValue,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    match &op.value {
        BinaryOp::D => make_d(Some(left), right, range),
        _ => todo!(),
    }
}

enum DLeftSide {
    Int(i32),
    Pool(Rc<Pool>),
}

enum DRightSide {
    List(Vec<i32>),
    Pool(Rc<Pool>),
}

/// Executes the unary or binary version of the _n_ `d` _m_ operator.
fn make_d(
    left: Option<&RuntimeValue>,
    right: &RuntimeValue,
    range: ast::Range,
) -> Result<RuntimeValue, RuntimeError> {
    let repeat = match left {
        Some(RuntimeValue::Int(i)) => DLeftSide::Int(*i),
        Some(RuntimeValue::List(list)) => DLeftSide::Int(list.iter().sum()),
        Some(RuntimeValue::Pool(d)) => DLeftSide::Pool(Rc::clone(d)),
        None => DLeftSide::Int(1),
        _ => panic!("make_d called with invalid left operand"),
    };
    let right = match right {
        RuntimeValue::Int(sides) => DRightSide::List((1..=*sides).collect()),
        RuntimeValue::List(list) => DRightSide::List(Rc::clone(list).to_vec()),
        RuntimeValue::Pool(d) => DRightSide::Pool(Rc::clone(d)),
        _ => panic!("make_d called with invalid right operand"),
    };
    if let (DLeftSide::Int(repeat), DRightSide::List(sides)) = (repeat, right) {
        return Ok(make_pool(repeat, sides).into());
    }
    // Otherwise we turn both sides into distributions, then cross-product-multiply them
    // let right_dist = match right {
    //     DRightSide::List(sides) => Pool::from_list(1, sides).sum().into(),
    //     DRightSide::Pool(d) => d,
    // };
    // let left_dist = match repeat {
    //     DLeftSide::Int(count) => Pool::from_list(1, vec![count]).into(),
    //     DLeftSide::Pool(d) => d.sum(),
    // };
    todo!()
}

fn make_pool(mut n: i32, mut sides: Vec<i32>) -> Pool {
    if n < 0 {
        for side in sides.iter_mut() {
            *side = -*side;
        }
        n = -n;
    }
    Pool::from_list(n as u32, sides)
}

impl From<ast::Range> for SourceSpan {
    fn from(range: ast::Range) -> Self {
        SourceSpan::new(range.start.into(), range.end - range.start)
    }
}

#[derive(Debug, Error, Diagnostic)]
#[error("Runtime error")]
pub enum RuntimeError {
    #[error("Empty input")]
    EmptyInput {
        #[label = "Empty input"]
        range: SourceSpan,
    },

    #[error("Mismatched binary op arguments")]
    MismatchedBinaryOpArgs {
        op: BinaryOp,
        left_size: usize,
        right_size: usize,

        #[label = "This argument has size {left_size}..."]
        left_range: SourceSpan,

        #[label = "...but this argument has size {right_size}"]
        right_range: SourceSpan,
    },

    // #[error("Probability error")]
    // ProbabilityError {
    //     #[label = "Probability error: {underlying}"]
    //     range: SourceSpan,

    //     underlying: probability::ProbabilityError,
    // },
    #[error("Output statement inside a function")]
    #[diagnostic(help("Output statements can only appear outside functions."))]
    OutputNotAtTopLevel {
        #[label = "Output statement inside a function"]
        range: SourceSpan,
    },

    #[error("Return statement outside a function")]
    ReturnOutsideFunction {
        #[label = "Return statement outside a function"]
        range: SourceSpan,
    },

    #[error("Loops must iterate over sequences")]
    LoopOverNonSequence {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: RuntimeType,
    },

    #[error("Reference to undefined to variable {name}")]
    UndefinedReference {
        #[label = "Variable not defined"]
        range: SourceSpan,
        name: String,
    },

    #[error("Conditions to `if` statements must be numbers.")]
    InvalidCondition {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: RuntimeType,
    },

    #[error("Both sides of a range constructor must evaluate to numbers.")]
    RangeHasNonSequenceEndpoints {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: RuntimeType,
    },

    #[error("Not yet implemented")]
    #[diagnostic(help("The developer is a lazy bum."))]
    NotYetImplemented {
        #[label = "Not yet implemented"]
        range: SourceSpan,
    },
}

#[derive(Debug, Error, Diagnostic)]
pub enum PrimitiveError {}
