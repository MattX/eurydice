//! A simple tree walking interpreter.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{self, BinaryOp, Expression, FunctionDefinition, Statement, UnaryOp, WithRange},
    probability::{self, Distribution},
};

#[derive(Debug, Clone)]
pub enum RuntimeValue {
    Int(i32),
    List(Rc<Vec<i32>>),
    Distribution(Rc<Vec<Distribution>>),
    Primitive(fn(&[RuntimeValue]) -> Result<RuntimeValue, PrimitiveError>),
    // TODO ideally this would instead be a reference into the AST, but what can we do
    Function {
        definition: Rc<FunctionDefinition>,
        env: RcValEnv,
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

impl From<Rc<Vec<Distribution>>> for RuntimeValue {
    fn from(value: Rc<Vec<Distribution>>) -> Self {
        RuntimeValue::Distribution(value)
    }
}

impl From<Vec<Distribution>> for RuntimeValue {
    fn from(value: Vec<Distribution>) -> Self {
        RuntimeValue::Distribution(Rc::new(value))
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
        debug_assert!(result.is_none(), "value returned from a top-level statement");
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
                let env = Rc::new(RefCell::new(ValEnv::with_parent(Rc::clone(&eval_context.env))));
                let value = RuntimeValue::Function {
                    // TODO make the parser parse the FD directly into an Rc
                    definition: Rc::new(fd.clone()),
                    env,
                };
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
                if let RuntimeValue::Int(0) = condition {
                    for statement in else_block.iter().flatten() {
                        let res = self.execute_statement(eval_context, statement)?;
                        if res.is_some() {
                            return Ok(res);
                        }
                    }
                } else {
                    for statement in then_block {
                        let res = self.execute_statement(eval_context, statement)?;
                        if res.is_some() {
                            return Ok(res);
                        }
                    }
                }
            }
            ast::Statement::Loop {
                variable,
                range_expression,
                body,
            } => {
                let range = self.evaluate(eval_context, range_expression)?;
                if let RuntimeValue::List(range) = range {
                    for value in range.iter() {
                        eval_context
                            .env
                            .borrow_mut()
                            .insert(variable.value.clone(), RuntimeValue::Int(value.clone()));
                        for statement in body {
                            self.execute_statement(eval_context, statement)?;
                        }
                    }
                } else {
                    return Err(RuntimeError::EmptyInput {
                        range: range_expression.range.into(),
                    });
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
                self.apply_unary_op(op, &value, expression.range)
            }
            Expression::BinaryOp { op, left, right } => {
                let left_value = self.evaluate(eval_context, left)?;
                let right_value = self.evaluate(eval_context, right)?;
                self.apply_binary_op(op, &left_value, &right_value, expression.range)
            }
            Expression::List(list) => {
                todo!()
            },
            Expression::FunctionCall { name, args } => todo!(),
            Expression::Reference(name) => {
                eval_context.env.borrow().get(name).ok_or_else(|| {
                    RuntimeError::UndefinedReference {
                        range: expression.range.into(),
                        name: name.clone(),
                    }
                })
            }
            Expression::Int(i) => Ok(RuntimeValue::Int(*i)),
        }
    }

    fn apply_unary_op(
        &self,
        op: &WithRange<UnaryOp>,
        operand: &RuntimeValue,
        range: ast::Range,
    ) -> Result<RuntimeValue, RuntimeError> {
        todo!()
    }

    fn apply_binary_op(
        &self,
        op: &WithRange<BinaryOp>,
        left: &RuntimeValue,
        right: &RuntimeValue,
        range: ast::Range,
    ) -> Result<RuntimeValue, RuntimeError> {
        todo!()
    }

    fn make_list(&self, values: Vec<RuntimeValue>) -> Result<RuntimeValue, RuntimeError> {
        todo!()
    }
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

    #[error("Probability error")]
    ProbabilityError {
        #[label = "Probability error: {underlying}"]
        range: SourceSpan,

        underlying: probability::ProbabilityError,
    },

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
        #[label = "Loops must iterate over sequences"]
        range: SourceSpan,
    },

    #[error("Reference to undefined to variable {name}")]
    UndefinedReference {
        #[label = "Variable not defined"]
        range: SourceSpan,
        name: String,
    },
}

#[derive(Debug, Error, Diagnostic)]
pub enum PrimitiveError {}
