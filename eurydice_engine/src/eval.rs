//! A simple tree walking interpreter.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use malachite::base::num::arithmetic::traits::Lcm;
use malachite::{
    base::num::basic::traits::{One, Zero},
    rational::Rational,
    Natural,
};
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{
        self, BareListItem, BinaryOp, Expression, FunctionDefinition, ListItem, PositionOrder,
        SetParam, Statement, StaticType, UnaryOp, WithRange,
    },
    dice::{MultisetCrossProductIterator, Pool},
    primitives::{register_primitives, Primitive},
};

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    Int(i32),
    List(Rc<Vec<i32>>),
    Pool(Rc<Pool>),
}

impl std::fmt::Display for RuntimeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeValue::Int(i) => write!(f, "{}", i),
            RuntimeValue::List(list) => write!(
                f,
                "{{{}}}",
                list.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            RuntimeValue::Pool(d) => write!(f, "{}", d),
        }
    }
}

impl RuntimeValue {
    fn runtime_type(&self) -> StaticType {
        match self {
            RuntimeValue::Int(_) => StaticType::Int,
            RuntimeValue::List(_) => StaticType::List,
            RuntimeValue::Pool(_) => StaticType::Pool,
        }
    }

    fn to_list(&self, repeat: usize) -> Vec<i32> {
        match self {
            RuntimeValue::Int(i) => vec![*i; repeat],
            RuntimeValue::List(list) => Rc::clone(list).repeat(repeat),
            RuntimeValue::Pool(d) => d
                .sum()
                .into_die_iter()
                .map(|(k, _)| k)
                .collect::<Vec<_>>()
                .repeat(repeat),
        }
    }

    pub fn map_outcomes(&self, f: impl Fn(i32) -> i32) -> Self {
        match self {
            RuntimeValue::Int(i) => f(*i).into(),
            RuntimeValue::List(list) => f(list.iter().sum()).into(),
            RuntimeValue::Pool(d) => (**d).clone().map_outcomes(f).into(),
        }
    }

    fn to_pool(&self) -> Pool {
        match self {
            RuntimeValue::Int(i) => Pool::from_list(1, vec![*i]),
            RuntimeValue::List(list) => Pool::from_list(1, vec![list.iter().sum()]),
            RuntimeValue::Pool(p) => (**p).clone(),
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

/// Represents the environment at some point during program execution.
///
/// Lambdas capture their arguments, so we need RCs to keep the environments alive;
/// we also need RefCells because the bindings are mutable.
#[derive(Debug)]
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

#[derive(Debug, Clone)]
pub enum Function {
    Primitive(&'static Primitive),
    UserDefined(Rc<FunctionDefinition>),
}

impl Function {
    fn get_arg_types(&self) -> Vec<Option<StaticType>> {
        match self {
            Function::Primitive(primitive) => primitive.arg_types.clone(),
            Function::UserDefined(fd) => fd.args.iter().map(|arg| arg.value.ty).collect(),
        }
    }
}

struct EvalContext {
    env: RcValEnv,
    recursion_depth: usize,
}

impl EvalContext {
    fn new(env: RcValEnv) -> Self {
        Self {
            env,
            recursion_depth: 0,
        }
    }
}

pub struct Evaluator {
    global_env: RcValEnv,
    outputs: Vec<(RuntimeValue, String)>,
    functions: HashMap<String, Function>,
    explode_depth: usize,
    recursion_depth: usize,
    lowest_first: bool,
    print_callback: Option<Box<dyn Fn(RuntimeValue, String)>>,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator {
    pub fn new() -> Self {
        let mut functions = HashMap::new();
        register_primitives(&mut functions);
        Self {
            global_env: Rc::new(RefCell::new(ValEnv::new())),
            outputs: Vec::new(),
            functions,
            explode_depth: 2,
            recursion_depth: 10,
            lowest_first: false,
            print_callback: None,
        }
    }

    pub fn set_print_callback(&mut self, callback: Box<dyn Fn(RuntimeValue, String)>) {
        self.print_callback = Some(callback);
    }

    pub fn execute(&mut self, statement: &WithRange<Statement>) -> Result<(), RuntimeError> {
        let eval_context = EvalContext::new(Rc::clone(&self.global_env));
        let result = self.execute_statement(&eval_context, statement)?;
        debug_assert!(
            result.is_none(),
            "value returned from a top-level statement"
        );
        Ok(())
    }

    pub fn take_outputs(&mut self) -> Vec<(RuntimeValue, String)> {
        std::mem::take(&mut self.outputs)
    }

    /// Executes a statement; returns a value if a `return` statement is encountered.
    fn execute_statement(
        &mut self,
        eval_context: &EvalContext,
        statement: &WithRange<Statement>,
    ) -> Result<Option<RuntimeValue>, RuntimeError> {
        match &statement.value {
            Statement::Assignment { name, value } => {
                let value = self.evaluate(eval_context, value)?;
                eval_context
                    .env
                    .borrow_mut()
                    .insert(name.value.clone(), value);
            }
            Statement::FunctionDefinition(fd) => {
                self.functions.insert(
                    fd.name.value.clone(),
                    Function::UserDefined(Rc::new(fd.clone())),
                );
            }
            Statement::Output { expr, named } => {
                if eval_context.recursion_depth != 0 {
                    return Err(RuntimeError::OutputNotAtTopLevel {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, expr)?;
                let name = match named {
                    Some(name) => interpolate_variable_names(name, &eval_context.env)?,
                    None => format!("output {}", self.outputs.len() + 1),
                };
                self.outputs.push((value, name));
            }
            Statement::Print { expr, named } => {
                let value = self.evaluate(eval_context, expr)?;
                let name = match named {
                    Some(name) => interpolate_variable_names(name, &eval_context.env)?,
                    None => "".to_string(),
                };
                if let Some(ref callback) = self.print_callback {
                    callback(value, name);
                }
            }
            Statement::Return { value } => {
                if self.recursion_depth == 0 {
                    return Err(RuntimeError::ReturnOutsideFunction {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, value)?;
                return Ok(Some(value));
            }
            Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition_value = self.evaluate(eval_context, condition)?;
                let cond_value = match condition_value {
                    RuntimeValue::Int(i) => i,
                    _ => {
                        return Err(RuntimeError::InvalidCondition {
                            range: condition.range.into(),
                            found: condition_value.runtime_type(),
                            value: condition_value,
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
            Statement::Loop {
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
            Statement::Set(s) => {
                if eval_context.recursion_depth != 0 {
                    return Err(RuntimeError::SetNotAtTopLevel {
                        range: statement.range.into(),
                    });
                }
                match s {
                    SetParam::PositionOrder(order) => {
                        self.lowest_first = *order == PositionOrder::Ascending
                    }
                    SetParam::ExplodeDepth(d) => self.explode_depth = *d,
                    SetParam::MaximumFunctionDepth(d) => self.recursion_depth = *d,
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
                Ok(apply_unary_op(op.value, &value))
            }
            Expression::BinaryOp { op, left, right } => {
                let left_value = self.evaluate(eval_context, left)?;
                let right_value = self.evaluate(eval_context, right)?;
                Ok(apply_binary_op(
                    op,
                    &left_value,
                    left.range,
                    &right_value,
                    self.lowest_first,
                )?)
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
            Expression::FunctionCall { name, args } => {
                let func = self
                    .functions
                    .get(&name.value)
                    .ok_or_else(|| RuntimeError::UndefinedFunction {
                        range: expression.range.into(),
                        name: name.value.clone(),
                    })?
                    .clone();
                let ranges = args.iter().map(|arg| arg.range).collect::<Vec<_>>();
                let args = args
                    .iter()
                    .map(|arg| self.evaluate(eval_context, arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let args = args
                    .into_iter()
                    .zip(ranges.iter())
                    .map(|(arg, range)| WithRange {
                        value: arg,
                        range: *range,
                    })
                    .collect::<Vec<_>>();
                let func_with_range = WithRange {
                    value: func,
                    range: name.range,
                };
                self.evaluate_function_call(eval_context, &func_with_range, args)
            }
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
        item: &ListItem,
    ) -> Result<Vec<i32>, RuntimeError> {
        let repeat_count = match &item.repeat {
            Some(repeat) => match self.evaluate(eval_context, repeat)? {
                RuntimeValue::Int(i) => {
                    usize::try_from(i.max(0)).expect("converting a positive i32 into usize")
                }
                repeat_value => {
                    return Err(RuntimeError::InvalidRepeatExpression {
                        range: repeat.range.into(),
                        found: repeat_value.runtime_type(),
                        value: repeat_value,
                    })
                }
            },
            None => 1,
        };
        let base = match &item.item {
            BareListItem::Expr(expression) => self.evaluate(eval_context, expression),
            BareListItem::Range(start_expr, end_expr) => {
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
        Ok(base.to_list(repeat_count))
    }

    fn evaluate_function_call(
        &mut self,
        eval_context: &EvalContext,
        function: &WithRange<Function>,
        args: Vec<WithRange<RuntimeValue>>,
    ) -> Result<RuntimeValue, RuntimeError> {
        if eval_context.recursion_depth >= self.recursion_depth {
            return Ok(vec![].into());
        }

        // First, convert the arguments to the expected types.
        let expected_types = function.value.get_arg_types();
        if expected_types.len() != args.len() {
            panic!(
                "wrong number of arguments; this should have been caught by function name matching"
            );
        }
        let arg_ranges = args.iter().map(|arg| arg.range).collect::<Vec<_>>();
        let args = args
            .into_iter()
            .zip(expected_types.iter())
            .map(|(arg, &expected)| coerce_arg(arg.value, expected))
            .collect::<Result<Vec<_>, _>>()?;

        // This vector will contain references to pools which match an int or list arguments. These
        // are the pools over which we need to iterate to get the argument values.
        let mut pools = Vec::new();
        // Contains (index, is_int) for the corresponding pool.
        let mut pool_iterator_info = Vec::new();
        for (i, (arg, expected_type)) in args.iter().zip(expected_types.iter()).enumerate() {
            if let (RuntimeValue::Pool(p), Some(StaticType::Int) | Some(StaticType::List)) =
                (&arg, expected_type)
            {
                // If the expected type is an Int, `coerce_arg` has already turned the pool into a sum,
                // with outcomes of length 1.
                pools.push(p);
                pool_iterator_info.push((i, *expected_type == Some(StaticType::Int)));
            }
        }

        if pools.is_empty() {
            return self.call_function(eval_context, function, &args, &arg_ranges);
        }

        let cross_product_iterator = MultisetCrossProductIterator::new(
            pools.iter().map(|p| p.multiset_iterator()).collect(),
        );

        let mut args = args.clone();
        let mut results = Vec::new();
        for (values, weight) in cross_product_iterator {
            for ((i, is_int), value) in pool_iterator_info.iter().zip(values.iter()) {
                if *is_int {
                    args[*i] = value[0].into();
                } else {
                    args[*i] = reverse_if(!self.lowest_first, value).into();
                }
            }
            results.push((
                self.call_function(eval_context, function, &args, &arg_ranges)?,
                weight,
            ));
        }

        // TODO this is similar to the logic in flat_map in Pool, find a way to use that?
        let mut total_results = BTreeMap::<i32, Rational>::new();
        let mut lcm = Natural::ONE;
        for (result, weight) in results {
            if let RuntimeValue::Pool(ref p) = result {
                // The empty die is ignored in this context, but the empty list is not.
                if p.ordered_outcomes().is_empty() {
                    continue;
                }
            }
            let summed = result.to_pool().sum();
            let total_count = summed
                .ordered_outcomes()
                .iter()
                .map(|(_, count)| count)
                .sum();
            lcm = lcm.lcm(&total_count);
            for (outcome, count) in summed.ordered_outcomes() {
                *total_results.entry(*outcome).or_insert(Rational::ZERO) +=
                    Rational::from_naturals(count * &weight, total_count.clone());
            }
        }
        Ok(total_results
            .into_iter()
            .map(|(outcome, weight)| {
                let (numerator, denominator) =
                    (weight * Rational::from(&lcm)).into_numerator_and_denominator();
                debug_assert_eq!(denominator, Natural::ONE);
                (outcome, numerator)
            })
            .collect::<Pool>()
            .into())
    }

    fn call_function(
        &mut self,
        eval_context: &EvalContext,
        function: &WithRange<Function>,
        args: &[RuntimeValue],
        arg_ranges: &[ast::Range],
    ) -> Result<RuntimeValue, RuntimeError> {
        match &function.value {
            Function::Primitive(primitive) => (primitive.execute)(
                args,
                arg_ranges,
                self.explode_depth,
                self.lowest_first,
                function.range,
            ),
            Function::UserDefined(user_function) => {
                let mut new_env = ValEnv::with_parent(Rc::clone(&eval_context.env));
                for (arg, formal) in args.iter().zip(user_function.args.iter()) {
                    new_env.insert(formal.value.name.clone(), arg.clone());
                }
                let new_context = EvalContext {
                    env: Rc::new(RefCell::new(new_env)),
                    recursion_depth: eval_context.recursion_depth + 1,
                };
                let mut result = None;
                for statement in &user_function.body {
                    result = self.execute_statement(&new_context, statement)?;
                    if result.is_some() {
                        break;
                    }
                }
                // If there's no result, there was no return statement in the function.
                Ok(result.unwrap_or(Pool::from_list(1, vec![]).into()))
            }
        }
    }
}

fn apply_unary_op(op: UnaryOp, operand: &RuntimeValue) -> RuntimeValue {
    match op {
        UnaryOp::D => make_d(None, operand),
        UnaryOp::Negate => operand.map_outcomes(|o| -o),
        UnaryOp::Invert => operand.map_outcomes(|o| if o == 0 { 1 } else { 0 }),
        UnaryOp::Length => match operand {
            RuntimeValue::Int(i) => i32::try_from(i.abs().to_string().len())
                .expect("vector length fits in i32")
                .into(),
            RuntimeValue::List(list) => i32::try_from(list.len())
                .expect("vector length fits in i32")
                .into(),
            RuntimeValue::Pool(d) => i32::try_from(d.dimension())
                .expect("vector length fits in i32")
                .into(),
        },
    }
}

fn apply_binary_op(
    op: &WithRange<BinaryOp>,
    left: &RuntimeValue,
    left_range: ast::Range,
    right: &RuntimeValue,
    lowest_first: bool,
) -> Result<RuntimeValue, RuntimeError> {
    match &op.value {
        BinaryOp::D => Ok(make_d(Some(left), right)),
        BinaryOp::At => {
            let left = match left {
                RuntimeValue::Int(i) => Rc::new(vec![*i]),
                RuntimeValue::List(lst) => Rc::clone(lst),
                RuntimeValue::Pool(_) => {
                    return Err(RuntimeError::InvalidArgumentToOperator {
                        operator_range: op.range.into(),
                        op: op.value,
                        expected: "an int or a list",
                        found_range: left_range.into(),
                        found: left.runtime_type(),
                    })
                }
            };
            match right {
                RuntimeValue::Int(i) => {
                    let digits: Vec<i32> = i
                        .abs()
                        .to_string()
                        .chars()
                        // Unwrap here is ok as all chars for a *positive* integer are valid digits.
                        .map(|c| {
                            i32::try_from(c.to_digit(10).unwrap()).expect("digit fits in i32")
                                * i.signum()
                        })
                        .collect();
                    Ok(select_positions(&left, &digits, lowest_first).into())
                }
                RuntimeValue::List(lst) => Ok(select_positions(&left, lst, false).into()),
                RuntimeValue::Pool(p) => {
                    Ok(select_in_dice(&left, (**p).clone(), lowest_first).into())
                }
            }
        }
        BinaryOp::Pow => math_binary_op(left, right, |a, b| {
            if b < 0 {
                Err(format!("Cannot raise {} to negative power {}", a, b))
            } else {
                a.checked_pow(b.unsigned_abs())
                    .ok_or_else(|| format!("Power overflow: {} ^ {}", a, b))
            }
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Add => math_binary_op(left, right, |a, b| {
            a.checked_add(b)
                .ok_or_else(|| format!("Addition overflow: {} + {}", a, b))
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Sub => math_binary_op(left, right, |a, b| {
            a.checked_sub(b)
                .ok_or_else(|| format!("Subtraction overflow: {} - {}", a, b))
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Mul => math_binary_op(left, right, |a, b| {
            a.checked_mul(b)
                .ok_or_else(|| format!("Multiplication overflow: {} * {}", a, b))
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Div => math_binary_op(left, right, |a, b| {
            if b == 0 {
                Err(format!("Cannot divide {} by zero", a))
            } else {
                Ok(a / b)
            }
        })
        .map_err(|msg| RuntimeError::MathError {
            range: op.range.into(),
            message: msg,
        }),
        BinaryOp::Eq => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a == b { 1 } else { 0 },
            |a, b| if a == b { 1 } else { 0 },
        )),
        BinaryOp::Ne => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a != b { 1 } else { 0 },
            |a, b| if a != b { 1 } else { 0 },
        )),
        BinaryOp::Lt => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a < b { 1 } else { 0 },
            |a, b| if a < b { 1 } else { 0 },
        )),
        BinaryOp::Le => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a <= b { 1 } else { 0 },
            |a, b| if a <= b { 1 } else { 0 },
        )),
        BinaryOp::Gt => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a > b { 1 } else { 0 },
            |a, b| if a > b { 1 } else { 0 },
        )),
        BinaryOp::Ge => Ok(comp_binary_op(
            left,
            right,
            |a, b| if a >= b { 1 } else { 0 },
            |a, b| if a >= b { 1 } else { 0 },
        )),
        BinaryOp::Or => {
            math_binary_op(left, right, |a, b| Ok(if a != 0 || b != 0 { 1 } else { 0 })).map_err(
                |msg| RuntimeError::MathError {
                    range: op.range.into(),
                    message: msg,
                },
            )
        }
        BinaryOp::And => {
            math_binary_op(left, right, |a, b| Ok(if a != 0 && b != 0 { 1 } else { 0 })).map_err(
                |msg| RuntimeError::MathError {
                    range: op.range.into(),
                    message: msg,
                },
            )
        }
    }
}

fn select_positions(indices: &[i32], vec: &[i32], lowest_first: bool) -> i32 {
    indices
        .iter()
        .map(|&i| {
            if i < 1 || i > i32::try_from(vec.len()).expect("vector length fits in i32") {
                return 0;
            }
            let i = usize::try_from(i)
                .expect("i is positive, and a positive i32 should fit in a usize");
            let i = if lowest_first { vec.len() - i } else { i - 1 };
            vec.get(i).copied().unwrap_or(0)
        })
        .sum()
}

fn select_in_dice(indices: &[i32], pool: Pool, lowest_first: bool) -> Pool {
    let dimension = usize::try_from(pool.dimension()).expect("usize is at least 32 bits");
    let mut keep_list = vec![false; dimension];
    for &i in indices {
        if i < 1 || i > i32::try_from(dimension).expect("vector length fits in i32") {
            continue;
        }
        let i =
            usize::try_from(i).expect("i is positive, and a positive i32 should fit in a usize");
        let i = if !lowest_first { dimension - i } else { i - 1 };
        keep_list[i] = true;
    }
    pool.sum_with_keep_list(&keep_list)
}

fn math_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    f: impl Fn(i32, i32) -> Result<i32, String>,
) -> Result<RuntimeValue, String> {
    let left = flatten_list(left);
    let right = flatten_list(right);
    let (left_pool, right_pool) = match (left, right) {
        (DLeftSide::Int(a), DLeftSide::Int(b)) => return Ok(f(a, b)?.into()),
        (left, right) => (left.to_pool().sum(), right.to_pool().sum()),
    };

    // For pool operations, we need to handle errors during mapping
    let mut results = Vec::new();
    for (left_outcome, left_weight) in left_pool.ordered_outcomes() {
        for (right_outcome, right_weight) in right_pool.ordered_outcomes() {
            match f(*left_outcome, *right_outcome) {
                Ok(result) => results.push((result, left_weight * right_weight)),
                Err(err) => return Err(err),
            }
        }
    }

    let mut pool_map = std::collections::BTreeMap::new();
    for (outcome, weight) in results {
        *pool_map.entry(outcome).or_insert(Natural::ZERO) += weight;
    }

    Ok(Pool::from(pool_map.into_iter().collect::<Vec<_>>()).into())
}

/// Binary ops behave differently depending on the types of their arguments.
/// (int, int) is the straightforward case
/// (list, list) is in lexicographic order
/// (list, int) or (int, list) are elementwise then summed (counted)
/// (pool, pool) sums both sides, then applies the binary op elementwise
fn comp_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    int_comp: impl Fn(i32, i32) -> i32,
    list_comp: impl Fn(&[i32], &[i32]) -> i32,
) -> RuntimeValue {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => int_comp(*a, *b).into(),
        (RuntimeValue::List(a), RuntimeValue::List(b)) => list_comp(a, b).into(),
        (RuntimeValue::List(a), RuntimeValue::Int(b)) => {
            a.iter().map(|&a| int_comp(a, *b)).sum::<i32>().into()
        }
        (RuntimeValue::Int(a), RuntimeValue::List(b)) => {
            b.iter().map(|&b| int_comp(*a, b)).sum::<i32>().into()
        }
        _ => {
            // At least one is a pool
            let left_pool = left.to_pool().sum();
            let right_pool = right.to_pool().sum();
            left_pool
                .flat_map(|left_outcome| {
                    right_pool
                        .clone()
                        .map_outcomes(|right_outcome| int_comp(left_outcome[0], right_outcome))
                        .into()
                })
                .into()
        }
    }
}

enum DLeftSide {
    Int(i32),
    Pool(Rc<Pool>),
}

impl DLeftSide {
    fn to_pool(&self) -> Rc<Pool> {
        match self {
            DLeftSide::Int(i) => Rc::new(Pool::from_list(1, vec![*i])),
            DLeftSide::Pool(p) => Rc::clone(p),
        }
    }
}

enum DRightSide {
    List(Vec<i32>),
    Pool(Rc<Pool>),
}

fn flatten_list(arg: &RuntimeValue) -> DLeftSide {
    match arg {
        RuntimeValue::Int(i) => DLeftSide::Int(*i),
        RuntimeValue::List(list) => DLeftSide::Int(list.iter().sum()),
        RuntimeValue::Pool(d) => DLeftSide::Pool(Rc::clone(d)),
    }
}

/// Executes the unary or binary version of the _n_ `d` _m_ operator.
fn make_d(left: Option<&RuntimeValue>, right: &RuntimeValue) -> RuntimeValue {
    let repeat = match left {
        Some(RuntimeValue::Int(i)) => DLeftSide::Int(*i),
        Some(RuntimeValue::List(list)) => DLeftSide::Int(list.iter().sum()),
        Some(RuntimeValue::Pool(d)) => DLeftSide::Pool(Rc::clone(d)),
        None => DLeftSide::Int(1),
    };
    let right = match right {
        RuntimeValue::Int(sides) => {
            if *sides > 0 {
                DRightSide::List((1..=*sides).collect())
            } else if *sides == 0 {
                DRightSide::List(vec![0])
            } else {
                DRightSide::List((*sides..=-1).collect())
            }
        }
        RuntimeValue::List(list) => DRightSide::List(Rc::clone(list).to_vec()),
        RuntimeValue::Pool(d) => DRightSide::Pool(Rc::clone(d)),
    };
    match (repeat, right) {
        (DLeftSide::Int(i), DRightSide::List(list)) => make_pool(i, list).into(),
        (DLeftSide::Int(i), DRightSide::Pool(p)) => {
            let mut new_pool = (*p).clone();
            if i < 0 {
                new_pool = new_pool.map_outcomes(|o| -o);
            }
            new_pool.set_dimension(new_pool.dimension() * i.unsigned_abs());
            new_pool.into()
        }
        (DLeftSide::Pool(left_p), right) => {
            let left_p = (*left_p).sum();
            let right = match right {
                DRightSide::List(list) => Pool::from_list(1, list),
                DRightSide::Pool(p) => (*p).sum(),
            };
            // At this point both |left_p| and |right| have a count of 1.
            // For each outcome in the left pool, sum the right pool with itself k times
            left_p
                .flat_map(|count| {
                    debug_assert_eq!(
                        count.len(),
                        1,
                        "summed distribution has count of {}",
                        count.len()
                    );
                    let mut dup_right = right.clone();
                    let multiplier = count[0];
                    if multiplier < 0 {
                        dup_right = dup_right.map_outcomes(|o| -o);
                    }
                    dup_right.set_dimension(dup_right.dimension() * multiplier.unsigned_abs());
                    dup_right.sum().into()
                })
                .into()
        }
    }
}

fn make_pool(mut n: i32, mut sides: Vec<i32>) -> Pool {
    if n < 0 {
        for side in sides.iter_mut() {
            *side = -*side;
        }
        n = -n;
    }
    if sides.is_empty() {
        n = 0;
    }
    Pool::from_list(u32::try_from(n).expect("n is positive"), sides)
}

fn coerce_arg(
    arg: RuntimeValue,
    expected: Option<StaticType>,
) -> Result<RuntimeValue, RuntimeError> {
    match (arg, expected) {
        (arg, None) => Ok(arg),
        (arg @ RuntimeValue::Int(_), Some(StaticType::Int)) => Ok(arg),
        (RuntimeValue::Int(i), Some(StaticType::List)) => Ok(vec![i].into()),
        (RuntimeValue::Int(i), Some(StaticType::Pool)) => Ok(Pool::from_list(1, vec![i]).into()),
        (RuntimeValue::List(lst), Some(StaticType::Int)) => Ok(lst.iter().sum::<i32>().into()),
        (list @ RuntimeValue::List(_), Some(StaticType::List)) => Ok(list),
        (RuntimeValue::List(lst), Some(StaticType::Pool)) => {
            Ok(Pool::from_list(1, (*lst).clone()).into())
        }
        (RuntimeValue::Pool(p), Some(StaticType::Int)) => Ok(p.sum().into()),
        (pool @ RuntimeValue::Pool(_), _) => Ok(pool),
    }
}

impl From<ast::Range> for SourceSpan {
    fn from(range: ast::Range) -> Self {
        SourceSpan::new(range.start.into(), range.end - range.start)
    }
}

fn interpolate_variable_names(
    template: &WithRange<String>,
    vars: &RcValEnv,
) -> Result<String, RuntimeError> {
    let mut result = String::new();
    let mut char_iter = template.value.chars();
    while let Some(chr) = char_iter.next() {
        if chr == '[' {
            let mut valid_end = false;
            let mut name = String::new();
            for chr in char_iter.by_ref() {
                if chr == ']' {
                    valid_end = true;
                    break;
                }
                name.push(chr);
            }
            if !valid_end || name.is_empty() || name.chars().any(|c| !c.is_ascii_uppercase()) {
                result.push('[');
                result.push_str(&name);
                if valid_end {
                    result.push(']');
                }
            } else if let Some(value) = vars.borrow().get(&name) {
                result.push_str(&value.to_string());
            } else {
                return Err(RuntimeError::UndefinedReference {
                    range: template.range.into(),
                    name,
                });
            }
        } else {
            result.push(chr);
        }
    }
    Ok(result)
}

fn reverse_if(should_reverse: bool, v: &[i32]) -> Vec<i32> {
    if should_reverse {
        v.iter().copied().rev().collect()
    } else {
        v.into()
    }
}

#[derive(Debug, Error, Diagnostic)]
#[error("Runtime error")]
pub enum RuntimeError {
    #[error("Output statement inside a function")]
    #[diagnostic(help("Output statements can only appear outside functions."))]
    OutputNotAtTopLevel {
        #[label = "Output statement inside a function"]
        range: SourceSpan,
    },

    #[error("Set statement inside a function")]
    #[diagnostic(help("Set statements can only appear outside functions."))]
    SetNotAtTopLevel {
        #[label = "Set statement inside a function"]
        range: SourceSpan,
    },

    #[error("Return statxement outside a function")]
    ReturnOutsideFunction {
        #[label = "Return statement outside a function"]
        range: SourceSpan,
    },

    #[error("Loops must iterate over sequences")]
    LoopOverNonSequence {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: StaticType,
    },

    #[error("Reference to undefined to variable [{name}]")]
    UndefinedReference {
        #[label = "Variable not defined"]
        range: SourceSpan,
        name: String,
    },

    #[error("Reference to undefined function [{name}]")]
    UndefinedFunction {
        #[label = "No function named [{name}]"]
        range: SourceSpan,
        name: String,
    },

    #[error("Conditions to `if` statements must be numbers.")]
    InvalidCondition {
        #[label = "This is a {found} with value {value}."]
        range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("Both sides of a range constructor must evaluate to numbers.")]
    RangeHasNonSequenceEndpoints {
        #[label = "This is a {found}."]
        range: SourceSpan,
        found: StaticType,
    },

    #[error("Invalid argument to operator")]
    InvalidArgumentToOperator {
        #[label = "Operator {op} expects {expected}."]
        operator_range: SourceSpan,
        op: BinaryOp,
        expected: &'static str,

        // TODO - show value
        #[label = "This is a {found}."]
        found_range: SourceSpan,
        found: StaticType,
    },

    #[error("Invalid argument to function")]
    NegativeArgumentToFunction {
        #[label = "Function [{name}] expects a nonnegative integer."]
        range: SourceSpan,
        name: String,

        #[label = "This evaluates to {value}."]
        found_range: SourceSpan,
        value: i32,
    },

    #[error("Invalid repeat expression")]
    #[diagnostic(help("The expression inside the repeat operator must evaluate to an int."))]
    InvalidRepeatExpression {
        #[label = "This is a {found}: {value}."]
        range: SourceSpan,
        found: StaticType,
        value: RuntimeValue,
    },

    #[error("A mathematical error occurred.")]
    MathError {
        #[label = "{message}"]
        range: SourceSpan,
        message: String,
    },
}

impl RuntimeError {
    pub fn range(&self) -> ast::Range {
        match self {
            RuntimeError::OutputNotAtTopLevel { range } => range.into(),
            RuntimeError::SetNotAtTopLevel { range } => range.into(),
            RuntimeError::ReturnOutsideFunction { range } => range.into(),
            RuntimeError::LoopOverNonSequence { range, .. } => range.into(),
            RuntimeError::UndefinedReference { range, .. } => range.into(),
            RuntimeError::UndefinedFunction { range, .. } => range.into(),
            RuntimeError::InvalidCondition { range, .. } => range.into(),
            RuntimeError::RangeHasNonSequenceEndpoints { range, .. } => range.into(),
            RuntimeError::InvalidArgumentToOperator { operator_range, .. } => operator_range.into(),
            RuntimeError::NegativeArgumentToFunction { range, .. } => range.into(),
            RuntimeError::InvalidRepeatExpression { range, .. } => range.into(),
            RuntimeError::MathError { range, .. } => range.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_variable_names() {
        fn ranged(s: &str) -> WithRange<String> {
            WithRange {
                value: s.to_string(),
                range: ast::Range {
                    start: 0,
                    end: s.len(),
                },
            }
        }

        let env = Rc::new(RefCell::new(ValEnv::new()));
        env.borrow_mut().insert("A".to_string(), 1.into());
        env.borrow_mut().insert("B".to_string(), 2.into());
        env.borrow_mut().insert("C".to_string(), 3.into());
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B]"), &env).unwrap(),
            "A + B = 1 + 2"
        );
        assert!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [INVALID]"), &env).is_err()
        );
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [invalid]"), &env).unwrap(),
            "A + B = 1 + 2 + [invalid]"
        );
        assert_eq!(
            interpolate_variable_names(&ranged("A + B = [A] + [B] + [C"), &env).unwrap(),
            "A + B = 1 + 2 + [C"
        );
    }

    #[test]
    fn test_division_by_zero() {
        use crate::ast::{BinaryOp, Range};

        let left = RuntimeValue::Int(5);
        let right = RuntimeValue::Int(0);
        let op = WithRange {
            value: BinaryOp::Div,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Cannot divide 5 by zero"));
            }
            _ => panic!("Expected MathError for division by zero"),
        }
    }

    #[test]
    fn test_negative_exponentiation() {
        use crate::ast::{BinaryOp, Range};

        let left = RuntimeValue::Int(2);
        let right = RuntimeValue::Int(-3);
        let op = WithRange {
            value: BinaryOp::Pow,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Cannot raise 2 to negative power -3"));
            }
            _ => panic!("Expected MathError for negative exponentiation"),
        }
    }

    #[test]
    fn test_valid_math_operations() {
        use crate::ast::{BinaryOp, Range};

        // Test valid division
        let left = RuntimeValue::Int(10);
        let right = RuntimeValue::Int(2);
        let op = WithRange {
            value: BinaryOp::Div,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Ok(RuntimeValue::Int(5)) => {}
            _ => panic!("Expected successful division result of 5"),
        }

        // Test valid exponentiation
        let left = RuntimeValue::Int(2);
        let right = RuntimeValue::Int(3);
        let op = WithRange {
            value: BinaryOp::Pow,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Ok(RuntimeValue::Int(8)) => {}
            _ => panic!("Expected successful exponentiation result of 8"),
        }
    }

    #[test]
    fn test_addition_overflow() {
        use crate::ast::{BinaryOp, Range};

        let left = RuntimeValue::Int(i32::MAX);
        let right = RuntimeValue::Int(1);
        let op = WithRange {
            value: BinaryOp::Add,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Addition overflow"));
                assert!(message.contains(&format!("{}", i32::MAX)));
            }
            _ => panic!("Expected MathError for addition overflow"),
        }
    }

    #[test]
    fn test_subtraction_overflow() {
        use crate::ast::{BinaryOp, Range};

        let left = RuntimeValue::Int(i32::MIN);
        let right = RuntimeValue::Int(1);
        let op = WithRange {
            value: BinaryOp::Sub,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Subtraction overflow"));
                assert!(message.contains(&format!("{}", i32::MIN)));
            }
            _ => panic!("Expected MathError for subtraction overflow"),
        }
    }

    #[test]
    fn test_multiplication_overflow() {
        use crate::ast::{BinaryOp, Range};

        let left = RuntimeValue::Int(i32::MAX);
        let right = RuntimeValue::Int(2);
        let op = WithRange {
            value: BinaryOp::Mul,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Multiplication overflow"));
                assert!(message.contains(&format!("{}", i32::MAX)));
            }
            _ => panic!("Expected MathError for multiplication overflow"),
        }
    }

    #[test]
    fn test_power_overflow() {
        use crate::ast::{BinaryOp, Range};

        let left = RuntimeValue::Int(2);
        let right = RuntimeValue::Int(32); // 2^32 overflows i32
        let op = WithRange {
            value: BinaryOp::Pow,
            range: Range { start: 0, end: 1 },
        };

        let result = apply_binary_op(&op, &left, Range { start: 0, end: 1 }, &right, false);
        match result {
            Err(RuntimeError::MathError { message, .. }) => {
                assert!(message.contains("Power overflow"));
                assert!(message.contains("2 ^ 32"));
            }
            _ => panic!("Expected MathError for power overflow"),
        }
    }
}
