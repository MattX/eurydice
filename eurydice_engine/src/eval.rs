//! A simple tree walking interpreter.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use malachite::num::arithmetic::traits::Lcm;
use malachite::{
    num::basic::traits::{One, Zero},
    Natural, Rational,
};
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{
        self, BareListItem, BinaryOp, Expression, FunctionDefinition, ListItem, PositionOrder,
        SetParam, Statement, StaticType, UnaryOp, WithRange,
    },
    dice::{explode, MultisetCrossProductIterator, Pool},
};

#[derive(Debug, Clone)]
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

    fn map_outcomes(&self, f: impl Fn(i32) -> i32) -> Self {
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
    Primitive(Primitive),
    UserDefined(Rc<FunctionDefinition>),
}

impl Function {
    fn get_arg_types(&self) -> Vec<Option<StaticType>> {
        match self {
            Function::Primitive(Primitive::Absolute) => vec![Some(StaticType::Int)],
            Function::Primitive(Primitive::Contains) => {
                vec![Some(StaticType::List), Some(StaticType::Int)]
            }
            Function::Primitive(Primitive::Count) => {
                vec![Some(StaticType::List), Some(StaticType::List)]
            }
            Function::Primitive(Primitive::Explode) => vec![Some(StaticType::Pool)],
            Function::Primitive(Primitive::Highest) => {
                vec![Some(StaticType::Int), Some(StaticType::Pool)]
            }
            Function::Primitive(Primitive::Lowest) => {
                vec![Some(StaticType::Int), Some(StaticType::Pool)]
            }
            Function::Primitive(Primitive::Middle) => {
                vec![Some(StaticType::Int), Some(StaticType::Pool)]
            }
            Function::Primitive(Primitive::HighestOf) => {
                vec![Some(StaticType::Int), Some(StaticType::Int)]
            }
            Function::Primitive(Primitive::LowestOf) => {
                vec![Some(StaticType::Int), Some(StaticType::Int)]
            }
            Function::Primitive(Primitive::Maximum) => vec![Some(StaticType::Pool)],
            Function::Primitive(Primitive::Reverse) => vec![Some(StaticType::List)],
            Function::Primitive(Primitive::Sort) => vec![Some(StaticType::List)],
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
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator {
    pub fn new() -> Self {
        let mut functions = HashMap::new();
        functions.insert(
            "absolute {}".to_string(),
            Function::Primitive(Primitive::Absolute),
        );
        functions.insert(
            "{} contains {}".to_string(),
            Function::Primitive(Primitive::Contains),
        );
        functions.insert(
            "count {} in {}".to_string(),
            Function::Primitive(Primitive::Count),
        );
        functions.insert(
            "explode {}".to_string(),
            Function::Primitive(Primitive::Explode),
        );
        functions.insert(
            "highest {} of {}".to_string(),
            Function::Primitive(Primitive::Highest),
        );
        functions.insert(
            "lowest {} of {}".to_string(),
            Function::Primitive(Primitive::Lowest),
        );
        functions.insert(
            "middle {} of {}".to_string(),
            Function::Primitive(Primitive::Middle),
        );
        functions.insert(
            "highest of {} and {}".to_string(),
            Function::Primitive(Primitive::HighestOf),
        );
        functions.insert(
            "lowest of {} and {}".to_string(),
            Function::Primitive(Primitive::LowestOf),
        );
        functions.insert(
            "maximum of {}".to_string(),
            Function::Primitive(Primitive::Maximum),
        );
        functions.insert(
            "reverse {}".to_string(),
            Function::Primitive(Primitive::Reverse),
        );
        functions.insert("sort {}".to_string(), Function::Primitive(Primitive::Sort));
        Self {
            global_env: Rc::new(RefCell::new(ValEnv::new())),
            outputs: Vec::new(),
            functions,
            explode_depth: 2,
            recursion_depth: 10,
            lowest_first: false,
        }
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
            Statement::Output { value, named } => {
                if eval_context.recursion_depth != 0 {
                    return Err(RuntimeError::OutputNotAtTopLevel {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, value)?;
                if let Some(name) = named {
                    self.outputs.push((
                        value.clone(),
                        interpolate_variable_names(name, &eval_context.env)?,
                    ));
                } else {
                    self.outputs
                        .push((value, format!("output {}", self.outputs.len() + 1)));
                }
            }
            // TODO handle named prints
            Statement::Print { expr, .. } => {
                let value = self.evaluate(eval_context, expr)?;
                println!("{}", value);
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
                apply_binary_op(op, &left_value, left.range, &right_value, self.lowest_first)
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
            results.push((self.call_function(eval_context, function, &args, &arg_ranges)?, weight));
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
            Function::Primitive(primitive) => {
                primitive.execute(args, arg_ranges, self.explode_depth, function.range)
            }
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
            RuntimeValue::Int(i) => (i.to_string().len() as i32).into(),
            RuntimeValue::List(list) => (list.len() as i32).into(),
            RuntimeValue::Pool(d) => ((*d).dimension() as i32).into(),
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
                        .map(|c| c.to_digit(10).unwrap() as i32 * i.signum())
                        .collect();
                    Ok(select_positions(&left, &digits, lowest_first).into())
                }
                RuntimeValue::List(lst) => Ok(select_positions(&left, lst, false).into()),
                RuntimeValue::Pool(p) => {
                    Ok(select_in_dice(&left, (**p).clone(), lowest_first).into())
                }
            }
        }
        // TODO should report an error if b < 0
        BinaryOp::Pow => Ok(math_binary_op(left, right, |a, b| a.pow(b.unsigned_abs()))),
        BinaryOp::Add => Ok(math_binary_op(left, right, |a, b| a + b)),
        BinaryOp::Sub => Ok(math_binary_op(left, right, |a, b| a - b)),
        BinaryOp::Mul => Ok(math_binary_op(left, right, |a, b| a * b)),
        BinaryOp::Div => Ok(math_binary_op(left, right, |a, b| a / b)),
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
        BinaryOp::Or => Ok(math_binary_op(left, right, |a, b| {
            if a != 0 || b != 0 {
                1
            } else {
                0
            }
        })),
        BinaryOp::And => Ok(math_binary_op(left, right, |a, b| {
            if a != 0 && b != 0 {
                1
            } else {
                0
            }
        })),
    }
}

fn select_positions(indices: &[i32], vec: &[i32], lowest_first: bool) -> i32 {
    indices
        .iter()
        .map(|&i| {
            if i < 1 || i > vec.len() as i32 {
                return 0;
            }
            let i = if lowest_first {
                vec.len() - i as usize
            } else {
                i as usize - 1
            };
            vec.get(i).copied().unwrap_or(0)
        })
        .sum()
}

fn select_in_dice(indices: &[i32], pool: Pool, lowest_first: bool) -> Pool {
    let size = pool.dimension() as usize;
    let mut keep_list = vec![false; size];
    for &i in indices {
        if i < 1 || i > size as i32 {
            continue;
        }
        let i = if !lowest_first {
            size - i as usize
        } else {
            i as usize - 1
        };
        keep_list[i] = true;
    }
    pool.sum_with_keep_list(&keep_list)
}

fn math_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    f: impl Fn(i32, i32) -> i32,
) -> RuntimeValue {
    let left = flatten_list(left);
    let right = flatten_list(right);
    let (left_pool, right_pool) = match (left, right) {
        (DLeftSide::Int(a), DLeftSide::Int(b)) => return f(a, b).into(),
        (left, right) => (left.to_pool().sum(), right.to_pool().sum()),
    };
    left_pool
        .flat_map(|left_outcome| {
            right_pool
                .clone()
                .map_outcomes(|right_outcome| f(left_outcome[0], right_outcome))
                .into()
        })
        .into()
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
        RuntimeValue::Int(sides) => DRightSide::List((1..=*sides).collect()),
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
    Pool::from_list(n as u32, sides)
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

#[derive(Debug, Clone, Copy)]
pub enum Primitive {
    Absolute,
    Contains,
    Count,
    Explode,
    Highest,
    Lowest,
    Middle,
    HighestOf,
    LowestOf,
    Maximum,
    Reverse,
    Sort,
}

impl Primitive {
    fn execute(
        self,
        args: &[RuntimeValue],
        arg_ranges: &[ast::Range],
        explode_depth: usize,
        function_range: ast::Range,
    ) -> Result<RuntimeValue, RuntimeError> {
        match self {
            Primitive::Absolute => {
                // arg: int
                let arg = &args[0];
                Ok(arg.map_outcomes(|o| o.abs()))
            }
            Primitive::Contains => {
                // args: list, int
                if let (RuntimeValue::List(haystack), RuntimeValue::Int(needle)) =
                    (&args[0], &args[1])
                {
                    Ok(if haystack.iter().any(|h| h == needle) {
                        1.into()
                    } else {
                        0.into()
                    })
                } else {
                    panic!("wrong argument types to [contains]");
                }
            }
            Primitive::Count => {
                // args: list, list
                if let (RuntimeValue::List(needle), RuntimeValue::List(haystack)) =
                    (&args[0], &args[1])
                {
                    let mut needle_map = HashMap::new();
                    for n in needle.iter() {
                        *needle_map.entry(n).or_insert(0) += 1;
                    }
                    Ok(haystack
                        .iter()
                        .map(|item| needle_map.get(item).copied().unwrap_or(0))
                        .sum::<i32>()
                        .into())
                } else {
                    panic!("wrong argument types to [count]");
                }
            }
            Primitive::Explode => {
                // args: pool
                if let RuntimeValue::Pool(d) = &args[0] {
                    let die: Vec<_> = (*d).clone().sum().into_die_iter().collect();
                    let highest_value = die.last().unwrap().0;
                    Ok(Pool::from(explode(die, &[highest_value], explode_depth)).into())
                } else {
                    panic!("wrong argument types to [explode]");
                }
            }
            Primitive::Highest | Primitive::Middle | Primitive::Lowest => {
                // args: int, pool
                if let (RuntimeValue::Int(i), RuntimeValue::Pool(d)) = (&args[0], &args[1]) {
                    let keep_list = keep_list_for_primitive(
                        self,
                        *i,
                        arg_ranges[0],
                        d.dimension() as usize,
                        function_range,
                    )?;
                    Ok(d.sum_with_keep_list(&keep_list).into())
                } else {
                    panic!("wrong argument types to [highest/lowest/middle]");
                }
            }
            Primitive::HighestOf => {
                // args: int, int
                if let (RuntimeValue::Int(i), RuntimeValue::Int(j)) = (&args[0], &args[1]) {
                    Ok((*(i.max(j))).into())
                } else {
                    panic!("wrong argument types to [highest of]");
                }
            }
            Primitive::LowestOf => {
                // args: int, int
                if let (RuntimeValue::Int(i), RuntimeValue::Int(j)) = (&args[0], &args[1]) {
                    Ok((*(i.min(j))).into())
                } else {
                    panic!("wrong argument types to [lowest of]");
                }
            }
            Primitive::Maximum => {
                // args: pool
                if let RuntimeValue::Pool(d) = &args[0] {
                    Ok(d.sum()
                        .ordered_outcomes()
                        .last()
                        .map(|(o, _)| *o)
                        .unwrap_or(0)
                        .into())
                } else {
                    panic!("wrong argument types to [maximum]");
                }
            }
            Primitive::Reverse => {
                // args: sequence
                if let RuntimeValue::List(lst) = &args[0] {
                    Ok(lst.iter().rev().copied().collect::<Vec<_>>().into())
                } else {
                    panic!("wrong argument types to [reverse]");
                }
            }
            Primitive::Sort => {
                // args: sequence
                if let RuntimeValue::List(lst) = &args[0] {
                    let mut lst = (**lst).clone();
                    lst.sort_unstable_by_key(|o| -o);
                    Ok(lst.into())
                } else {
                    panic!("wrong argument types to [sort]");
                }
            }
        }
    }
}

fn keep_list_for_primitive(
    primitive: Primitive,
    keep: i32,
    keep_range: ast::Range,
    outcomes_size: usize,
    function_range: ast::Range,
) -> Result<Vec<bool>, RuntimeError> {
    if keep < 0 {
        return Err(RuntimeError::NegativeArgumentToFunction {
            range: function_range.into(),
            name: match primitive {
                Primitive::Highest => "highest".to_string(),
                Primitive::Lowest => "lowest".to_string(),
                Primitive::Middle => "middle".to_string(),
                _ => unreachable!(),
            },
            found_range: keep_range.into(),
            value: keep,
        });
    }
    let mut keep_list = vec![false; outcomes_size];
    if keep == 0 {
        return Ok(keep_list);
    }
    match primitive {
        Primitive::Highest => {
            keep_list[outcomes_size - keep as usize..].fill(true);
        }
        Primitive::Lowest => {
            keep_list[..keep as usize].fill(true);
        }
        Primitive::Middle => {
            // This is rounding down
            let start = (outcomes_size - keep as usize).div_ceil(2);
            keep_list[start..start + keep as usize].fill(true);
        }
        _ => unreachable!(),
    }
    Ok(keep_list)
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
            if !valid_end {
                result.push('[');
                result.push_str(&name);
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
        v.into_iter().copied().rev().collect()
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

    #[error("Not yet implemented")]
    #[diagnostic(help("The developer is a lazy bum."))]
    NotYetImplemented {
        #[label = "Not yet implemented"]
        range: SourceSpan,
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
}

#[derive(Debug, Error, Diagnostic)]
pub enum PrimitiveError {}

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
        env.borrow_mut().insert("a".to_string(), 1.into());
        env.borrow_mut().insert("b".to_string(), 2.into());
        env.borrow_mut().insert("c".to_string(), 3.into());
        assert_eq!(
            interpolate_variable_names(&ranged("a + b = [a] + [b]"), &env).unwrap(),
            "a + b = 1 + 2"
        );
        assert!(
            interpolate_variable_names(&ranged("a + b = [a] + [b] + [invalid]"), &env,).is_err()
        );
        assert_eq!(
            interpolate_variable_names(&ranged("a + b = [a] + [b] + [c"), &env,).unwrap(),
            "a + b = 1 + 2 + [c"
        );
    }
}
