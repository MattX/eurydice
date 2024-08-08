//! A simple tree walking interpreter.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use malachite::{
    num::basic::traits::{One, Zero},
    Natural,
};
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::{
    ast::{
        self, BinaryOp, Expression, FunctionDefinition, ListLiteralItem, StaticType, UnaryOp,
        WithRange,
    },
    dice::{explode, Pool},
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
            RuntimeValue::Pool(d) => (**d)
                .sum()
                .into_iter()
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
    env: HashMap<String, (RuntimeValue, ast::Range)>,
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
            Some(value.0.clone())
        } else if let Some(parent) = &self.parent {
            parent.borrow().get(key)
        } else {
            None
        }
    }

    fn get_range(&self, key: &str) -> Option<ast::Range> {
        if let Some(value) = self.env.get(key) {
            Some(value.1)
        } else if let Some(parent) = &self.parent {
            parent.borrow().get_range(key)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, value: RuntimeValue, range: ast::Range) {
        self.env.insert(key, (value, range));
    }
}

#[derive(Debug, Clone)]
pub enum Function {
    Primitive(Primitive),
    UserDefined(Rc<FunctionDefinition>),
}

impl Function {
    fn get_arg_types(&self) -> Vec<StaticType> {
        match self {
            Function::Primitive(Primitive::Absolute) => vec![StaticType::Int],
            Function::Primitive(Primitive::Contains) => vec![StaticType::List, StaticType::Int],
            Function::Primitive(Primitive::Count) => vec![StaticType::List, StaticType::List],
            Function::Primitive(Primitive::Explode) => vec![StaticType::Pool],
            Function::Primitive(Primitive::Highest) => vec![StaticType::Int, StaticType::Pool],
            Function::Primitive(Primitive::Lowest) => vec![StaticType::Int, StaticType::Pool],
            Function::Primitive(Primitive::Middle) => vec![StaticType::Int, StaticType::Pool],
            Function::Primitive(Primitive::HighestOf) => vec![StaticType::Int, StaticType::Int],
            Function::Primitive(Primitive::LowestOf) => vec![StaticType::Int, StaticType::Int],
            Function::Primitive(Primitive::Maximum) => vec![StaticType::Pool],
            Function::Primitive(Primitive::Reverse) => vec![StaticType::List],
            Function::Primitive(Primitive::Sort) => vec![StaticType::List],
            Function::UserDefined(fd) => fd.args.iter().map(|arg| arg.value.1).collect(),
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
        functions.insert("absolute {}".to_string(), Function::Primitive(Primitive::Absolute));
        functions.insert("{} contains {}".to_string(), Function::Primitive(Primitive::Contains));
        functions.insert("count {} in {}".to_string(), Function::Primitive(Primitive::Count));
        functions.insert("explode {}".to_string(), Function::Primitive(Primitive::Explode));
        functions.insert("highest {} of {}".to_string(), Function::Primitive(Primitive::Highest));
        functions.insert("lowest {} of {}".to_string(), Function::Primitive(Primitive::Lowest));
        functions.insert("middle {} of {}".to_string(), Function::Primitive(Primitive::Middle));
        functions.insert(
            "highest of {} and {}".to_string(),
            Function::Primitive(Primitive::HighestOf),
        );
        functions.insert(
            "lowest of {} and {}".to_string(),
            Function::Primitive(Primitive::LowestOf),
        );
        functions.insert("maximum of {}".to_string(), Function::Primitive(Primitive::Maximum));
        functions.insert("reverse {}".to_string(), Function::Primitive(Primitive::Reverse));
        functions.insert("sort {}".to_string(), Function::Primitive(Primitive::Sort));
        Self {
            global_env: Rc::new(RefCell::new(ValEnv::new())),
            outputs: Vec::new(),
            functions: functions,
            explode_depth: 2,
            recursion_depth: 10,
            lowest_first: false,
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

    pub fn get_outputs(&mut self) -> Vec<(RuntimeValue, String)> {
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
                    .insert(name.value.clone(), value, name.range);
            }
            ast::Statement::FunctionDefinition(fd) => {
                self.functions.insert(
                    fd.name.value.clone(),
                    Function::UserDefined(Rc::new(fd.clone())),
                );
            }
            ast::Statement::Output { value, named } => {
                if eval_context.recursion_depth != 0 {
                    return Err(RuntimeError::OutputNotAtTopLevel {
                        range: statement.range.into(),
                    });
                }
                let value = self.evaluate(eval_context, value)?;
                self.outputs.push((
                    value,
                    named
                        .as_ref()
                        .map(|v| v.value.clone())
                        .unwrap_or_else(|| format!("output {}", self.outputs.len() + 1)),
                ));
            }
            ast::Statement::Return { value } => {
                if self.recursion_depth == 0 {
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
                    eval_context.env.borrow_mut().insert(
                        variable.value.clone(),
                        RuntimeValue::Int(*value),
                        variable.range,
                    );
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

        // For any argument where the provided argument is a pool, but the expected argument is an int or a list,
        // create an iterator over all possible outcomes of the pool.
        // |pool_iterators| contains triples of (iterator, index, expected_type == int).
        // pool_iterators[i].index = k if the i-th element of pool_iterators corresponds to the k-th argument.
        let mut pool_iterators = Vec::new();
        for (i, (arg, expected_type)) in args.iter().zip(expected_types).enumerate() {
            match (&arg, expected_type) {
                (RuntimeValue::Pool(p), StaticType::Int | StaticType::List) => {
                    // If the expected type is an Int, `coerce_arg` has already turned the pool into a sum.
                    pool_iterators.push((
                        p.outcome_iterator(),
                        i,
                        expected_type == StaticType::Int,
                    ));
                }
                _ => {}
            }
        }

        let mut results = Vec::new();
        let mut args = args.clone();
        // This loop executes exactly once if there are no pool iterators.
        loop {
            let mut weight = Natural::ONE;
            let mut finished = !pool_iterators.is_empty();

            // Advance the pool iterators.
            for (pool_iterator, arg_index, is_int) in pool_iterators.iter_mut() {
                if let Some((outcome, outcome_weight)) = pool_iterator.next() {
                    args[*arg_index] = if *is_int {
                        outcome[0].into()
                    } else {
                        outcome.to_vec().into()
                    };
                    weight *= outcome_weight;
                    finished = false;
                    break;
                } else {
                    pool_iterator.reset();
                    if let Some((outcome, outcome_weight)) = pool_iterator.next() {
                        args[*arg_index] = if *is_int {
                            outcome[0].into()
                        } else {
                            outcome.to_vec().into()
                        };
                        weight *= outcome_weight;
                    } else {
                        // This pool iterator didn't return a value after being reset, so it's empty.
                        // The function doesn't get called at all, return an empty list.
                        return Ok(vec![].into());
                    }
                }
            }
            // If we have pool iterators, but all of them reached the end and have to be reset, break.
            if finished {
                break;
            }

            let current_result = match &function.value {
                Function::Primitive(primitive) => {
                    primitive.execute(&args, &arg_ranges, self.explode_depth, function.range)?
                }
                Function::UserDefined(user_function) => {
                    let mut new_env = ValEnv::with_parent(Rc::clone(&eval_context.env));
                    for (arg, formal) in args.iter().zip(user_function.args.iter()) {
                        new_env.insert(formal.value.0.clone(), arg.clone(), formal.range);
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
                    result.unwrap_or(vec![].into())
                }
            };
            results.push((current_result, weight));
            // If there are no pool iterators, run the loop only once.
            if pool_iterators.is_empty() {
                break;
            }
        }

        // If there were any pool iterators, collect the results into a dice
        // Otherwise, there is exactly one result, which we should return as is
        if pool_iterators.is_empty() {
            debug_assert!(results.len() == 1);
            Ok(results.pop().unwrap().0)
        } else {
            let mut total_results = BTreeMap::<i32, Natural>::new();
            for (result, weight) in results {
                for (outcome, count) in result.to_pool().sum().ordered_outcomes() {
                    *total_results.entry(*outcome).or_insert(Natural::ZERO) += count * &weight;
                }
            }
            Ok(total_results.into_iter().collect::<Pool>().into())
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
            RuntimeValue::Pool(d) => ((*d).get_n() as i32).into(),
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
                        .to_string()
                        .chars()
                        .map(|c| c.to_digit(10).unwrap() as i32)
                        .collect();
                    Ok(select_positions(&left, &digits, lowest_first).into())
                }
                RuntimeValue::List(lst) => Ok(select_positions(&left, lst, false).into()),
                RuntimeValue::Pool(p) => {
                    Ok(select_in_dice(&left, (**p).clone(), lowest_first).into())
                }
            }
        }
        BinaryOp::Pow => Ok(math_binary_op(left, right, |a, b| a.pow(b as u32))),
        BinaryOp::Add => Ok(math_binary_op(left, right, |a, b| a + b)),
        BinaryOp::Sub => Ok(math_binary_op(left, right, |a, b| a - b)),
        BinaryOp::Mul => Ok(math_binary_op(left, right, |a, b| a * b)),
        BinaryOp::Div => Ok(math_binary_op(left, right, |a, b| a / b)),
        BinaryOp::Eq => Ok(math_binary_op(
            left,
            right,
            |a, b| if a == b { 1 } else { 0 },
        )),
        BinaryOp::Ne => Ok(math_binary_op(
            left,
            right,
            |a, b| if a != b { 1 } else { 0 },
        )),
        BinaryOp::Lt => Ok(math_binary_op(
            left,
            right,
            |a, b| if a < b { 1 } else { 0 },
        )),
        BinaryOp::Le => Ok(math_binary_op(
            left,
            right,
            |a, b| if a <= b { 1 } else { 0 },
        )),
        BinaryOp::Gt => Ok(math_binary_op(
            left,
            right,
            |a, b| if a > b { 1 } else { 0 },
        )),
        BinaryOp::Ge => Ok(math_binary_op(
            left,
            right,
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
            if i < 0 || i >= vec.len() as i32 {
                return 0;
            }
            let i = if !lowest_first {
                vec.len() - i as usize - 1
            } else {
                i as usize
            };
            vec.get(i).copied().unwrap_or(0)
        })
        .sum()
}

fn select_in_dice(indices: &[i32], mut pool: Pool, lowest_first: bool) -> Pool {
    let size = pool.get_n() as usize;
    let mut keep_list = vec![false; size];
    for &i in indices {
        if let Ok(mut i) = usize::try_from(i) {
            if lowest_first {
                i = size - i - 1;
            }
            keep_list[i] = true;
        }
    }
    pool.set_keep_list(keep_list);
    pool.sum()
}

fn math_binary_op(
    left: &RuntimeValue,
    right: &RuntimeValue,
    f: impl Fn(i32, i32) -> i32,
) -> RuntimeValue {
    let left = convert_arg(left);
    let right = convert_arg(right);
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

fn convert_arg(arg: &RuntimeValue) -> DLeftSide {
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
            new_pool.set_n(new_pool.get_n() * i.unsigned_abs());
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
                    let mut dup_right = right.clone();
                    dup_right.set_n(count[0] as u32);
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

fn coerce_arg(arg: RuntimeValue, expected: StaticType) -> Result<RuntimeValue, RuntimeError> {
    match (arg, expected) {
        (arg @ RuntimeValue::Int(_), StaticType::Int) => Ok(arg),
        (RuntimeValue::Int(i), StaticType::List) => Ok(vec![i].into()),
        (RuntimeValue::Int(i), StaticType::Pool) => Ok(Pool::from_list(1, vec![i]).into()),
        (RuntimeValue::List(lst), StaticType::Int) => Ok(lst.iter().sum::<i32>().into()),
        (list @ RuntimeValue::List(_), StaticType::List) => Ok(list),
        (RuntimeValue::List(lst), StaticType::Pool) => {
            Ok(Pool::from_list(1, (*lst).clone()).into())
        }
        (RuntimeValue::Pool(p), StaticType::Int) => Ok(p.sum().into()),
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
                if let (RuntimeValue::List(needle), RuntimeValue::List(haystack)) = (&args[0], &args[1]) {
                    let mut needle_map = HashMap::new();
                    for n in needle.iter() {
                        *needle_map.entry(n).or_insert(0) += 1;
                    }
                    Ok(haystack.iter().map(|item| needle_map.get(item).copied().unwrap_or(0)).sum::<i32>().into())
                } else {
                    panic!("wrong argument types to [count]");
                }
            }
            Primitive::Explode => {
                // args: pool
                if let RuntimeValue::Pool(d) = &args[0] {
                    let die: Vec<_> = (*d).clone().sum().into_iter().collect();
                    let highest_value = die.last().unwrap().0;
                    Ok(Pool::from_die(explode(die, &[highest_value], explode_depth)).into())
                } else {
                    panic!("wrong argument types to [explode]");
                }
            }
            Primitive::Highest => {
                // args: int, pool
                if let (RuntimeValue::Int(i), RuntimeValue::Pool(d)) = (&args[0], &args[1]) {
                    if *i < 0 {
                        return Err(RuntimeError::NegativeArgumentToFunction {
                            range: function_range.into(),
                            name: "highest".to_string(),
                            found_range: arg_ranges[0].into(),
                            value: *i,
                        });
                    } else if *i == 0 {
                        return Ok(Pool::from_list(1, vec![0]).into());
                    }
                    let mut d = (**d).clone();
                    let mut keep_list = vec![false; d.get_n() as usize];
                    keep_list[d.get_n() as usize - *i as usize..].fill(true);
                    d.set_keep_list(keep_list);
                    Ok(d.sum().into())
                } else {
                    panic!("wrong argument types to [highest]");
                }
            }
            Primitive::Lowest => {
                Err(RuntimeError::NotYetImplemented {
                    range: function_range.into(),
                })
            }
            Primitive::Middle => {
                Err(RuntimeError::NotYetImplemented {
                    range: function_range.into(),
                })
            }
            Primitive::HighestOf => {
                Err(RuntimeError::NotYetImplemented {
                    range: function_range.into(),
                })
            }
            Primitive::LowestOf => {
                Err(RuntimeError::NotYetImplemented {
                    range: function_range.into(),
                })
            }
            Primitive::Maximum => {
                Err(RuntimeError::NotYetImplemented {
                    range: function_range.into(),
                })
            }
            Primitive::Reverse => {
                Err(RuntimeError::NotYetImplemented {
                    range: function_range.into(),
                })
            }
            Primitive::Sort => {
                Err(RuntimeError::NotYetImplemented {
                    range: function_range.into(),
                })
            },
        }
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
    #[error("Output statement inside a function")]
    #[diagnostic(help("Output statements can only appear outside functions."))]
    OutputNotAtTopLevel {
        #[label = "Output statement inside a function"]
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
}

#[derive(Debug, Error, Diagnostic)]
pub enum PrimitiveError {}
